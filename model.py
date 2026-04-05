import abc
import os
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import jvp, make_functional_with_buffers


class LinearizedModel(nn.Module):
    """Creates a linearized version of a nn.Module.

    The linearized version of a model is a proper PyTorch model and can be
    trained as any other nn.Module.

    Args:
        model (nn.Module): The model to linearize. The trainable parameters of
            the linearized model will be initialized to the parameters of this
            model.
        init_model (nn.Module): A model of the same type as `model` containing
            the parameters around which the model is initialized. If not
            provided, `model` is used as the initialization model.
    """

    def __init__(self, model: nn.Module, init_model: nn.Module = None) -> None:
        """Initializes the linearized model."""
        super().__init__()
        if init_model is None:
            init_model = model

        func0, params0, self.buffers0 = make_functional_with_buffers(
            init_model.eval(), disable_autograd_tracking=True
        )
        # Forward Hugging Face-style (and generic) *args, **kwargs to the module.
        self._functional = lambda params, *a, **kw: func0(
            params, self.buffers0, *a, **kw
        )

        _, params, _ = make_functional_with_buffers(
            model, disable_autograd_tracking=True
        )

        self.params = nn.ParameterList(params)
        self.params0 = nn.ParameterList(params0)
        self._model_name = model.__class__.__name__

        # The intial parameters are not trainable.
        for p in self.params0:
            p.requires_grad = False

        # The params are.
        for p in self.params:
            p.requires_grad = True

    @staticmethod
    @contextmanager
    def _sdpa_math_only() -> Any:
        """Use only the SDPA math path so ``functorch.jvp`` (forward AD) works.

        ``attn_implementation=\"sdpa\"`` still dispatches to Flash or mem-efficient
        kernels unless this is set; those ops do not implement forward AD.
        """
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel(backends=[SDPBackend.MATH]):
                yield
        except ImportError:
            # PyTorch < 2.4: prefer legacy API (still disables flash / mem-efficient).
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            ):
                yield

    @staticmethod
    def _autocast_device_type(*args: Any, **kwargs: Any) -> str:
        for obj in list(args) + list(kwargs.values()):
            if torch.is_tensor(obj):
                return obj.device.type
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _forward_to_tensor(y: Any) -> torch.Tensor:
        """``jvp`` requires tensor outputs. Hugging Face LMs return ``ModelOutput``; use logits."""
        if torch.is_tensor(y):
            return y
        logits = getattr(y, "logits", None)
        if torch.is_tensor(logits):
            return logits
        raise TypeError(
            "LinearizedModel needs a tensor forward output or a .logits field "
            f"(got {type(y).__name__})."
        )

    @staticmethod
    def _forward_to_logprobs(y: Any, dim: int = -1) -> torch.Tensor:
        """Map model output to log-probs for JVP (vocabulary dimension ``dim``)."""
        logits = LinearizedModel._forward_to_tensor(y)
        return F.log_softmax(logits / 0.6, dim=dim)

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Linearized first-order Taylor step. Passes *args, **kwargs to the inner model
        (e.g. ``input_ids``, ``attention_mask``). For Hugging Face causal LMs, logits are
        returned; ``use_cache`` is forced to ``False`` so ``jvp`` does not see caches.

        Uses ``torch.autocast`` with bfloat16 on the inferred device (from a tensor arg).
        """
        # Caches / ModelOutput fields break functorch jvp unless we only propagate logits.
        alpha = kwargs.pop("alpha", 1.0)
        kw = {**kwargs, "use_cache": False}
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        device_type = self._autocast_device_type(*args, **kw)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            with self._sdpa_math_only():
                out, dp = jvp(
                    lambda p: self._forward_to_tensor(
                        self._functional(p, *args, **kw)
                    ),
                    (tuple(self.params0),),
                    (tuple(dparams),),
                )
        return out + alpha * dp
    
    def dp(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Linearized first-order Taylor step. Passes *args, **kwargs to the inner model
        (e.g. ``input_ids``, ``attention_mask``). For Hugging Face causal LMs, logits are
        returned; ``use_cache`` is forced to ``False`` so ``jvp`` does not see caches.

        Uses ``torch.autocast`` with bfloat16 on the inferred device (from a tensor arg).
        """
        # Caches / ModelOutput fields break functorch jvp unless we only propagate logits.
        alpha = kwargs.pop("alpha", 1.0)
        kw = {**kwargs, "use_cache": False}
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        device_type = self._autocast_device_type(*args, **kw)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            with self._sdpa_math_only():
                _, dp = jvp(
                    lambda p: self._forward_to_tensor(
                        self._functional(p, *args, **kw)
                    ),
                    (tuple(self.params0),),
                    (tuple(dparams),),
                )
        return dp

    def dp_logprobs(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """First-order directional derivative (JVP) in **log-probability** space.

        Same as ``dp`` (logits JVP) but the inner map is ``log_softmax(logits)``, so this
        is the JVP of log-probs w.r.t. parameters dotted with ``p - p_0``.

        Keyword-only (popped before the inner forward): ``logprobs_dim`` (default ``-1``)
        is the vocabulary axis for ``log_softmax``.
        """
        kwargs.pop("alpha", 1.0)
        logprobs_dim = kwargs.pop("logprobs_dim", -1)
        kw = {**kwargs, "use_cache": False}
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        device_type = self._autocast_device_type(*args, **kw)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            with self._sdpa_math_only():
                _, dp = jvp(
                    lambda p: self._forward_to_logprobs(
                        self._functional(p, *args, **kw), dim=logprobs_dim
                    ),
                    (tuple(self.params0),),
                    (tuple(dparams),),
                )
        return dp


def _index_of_name(names: Tuple[str, ...], target: str) -> Optional[int]:
    try:
        return names.index(target)
    except ValueError:
        return None


def _check_ab_rank(A: torch.Tensor, B: torch.Tensor, rank: int) -> None:
    if rank not in A.shape or rank not in B.shape:
        raise ValueError(
            f"rank={rank} must appear in both LoRA factor shapes (got A {tuple(A.shape)}, "
            f"B {tuple(B.shape)})."
        )


class LoRAFactors(nn.Module):
    """One linear LoRA module: low-rank factors ``A`` and ``B`` (PEFT-style)."""

    def __init__(self, A: torch.Tensor, B: torch.Tensor) -> None:
        super().__init__()
        self.A = nn.Parameter(A.clone())
        self.B = nn.Parameter(B.clone())


class LoRAEmbeddingFactor(nn.Module):
    """Single low-rank tensor (e.g. ``lora_embedding``) when there is no A/B pair."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.clone())


class LinearizedLowRankModel(nn.Module):
    """Linearized model like :class:`LinearizedModel`, but trainable LoRA state is **A/B factors**.

    JVP tangents on LoRA slots are ``A - A_0`` and ``B - B_0`` (not a stored delta tensor).
    ``self.lora_parameters`` is a :class:`nn.ModuleList` of :class:`LoRAFactors` (and
    optionally :class:`LoRAEmbeddingFactor` for unpaired embedding weights).

    Frozen full ``params0`` is still used as the JVP primal at ``p_0``.

    Args:
        model: PEFT-style module with ``lora_A`` / ``lora_B`` (and optionally
            ``lora_embedding``) parameters.
        init_model: Linearization point ``p_0``.
        rank: LoRA rank ``r``; must appear in both ``A`` and ``B`` shapes (and in
            embedding tensors if used).
        lora_param_patterns: Substrings used to select LoRA parameters from flat ``named_parameters``.
    """

    def __init__(
        self,
        model: nn.Module,
        init_model: nn.Module = None,
        *,
        rank: int,
        lora_param_patterns: Sequence[str] = (
            "lora_A",
            "lora_B",
            "lora_embedding",
        ),
    ) -> None:
        super().__init__()
        if init_model is None:
            init_model = model
        if rank < 1:
            raise ValueError("rank must be >= 1.")

        func0, params0, self.buffers0 = make_functional_with_buffers(
            init_model.eval(), disable_autograd_tracking=True
        )
        self._functional = lambda params, *a, **kw: func0(
            params, self.buffers0, *a, **kw
        )

        _, params_model, _ = make_functional_with_buffers(
            model, disable_autograd_tracking=True
        )

        names = tuple(n for n, _ in model.named_parameters())
        if len(names) != len(params0) or len(params0) != len(params_model):
            raise RuntimeError(
                "named_parameters / functional param counts disagree; use matching "
                "`model` and `init_model` topologies."
            )

        self.params0 = nn.ParameterList(params0)
        for p in self.params0:
            p.requires_grad = False

        self._param_names = names
        self.rank = rank
        self._lora_param_patterns: tuple[str, ...] = tuple(lora_param_patterns)

        lora_idx_set = {
            i
            for i, name in enumerate(names)
            if any(pat in name for pat in self._lora_param_patterns)
        }
        if not lora_idx_set:
            raise ValueError(
                "LinearizedLowRankModel: no parameter name matched "
                f"lora_param_patterns={self._lora_param_patterns}."
            )

        modules: List[Union[LoRAFactors, LoRAEmbeddingFactor]] = []
        # (module_index, iA, iB) for LoRAFactors; (module_index, i) for LoRAEmbeddingFactor
        self._ab_specs: list[tuple[int, int, int]] = []
        self._single_specs: list[tuple[int, int]] = []

        used: set[int] = set()
        mod_i = 0

        for i in sorted(lora_idx_set):
            if i in used:
                continue
            n = names[i]
            partner_name: Optional[str] = None
            if "lora_A" in n:
                partner_name = n.replace("lora_A", "lora_B", 1)
            elif "lora_B" in n:
                partner_name = n.replace("lora_B", "lora_A", 1)

            if partner_name is not None:
                j = _index_of_name(names, partner_name)
                if j is None or j not in lora_idx_set or j in used:
                    raise ValueError(
                        f"Could not pair LoRA param {n!r} with {partner_name!r}."
                    )
                if "lora_A" in n:
                    i_a, i_b = i, j
                else:
                    i_a, i_b = j, i
                if "lora_A" not in names[i_a] or "lora_B" not in names[i_b]:
                    raise ValueError(f"Expected lora_A / lora_B pair; got {names[i_a]!r}, {names[i_b]!r}.")
                A = params_model[i_a]
                B = params_model[i_b]
                _check_ab_rank(A, B, rank)
                modules.append(LoRAFactors(A, B))
                self._ab_specs.append((mod_i, i_a, i_b))
                mod_i += 1
                used.add(i_a)
                used.add(i_b)
                continue

            # e.g. lora_embedding only
            W = params_model[i]
            if rank not in W.shape:
                raise ValueError(
                    f"rank={rank} must appear in embedding LoRA shape {tuple(W.shape)} for {n!r}."
                )
            modules.append(LoRAEmbeddingFactor(W))
            self._single_specs.append((mod_i, i))
            mod_i += 1
            used.add(i)

        if used != lora_idx_set:
            raise ValueError(
                f"Unpaired or unmatched LoRA indices: {sorted(lora_idx_set - used)}."
            )

        self.lora_parameters = nn.ModuleList(modules)
        self._model_name = model.__class__.__name__

    def _jvp_tangents(self) -> Tuple[torch.Tensor, ...]:
        """Tangent per flat param: ``A - A0`` / ``B - B0`` / ``W - W0`` on LoRA slots; else zeros."""
        tan = [torch.zeros_like(p0) for p0 in self.params0]
        for mi, i_a, i_b in self._ab_specs:
            m = self.lora_parameters[mi]
            assert isinstance(m, LoRAFactors)
            tan[i_a] = m.A - self.params0[i_a]
            tan[i_b] = m.B - self.params0[i_b]
        for mi, i in self._single_specs:
            m = self.lora_parameters[mi]
            assert isinstance(m, LoRAEmbeddingFactor)
            tan[i] = m.weight - self.params0[i]
        return tuple(tan)

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        alpha = kwargs.pop("alpha", 1.0)
        kw = {**kwargs, "use_cache": False}
        device_type = LinearizedModel._autocast_device_type(*args, **kw)
        tangents = self._jvp_tangents()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            out, dp = jvp(
                lambda p: LinearizedModel._forward_to_tensor(
                    self._functional(p, *args, **kw)
                ),
                (tuple(self.params0),),
                (tangents,),
            )
        return out + alpha * dp

    def dp(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs.pop("alpha", 1.0)
        kw = {**kwargs, "use_cache": False}
        device_type = LinearizedModel._autocast_device_type(*args, **kw)
        tangents = self._jvp_tangents()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, dp = jvp(
                lambda p: LinearizedModel._forward_to_tensor(
                    self._functional(p, *args, **kw)
                ),
                (tuple(self.params0),),
                (tangents,),
            )
        return dp

    def dp_logprobs(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs.pop("alpha", 1.0)
        logprobs_dim = kwargs.pop("logprobs_dim", -1)
        kw = {**kwargs, "use_cache": False}
        device_type = LinearizedModel._autocast_device_type(*args, **kw)
        tangents = self._jvp_tangents()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, dp = jvp(
                lambda p: LinearizedModel._forward_to_logprobs(
                    self._functional(p, *args, **kw), dim=logprobs_dim
                ),
                (tuple(self.params0),),
                (tangents,),
            )
        return dp

    def named_lora_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for mi, i_a, i_b in self._ab_specs:
            m = self.lora_parameters[mi]
            assert isinstance(m, LoRAFactors)
            yield self._param_names[i_a], m.A
            yield self._param_names[i_b], m.B
        for mi, i in self._single_specs:
            m = self.lora_parameters[mi]
            assert isinstance(m, LoRAEmbeddingFactor)
            yield self._param_names[i], m.weight