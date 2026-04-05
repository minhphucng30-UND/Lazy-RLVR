import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import torch
from datasets import load_from_disk
from datasets import Dataset
import numpy as np
from verl.utils.reward_score.deepscaler import rllm_reward_fn_math
from model import LinearizedModel
from typing import Dict, List
from transformers import AutoTokenizer
# torch.backends.cuda.matmul.allow_tf32 = True

def _tokenize(batch: Dict[str, List], tokenizer: AutoTokenizer) -> Dict[str, List[List[int]]]:
    rows = [{k: batch[k][i] for k in batch.keys()} for i in range(len(next(iter(batch.values()))))]
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    all_advantages = []

    for row in rows:
        input_ids = []
        labels = []
        prompt = row["prompt"]
        # prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt + instruction_following}], add_generation_prompt=True, tokenize=False, enable_thinking=True)
        prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False, enable_thinking=True)
        response = row['response']
        advantage = row['advantage']
        all_advantages.append(advantage)
        prompt_enc = tokenizer(
            prompt,
            truncation=True,
            add_special_tokens=False,
            max_length=1024,
        )["input_ids"]
        input_ids.extend(prompt_enc)
        labels.extend([-100] * len(prompt_enc))

        response = tokenizer(
            response + tokenizer.eos_token,
            truncation=True,
            add_special_tokens=False,
            max_length=16384,
        )["input_ids"]
        input_ids.extend(response)
        labels.extend(response)
        attention_masks = [1] * len(input_ids)

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_masks)
        all_labels.append(labels)
    return {"input_ids": all_input_ids, "attention_mask": all_attention_masks, "labels": all_labels, "advantages": all_advantages}

def _collate(batch, tokenizer: AutoTokenizer):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []
    advantages = []
    for item in batch:
        padding_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding_len)
        attention_mask.append(item["attention_mask"] + [0] * padding_len)
        labels.append(item["labels"] + [-100] * padding_len)
        advantages.append(item["advantages"])

    return {
        "input_ids": torch.tensor(input_ids),
        "advantages": torch.tensor(advantages),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }
if __name__ == "__main__":
    import wandb
    wandb.init(project="lazyNTK-RLVR", name="linearized-train")
    tokenized_ds = load_from_disk("lazyNTK-RLVR/data/math/math_vllm_advantages_tokenized")
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen2.5-3B-Instruct")

    batch_size = 1
    dataloader = torch.utils.data.DataLoader(tokenized_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: _collate(x, tokenizer), num_workers=8, pin_memory=True)

    model_kwargs = {
        "torch_dtype": torch.float32,
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }
    beta = 1e-3
    model = AutoModelForCausalLM.from_pretrained("models/Qwen2.5-3B-Instruct", **model_kwargs)
    init_model = AutoModelForCausalLM.from_pretrained("models/Qwen2.5-3B-Instruct", **model_kwargs)
    model.gradient_checkpointing_enable()

    linearized_model = LinearizedModel(model, init_model)
    linearized_model.load_state_dict(torch.load("lazyNTK-RLVR/models/linearized_model_512.pth"))
    optimizer = torch.optim.AdamW(linearized_model.parameters(), lr=1e-6, weight_decay=1e-4, betas=(0.9, 0.95))
    gradient_accumulation_steps = 256
    train_steps = 384

    iter_loader = iter(dataloader)
    progress_bar = tqdm(total=train_steps, desc="Training")
    cnt = 0
    step = 0
    total_loss = 0.0
    import os
    os.makedirs("lazyNTK-RLVR/models", exist_ok=True)
    while cnt < train_steps:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dataloader)
            batch = next(iter_loader)
        
        batch = {k: v.to(model.device) for k, v in batch.items()}
        labels = batch.pop('labels')
        labels = labels[:, 1:].contiguous()
        loss_mask = labels != -100
        labels[labels == -100] = 0
        advantages = batch.pop('advantages')
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits_dp = linearized_model.dp_logprobs(**batch, logprobs_dim=-1)[:, :-1, :].to(torch.float32)
        logits_dp = torch.gather(logits_dp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        logits_dp = beta * (logits_dp * loss_mask).sum(dim=-1)
        loss = (logits_dp - advantages).pow(2).mean() / gradient_accumulation_steps
        total_loss += loss.item()
        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(linearized_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            cnt += 1
            with torch.no_grad():
                progress_bar.set_postfix(loss=loss.item())
                log_payload = {
                    "sequence_length": batch["input_ids"].shape[1],
                    "loss": total_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                total_loss = 0.0
                log_payload["grad_norm"] = grad_norm.item()
                wandb.log(log_payload)
            if (cnt + 1) % 64 == 0:
                torch.save(linearized_model.state_dict(), f"lazyNTK-RLVR/models/linearized_model_{cnt+1}.pth")
        step += 1

    
    torch.save(linearized_model.state_dict(), f"lazyNTK-RLVR/models/linearized_model_512.pth")

    # df = pd.read_parquet("lazyNTK-RLVR/data/math/math_vllm_advantages.parquet")
    # prompt = df['prompt'].tolist()
    # response_lst = df['response'].tolist()
    # advantage_lst = df['advantage'].tolist()

    # ds = Dataset.from_dict({
    #     'prompt': prompt,
    #     'response': response_lst,
    #     'advantage': advantage_lst
    # })
    # del df
    # print(ds[0])
    # ds = ds.map(lambda x: _tokenize(x, tokenizer), batched=True, remove_columns=ds.column_names, num_proc=8)
    # ds = ds.shuffle(seed=42)
    # ds.save_to_disk("data/math/math_vllm_advantages_tokenized")

