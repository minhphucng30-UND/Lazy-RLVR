from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import os
import pandas as pd
import sys
sys.path.append("/blue/sgao1/thua.nd/phuc/lazyNTK-RLVR")
from model import LinearizedModel
from verl.utils.reward_score.deepscaler import rllm_reward_fn_math
import torch
import argparse
from utils import TokenizedDataset, collate_fn
torch.backends.cuda.matmul.allow_tf32 = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/Qwen3-8B-Base-MATH")
    parser.add_argument("--base_model", type=str, default="models/Qwen3-8B-Base")
    parser.add_argument("--data", type=str, default="polaris")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_kwargs = {
        "torch_dtype": torch.float32,
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    linearized_model = LinearizedModel(model, base_model)

    model_name = args.model.split("/")[-1]
    df = pd.read_parquet(f"data/{args.data}/synthetic/{model_name}_vllm.parquet")
    prompts = df['prompt'].tolist()
    prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=True) for prompt in prompts]
    outputs = df['output'].tolist()

    all_prompts = []
    all_outputs = []
    for prompt, output_lst in zip(prompts, outputs):
        for output in output_lst:
            all_prompts.append(prompt)
            all_outputs.append(output)
    
    dataset = TokenizedDataset(all_prompts, all_outputs, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer))

    all_kls = 0.0
    progress_bar = tqdm(dataloader, desc="Calculating KL")
    for batch in progress_bar:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        labels = batch.pop("labels")[:, 1:].contiguous()
        loss_mask = labels != -100
        labels[labels == -100] = 0

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            linearized_logits = linearized_model(**batch)[:, :-1, :].to(torch.float32)
            linearized_logits /= 0.6
            linearized_logprobs = torch.gather(linearized_logits.log_softmax(dim=-1), dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            linearized_logprobs = (linearized_logprobs * loss_mask).sum(dim=-1)

            model_logits = model(**batch).logits[:, :-1, :].to(torch.float32)
            model_logits /= 0.6
            model_logprobs = torch.gather(model_logits.log_softmax(dim=-1), dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            model_logprobs = (model_logprobs * loss_mask).sum(dim=-1)
            batch_kl = (model_logprobs - linearized_logprobs) / (loss_mask.sum(dim=-1) + 1e-8)
            all_kls += batch_kl.sum().item()
        
        progress_bar.set_postfix(kl=f"{all_kls / (progress_bar.n + 1):.4f}")
    
    print(f"Average KL: {all_kls / len(dataset)}")
    row_data = {
        "model": args.model.split("/")[-1],
        "base_model": args.base_model.split("/")[-1],
        "type": "RLVR",
        "data": args.data,
        "kl": all_kls / len(dataset),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/linearized_kl.jsonl", "a") as f:
        f.write(json.dumps(row_data) + "\n")