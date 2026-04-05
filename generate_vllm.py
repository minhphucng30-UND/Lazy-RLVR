from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
import os
import argparse
os.environ["VLLM_USE_DEEP_GEMM"] = "0"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/Qwen3-8B-Base-MATH")
    parser.add_argument("--data", type=str, default="polaris")
    # parser.add_argument("--enforce_thinking", action="store_true")

    args = parser.parse_args()
    os.makedirs(f"data/{args.data}/synthetic", exist_ok=True)
    llm = LLM(args.model, enforce_eager=False, enable_prefix_caching=True, max_model_len=32768)
    tokenizer = llm.get_tokenizer()

    df = pd.read_parquet(f"data/{args.data}/train.parquet")
    df = df.sample(n=128, random_state=42)
    prompts = df['prompt'].tolist()
    prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=True) for prompt in prompts]
    # if args.enforce_thinking:
    #     print("Enforcing thinking...")
    #     prompts = [prompt + "\n<think>\n" for prompt in prompts]
    print(prompts[0])

    outputs = llm.generate(prompts, sampling_params=SamplingParams(max_tokens=8192, temperature=0.6, n=16))
    outputs = [[output.outputs[i].text for i in range(len(output.outputs))] for output in outputs]
    df['output'] = outputs
    df.to_parquet(f"data/{args.data}/synthetic/{args.model.split('/')[-1]}_vllm.parquet")