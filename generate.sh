# model_paths=("DeepScaleR-1.5B" "DeepSeek-R1-Distill-Qwen-1.5B" "Klear-Reasoner-8B-SFT" "Qwen3-8B-Base-MATH" "Polaris-4B")
model_paths=("Polaris-4B")


for model_path in ${model_paths[@]}; do
    # if [[ "$model_path" == "DeepScaleR-1.5B" || "$model_path" == "DeepSeek-R1-Distill-Qwen-1.5B" ]]; then
    #     enforce_thinking="--enforce_thinking"
    # else
    #     enforce_thinking=""
    # fi
    # python generate_vllm.py --model models/${model_path} --data polaris
    python generate_vllm.py --model models/${model_path} --data math
done