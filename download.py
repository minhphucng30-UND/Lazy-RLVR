from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="TMLR-Group-HF/GT-Qwen3-8B-Base-MATH",
    local_dir="models/Qwen3-8B-Base-MATH",
)