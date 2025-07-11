import os
from huggingface_hub import snapshot_download
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import huggingface_hub
huggingface_hub.login("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") 
#huggingface-cli download --token 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 --resume-download meta-llama/Llama-2-7b-hf --local-dir Llama-2-7b-hf

# 定义需要下载的所有模型库 ID
# 该列表综合了您提供的 CKPT_PATH 中提到的所有模型
MODELS_TO_DOWNLOAD = [
    # --- 主要的多模态模型 ---
    "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2-VL-7B",
    "llava-hf/llava-onevision-qwen2-7b-si-hf",
    "lmms-lab/llava-onevision-qwen2-7b-si",
    "THUDM/cogvlm-chat-hf",
    "llava-hf/llava-1.5-7b-hf",
    "liuhaotian/llava-v1.5-7b",
    "MAGAer13/mplug-owl2-llama2-7b",
    "THUDM/cogvlm-grounding-generalist-hf", # 新增
    "Lin-Chen/ShareGPT4V-7B",             # 新增 (对应 ShareGPT4V-7B-llava)
    "THUDM/cogvlm-base-490-hf",
 
    # --- 作为基座的语言模型 ---
    "lmsys/vicuna-7b-v1.5",               # 新增
    "Qwen/Qwen2-7B-Instruct",                # 新增
    "meta-llama/Llama-2-7b-hf",         # 新增 (对应 Llama-2-7b-hf-llava)
]

# 定义模型下载的基础路径
base_download_path = os.path.join(os.getcwd(), "downloaded_models")
os.makedirs(base_download_path, exist_ok=True)

print(f"Models will be downloaded to: {base_download_path}\n")

# 遍历并下载每个模型
for repo_id in MODELS_TO_DOWNLOAD:
    # 从repo_id中获取一个干净的文件夹名, 将斜杠替换为下划线
    model_folder_name = repo_id.replace("/", "_")
    local_model_path = os.path.join(base_download_path, model_folder_name)
    
    print(f"--- Syncing repository: {repo_id} ---")
    print(f"Target directory: {local_model_path}")
    
    try:
        # 每次都调用 snapshot_download。
        # 它会自动检查本地文件，仅下载缺失或不完整的部分。
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_model_path,
            repo_type="model",
            local_dir_use_symlinks=False,
            resume_download=True 
        )
        print(f"--- Repository {repo_id} is up to date. ---\n")
    except Exception as e:
        print(f"!!! Failed to sync {repo_id}. Error: {e}\n")

print("All synchronization tasks are complete.")
print("IMPORTANT: Remember to update the CKPT_PATH in ALL relevant merge scripts!")