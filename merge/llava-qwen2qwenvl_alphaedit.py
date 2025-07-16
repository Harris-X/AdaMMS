# llava_merging_with_alphaedit.py

import os
import sys
import json
import torch
import safetensors.torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# 关键依赖：从 AlphaEdit 项目中引入用于计算协方差的工具
# 请确保 AlphaEdit 项目中的 "rome" 文件夹与此脚本位于同一目录或在 Python 路径下
try:
    from rome.layer_stats import layer_stats
except ImportError:
    print("="*80)
    print("错误：无法导入 'rome.layer_stats'。")
    print("请确保您已经将 AlphaEdit 项目中的 'rome' 文件夹放置在您的工作目录或 Python 路径中。")
    print("这个模块对于计算 'null_space' 策略所需的协方差矩阵至关重要。")
    print("您可以从 AlphaEdit 的 GitHub 仓库获取相关文件: https://github.com/jianghoucheng/AlphaEdit")
    print("="*80)
    sys.exit(1)

# --- 模型与路径配置 (遵循您提供的脚本) ---
# 请在使用前将 "your/path/to/..." 替换为您的实际模型路径
CKPT_PATH = {
    "qwen2_vl": "/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct",
    "llava-onevision-qwen": "/home/user/xieqiuhao/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si"
}

INDEX_FILENAME = {
    "qwen2_vl": "model.safetensors.index.json",
    "llava-onevision-qwen": "model.safetensors.index.json"
}

# 用于缓存统计数据和投影矩阵的目录
STATS_DIR = "hparams_cache"
os.makedirs(STATS_DIR, exist_ok=True)

# --- 权重加载函数 (遵循您提供的脚本) ---
def load_safetensors_weights(base_path, file_list):
    weights = {}
    for file in tqdm(file_list, desc=f"Loading weights from {os.path.basename(base_path)}"):
        path = os.path.join(base_path, file)
        weights.update(safetensors.torch.load_file(path))
    return weights

def load_qwenvl_weights(base_path):
    with open(os.path.join(base_path, INDEX_FILENAME["qwen2_vl"]), 'r') as f:
        index = json.load(f)
    file_list = sorted(list(set(index["weight_map"].values())))
    return load_safetensors_weights(base_path, file_list)

def load_minicpm_weights(base_path):
    with open(os.path.join(base_path, INDEX_FILENAME["llava-onevision-qwen"]), 'r') as f:
        index = json.load(f)
    file_list = sorted(list(set(index["weight_map"].values())))
    return load_safetensors_weights(base_path, file_list)

# --- 新增：AlphaEdit 核心逻辑实现 ---
PROJECTOR_CACHE = {}

def compute_covariance_and_projector(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    hparams: argparse.Namespace,
) -> torch.Tensor:
    """
    计算给定层的协方差矩阵，进行SVD分解，并返回零空间投影矩阵 P。
    这是“零空间嫁接”方法的核心。
    """
    model_name_safe = hparams.base_model_path.replace("/", "_")
    key = (model_name_safe, layer_name, hparams.null_space_threshold)
    
    # 使用文件缓存来避免昂贵的重复计算
    cache_path = os.path.join(STATS_DIR, f"projector__{model_name_safe}__{layer_name.replace('.', '_')}__{hparams.null_space_threshold}.pt")

    if os.path.exists(cache_path) and not hparams.force_recompute:
        print(f"Loading cached projector for {model_name_safe} @ {layer_name}.")
        return torch.load(cache_path)

    print(f"Computing covariance for {model_name_safe} @ {layer_name}.")
    stat = layer_stats(
        model, tok, layer_name, STATS_DIR,
        hparams.mom2_dataset, to_collect=["mom2"],
        sample_size=hparams.mom2_n_samples,
        precision=hparams.mom2_dtype,
        force_recompute=hparams.force_recompute,
    )
    cov = stat.mom2.moment().float().cuda()

    print(f"Computing SVD and projector for {layer_name}.")
    U, S, _ = torch.linalg.svd(cov)
    
    threshold = hparams.null_space_threshold
    null_space_vectors = U[:, S < threshold]
    
    projector = null_space_vectors @ null_space_vectors.T
    
    print(f"Finished projector for {layer_name}. Original dim: {cov.shape[0]}, Null-space dim: {null_space_vectors.shape[1]}")
    
    torch.save(projector.cpu(), cache_path)
    
    return projector

# --- 合并逻辑与文件操作 (遵循您提供的脚本) ---
def need_merge(name:str) -> bool:
    if name in ['model.norm.weight']:
        return True
    if name in ['lm_head.weight', 'model.embed_tokens.weight']:
        return False
    if name.startswith("model.layers."):
        if name.endswith(".self_attn.rotary_emb.inv_freq"):
            return False
        return True
    return False

def create_soft_link(source_path, link_path):
    # Check if source path exists
    if not os.path.exists(source_path):
        print(f"Error: Source path '{source_path}' does not exist.")
        return

    # Check if link path exists, if not create it
    if not os.path.exists(link_path):
        os.makedirs(link_path)
        print(f"Created directory '{link_path}'")

    # Iterate through all files and directories in the source path
    for item in os.listdir(source_path):
        source_item = os.path.join(source_path, item)
        link_item = os.path.join(link_path, item)

        # Skip files that end with '.bin'
        if item.endswith('.bin'):
            print(f"Skipping '{item}' as it ends with '.bin'")
            continue

        # If it's a file, create a symbolic link
        if os.path.isfile(source_item):
            try:
                os.symlink(source_item, link_item)
                print(f"Created soft link '{link_item}' -> '{source_item}'")
            except OSError as e:
                print(f"Error creating soft link for '{item}': {e}")

        # If it's a directory, ignore it
        elif os.path.isdir(source_item):
            continue

# --- 主转换函数 ---
def convert(args):
    # 输出路径设置
    output_dir = "../merged_models"
    if args.output is not None:
        model_name = os.path.basename(args.output)
        output_dir = os.path.dirname(args.output)
    else:
        strategy_name = args.strategy if args.strategy else "interpolation"
        if strategy_name == "null_space":
            model_name = f"qwen-grafted-t{args.null_space_threshold}"
        else:
            model_name = f"qwen-merged-{strategy_name}-a{args.alpha}"
    
    OUTPUT_PATH = os.path.join(output_dir, model_name)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"Merging output path: {OUTPUT_PATH}")

    # 加载模型权重
    print("Loading base model (Qwen2-VL)...")
    base_weights = load_qwenvl_weights(args.base_model_path)
    print("Loading donor model (LLaVA-OneVision-Qwen)...")
    donor_weights = load_minicpm_weights(args.donor_model_path)
    
    # 选择合并策略
    if args.strategy == "null_space":
        print("="*80)
        print("Applying 'null_space' grafting strategy.")
        print("="*80)
        
        print(f"Loading base model '{args.base_model_path}' for covariance computation...")
        base_model_for_cov = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16).cuda()
        base_tok = AutoTokenizer.from_pretrained(args.base_model_path)

        for key in tqdm(base_weights.keys(), desc="Applying Null-Space Grafting"):
            if need_merge(key) and key in donor_weights:
                print(f"\nProcessing layer: {key}")
                module_path = key.rsplit('.', 1)[0]
                
                projector = compute_covariance_and_projector(
                    base_model_for_cov, base_tok, module_path, args
                ).cuda()

                w_a = base_weights[key].float().cuda()
                w_b = donor_weights[key].float().cuda()
                delta = w_b - w_a
                
                # 对于权重 W (d_out, d_in)，我们假设其作用于输入特征 x (d_in, 1)
                # 即 y = Wx。我们希望扰动不影响原有输入，因此投影应作用于 delta 的行
                # (W_A + Δ')x = W_A x + Δ'x。扰动 Δ' 应在 K_A 的左零空间
                # Δ'K_A = 0，所以 P 作用于 Δ' 的左侧
                projected_delta = projector @ delta
                
                base_weights[key] = (w_a + projected_delta).to(base_weights[key].dtype).cpu()
                
                del projector, w_a, w_b, delta, projected_delta
                gc.collect()
                torch.cuda.empty_cache()

        del base_model_for_cov, base_tok
        gc.collect()
        torch.cuda.empty_cache()

    else: 
        print("="*80)
        print("Applying default linear interpolation strategy.")
        print("="*80)
        for key in tqdm(base_weights.keys(), desc="Applying Linear Interpolation"):
            if key in donor_weights and need_merge(key):
                 base_weights[key] = (1 - args.alpha) * base_weights[key] + args.alpha * donor_weights[key]

    # --- 保存合并后的模型 (严格遵循您提供的分块保存逻辑) ---
    print("\nSaving merged model...")
    # 首先，获取基础模型的权重映射表
    llava_index_path = os.path.join(args.base_model_path, INDEX_FILENAME["qwen2_vl"])
    with open(llava_index_path, "r") as f:
        llava_index = json.load(f)["weight_map"]
    
    # 创建一个字典来存放每个分片文件的权重
    file_list = sorted(list(set(llava_index.values())))
    split_llava = {file: {} for file in file_list}
    
    # 将合并后的权重分配到相应的分片字典中
    for key, value in base_weights.items():
        if key in llava_index:
            split_llava[llava_index[key]][key] = value
        else:
            print(f"Warning: key '{key}' not found in weight map, will not be saved.", file=sys.stderr)
            
    # 逐一保存每个分片文件
    for file in file_list:
        save_path = os.path.join(OUTPUT_PATH, file)
        safetensors.torch.save_file(split_llava[file], save_path)
    
    # 复制或链接其他必要文件 (如 config.json, tokenizer_config.json 等)
    create_soft_link(source_path=args.base_model_path, link_path=OUTPUT_PATH)
    # 确保 index 文件也被复制
    with open(os.path.join(args.base_model_path, INDEX_FILENAME["qwen2_vl"]), "r") as f_in, \
         open(os.path.join(OUTPUT_PATH, INDEX_FILENAME["qwen2_vl"]), "w") as f_out:
        f_out.write(f_in.read())

    print("Convert Done.")
    print(f"Merged model and associated files saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LLaVA-OneVision-Qwen into Qwen2-VL using various strategies.")
    
    parser.add_argument('--output', type=str, default=None, help="Output directory and name for the merged model.")
    parser.add_argument('--strategy', type=str, default="interpolation", 
                        choices=['interpolation', 'ties', 'dare_ties', 'dare_linear', 'null_space'], 
                        help="Merging strategy to use. Default is 'interpolation'.")
    
    parser.add_argument('--alpha', type=float, default=0.5, help="Coefficient for linear interpolation.")
    parser.add_argument('-K', type=float, default=0.5, help="Top-K parameter for TIES/DARE merging strategies.")
    
    # Null-space grafting specific arguments
    parser.add_argument('--base_model_path', type=str, default=CKPT_PATH["qwen2_vl"], 
                        help="Path to the base model (Qwen2-VL) for grafting and covariance computation.")
    parser.add_argument('--donor_model_path', type=str, default=CKPT_PATH["llava-onevision-qwen"], 
                        help="Path to the donor model (LLaVA-OneVision).")
    parser.add_argument('--mom2_dataset', type=str, default="wikipedia", 
                        help="Dataset for covariance statistics ('wikipedia' or 'c4').")
    parser.add_argument('--mom2_n_samples', type=int, default=10000, 
                        help="Number of samples for statistics. Reduce if VRAM is limited.")
    parser.add_argument('--mom2_dtype', type=str, default="bfloat16", 
                        help="Precision for statistics computation (bfloat16, float16).")
    parser.add_argument('--null_space_threshold', type=float, default=1e-2, 
                        help="SVD singular value threshold to define the null-space.")
    parser.add_argument('--force_recompute', action='store_true', 
                        help="Force recomputation of covariance and projectors, ignoring caches.")

    args = parser.parse_args()
    
    # 更新路径，如果用户通过命令行提供了
    CKPT_PATH['qwen2_vl'] = args.base_model_path
    CKPT_PATH['llava-onevision-qwen'] = args.donor_model_path

    print("--- Configuration ---")
    print(f"Strategy: {args.strategy}")
    print(f"Base Model (M_A): {args.base_model_path}")
    print(f"Donor Model (M_B): {args.donor_model_path}")
    if args.strategy == 'null_space':
        print(f"Null-Space Grafting Threshold: {args.null_space_threshold}")
        print(f"Covariance Dataset: {args.mom2_dataset} ({args.mom2_n_samples} samples)")
    else:
        print(f"Interpolation Alpha: {args.alpha}")
    print("--------------------")

    convert(args)