# llava_merging_with_task_vectors.py

import os
import sys
import json
import torch
import safetensors.torch
import argparse
from tqdm import tqdm
import gc
import shutil

# --- 模型与路径配置 ---
CKPT_PATH = {
    "original_model": "/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-7B-Instruct",
    "qwen2_vl": "/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct",
    "llava-onevision-qwen": "/home/user/xieqiuhao/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si-hf"
}
INDEX_FILENAME = "model.safetensors.index.json"

# --- 权重加载与处理函数 ---

def load_weights_from_index(base_path: str) -> dict:
    """根据 index.json 文件动态加载所有权重分片。"""
    index_path = os.path.join(base_path, INDEX_FILENAME)
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"错误: 在 '{base_path}' 中未找到 '{INDEX_FILENAME}'。")
    
    with open(index_path, 'r') as f:
        index = json.load(f)
    
    file_list = sorted(list(set(index["weight_map"].values())))
    weights = {}
    for file in tqdm(file_list, desc=f"正在加载 {os.path.basename(base_path)} 的权重"):
        weights.update(safetensors.torch.load_file(os.path.join(base_path, file)))
    return weights

def normalize_keys(weights: dict, prefix_to_remove: str, prefix_to_add: str = "") -> dict:
    """
    标准化权重字典中的 key。
    - 移除 'prefix_to_remove'。
    - （可选）在开头添加 'prefix_to_add'。
    """
    if not prefix_to_remove and not prefix_to_add:
        return weights
    
    normalized_weights = {}
    for key, value in weights.items():
        new_key = key
        if prefix_to_remove and new_key.startswith(prefix_to_remove):
            new_key = new_key[len(prefix_to_remove):]
        
        if prefix_to_add:
            new_key = prefix_to_add + new_key
            
        normalized_weights[new_key] = value
    return normalized_weights

# --- 辅助函数 ---
def need_merge(name: str) -> bool:
    """
    判断一个层是否需要合并。
    根据 Qwen2-VL 的结构，语言模型层以 'model.layers.' 开头。
    视觉部分 ('visual.') 和 lm_head 不参与合并。
    """
    # 排除视觉塔和最终的分类头
    if name.startswith('visual.') or name.startswith('lm_head'):
        return False
    # 目标是语言模型的 Transformer 层
    if name.startswith("model.layers."):
        # 排除旋转位置编码的频率缓存
        if name.endswith(".self_attn.rotary_emb.inv_freq"):
            return False
        if name.endswith(".self_attn.q_proj.weight") or name.endswith(".self_attn.k_proj.weight") or name.endswith(".self_attn.v_proj.weight") or name.endswith(".self_attn.o_proj.weight"):
            return False # 修改了此处
        return True
    # 合并最终的 LayerNorm
    if name == 'model.norm.weight':
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

# --- 主转换与合并逻辑 ---
def convert(args, device):
    # --- 输出路径设置 ---
    output_dir = "merged_models"
    if args.output:
        model_name = os.path.basename(args.output)
        output_dir = os.path.dirname(args.output)
    else:
        strategy_name = args.strategy
        if strategy_name == "task_vector_grafting":
            model_name = f"grafted-s{args.lambda_s}-c{args.lambda_c}"
        else:
            model_name = f"interpolated-a{args.alpha}"
    
    OUTPUT_PATH = os.path.join(output_dir, model_name)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"合并模型将保存至: {OUTPUT_PATH}")

    # --- 模型加载与标准化 ---
    print("加载基础模型 (M_A: Qwen2-VL)...")
    base_weights = load_weights_from_index(args.base_model_path)
    
    print("加载增量模型 (M_B: llava-onevision-qwen)...")
    donor_weights_raw = load_weights_from_index(args.donor_model_path)
    # 标准化 Key：移除 'language_model.' 前缀，并添加 'model.' 前缀
    donor_weights = normalize_keys(donor_weights_raw, "language_model.")
    print("增量模型 Key 标准化完成。")
    
    merged_weights = {}

    # --- 策略选择 ---
    if args.strategy == "task_vector_grafting":
        print("="*80)
        print("应用 '任务向量嫁接' 策略。")
        print(f"协同系数 (lambda_s): {args.lambda_s}, 冲突系数 (lambda_c): {args.lambda_c}")
        print("="*80)

        print("加载原始预训练模型 (M_C: Qwen2-Instruct)...")
        original_weights_raw = load_weights_from_index(args.original_model_path)
        # 标准化 Key：移除 'model.' 前缀
        # original_weights = normalize_keys(original_weights_raw, "model.")
        print("原始模型 Key 标准化完成。")

        # 迭代基础模型的 key，因为它是我们最终要保存的结构
        for key in tqdm(base_weights.keys(), desc="应用任务向量嫁接"):
            # 检查标准化后的 key 是否存在于所有模型中
            if key in original_weights_raw and key in donor_weights and need_merge(key) and base_weights[key].shape == donor_weights[key].shape:
                w_c = original_weights_raw[key].float().to(device)
                w_a = base_weights[key].float().to(device)
                w_b = donor_weights[key].float().to(device)

                tau_a = w_a - w_c
                tau_b = w_b - w_c

                tau_a_norm_sq = torch.sum(tau_a * tau_a)
                if tau_a_norm_sq > 1e-9:
                    proj_scalar = torch.sum(tau_a * tau_b) / tau_a_norm_sq
                    tau_b_synergy = torch.clamp(proj_scalar, min=0) * tau_a
                    tau_b_conflict = torch.clamp(-proj_scalar, min=0) * tau_a
                else:
                    tau_b_synergy = torch.zeros_like(tau_b)
                    tau_b_conflict = torch.zeros_like(tau_b)
                
                tau_b_ortho = tau_b - (tau_b_synergy - tau_b_conflict)

                w_star = w_a + args.lambda_s * tau_b_synergy + args.lambda_c * tau_b_conflict + tau_b_ortho
                
                merged_weights[key] = w_star.to(base_weights[key].dtype).cpu()
            else:
                # 对于不合并或不存在于所有模型中的层，保留基础模型的权重
                merged_weights[key] = base_weights[key]
            
            gc.collect()
            torch.cuda.empty_cache()

    else: # 默认线性插值
        print("="*80)
        print(f"应用线性插值策略，alpha = {args.alpha}")
        print("="*80)
        for key in tqdm(base_weights.keys(), desc="应用线性插值"):
            if key in donor_weights and need_merge(key) and base_weights[key].shape == donor_weights[key].shape:
                 merged_weights[key] = (1 - args.alpha) * base_weights[key] + args.alpha * donor_weights[key]
            else:
                 merged_weights[key] = base_weights[key]

    # --- 保存合并后的模型 ---
    print("\n正在保存合并后的模型...")
    index_path = os.path.join(args.base_model_path, INDEX_FILENAME)
    with open(index_path, "r") as f:
        weight_map = json.load(f)["weight_map"]
    
    sharded_weights = {filename: {} for filename in set(weight_map.values())}
    for key, value in merged_weights.items():
        if key in weight_map:
            sharded_weights[weight_map[key]][key] = value
        else:
            print(f"警告: 在 weight_map 中找不到 key '{key}'，该权重将不会被保存。")

    for filename, weights_dict in sharded_weights.items():
        if weights_dict:
            save_path = os.path.join(OUTPUT_PATH, filename)
            safetensors.torch.save_file(weights_dict, save_path)
        
    create_soft_link(source_path=args.base_model_path, link_path=OUTPUT_PATH)

    print("合并完成。")
    print(f"模型已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用高级任务向量投影合并两个模型。")
    
    parser.add_argument('--strategy', type=str, default="task_vector_grafting", 
                        choices=['interpolation', 'task_vector_grafting'], 
                        help="使用的合并策略。")
    
    # 修复点：添加 cuda_device 参数
    parser.add_argument('--cuda_device', type=int, default=0, help="要使用的 CUDA 设备号 (例如, 0, 1, 2)。")

    # 模型路径
    parser.add_argument('--base_model_path', type=str, default=CKPT_PATH["qwen2_vl"], 
                        help="基础模型 (M_A) 的路径。")
    parser.add_argument('--donor_model_path', type=str, default=CKPT_PATH["llava-onevision-qwen"], 
                        help="增量模型 (M_B) 的路径。")
    parser.add_argument('--original_model_path', type=str, default=CKPT_PATH["original_model"], 
                        help="原始预训练模型 (M_C) 的路径。'task_vector_grafting' 策略需要。")

    # 策略特定参数
    parser.add_argument('--alpha', type=float, default=0.3, help="'interpolation' 策略的系数。")
    parser.add_argument('--lambda_s', type=float, default=1.2, help="'task_vector_grafting' 的协同系数。")
    parser.add_argument('--lambda_c', type=float, default=0.0, help="'task_vector_grafting' 的冲突系数。")

    parser.add_argument('--output', type=str, default=None, help="合并后模型的输出目录和名称。")
    
    args = parser.parse_args()

    # 修复点：根据参数设置设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_device}")
        print(f"正在使用 CUDA 设备: {device}")
    else:
        device = torch.device("cpu")
        print("警告: CUDA 不可用，将使用 CPU。计算会非常慢。")
    
    # 从参数更新路径
    CKPT_PATH['qwen2_vl'] = args.base_model_path
    CKPT_PATH['llava-onevision-qwen'] = args.donor_model_path
    CKPT_PATH['original_model'] = args.original_model_path

    print("--- 配置 ---")
    print(f"策略: {args.strategy}")
    print(f"基础模型 (M_A): {args.base_model_path}")
    print(f"增量模型 (M_B): {args.donor_model_path}")
    if args.strategy == 'task_vector_grafting':
        print(f"原始模型 (M_C): {args.original_model_path}")
        print(f"协同系数 (λ_s): {args.lambda_s}")
        print(f"冲突系数 (λ_c): {args.lambda_c}")
    else:
        print(f"插值 Alpha: {args.alpha}")
    print("--------------------")

    convert(args, device)