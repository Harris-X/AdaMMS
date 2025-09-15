# llava_merging_with_sosm.py

import os
import shutil
import sys
import json
import torch
import safetensors.torch
import argparse
from tqdm import tqdm
import gc


# --- 模型与路径配置 (请根据您的环境更新路径) ---
CKPT_PATH = {
    "original_model": "/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-7B-Instruct",  # 原始预训练模型 M_C
    "qwen2_vl": "/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct",      # 基础模型 M_A (例如，医疗专家)
    "llava-onevision-qwen": "/home/user/xieqiuhao/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si" # 增量模型 M_B (例如，法律专家)
}

# --- 权重加载函数 (保持不变) ---
def load_safetensors_weights(base_path, file_list):
    """从.safetensors文件加载权重"""
    weights = {}
    # 从index.json文件获取权重分片列表
    index_path = os.path.join(base_path, 'model.safetensors.index.json')
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found at {index_path}")
    with open(index_path, 'r') as f:
        index_data = json.load(f)
    file_list = sorted(list(set(index_data['weight_map'].values())))

    for file in tqdm(file_list, desc=f"Loading weights from {os.path.basename(base_path)}"):
        path = os.path.join(base_path, file)
        try:
            weights.update(safetensors.torch.load_file(path))
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
    return weights

# --- 辅助函数 (保持不变) ---
def need_merge(name: str) -> bool:
    """判断一个层是否需要被合并"""
    # 黑名单：不合并embedding和lm_head，因为它们特定于词表
    if 'lm_head.weight' in name or 'model.embed_tokens.weight' in name:
        return False
    # 白名单：通常合并Transformer块中的大部分权重
    if name.startswith("model.layers.") or name.startswith("model.norm.weight"):
        # 黑名单：旋转位置编码的频率是计算出来的，不是训练的
        if name.endswith(".rotary_emb.inv_freq"):
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

# --- 主要转换与合并逻辑 ---
def convert(args):
    """主函数，执行模型加载、合并和保存"""
    # --- 输出路径设置 ---
    output_dir = "merged_models"
    if args.output:
        model_name = os.path.basename(args.output)
        output_dir = os.path.dirname(args.output)
    else:
        model_name = args.strategy

    OUTPUT_PATH = os.path.join(output_dir, model_name)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"Merging output path: {OUTPUT_PATH}")

    # --- 模型加载 ---
    print("Loading Base Model (M_A)...")
    base_weights = load_safetensors_weights(args.base_model_path, [])

    print("Loading Donor Model (M_B)...")
    donor_weights = load_safetensors_weights(args.donor_model_path, [])

    # 为所有策略加载原始模型
    print("Loading Original Pretrained Model (M_C)...")
    original_weights = load_safetensors_weights(args.original_model_path, [])
    
    merged_weights = {}

    # --- 策略选择与执行 ---
    # 使用基础模型的键作为迭代的参考标准
    all_keys = tqdm(base_weights.keys(), desc=f"Applying '{args.strategy}' Strategy")
    
    for key in all_keys:
        # 检查所有模型是否都包含该键，并且该键是需要合并的
        if key in original_weights and key in donor_weights and need_merge(key) and base_weights[key].shape == donor_weights[key].shape:
            # 将权重移动到指定GPU并转换为float32以便精确计算
            w_c = original_weights[key].float().to(DEVICE)
            w_a = base_weights[key].float().to(DEVICE)
            w_b = donor_weights[key].float().to(DEVICE)

            # 计算任务向量
            tau_a = w_a - w_c
            tau_b = w_b - w_c

            if args.strategy == "sosm":
                # --- SOSM 核心逻辑 ---
                # 1. 创建协同与冲突掩码
                sign_a = torch.sign(tau_a)
                sign_b = torch.sign(tau_b)
                synergy_mask = (sign_a == sign_b).float()
                conflict_mask = 1.0 - synergy_mask

                # 2. 协同知识的增强合并
                tau_s = synergy_mask * (tau_a + tau_b) / 2.0
                
                # 3. 冲突知识的选择性正交化
                tau_a_k = conflict_mask * tau_a
                tau_b_k = conflict_mask * tau_b

                # 计算范数和内积以进行投影
                tau_a_k_norm_sq = torch.sum(tau_a_k * tau_a_k)
                
                tau_k = torch.zeros_like(tau_a)
                if tau_a_k_norm_sq > 1e-9:
                    inner_product_k = torch.sum(tau_a_k * tau_b_k)
                    proj_scalar_k = inner_product_k / tau_a_k_norm_sq
                    
                    # 从 tau_b_k 中减去其在 tau_a_k 上的投影
                    tau_b_k_ortho = tau_b_k - (proj_scalar_k * tau_a_k)
                    
                    # 冲突部分 = 基础向量 + 正交化后的增量向量
                    tau_k = tau_a_k + tau_b_k_ortho
                else:
                    # 如果基础向量的冲突部分为0，则直接保留
                    tau_k = tau_a_k 
                
                # 4. 最终合并
                tau_star = tau_s + tau_k
                w_star = w_c + tau_star
                
            elif args.strategy == "task_vector_grafting":
                # --- 原始的整体正交化逻辑 (保留作为对比) ---
                tau_a_norm_sq = torch.sum(tau_a * tau_a)
                if tau_a_norm_sq > 1e-9:
                    inner_product = torch.sum(tau_a * tau_b)
                    proj_scalar = inner_product / tau_a_norm_sq
                    tau_b_ortho = tau_b - proj_scalar * tau_a
                    w_star = w_a + tau_b_ortho # W_c + tau_a + (tau_b - proj)
                else:
                    w_star = w_a + tau_b # 如果tau_a为0，则直接相加
            
            else: # 默认为线性插值
                w_star = (1 - args.alpha) * w_a + args.alpha * w_b

            # 将计算后的权重转换回原始数据类型并移动到CPU
            merged_weights[key] = w_star.to(base_weights[key].dtype).cpu()
        
        else:
            # 对于不需要合并或不共同存在的层，直接使用基础模型的权重
            merged_weights[key] = base_weights[key]
        
        # 清理GPU缓存
        gc.collect()
        torch.cuda.empty_cache()

    # --- 保存合并后的模型 ---
    print("\nSaving merged model...")
    # 从基础模型获取权重分片结构
    index_path = os.path.join(args.base_model_path, 'model.safetensors.index.json')
    with open(index_path, "r") as f:
        index = json.load(f)["weight_map"]

    # 将合并后的权重按原始分片结构重新组织
    sharded_weights = {}
    for key, filename in index.items():
        if filename not in sharded_weights:
            sharded_weights[filename] = {}
        # 确保merged_weights中有这个键，否则使用原始基础模型的权重
        if key in merged_weights:
            sharded_weights[filename][key] = merged_weights[key]
        else:
            sharded_weights[filename][key] = base_weights[key]
            print(f"Warning: Key '{key}' not found in merged weights, using from base model.", file=sys.stderr)

    # 逐个保存分片文件
    for filename, tensors in tqdm(sharded_weights.items(), desc="Saving sharded weights"):
        save_path = os.path.join(OUTPUT_PATH, filename)
        safetensors.torch.save_file(tensors, save_path)

    # 复制或链接其他非权重文件 (如config.json, tokenizer等)
    shutil.copyfile(index_path, os.path.join(OUTPUT_PATH, 'model.safetensors.index.json'))
    create_soft_link(source_path=args.base_model_path, link_path=OUTPUT_PATH)
    
    print("\nMerge Done.")
    print(f"Merged model saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two models using advanced task vector strategies.")
    
    parser.add_argument('--strategy', type=str, default="sosm",
                        choices=['sosm', 'interpolation', 'task_vector_grafting'],
                        help="Merging strategy to use. 'sosm' is the recommended advanced strategy.")
    
    # 模型路径参数
    parser.add_argument('--base_model_path', type=str, default=CKPT_PATH["qwen2_vl"],
                        help="Path to the base model (M_A), whose capabilities are to be preserved.")
    parser.add_argument('--donor_model_path', type=str, default=CKPT_PATH["llava-onevision-qwen"],
                        help="Path to the donor model (M_B), from which knowledge is grafted.")
    parser.add_argument('--original_model_path', type=str, default=CKPT_PATH["original_model"],
                        help="Path to the original pre-trained model (M_C). Required for all task vector strategies.")

    # 策略特定参数
    parser.add_argument('--alpha', type=float, default=0.5, help="Coefficient for 'interpolation' strategy.")
    
    # 输出路径
    parser.add_argument('--output', type=str, default=None, help="Output directory and name for the merged model (e.g., /path/to/my_sosm_model).")
    
    # 添加CUDA设备选择参数
    parser.add_argument('--cuda_device', type=int, default=0, help="CUDA device to use (default: 4).")
    
    args = parser.parse_args()
    
    # 更新CUDA设备设置
    if torch.cuda.is_available() and args.cuda_device < torch.cuda.device_count():
        torch.cuda.set_device(args.cuda_device)
        DEVICE = f'cuda:{args.cuda_device}'
        print(f"Using CUDA device: {DEVICE}")
    else:
        DEVICE = 'cpu'
        print(f"CUDA device {args.cuda_device} not available, using CPU")
    
    # 更新配置字典
    CKPT_PATH['qwen2_vl'] = args.base_model_path
    CKPT_PATH['llava-onevision-qwen'] = args.donor_model_path
    CKPT_PATH['original_model'] = args.original_model_path

    print("--- Configuration ---")
    print(f"Strategy: {args.strategy}")
    print(f"CUDA Device: {DEVICE}")
    print(f"Base Model (M_A): {args.base_model_path}")
    print(f"Donor Model (M_B): {args.donor_model_path}")
    print(f"Original Model (M_C): {args.original_model_path}")
    if args.strategy == 'interpolation':
        print(f"Interpolation Alpha: {args.alpha}")
    print("---------------------\n")

    convert(args)