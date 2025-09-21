from collections import defaultdict
import json
import os
import shutil
import safetensors
import torch
from tqdm import tqdm

from utils import load_weights, normalize_llm_keys

EPS = 1e-9

def disentangled_reprojection_fusion(args):
    """阶段三：【SAM-S-DREAM】执行解耦重投影融合。"""
    print("\n--- [阶段三: SAM-S-DREAM 解耦重投影融合] ---")
    
    print("加载所有权重、掩码和激活...")
    weights_A = load_weights(args.base_model_path)
    weights_B_raw = load_weights(args.donor_model_path)
    weights_C_raw = load_weights(args.original_model_path)
    
    weights_B = normalize_llm_keys(weights_B_raw, list(weights_A.keys())); del weights_B_raw
    weights_C = normalize_llm_keys(weights_C_raw, list(weights_A.keys())); del weights_C_raw

    mask_cache_path = os.path.join(args.cache_dir, f"sams-dream_disjoint_mask_r{args.top_k_ratio}_alpha{args.alpha}.pt")
    disjoint_masks = torch.load(mask_cache_path)
    
    activations_A = torch.load(os.path.join(args.cache_dir, "activations_A.pt"))
    # 新增：bias 投影需要 B/C 的输出方向
    activations_B = torch.load(os.path.join(args.cache_dir, "activations_B.pt"))
    activations_C = torch.load(os.path.join(args.cache_dir, "activations_C.pt"))

    final_merged_weights = weights_A.copy()
    
    pbar = tqdm(disjoint_masks.items(), desc="【SAM-S-DREAM】执行重投影融合")
    processed_keys = set()
    for key, M_prime_B in pbar:
        if not (key in weights_B and key in weights_C): 
            continue
        
        processed_keys.add(key)
        W_A, W_B, W_C = weights_A[key].float(), weights_B[key].float(), weights_C[key].float()
        M_prime_B = M_prime_B.to(args.device)

        module_name = ".".join(key.replace("language_model.model.", "model.").split('.')[1:-1])
        tau_B = (W_B - W_C).to(args.device)
        tau_B_update = tau_B * M_prime_B.to(args.device)

        if W_A.ndim == 2 and key.endswith(".weight"):
            # 2D：沿输入激活投影
            d_i = activations_A[module_name]['input'].to(args.device).float()
            d_i_norm_sq = torch.sum(d_i * d_i)
            if d_i_norm_sq > EPS:
                proj_scalar = (tau_B_update @ d_i) / d_i_norm_sq
                tau_proj = torch.outer(proj_scalar, d_i)
                tau_ortho = tau_B_update - tau_proj
            else:
                tau_proj = torch.zeros_like(tau_B_update)
                tau_ortho = tau_B_update

        elif W_A.ndim == 1 and key.endswith(".bias"):
            # 1D bias：沿输出差分方向投影（g_bias = Y_B - Y_C）
            out_B = activations_B.get(module_name, {}).get('output', None)
            out_C = activations_C.get(module_name, {}).get('output', None)
            if out_B is None or out_C is None:
                # 回退：若方向缺失，使用简单加权更新已掩码的分量
                tau_proj = torch.zeros_like(tau_B_update)
                tau_ortho = tau_B_update
            else:
                g_dir = (out_B - out_C).to(args.device).float()
                g_norm_sq = torch.sum(g_dir * g_dir)
                if g_norm_sq > EPS:
                    proj_scalar = torch.sum(tau_B_update * g_dir) / g_norm_sq
                    tau_proj = proj_scalar * g_dir
                    # 仅在被掩码的位置更新；未掩码位置置零
                    tau_proj = tau_proj * M_prime_B
                    tau_ortho = tau_B_update - tau_proj
                else:
                    tau_proj = torch.zeros_like(tau_B_update)
                    tau_ortho = tau_B_update
        else:
            # 非预期形状
            continue

        W_star = W_A.to(args.device) + args.lambda_proj * tau_proj + args.lambda_ortho * tau_ortho
        final_merged_weights[key] = W_star.cpu().to(weights_A[key].dtype)

    # Part 2: 其余参数 (含 norm) 的保守加权平均
    print("\n正在使用简单加权平均处理其余参数 (norm, 其他未入选的参数)...")
    lam_default = args.lambda_proj
    lam_norm = getattr(args, "lambda_norm", 0.0)  # 新增: norm 的保守合并系数

    other_keys_pbar = tqdm(weights_A.keys(), desc="简单平均合并")
    for key in other_keys_pbar:
        # print(f"keys: {key}")
        if key in processed_keys:
            # print(f"跳过已处理的键: {key}")
            continue
        if key not in weights_B:
            continue
        if "lm_head" in key or "model.embed_tokens.weight" in key:
            # print(f"特殊键: {key}")
            continue

        W_A = weights_A[key].float()
        W_B = weights_B[key].float()

        lam = lam_default
        key_l = key.lower()
        if ('norm' in key_l) : # or ('layernorm' in key_l)
            # print(f"norm层: {key}")
            lam = lam_norm  # norm 更保守

        final_merged_weights[key] = ((1 - lam) * W_A + lam * W_B).to(W_A.dtype)
    
    _save_model(final_merged_weights)

def _save_model(args, merged_weights):
    """保存模型权重。"""
    print("\n正在保存合并后的模型...")
    index_path = os.path.join(args.base_model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_map = json.load(f)["weight_map"]
    
    sharded_weights = defaultdict(dict)
    for key, value in merged_weights.items():
        if key in index_map:
            sharded_weights[index_map[key]][key] = value
    
    for filename, weights_dict in sharded_weights.items():
        safetensors.torch.save_file(weights_dict, os.path.join(args.output_dir, filename))
    
    shutil.copy(index_path, os.path.join(args.output_dir, os.path.basename(index_path)))
    for filename in os.listdir(args.base_model_path):
        if filename.endswith(('.json', '.model', '.py', '.md')):
            source_file = os.path.join(args.base_model_path, filename)
            dest_file = os.path.join(args.output_dir, filename)
            if not os.path.exists(dest_file):
                    shutil.copy(source_file, dest_file)
    print(f"模型成功合并并保存至: {args.output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, default="./downloaded_models/Qwen2-VL-7B-Instruct", help="基础模型A的路径。")
    parser.add_argument('--donor_model_path', type=str, default="./downloaded_models/llava-onevision-qwen2-7b-si-hf", help="贡献模型B的路径。")
    parser.add_argument('--original_model_path', type=str, default="./downloaded_models/Qwen2-7B-Instruct", help="原始共同祖先模型C的路径。")
    parser.add_argument('--args.cache_dir', type=str, default="./merged_models", help="cache目录")
    parser.add_argument('--args.output_dir', type=str, default="./merged_models", help="合并后模型的输出目录。")
    parser.add_argument('--top_k_ratio', type=float, default=0.1, help="【阶段二】用于选举关键神经元的Top-K比率。")
    parser.add_argument('--alpha', type=float, default=0.1, help="【阶段二】夏普斯惩罚系数，控制对高曲率区域的惩罚力度。")
    parser.add_argument('--lambda_proj', type=float, default=1.0, help="【阶段三】投影（相关）分量的合并系数。")
    parser.add_argument('--lambda_ortho', type=float, default=0.8, help="【阶段三】正交（无关）分量的合并系数，保护泛化性。")
    parser.add_argument('--lambda_norm', type=float, default=0.0, help="norm 参数的加权平均系数（不走梯度合并）。")
    parser.add_argument('--mode', type=str, default="SAFE", help="为本次合并配置命名。")
    args = parser.parse_args()