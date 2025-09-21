from collections import defaultdict
import json
import os
import os.path as osp
import shutil
import safetensors
import torch
from tqdm import tqdm

from utils import load_weights, need_merge

EPS = 1e-9

def disentangled_reprojection_fusion(args):
    """阶段三：执行解耦重投影融合。"""
    print("\n--- [阶段三: 解耦重投影融合] ---")
    
    print("加载所有权重、掩码和激活...")
    weights_A = load_weights(args.base_model_path)
    weights_B_raw = load_weights(args.donor_model_path)
    weights_C_raw = load_weights(args.original_model_path)

    # 与阶段二一致：掩码文件名
    mask_cache_path = os.path.join(args.cache_dir, f"mask_r{args.top_k_ratio}_alpha{args.alpha}.pt")
    disjoint_masks = torch.load(mask_cache_path, map_location="cpu")

    # 激活文件：按 basename(model_path)_meta.pt 加载，并做键名规范化
    def canon_module_name(name: str) -> str:
        k = name.replace("language_model.model.", "model.").replace("language_model.", "model.")
        if "layers" in k:
            pos = k.find("layers")
            k = "model." + k[pos:]
        return k

    def canon_activations(acts: dict) -> dict:
        return {canon_module_name(k): v for k, v in acts.items()}

    A_activations_path = osp.basename(args.base_model_path.rstrip(os.sep)) + "_meta.pt"
    B_activations_path = osp.basename(args.donor_model_path.rstrip(os.sep)) + "_meta.pt"
    C_activations_path = osp.basename(args.original_model_path.rstrip(os.sep)) + "_meta.pt"

    activations_A = canon_activations(torch.load(osp.join(args.cache_dir, A_activations_path), map_location="cpu"))
    activations_B = canon_activations(torch.load(osp.join(args.cache_dir, B_activations_path), map_location="cpu"))
    activations_C = canon_activations(torch.load(osp.join(args.cache_dir, C_activations_path), map_location="cpu"))

    # 参数键的规范化 + B/C 原始键映射
    def canon_param_key(param_key: str) -> str:
        k = param_key.replace("language_model.model.", "model.").replace("language_model.", "model.")
        if "layers" in k:
            pos = k.find("layers")
            k = "model." + k[pos:]
        return k

    def canon_module_from_param_key(param_key: str) -> str:
        k = canon_param_key(param_key)
        parts = k.split('.')
        if len(parts) >= 2:
            parts = parts[:-1]
        return '.'.join(parts)

    b_canon_to_orig = {}
    for k in weights_B_raw.keys():
        ck = canon_param_key(k)
        if ck not in b_canon_to_orig:
            b_canon_to_orig[ck] = k
    c_canon_to_orig = {}
    for k in weights_C_raw.keys():
        ck = canon_param_key(k)
        if ck not in c_canon_to_orig:
            c_canon_to_orig[ck] = k

    # 设备
    device = torch.device(getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    final_merged_weights = weights_A.copy()

    pbar = tqdm(weights_A.keys(), desc="执行重投影融合(仅need_merge)")
    for key in pbar:
        # 不需要复杂合并的参数：保留 A 原值
        if not need_merge(key):
            continue

        # 映射 B/C 原始键
        a_canon = canon_param_key(key)
        b_key = b_canon_to_orig.get(a_canon, None)
        c_key = c_canon_to_orig.get(a_canon, None)
        if b_key is None or c_key is None:
            # 找不到对应权重，保留 A
            continue

        # 掩码必须存在
        if key not in disjoint_masks:
            continue
        M_prime_B = disjoint_masks[key].to(device)

        # 读取权重
        W_A = weights_A[key].float()
        W_B = weights_B_raw[b_key].float()
        W_C = weights_C_raw[c_key].float()

        # 激活模块名
        module_name = canon_module_from_param_key(key)

        tau_B = (W_B - W_C).to(device)
        tau_B_update = tau_B * M_prime_B

        if W_A.ndim == 2 and key.endswith(".weight"):
            # 需要 A 的输入激活
            if module_name not in activations_A or 'input' not in activations_A[module_name]:
                continue
            d_i = activations_A[module_name]['input'].to(device).float()
            d_i_norm_sq = torch.sum(d_i * d_i)
            if d_i_norm_sq > EPS:
                proj_scalar = (tau_B_update @ d_i) / d_i_norm_sq
                tau_proj = torch.outer(proj_scalar, d_i)
                tau_ortho = tau_B_update - tau_proj
            else:
                tau_proj = torch.zeros_like(tau_B_update)
                tau_ortho = tau_B_update

        elif W_A.ndim == 1 and key.endswith(".bias"):
            # 需要 B/C 的输出激活方向
            out_B = activations_B.get(module_name, {}).get('output', None)
            out_C = activations_C.get(module_name, {}).get('output', None)
            if out_B is None or out_C is None:
                # 没有方向信息则跳过，保留 A
                continue
            g_dir = (out_B - out_C).to(device).float()
            g_norm_sq = torch.sum(g_dir * g_dir)
            if g_norm_sq > EPS:
                proj_scalar = torch.sum(tau_B_update * g_dir) / g_norm_sq
                tau_proj = proj_scalar * g_dir
                tau_proj = tau_proj * M_prime_B  # 仅掩码位置
                tau_ortho = tau_B_update - tau_proj
            else:
                tau_proj = torch.zeros_like(tau_B_update)
                tau_ortho = tau_B_update
        else:
            # 其他形状不处理
            continue

        W_star = W_A.to(device) + args.lambda_proj * tau_proj + args.lambda_ortho * tau_ortho
        final_merged_weights[key] = W_star.cpu().to(weights_A[key].dtype)

    _save_model(args, final_merged_weights)

def _save_model(args, merged_weights):
    """保存模型权重。"""
    print("\n正在保存合并后的模型...")
    output_dir = osp.basename(args.base_model_path.rstrip(os.sep))
    output_dir = osp.join(args.output_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    sft_index = os.path.join(args.base_model_path, "model.safetensors.index.json")
    bin_index = os.path.join(args.base_model_path, "pytorch_model.bin.index.json")

    def copy_side_files():
        for filename in os.listdir(args.base_model_path):
            if filename.endswith((".json", ".model", ".py", ".md")):
                src = os.path.join(args.base_model_path, filename)
                dst = os.path.join(output_dir, filename)
                if not os.path.exists(dst):
                    shutil.copy(src, dst)

    if os.path.exists(sft_index):
        # Save as safetensors shards following index map
        with open(sft_index, "r") as f:
            index_map = json.load(f)["weight_map"]
        sharded_weights = defaultdict(dict)
        for key, value in merged_weights.items():
            if key in index_map:
                sharded_weights[index_map[key]][key] = value
        for filename, weights_dict in sharded_weights.items():
            safetensors.torch.save_file(weights_dict, os.path.join(output_dir, filename))
        shutil.copy(sft_index, os.path.join(output_dir, os.path.basename(sft_index)))
        copy_side_files()
        print(f"模型成功合并并保存至: {output_dir} (safetensors 分片)")
        return

    if os.path.exists(bin_index):
        # Save as PyTorch .bin shards following index map
        with open(bin_index, "r") as f:
            index_map = json.load(f)["weight_map"]
        sharded_weights = defaultdict(dict)
        for key, value in merged_weights.items():
            if key in index_map:
                sharded_weights[index_map[key]][key] = value
        for filename, weights_dict in sharded_weights.items():
            torch.save(weights_dict, os.path.join(output_dir, filename))
        shutil.copy(bin_index, os.path.join(output_dir, os.path.basename(bin_index)))
        copy_side_files()
        print(f"模型成功合并并保存至: {output_dir} (.bin 分片)")
        return

    # Fallback: single-file save
    sft_single = os.path.join(args.base_model_path, "model.safetensors")
    bin_single = os.path.join(args.base_model_path, "pytorch_model.bin")
    if os.path.exists(sft_single):
        out_path = os.path.join(output_dir, os.path.basename(sft_single))
        safetensors.torch.save_file(merged_weights, out_path)
        copy_side_files()
        print(f"模型成功合并并保存至: {out_path} (单一 safetensors)")
        return
    if os.path.exists(bin_single):
        out_path = os.path.join(output_dir, os.path.basename(bin_single))
        torch.save(merged_weights, out_path)
        copy_side_files()
        print(f"模型成功合并并保存至: {out_path} (单一 .bin)")
        return

    # If none detected, default to safetensors single-file name
    out_path = os.path.join(output_dir, "model.safetensors")
    safetensors.torch.save_file(merged_weights, out_path)
    copy_side_files()
    print(f"模型成功合并并保存至: {out_path} (默认 safetensors)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, default="/root/autodl-tmp/AdaMMS/downloaded_models/mplug-owl2-llama2-7b", help="基础模型A的路径。")
    parser.add_argument('--donor_model_path', type=str, default="/root/autodl-tmp/AdaMMS/downloaded_models/llava-v1.5-7b", help="贡献模型B的路径。")
    parser.add_argument('--original_model_path', type=str, default="/root/autodl-tmp/AdaMMS/downloaded_models/Llama-2-7b-hf", help="原始共同祖先模型C的路径。")
    parser.add_argument('--cache_dir', type=str, default="/root/autodl-tmp/AdaMMS/merge/SAFE/activations", help="cache目录（掩码/激活存放处）")
    parser.add_argument('--output_dir', type=str, default="/root/autodl-tmp/AdaMMS/merge/SAFE/output", help="合并后模型的输出目录。")
    parser.add_argument('--top_k_ratio', type=float, default=0.1, help="【阶段二】用于选举关键神经元的Top-K比率。")
    parser.add_argument('--alpha', type=float, default=0.1, help="【阶段二】夏普斯惩罚系数，控制对高曲率区域的惩罚力度。")
    parser.add_argument('--lambda_proj', type=float, default=1.0, help="【阶段三】投影（相关）分量的合并系数。")
    parser.add_argument('--lambda_ortho', type=float, default=0.8, help="【阶段三】正交（无关）分量的合并系数，保护泛化性。")
    parser.add_argument('--lambda_norm', type=float, default=0.0, help="norm 参数的加权平均系数（不走梯度合并）。")
    parser.add_argument('--mode', type=str, default="SAFE", help="为本次合并配置命名。")
    parser.add_argument('--device', type=str, default="cuda", help="PyTorch 设备，如 cuda:0 或 cpu；默认自动选择。")
    args = parser.parse_args()

    disentangled_reprojection_fusion(args)