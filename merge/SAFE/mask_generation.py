import json
import os
import safetensors
import torch
from tqdm import tqdm
from utils import load_weights, need_merge, normalize_llm_keys
import os.path as osp


def regularized_disjoint_mask_generation(args):
        """阶段二：【SAfe】生成夏普斯感知的非冲突更新掩码。"""
        print("\n--- [阶段二: SAfe 夏普斯感知评分与掩码生成] ---")
        mask_cache_path = os.path.join(args.cache_dir, f"mask_r{args.top_k_ratio}_alpha{args.alpha}.pt")
        if os.path.exists(mask_cache_path) and not args.force_recompute:
            print("SAfe 非冲突掩码缓存文件已存在, 跳过。")
            return

        print("加载所有权重和缓存的激活...")
        weights_A = load_weights(args.base_model_path)
        weights_B_raw = load_weights(args.donor_model_path)
        weights_C_raw = load_weights(args.original_model_path)
        
        weights_B = normalize_llm_keys(weights_B_raw, list(weights_A.keys())); del weights_B_raw
        weights_C = normalize_llm_keys(weights_C_raw, list(weights_A.keys())); del weights_C_raw

        
        A_activations_path = osp.basename(args.base_model_path.rstrip(os.sep)) + "_meta.pt"
        if osp.exists(osp.join(args.cache_dir, "activations", A_activations_path)):
            print(f"从缓存加载 A 模型激活: {A_activations_path}")
            activations_A = torch.load(osp.join(args.cache_dir, "activations", A_activations_path))
        else:
            raise FileNotFoundError(f"A 模型激活文件未找到: {osp.join(args.cache_dir, 'activations', A_activations_path)}")
        B_activations_path = osp.basename(args.donor_model_path.rstrip(os.sep)) + "_meta.pt"
        if osp.exists(osp.join(args.cache_dir, "activations", B_activations_path)):
            print(f"从缓存加载 B 模型激活: {B_activations_path}")
            activations_B = torch.load(osp.join(args.cache_dir, "activations", B_activations_path))
        else:
            raise FileNotFoundError(f"B 模型激活文件未找到: {osp.join(args.cache_dir, 'activations', B_activations_path)}")
        C_activations_path = osp.basename(args.original_model_path.rstrip(os.sep)) + "_meta.pt"
        if osp.exists(osp.join(args.cache_dir, "activations", C_activations_path)):
            print(f"从缓存加载 C 模型激活: {C_activations_path}")
            activations_C = torch.load(osp.join(args.cache_dir, "activations", C_activations_path))
        else:
            raise FileNotFoundError(f"C 模型激活文件未找到: {osp.join(args.cache_dir, 'activations', C_activations_path)}")

        activations = {
            'A': activations_A,
            'B': activations_B,
            'C': activations_C
        }

        disjoint_masks = {}
        pbar = tqdm(weights_A.keys(), desc="【SAFE】分析神经元")
        for key in pbar:
            if not need_merge(key): 
                continue
            if not (key in weights_B and key in weights_C): 
                continue

            module_name = ".".join(key.replace("language_model.model.", "model.").split('.')[1:-1])
            if module_name not in activations['A'] or 'output' not in activations['A'][module_name]:
                pbar.set_description(f"警告: 模块 {module_name} 激活缺失，跳过 {key}")
                continue

            try:
                W_A, W_B, W_C = weights_A[key].float(), weights_B[key].float(), weights_C[key].float()
                # 1) 构造近似“梯度方向”
                if W_A.ndim == 2 and key.endswith(".weight"):
                    # 2D: outer(output_A, input_A) / outer((output_B-output_C), input_A)
                    if 'input' not in activations['A'][module_name]:
                        pbar.set_description(f"警告: {module_name} 无 input 激活，跳过 {key}")
                        continue
                    out_A = activations['A'][module_name]['output']
                    in_A  = activations['A'][module_name]['input']
                    out_B = activations['B'][module_name].get('output', None)
                    out_C = activations['C'][module_name].get('output', None)
                    if out_B is None or out_C is None:
                        pbar.set_description(f"警告: {module_name} 无 B/C 输出，跳过 {key}")
                        continue

                    g_approx_A = torch.outer(out_A, in_A)
                    g_approx_B = torch.outer(out_B - out_C, in_A)

                elif W_A.ndim == 1 and key.endswith(".bias"):
                    # 1D bias: 用输出差向量作为方向
                    out_A = activations['A'][module_name]['output']
                    out_B = activations['B'][module_name].get('output', None)
                    out_C = activations['C'][module_name].get('output', None)
                    if out_B is None or out_C is None:
                        pbar.set_description(f"警告: {module_name} 无 B/C 输出，跳过 {key}")
                        continue
                    g_approx_A = out_A        # A 的输出强度作为显著性参考
                    g_approx_B = (out_B - out_C)  # B 相对 C 的输出变化决定注入方向
                else:
                    # 非预期形状，跳过
                    continue

                # 2) 夏普斯感知的显著性评分与掩码
                saliency_A = (W_A * g_approx_A).abs()
                sharpness_penalty_A = 1 + args.alpha * (g_approx_A**2)
                s_sas_A = saliency_A / sharpness_penalty_A

                saliency_B = (W_B * g_approx_B).abs()
                sharpness_penalty_B = 1 + args.alpha * (g_approx_B**2)
                s_sas_B = saliency_B / sharpness_penalty_B

                k = max(1, int(s_sas_A.numel() * args.top_k_ratio))
                if k <= 0: 
                    continue

                # 维度自适应 top-k
                th_A = torch.topk(s_sas_A.flatten(), k=k, sorted=False).values.min()
                th_B = torch.topk(s_sas_B.flatten(), k=k, sorted=False).values.min()
                mask_A = (s_sas_A >= th_A)
                mask_B = (s_sas_B >= th_B)

                tau_A = W_A - W_C
                tau_B = W_B - W_C

                # 冲突：同一位置两侧入选但方向相反
                conflict_mask = mask_A & mask_B & (torch.sign(tau_A) != torch.sign(tau_B))
                disjoint_mask_B = mask_B & (~conflict_mask)

                disjoint_masks[key] = disjoint_mask_B.cpu()

            except KeyError:
                pbar.set_description(f"警告: 模块 {module_name} 激活数据不完整，跳过 {key}")
                continue

        torch.save(disjoint_masks, mask_cache_path)
        print(f"SAFE非冲突掩码计算完成并缓存至: {mask_cache_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, default="/root/autodl-tmp/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct", help="基础模型A的路径。")
    parser.add_argument('--donor_model_path', type=str, default="/root/autodl-tmp/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si-hf", help="贡献模型B的路径。")
    parser.add_argument('--original_model_path', type=str, default="/root/autodl-tmp/AdaMMS/downloaded_models/Qwen2-7B-Instruct", help="原始共同祖先模型C的路径。")
    parser.add_argument("--cache_dir", type=str, default="/root/autodl-tmp/AdaMMS/merge/SAFE/activations", help="缓存目录")
    parser.add_argument("--top_k_ratio", type=float, default=0.1, help="选择的神经元比例")
    parser.add_argument("--alpha", type=float, default=0.1, help="夏普斯惩罚系数")
    parser.add_argument("--force_recompute", action='store_true', help="强制重新计算掩码")
    args = parser.parse_args()

    regularized_disjoint_mask_generation(args)