################################################################################
#
#       完整、独立的异构大语言模型合并脚本 (直接操作层版本)
#       LLaMA2-7B & Qwen2-7B
#       保证核心算法与官方 GitHub 代码一致，且流程清晰。
#
################################################################################

import os
import random
import torch
import torch.nn as nn
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from copy import deepcopy

# --- 1. 核心配置 ---
CKPT_PATH = {
    "llama2": "./downloaded_models/Llama-2-7b-hf",
    "qwen2": "./downloaded_models/Qwen2-7B-Instruct",
}
# 定义设备。如果显存不足，可将一个模型加载到CPU
MODEL_DEVICE_A = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
MODEL_DEVICE_B = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
COMPUTE_DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(f"模型A设备: {MODEL_DEVICE_A}, 模型B设备: {MODEL_DEVICE_B}, 计算设备: {COMPUTE_DEVICE}")

# --- 2. 辅助函数 (模型与数据加载) ---

def load_model_and_tokenizer(model_id, device):
    """通用模型加载函数"""
    print(f"正在加载模型: {model_id} -> {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"{model_id.split('/')[-1]} 模型加载完成。")
    return tokenizer, model

def prepare_dataloader(tokenizer_a, tokenizer_b, dataset_name="wikitext", max_samples=32, batch_size=2, max_length=128):
    """加载并为两个不同的分词器准备数据集"""
    print(f"正在加载并准备数据集: {dataset_name}...")
    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split="test").select(range(max_samples))
    dataset = dataset.filter(lambda ex: ex.get('text') and ex['text'].strip())

    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        inputs_a = tokenizer_a(texts, padding="max_length", truncation=True, return_tensors="pt", max_length=max_length)
        inputs_b = tokenizer_b(texts, padding="max_length", truncation=True, return_tensors="pt", max_length=max_length)
        return {"inputs_a": inputs_a, "inputs_b": inputs_b}

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

def get_module_by_name(model, module_name):
    """通过字符串名称安全地获取模块"""
    for part in module_name.split('.'):
        model = getattr(model, part, None)
        if model is None: return None
    return model

def register_hooks_for_reps(model, layer_names):
    """为指定层注册钩子以捕获输出激活，并立即转移到CPU以节省显存"""
    reps, hooks = {name: [] for name in layer_names}, []
    
    def hook_fn(name):
        def hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            reps[name].append(hidden_states.detach().cpu())
        return hook

    for name in layer_names:
        module = get_module_by_name(model, name)
        if module:
            hooks.append(module.register_forward_hook(hook_fn(name)))
        else:
            print(f"警告: 在模型中未找到层 '{name}'，无法注册钩子。")
            
    return reps, hooks


# --- 3. 核心算法：CKA & 深度对齐 (与官方仓库逻辑等价) ---

def center_gram(gram):
    """中心化Gram矩阵"""
    if gram.shape[0] == 0: return gram
    gram = gram.clone().float()
    gram -= gram.mean(dim=0, keepdim=True)
    gram -= gram.mean(dim=1, keepdim=True)
    return gram

def cka(gram_k, gram_l):
    """计算中心核对齐(CKA)"""
    gram_k = center_gram(gram_k)
    gram_l = center_gram(gram_l)
    scaled_hsic = torch.sum(gram_k * gram_l)
    norm_k = torch.norm(gram_k)
    norm_l = torch.norm(gram_l)
    return scaled_hsic / (norm_k * norm_l) if norm_k * norm_l != 0 else torch.tensor(0.0)

def compute_cka_matrix(reps1, reps2, names1, names2, max_tokens=4096):
    """高效计算CKA相似度矩阵，通过采样token来管理内存"""
    print("开始计算CKA矩阵...")
    cka_matrix = torch.zeros(len(names1), len(names2))
    
    processed_reps1 = {name: torch.cat(reps1[name], dim=0).flatten(0, 1).to(torch.float32) for name in names1 if reps1[name]}
    processed_reps2 = {name: torch.cat(reps2[name], dim=0).flatten(0, 1).to(torch.float32) for name in names2 if reps2[name]}

    for i, name1 in enumerate(tqdm(names1, desc="计算 CKA (Deep Model)")):
        if name1 not in processed_reps1: continue
        feat1_full = processed_reps1[name1]
        feat1 = feat1_full[torch.randperm(feat1_full.shape[0])[:max_tokens]].to(COMPUTE_DEVICE)
        gram_k = feat1 @ feat1.T
        
        for j, name2 in enumerate(names2):
            if name2 not in processed_reps2: continue
            feat2_full = processed_reps2[name2]
            feat2 = feat2_full[torch.randperm(feat2_full.shape[0])[:max_tokens]].to(COMPUTE_DEVICE)
            gram_l = feat2 @ feat2.T
            
            min_dim = min(gram_k.shape[0], gram_l.shape[0])
            if min_dim > 0:
              cka_matrix[i, j] = cka(gram_k[:min_dim, :min_dim], gram_l[:min_dim, :min_dim])
        
        del gram_k, feat1; gc.collect(); torch.cuda.empty_cache()
    print("CKA矩阵计算完成。")
    return cka_matrix.cpu()

def align_layers_lma(C):
    """使用LMA算法进行深度对齐 (逻辑等价于官方仓库的 align2)"""
    m, n = C.shape # m: deep, n: shallow
    F = torch.full((n + 1, m + 1), -torch.inf)
    F[0, :] = 0
    path = torch.zeros((n + 1, m + 1), dtype=torch.long)
    
    for i in range(1, n + 1):
        for j in range(i, m + 1):
            max_val, best_k = -torch.inf, -1
            for k in range(i - 1, j):
                segment_sim = C[k:j, i - 1].sum()
                current_val = F[i - 1, k] + segment_sim
                if current_val > max_val:
                    max_val, best_k = current_val, k
            F[i, j] = max_val
            path[i, j] = best_k

    alignment, i_idx, j_idx = [], n, m
    while i_idx > 0:
        k_idx = path[i_idx, j_idx]
        alignment.insert(0, list(range(k_idx, j_idx)))
        j_idx = k_idx
        i_idx -= 1
        
    return alignment

# --- 4. 宽度对齐与权重变换 (与官方仓库逻辑等价) ---
def match_tensors_zipit_cpu(reps_a, reps_b, target_dim):
    """在CPU上执行ZipIt!算法，以节省GPU显存"""
    dim_a, dim_b = reps_a.shape[1], reps_b.shape[1]
    
    reps_a_cpu = (reps_a / (torch.norm(reps_a, p=2, dim=0, keepdim=True) + 1e-6)).cpu().to(torch.float32)
    reps_b_cpu = (reps_b / (torch.norm(reps_b, p=2, dim=0, keepdim=True) + 1e-6)).cpu().to(torch.float32)
    
    all_reps = torch.cat([reps_a_cpu, reps_b_cpu], dim=1)
    sim_matrix = all_reps.T @ all_reps
    sim_matrix.fill_diagonal_(-torch.inf)
    
    total_dim = dim_a + dim_b
    perm_matrix = torch.eye(total_dim, device='cpu', dtype=torch.bfloat16)

    num_merges = total_dim - target_dim
    for _ in tqdm(range(num_merges), desc=f"Zipping to {target_dim} dims on CPU", leave=False):
        flat_idx = torch.argmax(sim_matrix)
        idx1, idx2 = np.unravel_index(flat_idx.item(), sim_matrix.shape)
        
        if sim_matrix[idx1, idx2] == -torch.inf: break

        perm_matrix[:, idx1] += perm_matrix[:, idx2]
        sim_matrix[:, idx1] = (sim_matrix[:, idx1] + sim_matrix[:, idx2]) / 2
        sim_matrix[idx1, :] = (sim_matrix[idx1, :] + sim_matrix[idx2, :]) / 2
        
        perm_matrix[:, idx2] = 0
        sim_matrix[idx2, :] = -torch.inf
        sim_matrix[:, idx2] = -torch.inf
        sim_matrix[idx1, idx1] = -torch.inf

    unmerge_matrix = perm_matrix[:, perm_matrix.sum(dim=0) != 0]
    merge_matrix = unmerge_matrix.T
    merge_matrix = merge_matrix / (torch.sum(merge_matrix, dim=1, keepdim=True) + 1e-6)

    Tm_a, Tm_b = merge_matrix[:, :dim_a], merge_matrix[:, dim_a:]
    Tu_a, Tu_b = unmerge_matrix[:dim_a, :], unmerge_matrix[dim_a:, :]

    return Tm_a, Tm_b, Tu_a, Tu_b

def apply_transform_and_merge(base_proj, donor_proj, T_out, T_in_inv, alpha):
    """
    应用变换 W_donor' = T_out @ W_donor @ T_in⁻¹ 并与 W_base 合并。
    智能处理偏置项。
    """
    dtype, device = base_proj.weight.dtype, base_proj.weight.device
    T_out, T_in_inv = T_out.to(device, dtype), T_in_inv.to(device, dtype)

    w_d_transformed = T_out @ donor_proj.weight.data.to(device, dtype) @ T_in_inv
    
    base_proj.weight.data.mul_(1 - alpha).add_(w_d_transformed * alpha)

    if hasattr(base_proj, 'bias') and base_proj.bias is not None and hasattr(donor_proj, 'bias') and donor_proj.bias is not None:
        bias_d_transformed = T_out @ donor_proj.bias.data.to(device, dtype)
        base_proj.bias.data.mul_(1 - alpha).add_(bias_d_transformed * alpha)
    # 如果只有 donor 有 bias 而 base 没有，则不进行操作
    # 如果只有 base 有 bias，则它会按 1-alpha 缩放


def set_seed(seed: int):
    """为CPU和GPU设置随机种子以确保可复现性。"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保卷积操作的可复现性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- 5. 主执行流程 ---

def main(alpha=0.5):
    """执行模型对齐与合并的主函数"""
    set_seed(42)

    # --- 步骤 1: 加载模型和数据 ---
    tokenizer_llama, model_llama = load_model_and_tokenizer(CKPT_PATH["llama2"], MODEL_DEVICE_A)
    tokenizer_qwen, model_qwen = load_model_and_tokenizer(CKPT_PATH["qwen2"], MODEL_DEVICE_B)
    
    if model_llama.config.num_hidden_layers >= model_qwen.config.num_hidden_layers:
        model_deep, model_shallow, tok_deep, tok_shallow = model_llama, model_qwen, tokenizer_llama, tokenizer_qwen
        deep_name, shallow_name = "llama2", "qwen2"
        deep_device, shallow_device = MODEL_DEVICE_A, MODEL_DEVICE_B
    else:
        model_deep, model_shallow = model_qwen, model_llama
        tok_deep, tok_shallow = tokenizer_qwen, tokenizer_llama
        deep_name, shallow_name = "qwen2", "llama2"
        deep_device, shallow_device = MODEL_DEVICE_B, MODEL_DEVICE_A

    names_deep = [f"model.layers.{i}" for i in range(model_deep.config.num_hidden_layers)]
    names_shallow = [f"model.layers.{i}" for i in range(model_shallow.config.num_hidden_layers)]
    
    dataloader = prepare_dataloader(tok_deep, tok_shallow, batch_size=2, max_samples=16)

    # --- 步骤 2: 提取激活值 ---
    print("\n--- 步骤 2: 为两个模型收集特征表示 ---")
    reps_deep, hooks_deep = register_hooks_for_reps(model_deep, names_deep)
    reps_shallow, hooks_shallow = register_hooks_for_reps(model_shallow, names_shallow)
    
    for batch in tqdm(dataloader, desc="特征提取"):
        with torch.no_grad():
            # 分别在各自的设备上运行
            model_deep(batch["inputs_a"].to(deep_device))
            model_shallow(batch["inputs_b"].to(shallow_device))
    
    for hook in hooks_deep + hooks_shallow: hook.remove()

    # --- 步骤 3: 深度对齐 ---
    print(f"\n--- 步骤 3: 深度异构对齐 (LMA) ---")
    cka_matrix = compute_cka_matrix(reps_deep, reps_shallow, names_deep, names_shallow)
    layer_alignment = align_layers_lma(cka_matrix)
    
    print("\n找到的层级映射关系 (Shallow Layer -> Deep Segment):")
    for i, segment in enumerate(layer_alignment):
        print(f"  {names_shallow[i]} -> {[names_deep[j] for j in segment]}")

    # --- 步骤 4: 宽度对齐与权重合并 ---
    print(f"\n--- 步骤 4: 宽度异构合并 (alpha={alpha}) ---")
    merged_model = deepcopy(model_shallow)

    for i, deep_segment_indices in enumerate(tqdm(layer_alignment, desc="合并所有层段")):
        shallow_layer_name = names_shallow[i]
        
        # 使用段的最后一层激活作为整个段的代表进行宽度对齐
        last_deep_layer_name = names_deep[deep_segment_indices[-1]]
        reps_deep_segment = torch.cat(reps_deep[last_deep_layer_name], dim=0).flatten(0, 1)
        reps_shallow_layer = torch.cat(reps_shallow[shallow_layer_name], dim=0).flatten(0, 1)
        
        # 计算宽度变换矩阵
        Tm_s, Tm_d, Tu_s, Tu_d = match_tensors_zipit_cpu(reps_shallow_layer, reps_deep_segment, merged_model.config.hidden_size)

        base_target_layer = get_module_by_name(merged_model, shallow_layer_name)
        
        with torch.no_grad():
            # 将段内所有层的贡献累加合并
            for k, deep_layer_idx in enumerate(deep_segment_indices):
                donor_layer = get_module_by_name(model_deep, names_deep[deep_layer_idx])
                
                T_out = Tu_s @ Tm_d
                T_in_inv = torch.linalg.pinv(Tu_d @ Tm_s)

                # 合并注意力层
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    apply_transform_and_merge(getattr(base_target_layer.self_attn, proj), getattr(donor_layer.self_attn, proj), T_out, T_in_inv, alpha / len(deep_segment_indices))
                
                # 合并MLP层
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                     apply_transform_and_merge(getattr(base_target_layer.mlp, proj), getattr(donor_layer.mlp, proj), T_out, T_in_inv, alpha / len(deep_segment_indices))

    # --- 步骤 5: 保存并测试 ---
    output_dir = f"./merged_{shallow_name}_as_base_alpha_{alpha}"
    print(f"\n--- 正在保存合并后的模型到 {output_dir} ---")
    merged_model.save_pretrained(output_dir)
    tok_shallow.save_pretrained(output_dir)
    print("模型保存完成。")

    del model_donor, model_base, reps_deep, reps_shallow, dataloader, merged_model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n--- 测试合并后的模型 ---")
    tokenizer_merged, model_merged = load_model_and_tokenizer(output_dir, COMPUTE_DEVICE)
    prompt = "The capital of France is"
    inputs = tokenizer_merged(prompt, return_tensors="pt").to(COMPUTE_DEVICE)
    with torch.no_grad():
        outputs = model_merged.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer_merged.eos_token_id)
    print("输入:", prompt)
    print("合并后模型输出:", tokenizer_merged.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main(alpha=0.5)