import os
import torch
import torch.nn as nn
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

# --- 1. 核心配置 ---
CKPT_PATH = {
    "llama2": "./downloaded_models/Llama-2-7b-hf",
    "qwen2": "./downloaded_models/Qwen2-7B-Instruct",
}
# 定义计算设备。如果显存不足，可将一个模型加载到CPU
MODEL_DEVICE_A = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
MODEL_DEVICE_B = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
COMPUTE_DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"模型A设备: {MODEL_DEVICE_A}, 模型B设备: {MODEL_DEVICE_B}, 计算设备: {COMPUTE_DEVICE}")

# --- 2. 辅助函数 (模型与数据加载) ---
# (这部分函数保持不变，为简洁起见，此处省略，请使用上一版代码中的函数)
def load_complete_model(model_id, device):
    print(f"正在加载模型: {model_id} -> {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"{model_id.split('/')[-1]} 模型加载完成。")
    return tokenizer, model

def load_and_prepare_dataset(tokenizer_a, tokenizer_b, dataset_name="wikitext", split="test", max_samples=16, max_length=128):
    print(f"正在加载数据集: {dataset_name}...")
    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split).select(range(max_samples))
    def tokenize_fn(examples):
        text = [t for t in examples["text"] if t and t.strip()]
        if not text: return {}
        inputs_a = tokenizer_a(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        inputs_b = tokenizer_b(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids_a": inputs_a.input_ids, "attention_mask_a": inputs_a.attention_mask, "input_ids_b": inputs_b.input_ids, "attention_mask_b": inputs_b.attention_mask}
    processed_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    processed_dataset.set_format(type='torch')
    return processed_dataset

def get_module_by_name(model, module_name):
    for part in module_name.split('.'):
        if not hasattr(model, part): return None
        model = getattr(model, part)
    return model

# 修正：为输入和输出都注册钩子
def register_hooks_for_reps(model, layer_names):
    reps_in, reps_out, hooks = {n: [] for n in layer_names}, {n: [] for n in layer_names}, []
    def get_hook_fn(name):
        def hook_fn(module, input, output):
            reps_in[name].append(input[0].detach().cpu())
            reps_out[name].append(output.detach().cpu())
        return hook_fn
    for name in layer_names:
        if module := get_module_by_name(model, name):
            hooks.append(module.register_forward_hook(get_hook_fn(name)))
    return reps_in, reps_out, hooks

# --- 3. 核心算法：CKA & 深度对齐 (LMA) ---
# (这部分函数保持不变，为简洁起见，此处省略，请使用上一版代码中的函数)
def cka(gram_k, gram_l):
    gram_k = center_gram(gram_k.float())
    gram_l = center_gram(gram_l.float())
    scaled_hsic = torch.sum(gram_k * gram_l)
    norm_k = torch.norm(gram_k)
    norm_l = torch.norm(gram_l)
    return scaled_hsic / (norm_k * norm_l) if norm_k != 0 and norm_l != 0 else torch.tensor(0.0)

def center_gram(gram):
    n = gram.shape[0]
    I = torch.eye(n, device=gram.device)
    H = I - 1/n * torch.ones(n, n, device=gram.device)
    return H @ gram @ H

def compute_cka_matrix(reps1, reps2, names1, names2, max_tokens=4096):
    print("开始计算CKA矩阵...")
    cka_matrix = torch.zeros(len(names1), len(names2))
    processed_reps1 = {name: torch.cat(reps1[name], dim=0).flatten(0, 1).to(torch.float32) for name in names1}
    processed_reps2 = {name: torch.cat(reps2[name], dim=0).flatten(0, 1).to(torch.float32) for name in names2}
    for i, name1 in enumerate(tqdm(names1, desc="Deep Model Layers")):
        feat1_full = processed_reps1[name1]
        feat1 = feat1_full[torch.randperm(feat1_full.shape[0])[:max_tokens]].to(COMPUTE_DEVICE)
        gram_k = feat1 @ feat1.T
        for j, name2 in enumerate(names2):
            feat2_full = processed_reps2[name2]
            feat2 = feat2_full[torch.randperm(feat2_full.shape[0])[:max_tokens]].to(COMPUTE_DEVICE)
            gram_l = feat2 @ feat2.T
            min_dim = min(gram_k.shape[0], gram_l.shape[0])
            cka_matrix[i, j] = cka(gram_k[:min_dim, :min_dim], gram_l[:min_dim, :min_dim])
        del gram_k; gc.collect(); torch.cuda.empty_cache()
    print("CKA矩阵计算完成。")
    return cka_matrix.cpu()

def align_layers_lma(C):
    m, n = C.shape; F = torch.full((n + 1, m + 1), -torch.inf); F[0, :] = 0
    path = torch.zeros((n + 1, m + 1), dtype=torch.long)
    for i in range(1, n + 1):
        for j in range(i, m + 1):
            max_val, best_k = -torch.inf, -1
            for k in range(i - 1, j):
                segment_sim = C[k:j, i - 1].sum()
                current_val = F[i - 1, k] + segment_sim
                if current_val > max_val: max_val, best_k = current_val, k
            F[i, j], path[i, j] = max_val, best_k
    alignment, i, j = [], n, m
    while i > 0:
        k = path[i, j]; alignment.insert(0, list(range(k, j))); j = k; i -= 1
    return alignment

# --- 4. 宽度异构合并：弹性神经元压缩 (ZipIt!) - CPU优化版 ---
# (此函数保持不变)
def match_tensors_zipit(reps_a, reps_b, target_dim):
    dim_a, dim_b = reps_a.shape[1], reps_b.shape[1]
    reps_a_cpu, reps_b_cpu = reps_a.cpu().to(torch.float32), reps_b.cpu().to(torch.float32)
    reps_a_norm = reps_a_cpu / (torch.norm(reps_a_cpu, p=2, dim=0, keepdim=True) + 1e-6)
    reps_b_norm = reps_b_cpu / (torch.norm(reps_b_cpu, p=2, dim=0, keepdim=True) + 1e-6)
    all_reps = torch.cat([reps_a_norm, reps_b_norm], dim=1) # covariance没有实现
    sim_matrix = all_reps.T @ all_reps
    sim_matrix.fill_diagonal_(-torch.inf)
    total_dim = dim_a + dim_b
    perm_matrix = torch.eye(total_dim, device='cpu', dtype=torch.bfloat16)
    num_merges = total_dim - target_dim
    for _ in tqdm(range(num_merges), desc=f"Zipping to {target_dim} dims on CPU", leave=False): # 能不能结合svd的方式来加速计算合并
        flat_idx = torch.argmax(sim_matrix)
        idx1, idx2 = np.unravel_index(flat_idx.item(), sim_matrix.shape)
        if sim_matrix[idx1, idx2] == -torch.inf: break
        perm_matrix[:, idx1] += perm_matrix[:, idx2]
        sim_matrix[:, idx1] = (sim_matrix[:, idx1] + sim_matrix[:, idx2]) / 2
        sim_matrix[idx1, :] = (sim_matrix[idx1, :] + sim_matrix[idx2, :]) / 2
        perm_matrix[:, idx2] = 0
        sim_matrix[idx2, :], sim_matrix[:, idx2] = -torch.inf, -torch.inf
        sim_matrix[idx1, idx1] = -torch.inf
    unmerge_matrix = perm_matrix[:, perm_matrix.sum(dim=0) != 0]
    merge_matrix = unmerge_matrix.T
    merge_matrix = merge_matrix / (torch.sum(merge_matrix, dim=1, keepdim=True) + 1e-6)
    Tm_a, Tm_b = merge_matrix[:, :dim_a], merge_matrix[:, dim_a:]
    Tu_a, Tu_b = unmerge_matrix[:dim_a, :], unmerge_matrix[dim_a:, :]
    return Tm_a, Tm_b, Tu_a, Tu_b

def match_tensors_zipit_optimized(reps_a, reps_b, target_dim, use_svd_init=True, batch_merges=True):
    """
    优化版的ZipIt算法实现，使用SVD进行初始降维并批量处理合并操作
    
    参数:
    - reps_a, reps_b: 需要对齐的两个表示
    - target_dim: 目标维度
    - use_svd_init: 是否使用SVD进行初始降维，加速后续迭代
    - batch_merges: 是否批量处理合并操作
    
    返回:
    - 变换矩阵: Tm_a, Tm_b, Tu_a, Tu_b
    """
    dim_a, dim_b = reps_a.shape[1], reps_b.shape[1]
    total_dim = dim_a + dim_b
    num_merges = total_dim - target_dim
    
    # 如果目标维度已满足或超过总维度，无需压缩
    if num_merges <= 0:
        # 直接返回单位变换矩阵
        Tm_a = torch.eye(target_dim, dim_a, dtype=torch.float32)
        Tm_b = torch.eye(target_dim, dim_b, dtype=torch.float32)
        Tu_a = torch.eye(dim_a, target_dim, dtype=torch.float32)
        Tu_b = torch.eye(dim_b, target_dim, dtype=torch.float32)
        return Tm_a, Tm_b, Tu_a, Tu_b
    
    # 转移到CPU并标准化
    reps_a_norm = reps_a.cpu().to(torch.float32)
    reps_b_norm = reps_b.cpu().to(torch.float32)
    
    # 按列标准化
    reps_a_norm = reps_a_norm / (torch.norm(reps_a_norm, p=2, dim=0, keepdim=True) + 1e-6)
    reps_b_norm = reps_b_norm / (torch.norm(reps_b_norm, p=2, dim=0, keepdim=True) + 1e-6)
    
    # 合并所有表示
    all_reps = torch.cat([reps_a_norm, reps_b_norm], dim=1)
    
    # 初始化置换矩阵
    perm_matrix = torch.eye(total_dim, device='cpu', dtype=torch.float32)
    
    # 使用SVD初始降维（如果启用）
    if use_svd_init and num_merges > total_dim // 3:  # 当需要大幅降维时，使用SVD
        print(f"使用SVD进行初始降维，从{total_dim}降至{target_dim*2}")
        try:
            # 计算协方差矩阵
            cov_matrix = all_reps.T @ all_reps
            
            # 进行特征值分解(SVD的简化版)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
            
            # 选择最大的target_dim*2个特征向量
            # 保留2倍的目标维度，以便后续细化
            keep_indices = torch.argsort(eigenvalues, descending=True)[:target_dim*2]
            reduced_basis = eigenvectors[:, keep_indices]
            
            # 使用SVD结果初始化置换矩阵
            # 此处不直接将维度降至target_dim，而是降至更大一些，然后再通过迭代细化
            perm_matrix = reduced_basis @ reduced_basis.T @ perm_matrix
            
            # 更新需要合并的数量
            num_merges = 2*target_dim - target_dim
            
            # 重新计算相似度矩阵
            sim_matrix = perm_matrix.T @ cov_matrix @ perm_matrix
            sim_matrix.fill_diagonal_(-torch.inf)
            
            del cov_matrix, eigenvalues, eigenvectors, reduced_basis
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
            
        except Exception as e:
            print(f"SVD初始化失败，回退到标准方法: {str(e)}")
            # 计算相似度矩阵
            sim_matrix = all_reps.T @ all_reps
            sim_matrix.fill_diagonal_(-torch.inf)
    else:
        # 标准方法：直接计算相似度矩阵
        sim_matrix = all_reps.T @ all_reps
        sim_matrix.fill_diagonal_(-torch.inf)
    
    # 批量合并处理
    if batch_merges and num_merges > 100:
        # 每次批量合并10%的神经元
        batch_size = max(int(num_merges * 0.1), 10)
        iterations = num_merges // batch_size
        
        for iteration in tqdm(range(iterations), desc=f"批量Zipping到{target_dim}维", leave=False):
            # 找出相似度最高的batch_size对神经元
            flat_indices = torch.topk(sim_matrix.flatten(), batch_size).indices
            
            # 转换为2D索引
            idx_pairs = [(idx // total_dim, idx % total_dim) for idx in flat_indices]
            
            # 过滤掉已经是-inf的对
            valid_pairs = [(i, j) for i, j in idx_pairs if sim_matrix[i, j] != -torch.inf]
            
            # 批量合并
            for idx1, idx2 in valid_pairs:
                # 合并操作
                perm_matrix[:, idx1] += perm_matrix[:, idx2]
                sim_matrix[:, idx1] = (sim_matrix[:, idx1] + sim_matrix[:, idx2]) / 2
                sim_matrix[idx1, :] = (sim_matrix[idx1, :] + sim_matrix[idx2, :]) / 2
                perm_matrix[:, idx2] = 0
                sim_matrix[idx2, :], sim_matrix[:, idx2] = -torch.inf, -torch.inf
                sim_matrix[idx1, idx1] = -torch.inf
        
        # 处理剩余的合并
        remaining = num_merges - batch_size * iterations
    else:
        remaining = num_merges
    
    # 处理剩余的合并（或所有合并，如果不使用批处理）
    for _ in tqdm(range(remaining), desc=f"精细Zipping到{target_dim}维", leave=False):
        # 找出相似度最高的一对神经元
        flat_idx = torch.argmax(sim_matrix)
        idx1, idx2 = flat_idx // total_dim, flat_idx % total_dim
        
        # 如果找不到有效的对，提前退出
        if sim_matrix[idx1, idx2] == -torch.inf:
            break
            
        # 合并操作
        perm_matrix[:, idx1] += perm_matrix[:, idx2]
        sim_matrix[:, idx1] = (sim_matrix[:, idx1] + sim_matrix[:, idx2]) / 2
        sim_matrix[idx1, :] = (sim_matrix[idx1, :] + sim_matrix[idx2, :]) / 2
        perm_matrix[:, idx2] = 0
        sim_matrix[idx2, :], sim_matrix[:, idx2] = -torch.inf, -torch.inf
        sim_matrix[idx1, idx1] = -torch.inf
    
    # 保留非零列构建变换矩阵
    nonzero_cols = perm_matrix.sum(dim=0) != 0
    unmerge_matrix = perm_matrix[:, nonzero_cols]
    
    # 转置得到合并矩阵
    merge_matrix = unmerge_matrix.T
    
    # 归一化
    merge_matrix = merge_matrix / (torch.sum(merge_matrix, dim=1, keepdim=True) + 1e-6)
    
    # 拆分为A和B的变换矩阵
    Tm_a, Tm_b = merge_matrix[:, :dim_a], merge_matrix[:, dim_a:]
    Tu_a, Tu_b = unmerge_matrix[:dim_a, :], unmerge_matrix[dim_a:, :]
    
    return Tm_a, Tm_b, Tu_a, Tu_b

# --- 5. 最终修复的核心函数：权重变换与合并 ---
def transform_and_merge_weights(base_proj, donor_proj, reps_in_b, reps_in_d, reps_out_b, reps_out_d, alpha):
    """对一对具体的投影层进行宽度对齐和合并, 使用输入和输出激活"""
    device = base_proj.weight.device
    dtype = base_proj.weight.dtype
    
    # 添加维度检查
    print(f"特征维度 - 输入基础: {reps_in_b.shape}, 输入捐赠: {reps_in_d.shape}")
    print(f"特征维度 - 输出基础: {reps_out_b.shape}, 输出捐赠: {reps_out_d.shape}")
    
    # 确保激活至少是2维的
    assert reps_in_b.dim() >= 2, f"输入基础特征维度不足: {reps_in_b.shape}"
    assert reps_in_d.dim() >= 2, f"输入捐赠特征维度不足: {reps_in_d.shape}"
    assert reps_out_b.dim() >= 2, f"输出基础特征维度不足: {reps_out_b.shape}"
    assert reps_out_d.dim() >= 2, f"输出捐赠特征维度不足: {reps_out_d.shape}"

    # 1. 计算输出空间的变换矩阵 (基于输出激活)
    target_dim_out = base_proj.out_features
    Tm_b_out, Tm_d_out, Tu_b_out, Tu_d_out = match_tensors_zipit_optimized(reps_out_b, reps_out_d, target_dim_out)
    T_out_d_to_b = (Tu_b_out @ Tm_d_out).to(device, dtype=dtype)

    # 2. 计算输入空间的变换矩阵 (基于输入激活)
    target_dim_in = base_proj.in_features
    Tm_b_in, Tm_d_in, Tu_b_in, Tu_d_in = match_tensors_zipit_optimized(reps_in_b, reps_in_d, target_dim_in)
    T_in_b_to_d = (Tu_d_in @ Tm_b_in).to(device, dtype=dtype)
    # T_in_inv = torch.linalg.pinv(T_in_b_to_d.to(torch.float32)).to(dtype)
    T_in_inv = T_in_b_to_d

    device = donor_proj.weight.device
    T_in_inv = T_in_inv.to(device, dtype=dtype)
    T_out_d_to_b = T_out_d_to_b.to(device, dtype=dtype)

    # 3. 应用正确的双向变换: W'_d = T_out @ W_d @ T_in⁻¹
    W_d_transformed = T_out_d_to_b @ donor_proj.weight.data @ T_in_inv

    assert base_proj.weight.data.shape == W_d_transformed.shape, \
        f"维度不匹配: base({base_proj.weight.data.shape}) vs transformed donor({W_d_transformed.shape})"

    device = base_proj.weight.device
    W_d_transformed = W_d_transformed.to(device, dtype=dtype)
    # 4. 加权平均权重
    base_proj.weight.data = (1 - alpha) * base_proj.weight.data + alpha * W_d_transformed
    
    # 5. Bias 只受输出变换的影响
    if hasattr(base_proj, 'bias') and base_proj.bias is not None and \
       hasattr(donor_proj, 'bias') and donor_proj.bias is not None:
        bias_d_transformed = T_out_d_to_b @ donor_proj.bias.data
        assert base_proj.bias.data.shape == bias_d_transformed.shape, "bias 维度不匹配"
        base_proj.bias.data = (1 - alpha) * base_proj.bias.data + alpha * bias_d_transformed
    
    del W_d_transformed, T_in_inv, T_out_d_to_b, Tm_b_out, Tm_d_out, Tu_b_out, Tu_d_out

# --- 6. 主执行流程 (已重构)---
def main(alpha=0.5, alignment_type='LMA'):
    tokenizer_donor, model_donor = load_complete_model(CKPT_PATH["llama2"], MODEL_DEVICE_A)
    tokenizer_base, model_base = load_complete_model(CKPT_PATH["qwen2"], MODEL_DEVICE_B)
    
    if model_donor.config.num_hidden_layers >= model_base.config.num_hidden_layers:
        model_deep, model_shallow, tok_deep, tok_shallow = model_donor, model_base, tokenizer_donor, tokenizer_base
        deep_name, shallow_name, deep_device, shallow_device = "llama2", "qwen2", MODEL_DEVICE_A, MODEL_DEVICE_B
    else:
        model_deep, model_shallow, tok_deep, tok_shallow = model_base, model_donor, tokenizer_base, tokenizer_donor
        deep_name, shallow_name, deep_device, shallow_device = "qwen2", "llama2", MODEL_DEVICE_B, MODEL_DEVICE_A

    names_deep_layers = [f"model.layers.{i}" for i in range(model_deep.config.num_hidden_layers)]
    names_shallow_layers = [f"model.layers.{i}" for i in range(model_shallow.config.num_hidden_layers)]

    print("\n--- 步骤 2: 为两个模型收集输入和输出激活 ---")
    proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    hook_names_deep = [f"{layer_name}.self_attn.{proj}" for layer_name in names_deep_layers for proj in proj_names]
    hook_names_shallow = [f"{layer_name}.self_attn.{proj}" for layer_name in names_shallow_layers for proj in proj_names]

    reps_in_d, reps_out_d, hooks_d = register_hooks_for_reps(model_deep, hook_names_deep)
    reps_in_s, reps_out_s, hooks_s = register_hooks_for_reps(model_shallow, hook_names_shallow)

    dataset = load_and_prepare_dataset(tok_deep, tok_shallow)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in tqdm(dataloader, desc="特征提取"):
        with torch.no_grad():
            inputs_a = {k: v.to(MODEL_DEVICE_A) for k, v in batch.items() if k.endswith('_a')}
            inputs_b = {k: v.to(MODEL_DEVICE_B) for k, v in batch.items() if k.endswith('_b')}
            inputs_deep_data = inputs_a if deep_name == 'llama2' else inputs_b
            inputs_shallow_data = inputs_b if shallow_name == 'qwen2' else inputs_a
            model_deep(input_ids=inputs_deep_data[f"input_ids_{'a' if deep_name=='llama2' else 'b'}"], attention_mask=inputs_deep_data[f"attention_mask_{'a' if deep_name=='llama2' else 'b'}"])
            model_shallow(input_ids=inputs_shallow_data[f"input_ids_{'b' if shallow_name=='qwen2' else 'a'}"], attention_mask=inputs_shallow_data[f"attention_mask_{'b' if shallow_name=='qwen2' else 'a'}"])
    
    for hook in hooks_d + hooks_s: hook.remove()
    
    print(f"\n--- 步骤 3: 深度异构对齐 (使用 {alignment_type}) ---")
    o_proj_names_deep = [name for name in hook_names_deep if name.endswith('o_proj')]
    o_proj_names_shallow = [name for name in hook_names_shallow if name.endswith('o_proj')]
    cka_matrix = compute_cka_matrix(reps_out_d, reps_out_s, o_proj_names_deep, o_proj_names_shallow)
    layer_alignment = align_layers_lma(cka_matrix)
    
    print("\n找到的层级映射关系 (Shallow Layer -> Deep Segment):")
    for i, segment in enumerate(layer_alignment):
        print(f"  {names_shallow_layers[i]} -> {[names_deep_layers[j] for j in segment]}")

    print(f"\n--- 步骤 4: 宽度异构合并 (alpha={alpha}) ---")
    merged_model = model_base

    for i, deep_segment_indices in enumerate(tqdm(layer_alignment, desc="合并所有层段")):
        shallow_layer_name = names_shallow_layers[i]
        
        for k, deep_layer_idx in enumerate(deep_segment_indices):
            deep_layer_name = names_deep_layers[deep_layer_idx]
            
            for proj_name in proj_names:
                print(f"  合并 {shallow_layer_name}.self_attn.{proj_name} 和 {deep_layer_name}.self_attn.{proj_name}")
                base_proj = get_module_by_name(merged_model, f"{shallow_layer_name}.self_attn.{proj_name}")
                donor_proj = get_module_by_name(model_deep, f"{deep_layer_name}.self_attn.{proj_name}")
                
                # 修复：使用不同的变量名存储处理后的特征
                key_shallow = f"{shallow_layer_name}.self_attn.{proj_name}"
                key_deep = f"{deep_layer_name}.self_attn.{proj_name}"
                
                # 检查键是否存在
                if key_shallow not in reps_in_s or key_deep not in reps_in_d:
                    print(f"警告：找不到 {key_shallow} 或 {key_deep} 的特征，跳过此合并")
                    continue
                    
                # 提取对应的输入和输出激活
                processed_reps_in_b = torch.cat(reps_in_s[key_shallow], dim=0).flatten(0, 1)
                processed_reps_in_d = torch.cat(reps_in_d[key_deep], dim=0).flatten(0, 1)
                processed_reps_out_b = torch.cat(reps_out_s[key_shallow], dim=0).flatten(0, 1)
                processed_reps_out_d = torch.cat(reps_out_d[key_deep], dim=0).flatten(0, 1)

                # 每个子层独立进行宽度合并
                transform_and_merge_weights(
                    base_proj, donor_proj,
                    processed_reps_in_b, processed_reps_in_d,
                    processed_reps_out_b, processed_reps_out_d,
                    alpha / len(deep_segment_indices) # 平均分配alpha
                )

    # --- 步骤 5: 保存并测试 ---
    output_dir = f"./merged_final_{shallow_name}_and_{deep_name}_alpha_{alpha}"
    print(f"\n--- 正在保存合并后的模型到 {output_dir} ---")
    merged_model.save_pretrained(output_dir)
    tok_shallow.save_pretrained(output_dir)
    print("模型保存完成。")

    del model_donor, model_base, merged_model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n--- 测试合并后的模型 ---")
    tokenizer_merged, model_merged = load_complete_model(output_dir, COMPUTE_DEVICE)
    prompt = "The capital of France is"
    inputs = tokenizer_merged(prompt, return_tensors="pt").to(COMPUTE_DEVICE)
    with torch.no_grad():
        outputs = model_merged.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer_merged.eos_token_id)
    print("输入:", prompt)
    print("合并后模型输出:", tokenizer_merged.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main(alpha=0.5)