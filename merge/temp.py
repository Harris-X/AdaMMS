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
MODEL_DEVICE_A = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_DEVICE_B = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
COMPUTE_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"模型A设备: {MODEL_DEVICE_A}, 模型B设备: {MODEL_DEVICE_B}, 计算设备: {COMPUTE_DEVICE}")

# --- 2. 辅助函数 (保持不变) ---
def load_complete_model(model_id, device):
    print(f"正在加载模型: {model_id} -> {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    print(f"{model_id.split('/')[-1]} 模型加载完成。")
    return tokenizer, model

def load_and_prepare_dataset(tokenizer_a, tokenizer_b, max_samples=16, max_length=128):
    print(f"正在加载数据集: wikitext...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test").select(range(max_samples))
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

def register_hooks_for_reps(model, layer_names):
    reps_in, reps_out, hooks = {n: [] for n in layer_names}, {n: [] for n in layer_names}, []
    def get_hook_fn(name):
        def hook_fn(module, input, output):
            reps_in[name].append(input[0].detach().cpu())
            reps_out[name].append((output[0] if isinstance(output, tuple) else output).detach().cpu())
        return hook_fn
    for name in layer_names:
        if module := get_module_by_name(model, name):
            hooks.append(module.register_forward_hook(get_hook_fn(name)))
    return reps_in, reps_out, hooks

# --- 3. CKA & LMA 对齐函数 (保持不变) ---
def cka(gram_k, gram_l):
    gram_k = center_gram(gram_k.float()); gram_l = center_gram(gram_l.float())
    scaled_hsic = torch.sum(gram_k * gram_l); norm_k = torch.norm(gram_k); norm_l = torch.norm(gram_l)
    return scaled_hsic / (norm_k * norm_l) if norm_k != 0 and norm_l != 0 else torch.tensor(0.0)

def center_gram(gram):
    n = gram.shape[0]; I = torch.eye(n, device=gram.device); H = I - 1/n * torch.ones(n, n, device=gram.device); return H @ gram @ H

def compute_cka_matrix(reps1, reps2, names1, names2, max_tokens=4096):
    print("开始计算CKA矩阵..."); cka_matrix = torch.zeros(len(names1), len(names2))
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
    print("CKA矩阵计算完成."); return cka_matrix.cpu()

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


# --- 4. 宽度异构合并：官方 ZipIt! 逻辑的重新实现 ---
def match_tensors_zipit_official(metric, model_dims, target_dim):
    """
    官方 `match_tensors_zipit` 逻辑的直接复现。
    计算一个统一的 merge 和 unmerge 矩阵。
    """
    if "covariance" not in metric:
        raise ValueError("Metric dictionary must contain 'covariance'")
    
    # 协方差 -> 相关性
    sims = metric["covariance"] / (torch.outer(torch.diag(metric["covariance"]).sqrt(), torch.diag(metric["covariance"]).sqrt()) + 1e-6)
    
    O = sims.shape[0]
    perm_matrix = torch.eye(O, O, device='cpu', dtype=torch.bfloat16)
    sims.fill_diagonal_(-torch.inf)

    num_merges = O - target_dim
    for _ in tqdm(range(num_merges), desc=f"Zipping to {target_dim} dims on CPU", leave=False):
        flat_idx = torch.argmax(sims)
        idx1, idx2 = np.unravel_index(flat_idx.item(), sims.shape)
        if sim_matrix[idx1, idx2] == -torch.inf: break
        
        perm_matrix[:, idx1] += perm_matrix[:, idx2]
        sims[:, idx1] = torch.min(sims[:, idx1], sims[:, idx2])
        sims[idx1, :] = torch.min(sims[idx1, :], sims[idx2, :])
        
        perm_matrix = torch.cat([perm_matrix[:, :idx2], perm_matrix[:, idx2+1:]], dim=1)
        sims = torch.cat([sims[:, :idx2], sims[:, idx2+1:]], dim=1)
        sims = torch.cat([sims[:idx2, :], sims[idx2+1:, :]], dim=0)

    unmerge = perm_matrix
    merge = unmerge.T / (unmerge.sum(dim=0, keepdim=True) + 1e-6)

    # 拆分矩阵以供不同模型使用
    merges, unmerges = [], []
    current_dim = 0
    for dim in model_dims:
        merges.append(merge[:, current_dim : current_dim + dim])
        unmerges.append(unmerge[current_dim : current_dim + dim, :])
        current_dim += dim
        
    return merges, unmerges

# --- 5. 最终修复的核心函数：权重变换与合并 ---
def transform_and_merge_weights(base_proj, donor_proj, reps_in_b, reps_in_d, reps_out_b, reps_out_d, alpha):
    """对一对具体的投影层进行正确的双向宽度对齐和合并"""
    device, dtype = base_proj.weight.device, base_proj.weight.dtype
    
    # 1. 计算输出空间的变换
    cov_out = torch.cov(torch.cat([reps_out_b, reps_out_d], dim=1))
    merges_out, unmerges_out = match_tensors_zipit_official(
        {"covariance": cov_out}, 
        model_dims=[base_proj.out_features, donor_proj.out_features],
        target_dim=base_proj.out_features # 目标维度是 base 模型的维度
    )
    Tm_b_out, Tm_d_out = merges_out[0].to(device, dtype=dtype), merges_out[1].to(device, dtype=dtype)
    Tu_b_out, Tu_d_out = unmerges_out[0].to(device, dtype=dtype), unmerges_out[1].to(device, dtype=dtype)

    # 2. 计算输入空间的变换
    cov_in = torch.cov(torch.cat([reps_in_b, reps_in_d], dim=1))
    merges_in, unmerges_in = match_tensors_zipit_official(
        {"covariance": cov_in},
        model_dims=[base_proj.in_features, donor_proj.in_features],
        target_dim=base_proj.in_features
    )
    Tm_b_in, Tm_d_in = merges_in[0].to(device, dtype=dtype), merges_in[1].to(device, dtype=dtype)
    Tu_b_in, Tu_d_in = unmerges_in[0].to(device, dtype=dtype), unmerges_in[1].to(device, dtype=dtype)
    
    # 3. 变换和合并权重
    W_b = base_proj.weight.data
    W_d = donor_proj.weight.data
    
    W_b_transformed = Tm_b_out @ W_b @ Tu_b_in
    W_d_transformed = Tm_b_out @ W_d @ Tu_d_in
    
    final_W = (1-alpha) * W_b_transformed + alpha * W_d_transformed
    
    # 4. 将合并后的权重投影回 base 空间
    base_proj.weight.data = torch.linalg.pinv(Tm_b_out.to(torch.float32)).to(dtype) @ final_W @ torch.linalg.pinv(Tu_b_in.to(torch.float32)).to(dtype)
    
    # 5. 合并 Bias (只受输出变换影响)
    if hasattr(base_proj, 'bias') and base_proj.bias is not None and hasattr(donor_proj, 'bias') and donor_proj.bias is not None:
        B_b = base_proj.bias.data
        B_d = donor_proj.bias.data
        
        B_b_transformed = Tm_b_out @ B_b
        B_d_transformed = Tm_b_out @ B_d # 注意：用base的merge矩阵来变换donor的bias
        
        final_B = (1-alpha) * B_b_transformed + alpha * B_d_transformed
        base_proj.bias.data = torch.linalg.pinv(Tm_b_out.to(torch.float32)).to(dtype) @ final_B


# --- 6. 主执行流程 (已重构)---
def main(alpha=0.5, alignment_type='LMA'):
    tokenizer_donor, model_donor = load_complete_model(CKPT_PATH["llama2"], MODEL_DEVICE_A)
    tokenizer_base, model_base = load_complete_model(CKPT_PATH["qwen2"], MODEL_DEVICE_B)
    
    model_deep, model_shallow, tok_deep, tok_shallow, deep_name, shallow_name, deep_device, shallow_device = \
        (model_donor, model_base, tokenizer_donor, tokenizer_base, "llama2", "qwen2", MODEL_DEVICE_A, MODEL_DEVICE_B) if model_donor.config.num_hidden_layers >= model_base.config.num_hidden_layers \
        else (model_base, model_donor, tokenizer_base, tokenizer_donor, "qwen2", "llama2", MODEL_DEVICE_B, MODEL_DEVICE_A)

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
            inputs_deep_data = batch[f"input_ids_{'a' if deep_name=='llama2' else 'b'}"].to(deep_device)
            inputs_shallow_data = batch[f"input_ids_{'a' if shallow_name=='llama2' else 'b'}"].to(shallow_device)
            model_deep(inputs_deep_data)
            model_shallow(inputs_shallow_data)
    
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
        
        for proj_name in proj_names:
            base_proj = get_module_by_name(merged_model, f"{shallow_layer_name}.self_attn.{proj_name}")
            
            # 对段内每一层进行加权合并
            for k, deep_layer_idx in enumerate(deep_segment_indices):
                deep_layer_name = names_deep_layers[deep_layer_idx]
                print(f"  合并 {shallow_layer_name}.{proj_name} 和 {deep_layer_name}.{proj_name}")
                
                donor_proj = get_module_by_name(model_deep, f"{deep_layer_name}.self_attn.{proj_name}")
                
                # 提取对应的输入和输出激活
                reps_in_b = torch.cat(reps_in_s[f"{shallow_layer_name}.self_attn.{proj_name}"], dim=0).flatten(0, 1)
                reps_in_d = torch.cat(reps_in_d[f"{deep_layer_name}.self_attn.{proj_name}"], dim=0).flatten(0, 1)
                reps_out_b = torch.cat(reps_out_s[f"{shallow_layer_name}.self_attn.{proj_name}"], dim=0).flatten(0, 1)
                reps_out_d = torch.cat(reps_out_d[f"{deep_layer_name}.self_attn.{proj_name}"], dim=0).flatten(0, 1)

                transform_and_merge_weights(
                    base_proj, donor_proj,
                    reps_in_b, reps_in_d,
                    reps_out_b, reps_out_d,
                    alpha / len(deep_segment_indices) # 平均分配alpha
                )

    # --- 步骤 5: 保存并测试 ---
    output_dir = f"./merged_final_{shallow_name}_and_{deep_name}_alpha_{alpha}"
    print(f"\n--- 正在保存合并后的模型到 {output_dir} ---")
    merged_model.save_pretrained(output_dir)
    tok_shallow.save_pretrained(output_dir)
    print("模型保存完成。")

    del model_donor, model_base, reps_deep, reps_shallow, dataset, dataloader, merged_model
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