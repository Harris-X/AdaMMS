import os
import torch
import torch.nn as nn
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import scipy
from functools import partial

# --- 1. 核心配置 ---
CKPT_PATH = {
    "llama2": "./downloaded_models/Llama-2-7b-hf",
    "qwen2": "./downloaded_models/Qwen2-7B-Instruct",
}
# 计算设备，如果显存不足，可考虑将模型加载到不同GPU
COMPUTE_DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(f"使用的计算设备: {COMPUTE_DEVICE}")


# --- 2. 辅助函数 (模型与数据加载) ---

def load_complete_model(model_id, device):
    """通用模型加载函数，使用bfloat16以节省内存"""
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

def load_and_prepare_dataset(tokenizer_a, tokenizer_b, dataset_name="wikitext", split="test", max_samples=128, max_length=128):
    """加载并处理数据集以进行特征提取"""
    print(f"正在加载数据集: {dataset_name}...")
    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)
    
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    def tokenize_fn(examples):
        text = [t for t in examples["text"] if t and t.strip()]
        if not text: return {}
        inputs_a = tokenizer_a(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        inputs_b = tokenizer_b(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids_a": inputs_a.input_ids, "attention_mask_a": inputs_a.attention_mask,
            "input_ids_b": inputs_b.input_ids, "attention_mask_b": inputs_b.attention_mask,
        }

    processed_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    processed_dataset.set_format(type='torch')
    return processed_dataset

def get_module_by_name(model, module_name):
    """通过字符串名称安全地获取模块"""
    for part in module_name.split('.'):
        if not hasattr(model, part): return None
        model = getattr(model, part)
    return model

def register_hooks_for_reps(model, layer_names):
    """为指定层注册钩子以捕获输出激活"""
    reps, hooks = {name: [] for name in layer_names}, []
    hook_fn = lambda name: (lambda module, input, output: reps[name].append((output[0] if isinstance(output, tuple) else output).detach().cpu()))
    for name in layer_names:
        if module := get_module_by_name(model, name):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    return reps, hooks


# --- 3. 核心算法：CKA & 深度对齐 ---

def cka(gram_k, gram_l):
    """计算中心核对齐(CKA)"""
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
    """高效计算CKA相似度矩阵"""
    print("开始计算CKA矩阵...")
    cka_matrix = torch.zeros(len(names1), len(names2))
    
    processed_reps1 = {name: torch.cat(reps1[name], dim=0).flatten(0, 1) for name in names1}
    processed_reps2 = {name: torch.cat(reps2[name], dim=0).flatten(0, 1) for name in names2}

    for i, name1 in enumerate(tqdm(names1, desc="Model Deep Layers")):
        feat1_full = processed_reps1[name1]
        feat1 = feat1_full[torch.randperm(feat1_full.shape[0])[:max_tokens]].to(COMPUTE_DEVICE)
        gram_k = feat1 @ feat1.T
        
        for j, name2 in enumerate(names2):
            feat2_full = processed_reps2[name2]
            feat2 = feat2_full[torch.randperm(feat2_full.shape[0])[:max_tokens]].to(COMPUTE_DEVICE)
            gram_l = feat2 @ feat2.T
            
            min_dim = min(gram_k.shape[0], gram_l.shape[0])
            cka_matrix[i, j] = cka(gram_k[:min_dim, :min_dim], gram_l[:min_dim, :min_dim])
            
    print("CKA矩阵计算完成。")
    return cka_matrix.cpu()

def align_layers_lma(C):
    """使用LMA（层对齐）算法进行深度对齐"""
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
                    max_val = current_val
                    best_k = k
            F[i, j] = max_val
            path[i, j] = best_k

    alignment, i, j = [], n, m
    while i > 0:
        k = path[i, j]
        alignment.insert(0, list(range(k, j)))
        j = k
        i -= 1
        
    return alignment


# --- 4. 宽度异构合并：弹性神经元压缩 (ZipIt!) ---

def match_tensors_zipit(reps_a, reps_b, target_dim):
    """
    ZipIt! 算法的实现，用于宽度异构合并
    返回四个变换矩阵：合并和解合并矩阵，分别对应两个模型
    """
    dim_a, dim_b = reps_a.shape[1], reps_b.shape[1]
    reps_a_norm = reps_a / (torch.norm(reps_a, p=2, dim=0, keepdim=True) + 1e-6)
    reps_b_norm = reps_b / (torch.norm(reps_b, p=2, dim=0, keepdim=True) + 1e-6)
    
    all_reps = torch.cat([reps_a_norm, reps_b_norm], dim=1)
    sim_matrix = all_reps.T @ all_reps
    sim_matrix.fill_diagonal_(-torch.inf)
    
    total_dim = dim_a + dim_b
    perm_matrix = torch.eye(total_dim, device=COMPUTE_DEVICE, dtype=torch.bfloat16)

    num_merges = total_dim - target_dim
    for _ in tqdm(range(num_merges), desc=f"Zipping to {target_dim} dims", leave=False):
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

def apply_hetero_merge_to_layer(base_layer, donor_layer, Tm_b, Tm_d, Tu_b, Tu_d, alpha):
    """
    对单个注意力层应用完整的异构合并。
    这个函数现在正确地处理权重变换，以避免维度不匹配。
    """
    # 将变换矩阵移动到计算设备并设置正确的类型
    dtype = base_layer.q_proj.weight.dtype
    Tm_b, Tm_d = Tm_b.to(COMPUTE_DEVICE, dtype=dtype), Tm_d.to(COMPUTE_DEVICE, dtype=dtype)
    Tu_b, Tu_d = Tu_b.to(COMPUTE_DEVICE, dtype=dtype), Tu_d.to(COMPUTE_DEVICE, dtype=dtype)
    
    # --- 变换和合并Q, K, V投影层 (输出变换) ---
    # W'_donor = W_donor @ T_unmerge_donor
    # W'_base  = W_base @ T_unmerge_base
    for proj in ['q_proj', 'k_proj', 'v_proj']:
        base_proj = getattr(base_layer, proj)
        donor_proj = getattr(donor_layer, proj)
        
        W_b_transformed = base_proj.weight.data @ Tu_b
        W_d_transformed = donor_proj.weight.data @ Tu_d

        # 加权平均
        base_proj.weight.data = (1 - alpha) * W_b_transformed + alpha * W_d_transformed
        if base_proj.bias is not None and donor_proj.bias is not None:
             base_proj.bias.data = (1-alpha) * base_proj.bias.data + alpha * donor_proj.bias.data

    # --- 变换和合并输出投影层 o_proj (输入变换) ---
    # W'_donor = T_merge_donor @ W_donor
    # W'_base  = T_merge_base @ W_base
    base_o_proj = base_layer.o_proj
    donor_o_proj = donor_layer.o_proj

    W_b_transformed = Tm_b @ base_o_proj.weight.data
    W_d_transformed = Tm_d @ donor_o_proj.weight.data

    base_o_proj.weight.data = (1 - alpha) * W_b_transformed + alpha * W_d_transformed
    if base_o_proj.bias is not None and donor_o_proj.bias is not None:
        base_o_proj.bias.data = (1 - alpha) * base_o_proj.bias.data + alpha * donor_o_proj.bias.data

# --- 5. 主执行流程 ---

def main(alpha=0.5, alignment_type='LMA'):
    """执行模型对齐与合并的主函数"""
    global model_base 
    tokenizer_donor, model_donor = load_complete_model(CKPT_PATH["llama2"], COMPUTE_DEVICE)
    tokenizer_base, model_base = load_complete_model(CKPT_PATH["qwen2"], COMPUTE_DEVICE)

    if model_donor.config.num_hidden_layers >= model_base.config.num_hidden_layers:
        model_deep, model_shallow, tok_deep, tok_shallow = model_donor, model_base, tokenizer_donor, tokenizer_base
        deep_name, shallow_name = "llama2", "qwen2"
    else:
        model_deep, model_shallow, tok_deep, tok_shallow = model_base, model_donor, tokenizer_base, tokenizer_donor
        deep_name, shallow_name = "qwen2", "llama2"

    names_deep = [f"model.layers.{i}" for i in range(model_deep.config.num_hidden_layers)]
    names_shallow = [f"model.layers.{i}" for i in range(model_shallow.config.num_hidden_layers)]
    
    dataset = load_and_prepare_dataset(tok_deep, tok_shallow, max_samples=32, max_length=256)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    print("\n--- 步骤 2: 为两个模型收集特征表示 ---")
    reps_deep, hooks_deep = register_hooks_for_reps(model_deep, names_deep)
    reps_shallow, hooks_shallow = register_hooks_for_reps(model_shallow, names_shallow)

    for batch in tqdm(dataloader, desc="特征提取"):
        with torch.no_grad():
            inputs_a = {"input_ids": batch["input_ids_a"].to(COMPUTE_DEVICE), "attention_mask": batch["attention_mask_a"].to(COMPUTE_DEVICE)}
            inputs_b = {"input_ids": batch["input_ids_b"].to(COMPUTE_DEVICE), "attention_mask": batch["attention_mask_b"].to(COMPUTE_DEVICE)}
            
            if deep_name == 'llama2':
                model_deep(**inputs_a)
                model_shallow(**inputs_b)
            else:
                model_deep(**inputs_b)
                model_shallow(**inputs_a)

    for hook in hooks_deep + hooks_shallow: hook.remove()

    print(f"\n--- 步骤 3: 深度异构对齐 (使用 {alignment_type}) ---")
    cka_matrix = compute_cka_matrix(reps_deep, reps_shallow, names_deep, names_shallow)
    layer_alignment = align_layers_lma(cka_matrix)
    
    print("\n找到的层级映射关系 (Shallow Layer -> Deep Segment):")
    for i, segment in enumerate(layer_alignment):
        print(f"  {names_shallow[i]} -> {[names_deep[j] for j in segment]}")

    print(f"\n--- 步骤 4: 宽度异构合并 (alpha={alpha}) ---")
    merged_model = model_base # 直接修改base model

    for i, deep_segment_indices in enumerate(tqdm(layer_alignment, desc="合并所有层段")):
        shallow_layer_name = names_shallow[i]
        
        # 1. 对整个段的激活进行ZipIt!
        # 我们将段内所有层的激活拼接起来，作为donor的整体表示
        segment_reps_deep = torch.cat(
            [torch.cat(reps_deep[names_deep[k]], dim=0).flatten(0, 1) for k in deep_segment_indices],
            dim=1
        ).to(COMPUTE_DEVICE)
        
        reps_shallow_layer = torch.cat(reps_shallow[shallow_layer_name], dim=0).flatten(0, 1).to(COMPUTE_DEVICE)
        
        # 目标宽度是base模型的隐藏层维度
        target_width = merged_model.config.hidden_size
        
        Tm_s, Tm_d, Tu_s, Tu_d = match_tensors_zipit(reps_shallow_layer, segment_reps_deep, target_width)

        # 2. 将变换应用到每一层，并进行加权
        base_target_layer = get_module_by_name(merged_model, shallow_layer_name)
        
        # 暂存base层的原始权重，用于加权
        original_base_weights = {name: p.data.clone() for name, p in base_target_layer.named_parameters()}
        
        # 将base层的权重清零，准备累加
        for p in base_target_layer.parameters(): p.data.zero_()
        
        # (1-alpha) * Base_Layer
        for name, p in base_target_layer.named_parameters():
             p.data += (1 - alpha) * original_base_weights[name]

        # alpha * sum(Transformed_Donor_Layers)
        for k, deep_layer_idx in enumerate(deep_segment_indices):
            donor_layer = get_module_by_name(model_deep, names_deep[deep_layer_idx])
            
            # 创建一个临时的、与base层结构相同的层来存放变换后的donor权重
            temp_donor_transformed = type(base_target_layer)(merged_model.config).to(COMPUTE_DEVICE, dtype=torch.bfloat16)
            temp_donor_transformed.load_state_dict(donor_layer.state_dict())
            
            # donor_k_reps = torch.cat(reps_deep[names_deep[deep_layer_idx]], dim=0).flatten(0, 1).to(COMPUTE_DEVICE)
            # Tms, Tmd, Tus, Tud = match_tensors_zipit(reps_shallow_layer, donor_k_reps, target_width)

            apply_hetero_merge_to_layer(
                 temp_donor_transformed.self_attn, # "base"
                 donor_layer.self_attn,            # "donor"
                 Tm_s, Tm_d, Tu_s, Tu_d,
                 1.0 # alpha=1.0,因为我们只想得到变换后的结果
            )

            # 将变换后的donor层权重累加到base层，并平均分配alpha
            for name, p_transformed in temp_donor_transformed.named_parameters():
                 getattr(base_target_layer, name.replace('.', '_')).data += (alpha / len(deep_segment_indices)) * p_transformed.data


    # --- 步骤 5: 保存并测试 ---
    output_dir = f"./merged_{shallow_name}_and_{deep_name}_alpha_{alpha}"
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