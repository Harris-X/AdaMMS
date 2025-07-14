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
# 定义计算设备和模型加载设备。如果显存不足，可将一个模型加载到CPU
MODEL_DEVICE_A = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
MODEL_DEVICE_B = torch.device("cuda:5" if torch.cuda.is_available() else "cpu") # 也可以是 "cpu"
COMPUTE_DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(f"模型A设备: {MODEL_DEVICE_A}, 模型B设备: {MODEL_DEVICE_B}, 计算设备: {COMPUTE_DEVICE}")
# COMPUTE_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"使用的计算设备: {COMPUTE_DEVICE}")


# --- 2. 辅助函数 (模型与数据加载) ---
# (这部分函数保持不变，为简洁起见，此处省略，请使用上一版代码中的函数)
def load_complete_model(model_id, device):
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

def load_and_prepare_dataset(tokenizer_a, tokenizer_b, dataset_name="wikitext", split="test", max_samples=32, max_length=128):
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
    """为指定层注册钩子以捕获输出激活，并立即转移到CPU"""
    reps, hooks = {name: [] for name in layer_names}, []
    # LLM的线性层输出通常就是我们需要的激活
    hook_fn = lambda name: (lambda module, input, output: reps[name].append(output.detach().cpu()))
    for name in layer_names:
        if module := get_module_by_name(model, name):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    return reps, hooks


# --- 3. 核心算法：CKA & 深度对齐 (LMA) ---
# (这部分函数保持不变，为简洁起见，此处省略，请使用上一版代码中的函数)
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
    
    # 预处理所有激活
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
            
        del gram_k
        gc.collect()
        torch.cuda.empty_cache()

    print("CKA矩阵计算完成。")
    return cka_matrix.cpu()

def align_layers_lma(C):
    """使用LMA算法进行深度对齐"""
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


# --- 4. 宽度异构合并：弹性神经元压缩 (ZipIt!) - CPU优化版 ---
# (此函数保持不变，为简洁起见，此处省略，请使用上一版代码中的函数)
def match_tensors_zipit(reps_a, reps_b, target_dim):
    """在CPU上执行ZipIt!算法，返回适用于两个模型的变换矩阵"""
    dim_a, dim_b = reps_a.shape[1], reps_b.shape[1]
    
    reps_a_cpu = reps_a.cpu().to(torch.float32)
    reps_b_cpu = reps_b.cpu().to(torch.float32)
    
    reps_a_norm = reps_a_cpu / (torch.norm(reps_a_cpu, p=2, dim=0, keepdim=True) + 1e-6)
    reps_b_norm = reps_b_cpu / (torch.norm(reps_b_cpu, p=2, dim=0, keepdim=True) + 1e-6)
    
    all_reps = torch.cat([reps_a_norm, reps_b_norm], dim=1)
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

# --- 5. 最终修复的核心函数：权重变换与合并 ---
def transform_and_merge_weights(base_proj, donor_proj, reps_base, reps_donor, alpha):
    """
    对一对具体的投影层（如q_proj）进行宽度对齐和合并。
    """
    # 目标维度是 base 层的输出维度
    target_dim = base_proj.out_features
    
    # 在CPU上为这对层专门计算变换矩阵
    Tm_b, Tm_d, Tu_b, Tu_d = match_tensors_zipit(reps_base, reps_donor, target_dim)
    
    # 移动到计算设备并设置正确的数据类型
    dtype = base_proj.weight.dtype
    device = base_proj.weight.device
    Tm_b, Tm_d = Tm_b.to(device, dtype=dtype), Tm_d.to(device, dtype=dtype)
    Tu_b, Tu_d = Tu_b.to(device, dtype=dtype), Tu_d.to(device, dtype=dtype)
    
    # 变换 donor 权重以匹配 base 权重
    # W_d' = T_out @ W_d @ T_in⁻¹
    # T_out: 从 donor 输出空间到 base 输出空间
    # T_in:  从 base 输入空间到 donor 输入空间
    T_out_d_to_b = Tu_b @ Tm_d
    T_in_b_to_d = Tu_d @ Tm_b

    # 计算输入变换的伪逆
    T_in_inv = torch.linalg.pinv(T_in_b_to_d.to(torch.float32)).to(dtype)
    
    W_d_transformed = T_out_d_to_b @ donor_proj.weight.data @ T_in_inv
    
    # 断言确保维度完全匹配
    assert base_proj.weight.data.shape == W_d_transformed.shape, \
        f"维度不匹配: base({base_proj.weight.data.shape}) vs transformed donor({W_d_transformed.shape})"
    
    # 加权平均权重
    base_proj.weight.data = (1 - alpha) * base_proj.weight.data + alpha * W_d_transformed
    
    # Bias 只受输出变换 T_out_d_to_b 的影响
    if hasattr(base_proj, 'bias') and base_proj.bias is not None and \
       hasattr(donor_proj, 'bias') and donor_proj.bias is not None:
        bias_d_transformed = T_out_d_to_b @ donor_proj.bias.data
        assert base_proj.bias.data.shape == bias_d_transformed.shape, "bias 维度不匹配"
        base_proj.bias.data = (1 - alpha) * base_proj.bias.data + alpha * bias_d_transformed


# --- 6. 主执行流程 ---
def main(alpha=0.5, alignment_type='LMA'):
    """执行模型对齐与合并的主函数"""
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

    print("\n--- 步骤 2: 为两个模型收集特征表示 ---")
    # 注册更精细的钩子，捕获每个子层的输出
    proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    hook_names_deep = [f"{layer_name}.self_attn.{proj}" for layer_name in names_deep_layers for proj in proj_names]
    hook_names_shallow = [f"{layer_name}.self_attn.{proj}" for layer_name in names_shallow_layers for proj in proj_names]

    reps_deep, hooks_deep = register_hooks_for_reps(model_deep, hook_names_deep)
    reps_shallow, hooks_shallow = register_hooks_for_reps(model_shallow, hook_names_shallow)

    dataset = load_and_prepare_dataset(tok_deep, tok_shallow, max_samples=16) # 减少样本以加快调试
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1) # 减小batch size

    for batch in tqdm(dataloader, desc="特征提取"):
        with torch.no_grad():
            inputs_a = {k: v.to(MODEL_DEVICE_A) for k, v in batch.items() if k.endswith('_a')}
            inputs_b = {k: v.to(MODEL_DEVICE_B) for k, v in batch.items() if k.endswith('_b')}
            if deep_name == 'llama2':
                model_deep(input_ids=inputs_a['input_ids_a'], attention_mask=inputs_a['attention_mask_a'])
                model_shallow(input_ids=inputs_b['input_ids_b'], attention_mask=inputs_b['attention_mask_b'])
            else:
                model_deep(input_ids=inputs_b['input_ids_b'], attention_mask=inputs_b['attention_mask_b'])
                model_shallow(input_ids=inputs_a['input_ids_a'], attention_mask=inputs_a['attention_mask_a'])
    
    for hook in hooks_deep + hooks_shallow: hook.remove()
    
    print(f"\n--- 步骤 3: 深度异构对齐 (使用 {alignment_type}) ---")
    # 使用o_proj的激活作为整个attention block的代表性激活
    o_proj_names_deep = [name for name in hook_names_deep if name.endswith('o_proj')]
    o_proj_names_shallow = [name for name in hook_names_shallow if name.endswith('o_proj')]
    cka_matrix = compute_cka_matrix(reps_deep, reps_shallow, o_proj_names_deep, o_proj_names_shallow)
    layer_alignment = align_layers_lma(cka_matrix)
    
    print("\n找到的层级映射关系 (Shallow Layer -> Deep Segment):")
    for i, segment in enumerate(layer_alignment):
        print(f"  {names_shallow_layers[i]} -> {[names_deep_layers[j] for j in segment]}")

    print(f"\n--- 步骤 4: 宽度异构合并 (alpha={alpha}) ---")
    merged_model = model_base

    for i, deep_segment_indices in enumerate(tqdm(layer_alignment, desc="合并所有层段")):
        shallow_layer_name = names_shallow_layers[i]
        
        # 对段内每一层进行加权合并
        for k, deep_layer_idx in enumerate(deep_segment_indices):
            deep_layer_name = names_deep_layers[deep_layer_idx]
            
            # 对q,k,v,o四个子层分别进行对齐和合并
            for proj_name in proj_names:
                print(f"  合并 {shallow_layer_name}.{proj_name} 和 {deep_layer_name}.{proj_name}")
                
                base_proj = get_module_by_name(merged_model, f"{shallow_layer_name}.self_attn.{proj_name}")
                donor_proj = get_module_by_name(model_deep, f"{deep_layer_name}.self_attn.{proj_name}")
                
                reps_base_proj = torch.cat(reps_shallow[f"{shallow_layer_name}.self_attn.{proj_name}"], dim=0).flatten(0, 1)
                reps_donor_proj = torch.cat(reps_deep[f"{deep_layer_name}.self_attn.{proj_name}"], dim=0).flatten(0, 1)

                # 每个子层独立进行宽度合并
                transform_and_merge_weights(
                    base_proj, 
                    donor_proj,
                    reps_base_proj,
                    reps_donor_proj,
                    # 平均分配alpha到段内的每一层
                    alpha / len(deep_segment_indices) if k == 0 else 1.0 # 第一次合并用alpha，之后累加
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