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
# 推荐使用不同的GPU设备以避免显存不足
# 如果只有一个GPU，请确保其显存足够大 (例如 >= 24GB)
DEVICE_DONOR = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
DEVICE_BASE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
DEVICE_COMPUTE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(f"Donor模型设备: {DEVICE_DONOR}, Base模型设备: {DEVICE_BASE}, 计算设备: {DEVICE_COMPUTE}")


# --- 2. 辅助函数 (模型加载, 数据集, CKA计算, 对齐) ---
def load_complete_model(model_id, device):
    """通用模型加载函数，使用bfloat16以节省内存"""
    print(f"正在加载模型: {model_id} 到 {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    ).to(device)
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
    parts = module_name.split('.')
    module = model
    for part in parts:
        if not hasattr(module, part): return None
        module = getattr(module, part)
    return module

def register_hooks_for_reps(model, layer_names):
    """为指定层注册钩子以捕获输出激活"""
    reps = {name: [] for name in layer_names}
    hooks = []
    
    # 使用偏函数固定name参数，避免作用域问题
    def hook_fn_wrapper(name):
        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            reps[name].append(hidden_states.detach().to('cpu', non_blocking=True))
        return hook_fn

    for name in layer_names:
        module = get_module_by_name(model, name)
        if module:
            hooks.append(module.register_forward_hook(hook_fn_wrapper(name)))
    return reps, hooks

def gram_linear(x):
    return x @ x.T

def center_gram(K):
    n = K.shape[0]
    I = torch.eye(n, device=K.device)
    H = I - 1/n * torch.ones(n, n, device=K.device)
    return H @ K @ H

def cka(K, L):
    hsic = (center_gram(K).T * center_gram(L)).sum()
    var_k = torch.sqrt((center_gram(K).T * center_gram(K)).sum())
    var_l = torch.sqrt((center_gram(L).T * center_gram(L)).sum())
    return hsic / (var_k * var_l + 1e-8)

def compute_cka_matrix(reps1, reps2, names1, names2, max_seq_len=1024):
    print("开始计算CKA矩阵...")
    cka_matrix = torch.zeros(len(names1), len(names2))
    for i, name1 in enumerate(tqdm(names1, desc="Model Deep Layers")):
        feat1_full = torch.cat(reps1[name1], dim=0).flatten(0, 1).to(torch.float32)
        feat1 = feat1_full[torch.randperm(feat1_full.shape[0])[:max_seq_len]]
        gram_k = gram_linear(feat1.to(DEVICE_COMPUTE))
        
        for j, name2 in enumerate(names2):
            feat2_full = torch.cat(reps2[name2], dim=0).flatten(0, 1).to(torch.float32)
            feat2 = feat2_full[torch.randperm(feat2_full.shape[0])[:max_seq_len]]
            gram_l = gram_linear(feat2.to(DEVICE_COMPUTE))
            
            min_dim = min(gram_k.shape[0], gram_l.shape[0])
            cka_matrix[i, j] = cka(gram_k[:min_dim, :min_dim], gram_l[:min_dim, :min_dim])
    print("CKA矩阵计算完成。")
    return cka_matrix

def align_layers_dp(C):
    """使用动态规划计算最优映射 (Segment-wise Model Alignment)"""
    m, n = C.shape
    if m < n: raise ValueError("深度模型A的层数应大于等于浅度模型B")
    
    F = torch.zeros((n + 1, m + 1))
    P = torch.zeros((n + 1, m + 1), dtype=torch.long) # 指针，记录路径

    for i in range(1, n + 1):
        for j in range(i, m + 1):
            # C的索引需要-1
            # F[i, j-1] -> 不将第j层分给第i段
            # F[i-1, k] + C[k:j, i-1].sum() -> 将k+1到j层分给第i段
            # 这里我们使用论文中简化的SMA，只考虑段尾部的相似度
            val1 = F[i, j - 1]
            val2 = F[i - 1, j - 1] + C[j - 1, i - 1]
            if val1 > val2:
                F[i,j] = val1
                P[i,j] = j - 1
            else:
                F[i,j] = val2
                P[i,j] = j - 1
                
    # 回溯
    alignment = [[] for _ in range(n)]
    i, j = n, m
    while i > 0 and j > 0:
        prev_j = P[i,j]
        if F[i,j] > F[i, j-1]: # 意味着第j层是对齐点
             # 从上一个对齐点到当前对齐点的所有层都属于这个段
            prev_alignment_j = P[i-1,j-1] if i > 1 else -1
            segment = list(range(prev_alignment_j + 1, j))
            alignment[i-1] = segment
            j = prev_alignment_j
            i -= 1
        else:
            j -= 1
            
    # 处理第一个段
    if i == 1:
        alignment[0] = list(range(j))
        
    return [s for s in alignment if s] # 过滤空列表


# --- 4. 宽度异构合并：弹性神经元压缩 ---
def elastic_neuron_zipping(reps_list, target_width):
    """
    实现弹性神经元压缩。
    :param reps_list: 包含多个模型激活的列表 [(N, dim_1), (N, dim_2), ...]
    :param target_width: 合并后的目标宽度
    :return: 变换矩阵列表 [T_1, T_2, ...]
    """
    num_models = len(reps_list)
    dims = [r.shape[1] for r in reps_list]
    
    # 归一化并拼接所有神经元的激活
    all_neurons_norm = torch.cat(
        [r / (torch.norm(r, dim=0, keepdim=True) + 1e-8) for r in reps_list], dim=1
    )
    
    sim_matrix = all_neurons_norm.T @ all_neurons_norm
    sim_matrix.fill_diagonal_(0)

    total_neurons = sum(dims)
    groups = [[i] for i in range(total_neurons)]

    num_merges = total_neurons - target_width
    for _ in tqdm(range(num_merges), desc="弹性神经元压缩", leave=False):
        max_sim, flat_idx = torch.max(sim_matrix.flatten(), 0)
        if max_sim < -1: break
        
        idx1, idx2 = np.unravel_index(flat_idx.item(), sim_matrix.shape)

        # 合并组和相似度
        group1_len, group2_len = len(groups[idx1]), len(groups[idx2])
        sim_matrix[idx1, :] = (sim_matrix[idx1, :] * group1_len + sim_matrix[idx2, :] * group2_len) / (group1_len + group2_len)
        sim_matrix[:, idx1] = sim_matrix[idx1, :]
        sim_matrix[idx2, :] = sim_matrix[:, idx2] = -torch.inf
        sim_matrix.fill_diagonal_(0)
        
        groups[idx1].extend(groups[idx2])
        groups[idx2] = []
        
    # 构建变换矩阵
    transforms = [torch.zeros(d, target_width, device=device, dtype=reps_list[0].dtype) for d in dims]
    offsets = np.cumsum([0] + dims)
    
    final_group_idx = 0
    for group in groups:
        if group:
            for neuron_idx in group:
                for model_idx in range(num_models):
                    if offsets[model_idx] <= neuron_idx < offsets[model_idx+1]:
                        original_idx = neuron_idx - offsets[model_idx]
                        transforms[model_idx][original_idx, final_group_idx] = 1.0
                        break
            final_group_idx += 1
            
    return transforms

def apply_hetero_merge(base_layer, donor_layers, T_base, T_donors, alpha):
    """
    应用宽度和深度异构合并。
    :param base_layer: 浅模型的层 (nn.Module)
    :param donor_layers: 深模型段中的层列表 [nn.Module]
    :param T_base: 浅模型层的变换矩阵
    :param T_donors: 深模型段中各层变换矩阵的列表
    :param alpha: 合并系数
    """
    # 1. 变换并融合donor段的权重
    # W_donor_fused = (W_d1 @ T_d1) @ (W_d2 @ T_d2) ...
    # 这是一个简化，更准确的实现需要逐层应用变换
    # 我们这里采用论文Eq.6的逻辑: 合并第一层，后续层只用深模型的
    
    # 合并第一层
    donor_first_layer = donor_layers[0]
    T_donor_first = T_donors[0]
    
    # -- Projections (q, k, v) --
    for proj_name in ["q_proj", "k_proj", "v_proj"]:
        base_proj = getattr(base_layer, proj_name)
        donor_proj = getattr(donor_first_layer, proj_name)

        w_base_transformed = base_proj.weight.data @ T_base.to(base_proj.weight.dtype)
        w_donor_transformed = donor_proj.weight.data @ T_donor_first.to(donor_proj.weight.dtype)

        base_proj.weight.data = (1 - alpha) * w_base_transformed + alpha * w_donor_transformed
        if base_proj.bias is not None and donor_proj.bias is not None:
            base_proj.bias.data = (1 - alpha) * base_proj.bias.data + alpha * donor_proj.bias.data

    # -- Output Projection --
    base_o_proj = base_layer.o_proj
    donor_o_proj = donor_first_layer.o_proj
    w_base_transformed = T_base.T.to(base_o_proj.weight.dtype) @ base_o_proj.weight.data
    w_donor_transformed = T_donor_first.T.to(donor_o_proj.weight.dtype) @ donor_o_proj.weight.data
    base_o_proj.weight.data = (1 - alpha) * w_base_transformed + alpha * w_donor_transformed
    
    # 2. 对于donor段中剩余的层, 权重只来自donor模型, 但需要经过变换
    # W'_j = (T_j-1)^-1 @ W_j @ T_j  (伪逆)
    # 简化：由于变换矩阵是稀疏的0/1矩阵，转置即为逆
    for i in range(1, len(donor_layers)):
        donor_layer = donor_layers[i]
        T_prev = T_donors[i-1]
        T_curr = T_donors[i]
        
        # 找到需要被替换的base_model中的层, 这里我们直接修改base_layer
        # 这意味着我们将深层网络的结构“注入”到了浅层网络中
        # 这是一个复杂的操作，简单起见，我们只打印信息
        print(f"  [信息] 深模型段的第 {i+1} 层 ({donor_layer}) 将被合并。")
        print(f"  其权重应使用 T_prev 和 T_curr 进行变换。")
        # 在一个更完整的框架中，这里需要动态地向base_model中插入新的层
        # 或者修改现有层的权重来模拟这个过程
        # 例如，可以将被合并层之后的层权重用这里的变换后权重替换

# --- 5. 主执行流程 ---
def main(alpha=0.5):
    """执行模型对齐与合并的主函数"""
    # --- 步骤 1: 加载模型和数据集 ---
    tokenizer_a, model_a = load_complete_model(CKPT_PATH["llama2"], DEVICE_DONOR)
    tokenizer_b, model_b = load_complete_model(CKPT_PATH["qwen2"], DEVICE_BASE)
    
    models = {"a": model_a, "b": model_b}

    if model_a.config.num_hidden_layers > model_b.config.num_hidden_layers:
        deep_key, shallow_key = 'a', 'b'
    else:
        deep_key, shallow_key = 'b', 'a'
        
    model_deep, model_shallow = models[deep_key], models[shallow_key]
    tokenizer_deep, tokenizer_shallow = (tokenizer_a, tokenizer_b) if deep_key == 'a' else (tokenizer_b, tokenizer_a)
        
    names_deep = [f"model.layers.{i}" for i in range(model_deep.config.num_hidden_layers)]
    names_shallow = [f"model.layers.{i}" for i in range(model_shallow.config.num_hidden_layers)]

    dataset = load_and_prepare_dataset(tokenizer_a, tokenizer_b, max_samples=16, max_length=64)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # --- 步骤 2: 收集特征表示 ---
    print("\n--- 步骤 2: 收集特征表示 ---")
    reps_deep, hooks_deep = register_hooks_for_reps(model_deep, names_deep)
    reps_shallow, hooks_shallow = register_hooks_for_reps(model_shallow, names_shallow)

    for batch in tqdm(dataloader, desc="特征提取"):
        with torch.no_grad():
            inputs_deep = {"input_ids": batch[f"input_ids_{deep_key}"].to(model_deep.device), "attention_mask": batch[f"attention_mask_{deep_key}"].to(model_deep.device)}
            inputs_shallow = {"input_ids": batch[f"input_ids_{shallow_key}"].to(model_shallow.device), "attention_mask": batch[f"attention_mask_{shallow_key}"].to(model_shallow.device)}
            model_deep(**inputs_deep)
            model_shallow(**inputs_shallow)
            
    for hook in hooks_deep + hooks_shallow: hook.remove()

    # --- 步骤 3: 深度对齐 ---
    print("\n--- 步骤 3: 深度异构对齐 ---")
    cka_matrix = compute_cka_matrix(reps_deep, reps_shallow, names_deep, names_shallow)
    layer_alignment = align_layers_dp(cka_matrix)
    
    print("\n找到的层级映射关系 (Shallow -> Deep Segments):")
    for i, segment_indices in enumerate(layer_alignment):
        print(f"  Shallow Layer {names_shallow[i]} -> Deep Segment {[names_deep[j] for j in segment_indices]}")

    # --- 步骤 4: 宽度对齐与合并 ---
    print(f"\n--- 步骤 4: 宽度异构合并 (alpha={alpha}) ---")
    merged_model = model_shallow # 我们修改浅模型作为基座

    for i, deep_segment_indices in enumerate(tqdm(layer_alignment, desc="合并层段")):
        if not deep_segment_indices: continue

        shallow_layer_name = names_shallow[i]
        
        # 1. 对段内每一层进行宽度对齐
        T_donors = []
        base_layer_reps = torch.cat(reps_shallow[shallow_layer_name], dim=0).flatten(0, 1).to(DEVICE_COMPUTE)
        target_width = base_layer_reps.shape[1]

        for donor_idx in deep_segment_indices:
            donor_layer_reps = torch.cat(reps_deep[names_deep[donor_idx]], dim=0).flatten(0, 1).to(DEVICE_COMPUTE)
            T_donor, T_base = elastic_neuron_zipping([donor_layer_reps, base_layer_reps], target_width)
            T_donors.append(T_donor)
        
        # 2. 应用合并
        base_layer_module = get_module_by_name(merged_model, shallow_layer_name).self_attn
        donor_layer_modules = [get_module_by_name(model_deep, names_deep[k]).self_attn for k in deep_segment_indices]
        
        # 注意：这里我们只合并了第一层，这是一个简化。
        # 一个完整的实现需要修改模型架构，将donor_layers[1:]的变换结果插入到merged_model中。
        # 这超出了简单的权重修改范围，需要动态构建网络，非常复杂。
        # 我们遵循论文的简化逻辑，合并第一层，并提示后续层的处理方式。
        print(f"正在合并 {shallow_layer_name} 与 {names_deep[deep_segment_indices[0]]}...")
        apply_hetero_merge(base_layer_module, donor_layer_modules, T_base, T_donors, alpha)
        if len(donor_layer_modules) > 1:
            print(f"  注意：段 {deep_segment_indices} 包含多于一层。")
            print(f"  在理想情况下，需要将 {len(donor_layer_modules)-1} 个额外的变换层注入到合并模型中。")
            print(f"  为简化起见，当前实现仅合并了第一层。")


    # --- 步骤 5: 保存并测试 ---
    output_dir = f"./merged_model_hetero_alpha_{alpha}"
    print(f"\n--- 正在保存合并后的模型到 {output_dir} ---")
    merged_model.save_pretrained(output_dir)
    tokenizer_shallow.save_pretrained(output_dir)
    print("模型保存完成。")

    del model_a, model_b, model_deep, model_shallow, reps_deep, reps_shallow
    gc.collect()
    torch.cuda.empty_cache()

    print("\n--- 测试合并后的模型 ---")
    tokenizer_merged, model_merged = load_complete_model(output_dir, DEVICE_BASE)
    prompt = "The capital of France is"
    inputs = tokenizer_merged(prompt, return_tensors="pt").to(model_merged.device)
    with torch.no_grad():
        outputs = model_merged.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer_merged.eos_token_id)
    print("输入:", prompt)
    print("合并后模型输出:", tokenizer_merged.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main(alpha=0.5)