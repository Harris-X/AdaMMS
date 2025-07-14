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
# 建议使用多个GPU来分担模型加载的显存压力
# 例如: device_a = "cuda:0", device_b = "cuda:1", compute_device="cuda:0"
# 如果只有一个GPU，请确保显存足够大
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# --- 2. 辅助函数 ---

def load_complete_model(model_id, device):
    """通用模型加载函数"""
    print(f"正在加载模型: {model_id} 到 {device}...")
    # 使用 bfloat16 以节省内存
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        low_cpu_mem_usage=True
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
        # 为两个模型分别进行分词
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
        if not hasattr(module, part):
            return None
        module = getattr(module, part)
    return module

def register_hooks_for_reps(model, layer_names):
    """为指定层注册钩子以捕获输出激活"""
    reps = {name: [] for name in layer_names}
    hooks = []
    
    def get_hook_fn(name):
        def hook_fn(module, input, output):
            # LLM的输出通常是一个元组 (hidden_states, ...)
            hidden_states = output[0] if isinstance(output, tuple) else output
            reps[name].append(hidden_states.detach().to('cpu', non_blocking=True))
        return hook_fn

    for name in layer_names:
        module = get_module_by_name(model, name)
        if module:
            hooks.append(module.register_forward_hook(get_hook_fn(name)))
    return reps, hooks

# --- 3. 核心算法：CKA & 对齐 (基于论文思想) ---
def gram_linear(x):
    """计算线性核的格拉姆矩阵"""
    return x @ x.T

def center_gram(K):
    """中心化格拉姆矩阵"""
    n = K.shape[0]
    I = torch.eye(n, device=K.device)
    H = I - 1/n * torch.ones(n, n, device=K.device)
    return H @ K @ H

def cka(K, L):
    """计算中心核对齐(CKA)"""
    K_c = center_gram(K)
    L_c = center_gram(L)
    # HSIC (Hilbert-Schmidt Independence Criterion)
    hsic = (K_c.T * L_c).sum()
    var_K = torch.sqrt((K_c.T * K_c).sum())
    var_L = torch.sqrt((L_c.T * L_c).sum())
    # 避免除零
    if var_K == 0.0 or var_L == 0.0:
        return torch.tensor(0.0)
    return hsic / (var_K * var_L)

def compute_cka_matrix(reps1, reps2, names1, names2, max_seq_len=2048):
    """计算两个模型层激活之间的CKA相似度矩阵"""
    print("开始计算CKA矩阵...")
    cka_matrix = torch.zeros(len(names1), len(names2))
    
    for i, name1 in enumerate(tqdm(names1, desc="Model A Layers")):
        # 拼接所有批次的激活，并展平 (batch_size * seq_len, hidden_dim)
        feat1_full = torch.cat(reps1[name1], dim=0).flatten(0, 1).to(torch.float32)
        
        # 为避免OOM，对序列长度进行采样
        if feat1_full.shape[0] > max_seq_len:
            indices = torch.randperm(feat1_full.shape[0])[:max_seq_len]
            feat1 = feat1_full[indices]
        else:
            feat1 = feat1_full
            
        gram_k = gram_linear(feat1.to(device))
        
        for j, name2 in enumerate(names2):
            feat2_full = torch.cat(reps2[name2], dim=0).flatten(0, 1).to(torch.float32)
            if feat2_full.shape[0] > max_seq_len:
                indices = torch.randperm(feat2_full.shape[0])[:max_seq_len]
                feat2 = feat2_full[indices]
            else:
                feat2 = feat2_full

            gram_l = gram_linear(feat2.to(device))
            
            # 确保两个格拉姆矩阵维度一致
            min_dim = min(gram_k.shape[0], gram_l.shape[0])
            cka_matrix[i, j] = cka(gram_k[:min_dim, :min_dim], gram_l[:min_dim, :min_dim])
            
    print("CKA矩阵计算完成。")
    return cka_matrix

def align_layers_dp(C, alignment_type='LMA'):
    """
    使用动态规划进行层对齐
    :param C: 相似度矩阵 (深模型层数 x 浅模型层数)
    :param alignment_type: 'SMA' (段对齐) 或 'LMA' (层对齐)
    :return: 浅模型每层对应深模型中的层索引列表
    """
    m, n = C.shape
    if m < n:
        raise ValueError("深度模型A的层数应大于等于浅度模型B")

    F = torch.zeros((n + 1, m + 1))
    
    # 动态规划
    for i in range(1, n + 1):
        for j in range(i, m + 1):
            if alignment_type == 'SMA': # 段对齐
                F[i, j] = max(F[i, j - 1], F[i - 1, j - 1] + C[j - 1, i - 1])
            elif alignment_type == 'LMA': # 层对齐，考虑段内每一层的贡献
                # 简化版LMA：将段内所有层与目标层的相似度求和作为匹配得分
                # 一个更复杂的实现需要回溯路径，这里我们用一个近似
                segment_sim = C[i-1:j, i-1].sum() # 假设段是从i-1到j-1
                F[i, j] = max(F[i, j - 1], F[i - 1, i-1] + segment_sim if i<=j else -torch.inf)
            else:
                raise ValueError("未知的对齐类型")

    # 回溯找到最优路径
    alignment = []
    i, j = n, m
    while i > 0 and j > 0:
        if alignment_type == 'SMA':
            if F[i, j] == F[i, j - 1] and j > i:
                j -= 1
            else:
                alignment.insert(0, list(range(j-1, F[i-1, :j].argmax().item() if i>1 else -1, -1)))
                j = F[i-1, :j].argmax().item()
                i -= 1
        else: # LMA
             if F[i, j] == F[i, j - 1] and j > i:
                j -= 1
             else:
                # 找到上一个分段点
                prev_j = i - 2 if i > 1 else -1
                start_j = 0
                max_val = -torch.inf
                for k in range(i-1, j):
                    val = F[i-1, k] + C[k:j, i-1].sum()
                    if val > max_val:
                        max_val = val
                        start_j = k
                
                alignment.insert(0, list(range(start_j, j)))
                j = start_j
                i -= 1
    return alignment

# --- 4. 宽度异构合并：弹性神经元压缩 ---
def elastic_neuron_zipping(A_reps, B_reps, target_width):
    """
    实现弹性神经元压缩
    :param A_reps: 模型A的激活 (N, dim_a)
    :param B_reps: 模型B的激活 (N, dim_b)
    :param target_width: 合并后的目标宽度
    :return: 变换矩阵 (T_A, T_B)
    """
    A_reps_norm = A_reps / (torch.norm(A_reps, dim=0, keepdim=True) + 1e-6)
    B_reps_norm = B_reps / (torch.norm(B_reps, dim=0, keepdim=True) + 1e-6)
    
    # 合并所有神经元，计算跨模型的相似度矩阵
    # [cite: 14]
    all_neurons = torch.cat([A_reps_norm, B_reps_norm], dim=1)
    sim_matrix = all_neurons.T @ all_neurons
    sim_matrix.fill_diagonal_(0) # 忽略自身与自身的相似度

    num_a, num_b = A_reps.shape[1], B_reps.shape[1]
    total_neurons = num_a + num_b
    
    # 使用一个列表记录每个神经元当前所属的组
    groups = [[i] for i in range(total_neurons)]

    # 贪心合并，直到达到目标宽度
    # [cite: 140]
    num_merges = total_neurons - target_width
    for _ in tqdm(range(num_merges), desc="弹性神经元压缩"):
        # 找到最相似的一对
        max_sim, flat_idx = torch.max(sim_matrix.flatten(), 0)
        if max_sim < -1: break # 没有更多可合并的了
        
        idx1, idx2 = np.unravel_index(flat_idx.item(), sim_matrix.shape)

        # 合并组
        groups[idx1].extend(groups[idx2])
        groups[idx2] = []
        
        # 更新相似度矩阵：合并后的组与其它组的相似度取平均
        for i in range(total_neurons):
            if i != idx1 and i != idx2 and groups[i]:
                sim_1 = sim_matrix[idx1, i]
                sim_2 = sim_matrix[idx2, i]
                new_sim = (sim_1 * len(groups[idx1]) + sim_2 * len(groups[idx2])) / (len(groups[idx1]) + len(groups[idx2]))
                sim_matrix[idx1, i] = sim_matrix[i, idx1] = new_sim

        # 屏蔽已合并的行和列
        sim_matrix[idx2, :] = sim_matrix[:, idx2] = -torch.inf
        sim_matrix[idx1, idx1] = 0 # 自身相似度置0
        
    # 构建变换矩阵
    T_A = torch.zeros(num_a, target_width, device=device, dtype=A_reps.dtype)
    T_B = torch.zeros(num_b, target_width, device=device, dtype=B_reps.dtype)
    
    final_group_idx = 0
    for group in groups:
        if group:
            for neuron_idx in group:
                if neuron_idx < num_a: # 来自模型A
                    T_A[neuron_idx, final_group_idx] = 1.0
                else: # 来自模型B
                    T_B[neuron_idx - num_a, final_group_idx] = 1.0
            final_group_idx += 1
            
    return T_A, T_B


def apply_width_hetero_merge(base_layer, donor_layer, T_base, T_donor, alpha):
    """应用宽度异构合并到具体层"""
    # 合并线性投影层
    for proj_name in ["q_proj", "k_proj", "v_proj"]:
        base_proj = getattr(base_layer, proj_name)
        donor_proj = getattr(donor_layer, proj_name)
        
        # 变换donor权重以匹配base的输入维度
        # W' = W_donor @ T_donor
        transformed_donor_w = donor_proj.weight.data @ T_donor.to(donor_proj.weight.dtype)
        
        # 新的权重是加权平均
        base_proj.weight.data = (1 - alpha) * (base_proj.weight.data @ T_base.to(base_proj.weight.dtype)) + alpha * transformed_donor_w
        
        # Bias合并
        if base_proj.bias is not None and donor_proj.bias is not None:
             base_proj.bias.data = (1 - alpha) * base_proj.bias.data + alpha * donor_proj.bias.data

    # 合并输出投影层
    # W' = T_base.T @ W_base
    base_o_proj = base_layer.o_proj
    donor_o_proj = donor_layer.o_proj
    
    transformed_base_w = T_base.T.to(base_o_proj.weight.dtype) @ base_o_proj.weight.data
    transformed_donor_w = T_donor.T.to(donor_o_proj.weight.dtype) @ donor_o_proj.weight.data
    
    base_o_proj.weight.data = (1 - alpha) * transformed_base_w + alpha * transformed_donor_w
    
    if base_o_proj.bias is not None and donor_o_proj.bias is not None:
        base_o_proj.bias.data = (1 - alpha) * base_o_proj.bias.data + alpha * donor_o_proj.bias.data


# --- 5. 主执行流程 ---

def main(alpha=0.5):
    """执行模型对齐与合并的主函数"""
    # --- 步骤 1: 加载模型和数据集 ---
    # Donor是提供知识的模型，Base是接受知识并最终被修改的模型
    tokenizer_donor, model_donor = load_complete_model(CKPT_PATH["llama2"], device) # Llama2
    tokenizer_base, model_base = load_complete_model(CKPT_PATH["qwen2"], device)   # Qwen2

    # 确定哪个模型更深
    if model_donor.config.num_hidden_layers > model_base.config.num_hidden_layers:
        model_deep, model_shallow = model_donor, model_base
        names_deep = [f"model.layers.{i}" for i in range(model_deep.config.num_hidden_layers)]
        names_shallow = [f"model.layers.{i}" for i in range(model_shallow.config.num_hidden_layers)]
        reps_deep_key, reps_shallow_key = 'a', 'b'
    else:
        model_deep, model_shallow = model_base, model_donor
        names_deep = [f"model.layers.{i}" for i in range(model_deep.config.num_hidden_layers)]
        names_shallow = [f"model.layers.{i}" for i in range(model_shallow.config.num_hidden_layers)]
        reps_deep_key, reps_shallow_key = 'b', 'a'

    dataset = load_and_prepare_dataset(tokenizer_donor, tokenizer_base, max_samples=32, max_length=128)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    # --- 步骤 2: 收集特征表示 ---
    print("\n--- 正在为两个模型收集特征表示 ---")
    reps_deep, hooks_deep = register_hooks_for_reps(model_deep, names_deep)
    reps_shallow, hooks_shallow = register_hooks_for_reps(model_shallow, names_shallow)

    for batch in tqdm(dataloader, desc="特征提取"):
        with torch.no_grad():
            inputs_a = {"input_ids": batch["input_ids_a"].to(device), "attention_mask": batch["attention_mask_a"].to(device)}
            inputs_b = {"input_ids": batch["input_ids_b"].to(device), "attention_mask": batch["attention_mask_b"].to(device)}
            if reps_deep_key == 'a':
                model_deep(**inputs_a)
                model_shallow(**inputs_b)
            else:
                model_deep(**inputs_b)
                model_shallow(**inputs_a)
    
    for hook in hooks_deep + hooks_shallow: hook.remove()

    # --- 步骤 3: 深度对齐 ---
    print("\n--- 步骤 3: 深度异构对齐 ---")
    cka_matrix = compute_cka_matrix(reps_deep, reps_shallow, names_deep, names_shallow)
    # 使用LMA进行对齐
    # 
    layer_alignment = align_layers_dp(cka_matrix, alignment_type='LMA') 
    
    print("\n找到的层级映射关系 (Shallow -> Deep Segments):")
    for i, segment in enumerate(layer_alignment):
        print(f"  Shallow Layer {i} -> Deep Segment (Layers {segment})")

    # --- 步骤 4: 宽度对齐与合并 ---
    print(f"\n--- 步骤 4: 宽度异构合并 (alpha={alpha}) ---")
    merged_model = model_base # 我们将修改base model
    
    for i, deep_segment_indices in enumerate(tqdm(layer_alignment, desc="合并层段")):
        shallow_layer_name = names_shallow[i]
        
        # 1. 准备该段的激活
        # 将一个段内所有深模型层的激活拼接起来
        segment_reps_deep = torch.cat(
            [torch.cat(reps_deep[names_deep[k]], dim=0).flatten(0, 1) for k in deep_segment_indices],
            dim=1 # 在特征维度上拼接
        ).to(device)
        
        reps_shallow_layer = torch.cat(reps_shallow[shallow_layer_name], dim=0).flatten(0, 1).to(device)
        
        # 2. 宽度对齐：弹性神经元压缩
        # 目标宽度是浅模型层的宽度
        # [cite: 144]
        target_w = model_shallow.config.hidden_size
        T_deep_segment, T_shallow = elastic_neuron_zipping(segment_reps_deep, reps_shallow_layer, target_w)
        
        # 3. 合并
        # 将一个段视为一个整体，应用变换
        base_layer = get_module_by_name(merged_model, shallow_layer_name).self_attn
        
        # 对于donor，我们需要将整个段的权重“融合”起来
        # 这是一个简化处理，实际中可能需要更复杂的融合方式
        # 这里我们取段的最后一层作为代表进行变换
        donor_layer = get_module_by_name(model_deep, names_deep[deep_segment_indices[-1]]).self_attn
        
        # 注意：这里的T_deep_segment的维度需要与融合后的donor层权重维度匹配
        # 这是一个复杂点，我们先假设可以找到一个代表性的变换
        # 一个简单的策略是只使用段最后一层的激活来计算变换
        final_donor_rep = torch.cat(reps_deep[names_deep[deep_segment_indices[-1]]], dim=0).flatten(0,1).to(device)
        final_T_donor, final_T_shallow = elastic_neuron_zipping(final_donor_rep, reps_shallow_layer, target_w)
        
        apply_width_hetero_merge(base_layer, donor_layer, final_T_shallow, final_T_donor, alpha)

    # --- 步骤 5: 保存并测试 ---
    output_dir = f"./merged_qwen2_llama2_hetero_alpha_{alpha}"
    print(f"\n--- 正在保存合并后的模型到 {output_dir} ---")
    merged_model.save_pretrained(output_dir)
    tokenizer_base.save_pretrained(output_dir)
    print("模型保存完成。")

    del model_donor, model_base, reps_deep, reps_shallow, dataset, dataloader, merged_model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n--- 测试合并后的模型 ---")
    tokenizer_merged, model_merged = load_complete_model(output_dir, device)
    prompt = "The capital of France is"
    inputs = tokenizer_merged(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model_merged.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer_merged.eos_token_id)
    print("输入:", prompt)
    print("合并后模型输出:", tokenizer_merged.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main(alpha=0.5)