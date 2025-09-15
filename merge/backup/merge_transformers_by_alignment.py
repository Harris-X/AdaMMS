import os
import torch
import torch.nn as nn
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# --- 1. 核心配置 ---
CKPT_PATH = {
    "llama2": "./downloaded_models/Llama-2-7b-hf",
    "qwen2": "./downloaded_models/Qwen2-7B-Instruct",
}
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# --- 2. 辅助函数 (模型加载, 数据集, CKA计算, 对齐) ---

def load_complete_model(model_id):
    """通用模型加载函数，使用bfloat16以节省内存"""
    print(f"正在加载模型: {model_id}...")
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
        text = [t for t in examples["text"] if t.strip()]
        if not text: return {}
        inputs_a = tokenizer_a(text, max_length=max_length, padding="max_length", truncation=True)
        inputs_b = tokenizer_b(text, max_length=max_length, padding="max_length", truncation=True)
        return {
            "input_ids_a": inputs_a.input_ids, "attention_mask_a": inputs_a.attention_mask,
            "input_ids_b": inputs_b.input_ids, "attention_mask_b": inputs_b.attention_mask,
        }

    processed_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    # **关键修复**: 设置数据集格式为PyTorch张量
    processed_dataset.set_format(type='torch', columns=['input_ids_a', 'attention_mask_a', 'input_ids_b', 'attention_mask_b'])
    return processed_dataset

def get_module_by_name(model, module_name):
    """通过字符串名称获取模块"""
    parts = module_name.split('.')
    module = model
    for part in parts:
        module = getattr(module, part)
    return module

def register_hooks_for_reps(model, layer_names):
    """为指定层注册钩子以捕获输出"""
    reps = {name: [] for name in layer_names}
    hooks = []
    def get_hook_fn(name):
        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            reps[name].append(hidden_states.detach().cpu())
        return hook_fn

    for name in layer_names:
        module = get_module_by_name(model, name)
        hooks.append(module.register_forward_hook(get_hook_fn(name)))
    return reps, hooks

def _HSIC(K, L):
    """计算HSIC"""
    K, L = K.float(), L.float()
    N = K.shape[0]
    if N < 4: return 0.0
    ones = torch.ones(N, 1, device=K.device)
    result = torch.trace(K @ L)
    result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
    result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
    return (1 / (N * (N - 3)) * result).item() if N > 3 else 0.0

def compute_cka(reps1, reps2, names1, names2, max_features=8192):
    """计算CKA相似度矩阵"""
    cka_matrix = torch.zeros(len(names1), len(names2))
    for i, name1 in enumerate(tqdm(names1, desc="计算CKA矩阵")):
        # **改进**: 将所有批次的激活连接起来，并展平
        feat1 = torch.cat(reps1[name1], dim=0).flatten(0, 1)
        
        # **优化**: 如果特征过多，随机抽样以避免OOM
        if feat1.shape[0] > max_features:
            indices = torch.randperm(feat1.shape[0])[:max_features]
            feat1 = feat1[indices]
        
        X = feat1.to(device)
        K = X @ X.t()
        K.fill_diagonal_(0.0)
        hsic_K_K = _HSIC(K, K)
        if hsic_K_K == 0: continue

        for j, name2 in enumerate(names2):
            feat2 = torch.cat(reps2[name2], dim=0).flatten(0, 1)
            if feat2.shape[0] > max_features:
                indices = torch.randperm(feat2.shape[0])[:max_features]
                feat2 = feat2[indices]
            
            Y = feat2.to(device)
            L = Y @ Y.t()
            L.fill_diagonal_(0.0)
            hsic_L_L = _HSIC(L, L)
            if hsic_L_L == 0: continue
            hsic_K_L = _HSIC(K, L)
            cka = hsic_K_L / (torch.sqrt(torch.tensor(hsic_K_K * hsic_L_L, device=device)) + 1e-10)
            cka_matrix[i, j] = cka
    return cka_matrix

def align(C):
    """使用动态规划计算最优映射"""
    m, n = C.shape
    if m < n: raise ValueError("成本矩阵行数m应>=列数n")
    F = torch.zeros((n + 1, m + 1))
    for k in range(1, n + 1):
        for l in range(k, m + 1):
            F[k, l] = max(F[k, l - 1], F[k - 1, l - 1] + C[l - 1, k - 1])
    A = torch.zeros(n, dtype=torch.long)
    k, l = n, m
    while k > 0 and l > 0:
        if l > k and F[k, l].item() == F[k, l - 1].item():
            l -= 1
        else:
            A[k - 1] = l - 1
            k -= 1; l -= 1
    return A

# --- 3. 核心功能：参数变换与合并 ---

def solve_linear_transform(A, B):
    """解线性方程 AX = B, 求变换矩阵 X"""
    A_f = A.to(torch.float32)
    B_f = B.to(torch.float32)
    X = torch.linalg.pinv(A_f) @ B_f
    return X.to(A.dtype)

def transform_and_merge_layer(base_layer, donor_layer, transform_matrix, alpha):
    """使用变换矩阵调整donor_layer的权重，并与base_layer合并"""
    base_o_proj_weight = base_layer.o_proj.weight.data
    donor_o_proj_weight = donor_layer.o_proj.weight.data
    
    transform_matrix = transform_matrix.to(donor_o_proj_weight.dtype)
    transformed_donor_weight = donor_o_proj_weight @ transform_matrix
    
    if base_o_proj_weight.shape == transformed_donor_weight.shape:
        base_layer.o_proj.weight.data = (1 - alpha) * base_o_proj_weight + alpha * transformed_donor_weight
    else:
        print(f"警告: o_proj变换后维度不匹配，跳过合并 {base_o_proj_weight.shape} vs {transformed_donor_weight.shape}")

    for proj in ['q_proj', 'k_proj', 'v_proj']:
        base_proj = getattr(base_layer, proj)
        donor_proj = getattr(donor_layer, proj)
        if base_proj.weight.shape == donor_proj.weight.shape:
             base_proj.weight.data = (1 - alpha) * base_proj.weight.data + alpha * donor_proj.weight.data
             # **修复**: 安全地合并bias
             if hasattr(base_proj, 'bias') and base_proj.bias is not None and hasattr(donor_proj, 'bias') and donor_proj.bias is not None:
                 base_proj.bias.data = (1 - alpha) * base_proj.bias.data + alpha * donor_proj.bias.data

# --- 4. 主执行流程 ---

def main(alpha=0.5):
    """执行模型对齐与合并的主函数"""
    # --- 步骤 1: 加载模型和数据集 ---
    tokenizer_a, model_a = load_complete_model(CKPT_PATH["llama2"]) # Donor
    tokenizer_b, model_b = load_complete_model(CKPT_PATH["qwen2"])  # Base
    
    dataset = load_and_prepare_dataset(tokenizer_a, tokenizer_b, max_samples=64, max_length=128)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4) # 减小batch size以防OOM

    attn_names_a = [name for name, module in model_a.named_modules() if name.endswith("self_attn")]
    attn_names_b = [name for name, module in model_b.named_modules() if name.endswith("self_attn")]

    # --- 步骤 2: 迭代数据集，收集特征表示 ---
    print("\n--- 正在为两个模型收集特征表示 ---")
    reps_a, hooks_a = register_hooks_for_reps(model_a, attn_names_a)
    reps_b, hooks_b = register_hooks_for_reps(model_b, attn_names_b)
    
    for batch in tqdm(dataloader, desc="特征提取"):
        with torch.no_grad():
            # **关键修复**: batch已经是tensor字典，直接使用
            inputs_a = {"input_ids": batch["input_ids_a"].to(device), "attention_mask": batch["attention_mask_a"].to(device)}
            model_a(**inputs_a)
            inputs_b = {"input_ids": batch["input_ids_b"].to(device), "attention_mask": batch["attention_mask_b"].to(device)}
            model_b(**inputs_b)
    
    for hook in hooks_a + hooks_b: hook.remove()

    # --- 步骤 3: 计算CKA并对齐 ---
    print("\n--- 正在计算CKA相似度并寻找层映射 ---")
    cka_matrix = compute_cka(reps_a, reps_b, attn_names_a, attn_names_b)
    alignment_indices = align(cka_matrix)
    
    layer_mapping = {attn_names_b[i]: attn_names_a[j] for i, j in enumerate(alignment_indices)}
    print("\n找到的层级映射关系 (Qwen2 -> Llama2):")
    for qwen_layer, llama_layer in layer_mapping.items():
        print(f"  {qwen_layer} -> {llama_layer}")

    # --- 步骤 4: 学习变换矩阵并合并 ---
    print(f"\n--- 正在学习变换矩阵并合并模型 (alpha={alpha}) ---")
    merged_model = model_b

    for base_name, donor_name in tqdm(layer_mapping.items(), desc="合并层"):
        base_rep = torch.cat(reps_b[base_name], dim=0).flatten(0, 1)
        donor_rep = torch.cat(reps_a[donor_name], dim=0).flatten(0, 1)
        
        min_len = min(base_rep.shape[0], donor_rep.shape[0])
        base_rep, donor_rep = base_rep[:min_len].to(device), donor_rep[:min_len].to(device)

        transform_matrix = solve_linear_transform(donor_rep, base_rep)
        
        base_layer = get_module_by_name(merged_model, base_name)
        donor_layer = get_module_by_name(model_a, donor_name)
        
        transform_and_merge_layer(base_layer, donor_layer, transform_matrix, alpha)

    # --- 步骤 5: 保存并测试 ---
    output_dir = f"./merged_qwen2_llama2_transformed_alpha_{alpha}"
    print(f"\n--- 正在保存合并后的模型到 {output_dir} ---")
    merged_model.save_pretrained(output_dir)
    tokenizer_b.save_pretrained(output_dir)
    print("模型保存完成。")

    del model_a, tokenizer_a, reps_a, reps_b, dataset, dataloader, merged_model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n--- 测试合并后的模型 ---")
    tokenizer_merged, model_merged = load_complete_model(output_dir)
    prompt = "The capital of France is"
    inputs = tokenizer_merged(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model_merged.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer_merged.eos_token_id)
    print("输入:", prompt)
    print("合并后模型输出:", tokenizer_merged.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main(alpha=0.5)