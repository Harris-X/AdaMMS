import os
import torch
import torch.nn as nn
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# --- 1. 核心配置 ---
CKPT_PATH = {
    "llama2": "./downloaded_models/Llama-2-7b-hf",
    "qwen2": "./downloaded_models/Qwen2-7B-Instruct",
}
MODEL_DEVICE_A = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_DEVICE_B = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
COMPUTE_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"模型A设备: {MODEL_DEVICE_A}, 模型B设备: {MODEL_DEVICE_B}, 计算设备: {COMPUTE_DEVICE}")

# --- 2. 辅助函数 ---
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

def register_hooks_for_reps(model, layer_names):
    reps_out, hooks = {n: [] for n in layer_names}, []
    
    def get_hook_fn(name):
        def hook_fn(module, input, output):
            # 处理不同类型的输出
            if isinstance(output, tuple):
                # 取第一个元素，通常是主要输出
                tensor_output = output[0]
            elif hasattr(output, "last_hidden_state"):
                # 处理一些模型的特殊输出格式
                tensor_output = output.last_hidden_state
            else:
                tensor_output = output
                
            # 确保是张量类型
            if not isinstance(tensor_output, torch.Tensor):
                print(f"警告: {name} 的输出不是张量，而是 {type(tensor_output)}")
                return
                
            # 安全地转移到CPU
            reps_out[name].append(tensor_output.detach().cpu())
        return hook_fn
        
    for name in layer_names:
        module = get_module_by_name(model, name)
        if module is not None:
            hooks.append(module.register_forward_hook(get_hook_fn(name)))
        else:
            print(f"警告: 找不到模块 {name}")
            
    return reps_out, hooks

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
    for i, name1 in enumerate(tqdm(names1, desc="Llama2 Layers")):
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

# --- 3. 创建堆叠模型 ---
def create_stacked_model(model_qwen, tokenizer_qwen, stack_strategy):
    """
    根据堆叠策略创建堆叠后的Qwen2模型（内存优化版）
    """
    print(f"开始创建堆叠模型，堆叠策略: {stack_strategy}")
    
    # 清理现有的GPU缓存，为新模型腾出空间
    gc.collect()
    torch.cuda.empty_cache()
    
    # 克隆原始模型配置
    config = deepcopy(model_qwen.config)
    original_num_layers = config.num_hidden_layers
    target_num_layers = original_num_layers + sum(stack_strategy.values())
    config.num_hidden_layers = target_num_layers
    
    # 先在CPU上创建空的堆叠模型
    print("在CPU上创建模型以节省GPU内存...")
    stacked_model = AutoModelForCausalLM.from_pretrained(
        CKPT_PATH["qwen2"], 
        config=config,
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True, 
        low_cpu_mem_usage=True,
        device_map="cpu"  # 强制加载到CPU
    )
    
    # 创建层映射关系
    layer_mapping = []
    for old_idx in range(original_num_layers):
        repeat_times = 1 + stack_strategy.get(old_idx, 0)
        for _ in range(repeat_times):
            layer_mapping.append(old_idx)
    
    print(f"层映射关系: {layer_mapping}")
    
    # --- 关键修复：将源模型移动到CPU ---
    print(f"将源模型从 {model_qwen.device} 移动到 CPU 以避免GPU内存溢出...")
    model_qwen.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    # --- 修复结束 ---

    # 从原始模型获取状态字典 (现在所有张量都在CPU上)
    print("从原始模型提取权重...")
    original_state_dict = model_qwen.state_dict()
    
    # 构建新的状态字典
    print("构建堆叠模型的权重...")
    new_state_dict = {}
    
    # 分批处理权重以减少内存使用
    non_layer_keys = [k for k in original_state_dict.keys() if "model.layers." not in k]
    
    layer_keys_pattern = set()
    for k in original_state_dict.keys():
        if "model.layers." in k:
            parts = k.split('.')
            suffix = '.'.join(parts[3:])
            if suffix:
                layer_keys_pattern.add(suffix)

    # 先处理非层参数
    for key in non_layer_keys:
        new_state_dict[key] = original_state_dict[key].clone()
    
    # 分批处理每个新层
    for new_idx, old_idx in enumerate(tqdm(layer_mapping, desc="构建层权重")):
        for suffix in layer_keys_pattern:
            old_key = f"model.layers.{old_idx}.{suffix}"
            new_key = f"model.layers.{new_idx}.{suffix}"
            if old_key in original_state_dict:
                new_state_dict[new_key] = original_state_dict[old_key].clone()
    
    # 加载状态字典
    print("加载新状态字典...")
    stacked_model.load_state_dict(new_state_dict)
    
    # 清理不需要的变量
    del new_state_dict, original_state_dict
    gc.collect()
    
    print(f"堆叠模型创建完成，原始层数: {original_num_layers}，堆叠后层数: {target_num_layers}")
    
    # 返回模型，仍在CPU上，可以在需要时再移动到GPU
    return stacked_model, tokenizer_qwen

# --- 4. 评估模型性能 ---
def evaluate_model(model, tokenizer, prompts, max_new_tokens=50, device=None):
    """内存安全的模型评估函数"""
    results = []
    
    # 确定设备
    if device is None:
        # 尝试找到有足够内存的GPU
        if torch.cuda.is_available():
            # 检查可用GPU内存
            device_id = -1
            max_free_mem = 0
            for i in range(torch.cuda.device_count()):
                free_mem = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                if free_mem > max_free_mem and free_mem > 2 * 1024 * 1024 * 1024:  # 至少需要2GB空闲
                    max_free_mem = free_mem
                    device_id = i
            
            if device_id >= 0:
                device = torch.device(f"cuda:{device_id}")
                print(f"使用GPU {device_id} 进行评估")
            else:
                device = torch.device("cpu")
                print("所有GPU内存不足，使用CPU进行评估")
        else:
            device = torch.device("cpu")
            print("无可用GPU，使用CPU进行评估")
    
    # 确保模型在正确的设备上
    model_device = next(model.parameters()).device
    if model_device != device:
        print(f"将模型从 {model_device} 移动到 {device}...")
        # 如果是大模型，可能需要逐层加载到GPU
        try:
            model.to(device)
        except RuntimeError:
            print("内存不足，使用CPU评估")
            device = torch.device("cpu")
            model.to(device)
    
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            print(f"生成回应: '{prompt}'")
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # 使用更保守的生成配置以节省内存
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True  # 启用KV缓存以加速生成
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append({"prompt": prompt, "generated": generated_text})
            
            # 输出示例
            print(f"生成: {generated_text}\n")
            
            # 每次生成后清理缓存
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results

# --- 5. 主执行流程 ---
def main():
    # 加载模型
    tokenizer_llama, model_llama = load_complete_model(CKPT_PATH["llama2"], MODEL_DEVICE_A)
    tokenizer_qwen, model_qwen = load_complete_model(CKPT_PATH["qwen2"], MODEL_DEVICE_B)
    
    # 提取模型层数
    llama_layers = model_llama.config.num_hidden_layers  # 应该是32
    qwen_layers = model_qwen.config.num_hidden_layers    # 应该是28
    print(f"Llama2层数: {llama_layers}, Qwen2层数: {qwen_layers}")
    
    # 准备数据
    dataset = load_and_prepare_dataset(tokenizer_llama, tokenizer_qwen, max_samples=8)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 注册钩子收集激活
    names_llama = [f"model.layers.{i}" for i in range(llama_layers)]
    names_qwen = [f"model.layers.{i}" for i in range(qwen_layers)]
    
    reps_llama, hooks_llama = register_hooks_for_reps(model_llama, names_llama)
    reps_qwen, hooks_qwen = register_hooks_for_reps(model_qwen, names_qwen)
    
    # 运行前向传播收集特征
    for batch in tqdm(dataloader, desc="特征提取"):
        with torch.no_grad():
            model_llama(input_ids=batch["input_ids_a"].to(MODEL_DEVICE_A),
                       attention_mask=batch["attention_mask_a"].to(MODEL_DEVICE_A))
            model_qwen(input_ids=batch["input_ids_b"].to(MODEL_DEVICE_B),
                      attention_mask=batch["attention_mask_b"].to(MODEL_DEVICE_B))
    
    # 移除钩子
    for hook in hooks_llama + hooks_qwen:
        hook.remove()
    
    # 计算CKA相似度矩阵
    cka_matrix = compute_cka_matrix(reps_llama, reps_qwen, names_llama, names_qwen)
    
    # 可视化CKA矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(cka_matrix, cmap='viridis')
    plt.colorbar(label='CKA Similarity')
    plt.xlabel('Qwen2 Layers')
    plt.ylabel('Llama2 Layers')
    plt.title('Layer-wise CKA Similarity between Llama2 and Qwen2')
    plt.savefig('llama2_qwen2_cka_similarity.png')
    plt.close()
    
    # 确定堆叠策略
    # 方法1: 使用CKA矩阵中最相似的层
    stack_strategy = {}
    
    # 计算需要额外堆叠的层数
    layers_to_add = llama_layers - qwen_layers  # 应该是4
    
    # 找出哪些Qwen层与多个Llama层高度相似
    layer_match_counts = {}
    threshold = 0.8 * torch.max(cka_matrix)  # 设置相似度阈值
    
    for llama_idx in range(llama_layers):
        max_sim_qwen_idx = torch.argmax(cka_matrix[llama_idx]).item()
        if cka_matrix[llama_idx, max_sim_qwen_idx] > threshold:
            if max_sim_qwen_idx not in layer_match_counts:
                layer_match_counts[max_sim_qwen_idx] = 0
            layer_match_counts[max_sim_qwen_idx] += 1
    
    # 根据匹配次数排序
    sorted_matches = sorted(layer_match_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 分配堆叠次数
    remaining_layers = layers_to_add
    for qwen_idx, match_count in sorted_matches:
        if remaining_layers <= 0:
            break
        # 每个高频匹配层最多堆叠1次
        stack_times = min(1, remaining_layers)
        stack_strategy[qwen_idx] = stack_times
        remaining_layers -= stack_times
    
    # 如果还有剩余需要堆叠的层，从模型中间开始堆叠
    if remaining_layers > 0:
        middle_start = qwen_layers // 3
        for i in range(remaining_layers):
            idx = (middle_start + i) % qwen_layers
            if idx not in stack_strategy:
                stack_strategy[idx] = 1
                remaining_layers -= 1
            if remaining_layers <= 0:
                break
    
    print(f"确定的堆叠策略: {stack_strategy}")
    total_added = sum(stack_strategy.values())
    print(f"总共添加的层数: {total_added}, 目标添加层数: {layers_to_add}")
    
    # 创建堆叠模型
    stacked_model, stacked_tokenizer = create_stacked_model(model_qwen, tokenizer_qwen, stack_strategy)
    
    # 保存堆叠模型（可选：以safetensors格式保存以减少内存使用）
    output_dir = "./stacked_qwen2_to_llama2_depth"
    print(f"\n--- 正在保存堆叠模型到 {output_dir} ---")

    # 确保模型在CPU上以节省GPU内存
    stacked_model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    # 保存模型（分片以减少内存使用）
    stacked_model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="2GB")
    stacked_tokenizer.save_pretrained(output_dir)
    print("模型保存完成。")
    
    # 测试评估模型
    test_prompts = [
        "The capital of France is",
        "Artificial intelligence can be defined as",
        "The main difference between deep learning and machine learning is",
        "In the context of climate change, renewable energy sources"
    ]
    
    print("\n--- 测试原始Qwen2模型 ---")
    # 清理内存以释放空间
    del model_llama
    gc.collect()
    torch.cuda.empty_cache()
    original_results = evaluate_model(model_qwen, tokenizer_qwen, test_prompts, device=MODEL_DEVICE_B)

    print("\n--- 测试堆叠后的Qwen2模型 ---")
    # 清理更多内存
    # model_qwen 已经在前面被移动到CPU，这里可以安全删除
    del model_qwen
    gc.collect()
    torch.cuda.empty_cache()
    # 尝试选择一个空闲的GPU
    stacked_results = evaluate_model(stacked_model, stacked_tokenizer, test_prompts, device=COMPUTE_DEVICE)
    
    # --- 修复开始 ---
    # 清理资源
    # model_llama 和 model_qwen 已经在前面被删除了，这里只删除 stacked_model
    print("清理最后的模型资源...")
    del stacked_model
    gc.collect()
    torch.cuda.empty_cache()
    # --- 修复结束 ---
    
    print("\n任务完成！")

if __name__ == "__main__":
    main()