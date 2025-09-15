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
MODEL_DEVICE_A = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
MODEL_DEVICE_B = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
COMPUTE_DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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

def load_and_prepare_qa_dataset(tokenizer_a, tokenizer_b, dataset_name="squad", split="validation", max_samples=16, max_length=128):
    """
    加载QA数据集并为两个模型准备输入
    
    Args:
        tokenizer_a, tokenizer_b: 两个模型的分词器
        dataset_name: 要加载的QA数据集名称
        split: 数据集分割
        max_samples: 最大样本数
        max_length: 序列最大长度
    
    Returns:
        处理后的数据集
    """
    print(f"正在加载QA数据集: {dataset_name}...")
    
    # 加载数据集
    if dataset_name == "squad":
        dataset = load_dataset("squad", split=split)
        
        flattened_data = []
        for example in dataset:
            question = example["question"]
            answer_text = example["answers"]["text"][0] if example["answers"]["text"] else ""
            context = example["context"]
            
            flattened_data.append({
                "question": question,
                "answer": answer_text,
                "context": context
            })
        
        from datasets import Dataset
        dataset = Dataset.from_dict({
            "question": [item["question"] for item in flattened_data],
            "answer": [item["answer"] for item in flattened_data],
            "context": [item["context"] for item in flattened_data]
        }).select(range(min(max_samples, len(flattened_data))))
    else:
        dataset = load_dataset(dataset_name, split=split).select(range(max_samples))
    
    def tokenize_qa_fn(examples):
        questions = examples["question"]
        answers = examples["answer"]
        
        # 确保问题非空
        valid_indices = [i for i, q in enumerate(questions) if q and q.strip()]
        if not valid_indices:
            return {}
        
        valid_questions = [questions[i] for i in valid_indices]
        valid_answers = [answers[i] for i in valid_indices]
        
        # 添加前缀使问题更明确
        formatted_questions = [f"Question: {q} Answer:" for q in valid_questions]
        
        # 使用两个模型的tokenizer处理输入
        inputs_a = tokenizer_a(formatted_questions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        inputs_b = tokenizer_b(formatted_questions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        
        return {
            "input_ids_a": inputs_a.input_ids,
            "attention_mask_a": inputs_a.attention_mask,
            "input_ids_b": inputs_b.input_ids,
            "attention_mask_b": inputs_b.attention_mask,
            "raw_question": valid_questions,
            "raw_answer": valid_answers
        }
    
    # 处理数据集
    processed_dataset = dataset.map(tokenize_qa_fn, batched=True, remove_columns=dataset.column_names)
    
    # --- 修复：包含所有字段在 set_format 中 ---
    processed_dataset.set_format(
        type='torch', 
        columns=["input_ids_a", "attention_mask_a", "input_ids_b", "attention_mask_b"],
        output_all_columns=True  # 确保所有列都可以访问
    )
    
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
        # 根据堆叠策略，确定每个原始层需要重复的次数
        repeat_times = 1 + stack_strategy.get(old_idx, 0)
        # 将该层索引重复相应次数添加到映射中
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

# --- LMA层对齐算法 ---
def align_layers_lma(C):
    """使用动态规划进行层对齐"""
    m, n = C.shape  # m是深层模型层数，n是浅层模型层数
    F = torch.full((n + 1, m + 1), -torch.inf)
    F[0, :] = 0
    path = torch.zeros((n + 1, m + 1), dtype=torch.long)
    
    for i in range(1, n + 1):
        for j in range(i, m + 1):
            max_val, best_k = -torch.inf, -1
            for k in range(i - 1, j):
                # 计算从深层模型的k到j-1的层与浅层模型的第i-1层的相似度总和
                segment_sim = C[k:j, i - 1].sum()
                current_val = F[i - 1, k] + segment_sim
                if current_val > max_val:
                    max_val, best_k = current_val, k
            F[i, j], path[i, j] = max_val, best_k
    
    # 回溯构建对齐方案
    alignment, i, j = [], n, m
    while i > 0:
        k = path[i, j].item()
        alignment.insert(0, list(range(k, j)))
        j = k
        i -= 1
    
    return alignment

def create_stack_strategy_threshold(cka_matrix, similarity_threshold=0.8, start_from_middle=True):
    """
    基于相似度阈值创建堆叠策略，可以从中间层开始。
    
    Args:
        cka_matrix: 相似度矩阵，形状为 [deep_layers, shallow_layers]
        similarity_threshold: 相似度阈值，高于此值的层被认为匹配良好
        start_from_middle: 是否从中间层开始处理
    
    Returns:
        堆叠策略字典 {shallow_layer_idx: num_repeats, ...}
    """
    deep_layers, shallow_layers = cka_matrix.shape
    stack_strategy = {}
    
    # 计算每个浅层模型层最相似的深层模型层
    best_matches = {}
    for j in range(shallow_layers):
        similarities = cka_matrix[:, j]
        best_idx = torch.argmax(similarities).item()
        best_sim = similarities[best_idx].item()
        best_matches[j] = (best_idx, best_sim)
    
    # 计算匹配矩阵：每个深层与哪些浅层匹配
    deep_to_shallow = {}
    for shallow_idx, (deep_idx, sim) in best_matches.items():
        if deep_idx not in deep_to_shallow:
            deep_to_shallow[deep_idx] = []
        deep_to_shallow[deep_idx].append((shallow_idx, sim))
    
    # 计算需要堆叠的层
    layers_to_stack = []
    for deep_idx in range(deep_layers):
        # 检查这个深层是否没有匹配的浅层
        if deep_idx not in deep_to_shallow or len(deep_to_shallow[deep_idx]) == 0:
            continue
            
        # 如果多个浅层都匹配到同一个深层，我们只保留相似度最高的一个
        if len(deep_to_shallow[deep_idx]) > 1:
            # 按相似度排序
            matched_layers = sorted(deep_to_shallow[deep_idx], key=lambda x: x[1], reverse=True)
            # 第一个是最匹配的，其余的需要堆叠
            for shallow_idx, sim in matched_layers[1:]:
                if sim >= similarity_threshold:  # 相似度超过阈值，认为是好的匹配
                    layers_to_stack.append(shallow_idx)
    
    # 从中间层开始处理
    if start_from_middle:
        middle_idx = shallow_layers // 2
        # 按照与中间层的距离排序
        layers_to_stack.sort(key=lambda x: abs(x - middle_idx))
    
    # 确定堆叠次数
    layers_to_add = deep_layers - shallow_layers
    for shallow_idx in layers_to_stack:
        if layers_to_add <= 0:
            break
        if shallow_idx not in stack_strategy:
            stack_strategy[shallow_idx] = 1
        else:
            stack_strategy[shallow_idx] += 1
        layers_to_add -= 1
    
    # 如果还需要更多层，从中间开始添加
    if layers_to_add > 0:
        middle_start = shallow_layers // 3
        for i in range(layers_to_add):
            idx = (middle_start + i) % shallow_layers
            if idx not in stack_strategy:
                stack_strategy[idx] = 1
            else:
                stack_strategy[idx] += 1
    
    return stack_strategy


def evaluate_qa_performance(model, tokenizer, dataset, max_new_tokens=50, device=None):
    """QA任务特定的评估函数"""
    results = []
    
    # 设备选择逻辑与evaluate_model相同
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
        for i in range(len(dataset)):
            question = dataset[i]["raw_question"]
            ground_truth = dataset[i]["raw_answer"]
            
            print(f"问题: '{question}'")
            inputs = tokenizer(question, return_tensors="pt").to(device)
            
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 对于QA任务，我们禁用采样以获得确定性回答
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的答案部分
            answer_part = generated_text[len(question):].strip()
            
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "generated": answer_part
            })
            
            print(f"真实答案: {ground_truth}")
            print(f"生成答案: {answer_part}\n")
            
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
    dataset = load_and_prepare_qa_dataset(
        tokenizer_llama, tokenizer_qwen, 
        dataset_name="squad",
        max_samples=16
    )
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
    
    # --- 保留LMA算法代码，但通过注释禁用 ---
    '''
    print("\n--- 使用LMA算法进行层对齐 ---")
    # LMA需要浅层模型作为对齐目标，所以如果llama2是深层，qwen2是浅层，直接使用cka_matrix
    # 如果qwen2是深层，llama2是浅层，需要转置cka_matrix
    if llama_layers > qwen_layers:  # llama2是深层模型
        layer_alignment = align_layers_lma(cka_matrix)
        deep_model_name, shallow_model_name = "Llama2", "Qwen2"
        deep_layers, shallow_layers = names_llama, names_qwen
    else:  # qwen2是深层模型
        layer_alignment = align_layers_lma(cka_matrix.T)
        deep_model_name, shallow_model_name = "Qwen2", "Llama2"
        deep_layers, shallow_layers = names_qwen, names_llama
    
    print("\n层对齐结果 (Shallow -> Deep Segments):")
    for i, segment in enumerate(layer_alignment):
        print(f"  {shallow_model_name} Layer {i} -> {deep_model_name} Layers {segment}")
    
    # 创建堆叠策略（针对qwen2）
    stack_strategy = {}
    
    if llama_layers > qwen_layers:  # 需要堆叠qwen2
        # 对于每个qwen2层，检查它对应几个llama2层
        for qwen_idx, llama_indices in enumerate(layer_alignment):
            # 如果一个qwen2层对应多个llama2层，则需要堆叠
            if len(llama_indices) > 1:
                # 需要堆叠的次数 = 对应的llama2层数 - 1
                stack_strategy[qwen_idx] = len(llama_indices) - 1
    else:  # 需要堆叠llama2（这种情况应该不会发生，但为完整性保留）
        # 对于每个llama2层，检查它对应几个qwen2层
        for llama_idx, qwen_indices in enumerate(layer_alignment):
            if len(qwen_indices) > 1:
                stack_strategy[llama_idx] = len(qwen_indices) - 1
    '''
    
    # --- 使用阈值对齐算法 ---
    print("\n--- 使用阈值方法进行层对齐（从中间层开始）---")
    if llama_layers > qwen_layers:  # 需要堆叠qwen2
        deep_model_name, shallow_model_name = "Llama2", "Qwen2"
        stack_strategy = create_stack_strategy_threshold(cka_matrix, similarity_threshold=0.7)
    else:  # 需要堆叠llama2
        deep_model_name, shallow_model_name = "Qwen2", "Llama2"
        stack_strategy = create_stack_strategy_threshold(cka_matrix.T, similarity_threshold=0.7)
        
    print(f"\n确定的堆叠策略: {stack_strategy}")
    total_added = sum(stack_strategy.values())
    layers_to_add = abs(llama_layers - qwen_layers)
    print(f"总共添加的层数: {total_added}, 目标添加层数: {layers_to_add}")
    
    # 如果添加的层数不匹配，调整策略
    if total_added != layers_to_add:
        print(f"调整堆叠策略以匹配目标层数...")
        if total_added < layers_to_add:
            # 需要增加更多层
            if llama_layers > qwen_layers:
                shallow_num_layers = qwen_layers
            else:
                shallow_num_layers = llama_layers
                
            # 从中间开始添加剩余层
            middle_start = shallow_num_layers // 3
            for i in range(layers_to_add - total_added):
                idx = (middle_start + i) % shallow_num_layers
                if idx not in stack_strategy:
                    stack_strategy[idx] = 1
                else:
                    stack_strategy[idx] += 1
                
                print(f"额外堆叠层: {idx}")
        elif total_added > layers_to_add:
            # 需要减少层数，从堆叠次数最少的层开始减
            layers_to_remove = total_added - layers_to_add
            stack_items = sorted(stack_strategy.items(), key=lambda x: x[1])
            
            for idx, count in stack_items:
                if layers_to_remove <= 0:
                    break
                if count <= layers_to_remove:
                    layers_to_remove -= count
                    del stack_strategy[idx]
                    print(f"移除堆叠层: {idx}")
                else:
                    stack_strategy[idx] -= layers_to_remove
                    layers_to_remove = 0
                    print(f"减少层 {idx} 的堆叠次数至 {stack_strategy[idx]}")
        
        print(f"最终堆叠策略: {stack_strategy}")
        total_added = sum(stack_strategy.values())
        print(f"最终添加层数: {total_added}, 目标层数: {layers_to_add}")
    
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
    
    # === 新增：测试提示评估 ===
    test_prompts = [
        "The capital of France is",
        "Artificial intelligence can be defined as",
        "小梅数她家的鸡与兔,数头有16个,数脚有44只.问小梅家的鸡与兔各有多少只？",
        "In the context of climate change, renewable energy sources"
    ]
    
    print("\n=== 开始对原始模型进行测试提示评估 ===")
    
    # 测试原始的两个模型
    original_test_results = test_models_with_prompts(
        model_llama, tokenizer_llama, 
        model_qwen, tokenizer_qwen, 
        test_prompts
    )
    
    print("\n=== 测试堆叠后的模型 ===")
    
    # 清理内存为堆叠模型测试做准备
    del model_llama  # 已经测试完成，可以删除
    gc.collect()
    torch.cuda.empty_cache()
    
    # 测试堆叠模型
    print("\n--- 测试堆叠后的Qwen2模型 ---")
    try:
        stacked_test_results = evaluate_model(
            stacked_model, 
            stacked_tokenizer, 
            test_prompts, 
            max_new_tokens=100,
            device=COMPUTE_DEVICE
        )
        
        print("堆叠模型测试结果:")
        for i, result in enumerate(stacked_test_results):
            print(f"  提示 {i+1}: {result['prompt']}")
            print(f"  生成: {result['generated']}")
            print()
            
    except Exception as e:
        print(f"堆叠模型测试失败: {e}")
        stacked_test_results = []
    
    # 保存测试结果
    save_test_results(original_test_results, stacked_test_results)
    
    # 显示对比结果
    print("\n=== 模型性能对比总结 ===")
    for i, prompt in enumerate(test_prompts):
        print(f"\n测试提示 {i+1}: '{prompt}'")
        print("-" * 60)
        
        # Llama2 结果
        if i < len(original_test_results.get("llama2", [])):
            llama_response = original_test_results["llama2"][i]["generated"]
            print(f"Llama2: {llama_response[:100]}{'...' if len(llama_response) > 100 else ''}")
        else:
            print("Llama2: [测试失败]")
        
        # Qwen2 结果
        if i < len(original_test_results.get("qwen2", [])):
            qwen_response = original_test_results["qwen2"][i]["generated"]
            print(f"Qwen2: {qwen_response[:100]}{'...' if len(qwen_response) > 100 else ''}")
        else:
            print("Qwen2: [测试失败]")
        
        # 堆叠模型结果
        if i < len(stacked_test_results):
            stacked_response = stacked_test_results[i]["generated"]
            print(f"堆叠Qwen2: {stacked_response[:100]}{'...' if len(stacked_response) > 100 else ''}")
        else:
            print("堆叠Qwen2: [测试失败]")
    
    # === 继续原有的QA数据集测试 ===
    print("\n=== 开始QA数据集测试 ===")
    
    # 创建评估数据集
    eval_dataset = load_and_prepare_qa_dataset(
        tokenizer_llama, tokenizer_qwen,
        dataset_name="squad", 
        max_samples=5
    )
    
    # 评估原始Qwen2模型
    print("\n--- QA测试：原始Qwen2模型 ---")
    original_qa_results = evaluate_qa_performance(model_qwen, tokenizer_qwen, eval_dataset, device=MODEL_DEVICE_B)

    # 清理更多内存
    del model_qwen
    gc.collect()
    torch.cuda.empty_cache()
    
    # 评估堆叠模型
    print("\n--- QA测试：堆叠后的Qwen2模型 ---")
    stacked_qa_results = evaluate_qa_performance(stacked_model, stacked_tokenizer, eval_dataset, device=COMPUTE_DEVICE)
    
    # QA测试结果对比
    print("\n--- QA模型性能比较 ---")
    for i, (orig, stacked) in enumerate(zip(original_qa_results, stacked_qa_results)):
        print(f"问题 {i+1}: {orig['question']}")
        print(f"  真实答案: {orig['ground_truth']}")
        print(f"  原始模型: {orig['generated']}")
        print(f"  堆叠模型: {stacked['generated']}")
        print()

    # --- 修复开始 ---
    # 清理资源
    # model_llama 和 model_qwen 已经在前面被删除了，这里只删除 stacked_model
    print("清理最后的模型资源...")
    # del stacked_model
    gc.collect()
    torch.cuda.empty_cache()
    # --- 修复结束 ---
    
    print("\n任务完成！")

    # --- 新增部分：收集和比较激活状态 ---
    print("\n--- 收集堆叠后模型的激活状态并与Llama2对比 ---")
    
    # 1. 保存原始Llama2的激活状态
    original_llama_activations = {}
    for name in names_llama:
        # 将所有批次的激活拼接成一个张量，并确保是 float32 类型
        original_llama_activations[name] = torch.cat(reps_llama[name], dim=0).float().cpu()
    
    # 清理不需要的变量以节省内存
    del reps_llama, reps_qwen
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. 为堆叠模型创建新的钩子
    stacked_layer_names = [f"model.layers.{i}" for i in range(stacked_model.config.num_hidden_layers)]
    stacked_reps, stacked_hooks = register_hooks_for_reps(stacked_model, stacked_layer_names)
    
    # 3. 在相同的数据上运行堆叠模型
    print("\n重新加载数据集进行堆叠模型激活收集...")
    # dataset = load_and_prepare_dataset(tokenizer_llama, stacked_tokenizer, max_samples=8)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    # 准备数据 - 使用QA数据集
    dataset = load_and_prepare_qa_dataset(
        tokenizer_llama, tokenizer_qwen, 
        dataset_name="squad",
        max_samples=16
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    stacked_model.to(COMPUTE_DEVICE)
    stacked_model.eval()
    
    for batch in tqdm(dataloader, desc="收集堆叠模型激活"):
        with torch.no_grad():
            stacked_model(
                input_ids=batch["input_ids_b"].to(COMPUTE_DEVICE),
                attention_mask=batch["attention_mask_b"].to(COMPUTE_DEVICE)
            )
    
    # 移除钩子
    for hook in stacked_hooks:
        hook.remove()
    
    # 4. 处理收集到的激活
    stacked_activations = {}
    for name in stacked_layer_names:
        # 确保激活数据是 float32 类型以避免后续问题
        stacked_activations[name] = torch.cat(stacked_reps[name], dim=0).float().cpu()
    
    # 5. 保存激活状态
    activations_dir = "./model_activations"
    os.makedirs(activations_dir, exist_ok=True)
    
    print(f"\n保存激活状态到 {activations_dir}...")
    
    # 保存Llama2激活
    torch.save(original_llama_activations, os.path.join(activations_dir, "llama2_activations.pt"))
    
    # 保存堆叠后Qwen2激活
    torch.save(stacked_activations, os.path.join(activations_dir, "stacked_qwen2_activations.pt"))
    
    # 6. 计算堆叠模型层与Llama2层之间的CKA相似度
    print("\n计算堆叠模型与Llama2的层间相似度...")
    stacked_cka_matrix = compute_cka_matrix(
        {name: [original_llama_activations[name]] for name in names_llama},
        {name: [stacked_activations[name]] for name in stacked_layer_names},
        names_llama,
        stacked_layer_names
    )
    
    # 7. 可视化新的CKA矩阵
    plt.figure(figsize=(12, 10))
    plt.imshow(stacked_cka_matrix, cmap='viridis')
    plt.colorbar(label='CKA Similarity')
    plt.xlabel('Stacked Qwen2 Layers')
    plt.ylabel('Llama2 Layers')
    plt.title('Layer-wise CKA Similarity between Llama2 and Stacked Qwen2')
    plt.savefig(os.path.join(activations_dir, 'llama2_stacked_qwen2_cka_similarity.png'))
    plt.close()
    
    # 8. 分析层对齐效果
    print("\n--- 分析层对齐效果 ---")
    # 寻找堆叠后每一层最相似的Llama2层
    best_matches = {}
    for j, stacked_name in enumerate(stacked_layer_names):
        best_i = torch.argmax(stacked_cka_matrix[:, j]).item()
        similarity = stacked_cka_matrix[best_i, j].item()
        best_matches[stacked_name] = (names_llama[best_i], similarity)
    
    # 输出对齐结果分析
    print("\n堆叠Qwen2模型的层与Llama2模型最相似的层:")
    for stacked_name, (llama_name, sim) in best_matches.items():
        stacked_idx = int(stacked_name.split('.')[-1])
        llama_idx = int(llama_name.split('.')[-1])
        print(f"  堆叠Qwen2层 {stacked_idx} -> Llama2层 {llama_idx} (相似度: {sim:.4f})")
    
    # 9. 将分析结果保存到文件
    with open(os.path.join(activations_dir, "layer_alignment_analysis.txt"), "w") as f:
        f.write("堆叠Qwen2模型的层与Llama2模型最相似的层:\n")
        for stacked_name, (llama_name, sim) in best_matches.items():
            stacked_idx = int(stacked_name.split('.')[-1])
            llama_idx = int(llama_name.split('.')[-1])
            f.write(f"堆叠Qwen2层 {stacked_idx} -> Llama2层 {llama_idx} (相似度: {sim:.4f})\n")
    
    # 10. 可视化激活状态的PCA (可选，仅对一部分层)
    try:
        from sklearn.decomposition import PCA
        
        print("\n执行神经元激活的PCA分析...")
        
        # 选择一些关键层进行可视化
        key_indices = [0, stacked_model.config.num_hidden_layers // 4, 
                       stacked_model.config.num_hidden_layers // 2,
                       stacked_model.config.num_hidden_layers - 1]
        
        for idx in key_indices:
            llama_layer = names_llama[idx] if idx < len(names_llama) else names_llama[-1]
            stacked_layer = stacked_layer_names[idx]
            
            # 提取激活并降维
            llama_act = original_llama_activations[llama_layer].flatten(1)
            stacked_act = stacked_activations[stacked_layer].flatten(1)
            
            # 对前1000个token进行PCA
            max_tokens = min(1000, llama_act.shape[0], stacked_act.shape[0])
            
            # --- 修复开始：分别处理不同维度的数据 ---
            # 将 BFloat16 转换为 float32，然后转换为 numpy
            llama_act_subset = llama_act[:max_tokens].float().numpy()
            stacked_act_subset = stacked_act[:max_tokens].float().numpy()
            
            print(f"层 {idx}: Llama2 特征维度: {llama_act_subset.shape[1]}, 堆叠Qwen2 特征维度: {stacked_act_subset.shape[1]}")
            
            # 检查数据有效性
            if np.any(np.isnan(llama_act_subset)) or np.any(np.isnan(stacked_act_subset)):
                print(f"警告: 层 {idx} 的激活数据包含 NaN，跳过此层的 PCA 分析")
                continue
                
            if np.any(np.isinf(llama_act_subset)) or np.any(np.isinf(stacked_act_subset)):
                print(f"警告: 层 {idx} 的激活数据包含无穷大值，跳过此层的 PCA 分析")
                continue
            
            # 执行PCA - 分别为每个模型创建 PCA
            try:
                # 为 Llama2 数据创建并拟合 PCA
                pca_llama = PCA(n_components=2)
                llama_pca = pca_llama.fit_transform(llama_act_subset)
                
                # 为堆叠 Qwen2 数据创建并拟合独立的 PCA
                pca_stacked = PCA(n_components=2)
                stacked_pca = pca_stacked.fit_transform(stacked_act_subset)
                
                # 可视化 - 分别显示两个模型的 PCA 结果
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                
                # 子图1: Llama2 PCA
                ax1.scatter(llama_pca[:, 0], llama_pca[:, 1], alpha=0.6, color='blue', s=20)
                ax1.set_title(f'Llama2 {llama_layer} PCA')
                ax1.set_xlabel('First Principal Component')
                ax1.set_ylabel('Second Principal Component')
                ax1.grid(True, alpha=0.3)
                
                # 子图2: 堆叠 Qwen2 PCA
                ax2.scatter(stacked_pca[:, 0], stacked_pca[:, 1], alpha=0.6, color='red', s=20)
                ax2.set_title(f'Stacked Qwen2 {stacked_layer} PCA')
                ax2.set_xlabel('First Principal Component')
                ax2.set_ylabel('Second Principal Component')
                ax2.grid(True, alpha=0.3)
                
                # 子图3: 重叠比较（标准化后）
                # 标准化 PCA 结果以便比较
                llama_pca_norm = (llama_pca - llama_pca.mean(axis=0)) / llama_pca.std(axis=0)
                stacked_pca_norm = (stacked_pca - stacked_pca.mean(axis=0)) / stacked_pca.std(axis=0)
                
                ax3.scatter(llama_pca_norm[:, 0], llama_pca_norm[:, 1], 
                           alpha=0.5, color='blue', s=20, label=f'Llama2 {llama_layer}')
                ax3.scatter(stacked_pca_norm[:, 0], stacked_pca_norm[:, 1], 
                           alpha=0.5, color='red', s=20, label=f'Stacked Qwen2 {stacked_layer}')
                ax3.set_title(f'Normalized PCA Comparison - Layer {idx}')
                ax3.set_xlabel('Normalized PC1')
                ax3.set_ylabel('Normalized PC2')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(activations_dir, f'pca_layer_{idx}_comparison.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # 计算并保存解释方差比
                with open(os.path.join(activations_dir, f'pca_layer_{idx}_variance.txt'), 'w') as f:
                    f.write(f"层 {idx} PCA 分析结果:\n")
                    f.write(f"Llama2 {llama_layer}:\n")
                    f.write(f"  特征维度: {llama_act_subset.shape[1]}\n")
                    f.write(f"  PC1 解释方差比: {pca_llama.explained_variance_ratio_[0]:.4f}\n")
                    f.write(f"  PC2 解释方差比: {pca_llama.explained_variance_ratio_[1]:.4f}\n")
                    f.write(f"  总解释方差比: {pca_llama.explained_variance_ratio_.sum():.4f}\n\n")
                    
                    f.write(f"堆叠Qwen2 {stacked_layer}:\n")
                    f.write(f"  特征维度: {stacked_act_subset.shape[1]}\n")
                    f.write(f"  PC1 解释方差比: {pca_stacked.explained_variance_ratio_[0]:.4f}\n")
                    f.write(f"  PC2 解释方差比: {pca_stacked.explained_variance_ratio_[1]:.4f}\n")
                    f.write(f"  总解释方差比: {pca_stacked.explained_variance_ratio_.sum():.4f}\n")
                
                print(f"层 {idx} 的 PCA 分析完成")
                
            except Exception as e:
                print(f"层 {idx} 的 PCA 分析失败: {e}")
                continue
            # --- 修复结束 ---
        
        print("PCA分析完成并保存在激活目录中")
    except ImportError:
        print("未安装scikit-learn，跳过PCA分析")
    except Exception as e:
        print(f"PCA分析过程中出现错误: {e}")
    
    # 清理资源
    # del stacked_activations, stacked_model # original_llama_activations
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n激活状态收集与分析完成！数据已保存到", activations_dir)

    # 在 PCA 分析之后添加这个可选的高级对比方法

    # 11. 可选：通过 CCA (Canonical Correlation Analysis) 对比不同维度的激活
    try:
        from sklearn.cross_decomposition import CCA
        
        print("\n执行 CCA 分析以直接比较不同维度的激活...")
        
        for idx in key_indices:
            llama_layer = names_llama[idx] if idx < len(names_llama) else names_llama[-1]
            stacked_layer = stacked_layer_names[idx]
            
            llama_act = original_llama_activations[llama_layer].flatten(1)
            stacked_act = stacked_activations[stacked_layer].flatten(1)
            
            max_tokens = min(500, llama_act.shape[0], stacked_act.shape[0])  # 使用更少的样本以加速CCA
            
            llama_act_subset = llama_act[:max_tokens].float().numpy()
            stacked_act_subset = stacked_act[:max_tokens].float().numpy()
            
            # 确定 CCA 的组件数量（不能超过最小的特征维度）
            n_components = min(2, llama_act_subset.shape[1], stacked_act_subset.shape[1])
            
            if n_components < 2:
                print(f"跳过层 {idx} 的 CCA 分析，特征维度太小")
                continue
            
            try:
                # 执行 CCA
                cca = CCA(n_components=n_components)
                llama_cca, stacked_cca = cca.fit_transform(llama_act_subset, stacked_act_subset)
                
                # 可视化 CCA 结果
                plt.figure(figsize=(10, 8))
                plt.scatter(llama_cca[:, 0], llama_cca[:, 1], alpha=0.5, 
                           color='blue', s=20, label=f'Llama2 {llama_layer}')
                plt.scatter(stacked_cca[:, 0], stacked_cca[:, 1], alpha=0.5, 
                           color='red', s=20, label=f'Stacked Qwen2 {stacked_layer}')
                plt.xlabel('First Canonical Component')
                plt.ylabel('Second Canonical Component')
                plt.title(f'CCA Analysis - Layer {idx}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(activations_dir, f'cca_layer_{idx}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # 计算典型相关系数
                correlations = []
                for i in range(n_components):
                    corr = np.corrcoef(llama_cca[:, i], stacked_cca[:, i])[0, 1]
                    correlations.append(corr)
                
                print(f"层 {idx} CCA 典型相关系数: {correlations}")
                
                # 保存CCA结果
                with open(os.path.join(activations_dir, f'cca_layer_{idx}_results.txt'), 'w') as f:
                    f.write(f"层 {idx} CCA 分析结果:\n")
                    f.write(f"典型相关系数: {correlations}\n")
                    f.write(f"平均相关系数: {np.mean(correlations):.4f}\n")
                
            except Exception as e:
                print(f"层 {idx} 的 CCA 分析失败: {e}")
                continue
        
        print("CCA分析完成")
        
    except ImportError:
        print("未安装scikit-learn的CCA模块，跳过CCA分析")
    except Exception as e:
        print(f"CCA分析过程中出现错误: {e}")

def test_models_with_prompts(model_llama, tokenizer_llama, model_qwen, tokenizer_qwen, test_prompts):
    """
    使用测试提示对两个模型进行评估和比较
    
    Args:
        model_llama: Llama2模型
        tokenizer_llama: Llama2分词器
        model_qwen: Qwen2模型
        tokenizer_qwen: Qwen2分词器
        test_prompts: 测试提示列表
    
    Returns:
        包含两个模型测试结果的字典
    """
    print("\n=== 开始对两个模型进行测试提示评估 ===")
    
    results = {
        "llama2": [],
        "qwen2": []
    }
    
    # 测试 Llama2 模型
    print("\n--- 测试 Llama2 模型 ---")
    try:
        llama_results = evaluate_model(
            model_llama, 
            tokenizer_llama, 
            test_prompts, 
            max_new_tokens=100,  # 增加生成长度以获得更完整的回答
            device=MODEL_DEVICE_A
        )
        results["llama2"] = llama_results
        
        print("Llama2 模型测试结果:")
        for i, result in enumerate(llama_results):
            print(f"  提示 {i+1}: {result['prompt']}")
            print(f"  生成: {result['generated']}")
            print()
            
    except Exception as e:
        print(f"Llama2 模型测试失败: {e}")
        results["llama2"] = []
    
    # 测试 Qwen2 模型
    print("\n--- 测试 Qwen2 模型 ---")
    try:
        qwen_results = evaluate_model(
            model_qwen, 
            tokenizer_qwen, 
            test_prompts, 
            max_new_tokens=100,
            device=MODEL_DEVICE_B
        )
        results["qwen2"] = qwen_results
        
        print("Qwen2 模型测试结果:")
        for i, result in enumerate(qwen_results):
            print(f"  提示 {i+1}: {result['prompt']}")
            print(f"  生成: {result['generated']}")
            print()
            
    except Exception as e:
        print(f"Qwen2 模型测试失败: {e}")
        results["qwen2"] = []
    
    return results

def save_test_results(test_results, stacked_results, output_dir="./test_results"):
    """
    保存测试结果到文件
    
    Args:
        test_results: 原始模型测试结果
        stacked_results: 堆叠模型测试结果
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细的测试结果
    import json
    
    # 创建完整的测试报告
    full_report = {
        "original_models": test_results,
        "stacked_model": stacked_results,
        "test_prompts": test_prompts,
        "timestamp": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    }
    
    # 保存 JSON 格式的结果
    with open(os.path.join(output_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    
    # 保存人类可读的文本格式
    with open(os.path.join(output_dir, "test_results.txt"), "w", encoding="utf-8") as f:
        f.write("=== 模型测试结果对比报告 ===\n\n")
        
        for i, prompt in enumerate(test_prompts):
            f.write(f"测试提示 {i+1}: {prompt}\n")
            f.write("=" * 50 + "\n")
            
            # Llama2 结果
            if i < len(test_results.get("llama2", [])):
                f.write(f"Llama2 生成:\n{test_results['llama2'][i]['generated']}\n\n")
            else:
                f.write("Llama2 生成: [测试失败]\n\n")
            
            # Qwen2 结果
            if i < len(test_results.get("qwen2", [])):
                f.write(f"Qwen2 生成:\n{test_results['qwen2'][i]['generated']}\n\n")
            else:
                f.write("Qwen2 生成: [测试失败]\n\n")
            
            # 堆叠模型结果
            if i < len(stacked_results):
                f.write(f"堆叠Qwen2 生成:\n{stacked_results[i]['generated']}\n\n")
            else:
                f.write("堆叠Qwen2 生成: [测试失败]\n\n")
            
            f.write("-" * 80 + "\n\n")
    
    print(f"测试结果已保存到 {output_dir}")

if __name__ == "__main__":
    main()