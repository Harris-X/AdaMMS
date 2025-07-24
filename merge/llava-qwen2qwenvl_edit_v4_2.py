# llava_merging_gradient_guided.py
# ------------------------------------------------------------------------------------------
# 本脚本实现了“梯度引导的自适应合并 (Gradient-Guided Adaptive Merging)”方法。
# 核心思想:
# 1. 不再使用激活值的相似度（CKA），而是使用模型在通用数据集上的“梯度”来判断功能上的一致性。
# 2. 两个模型在同一数据上的梯度方向越一致，说明它们的功能协同性越高。
# 3. 合并策略是一个由梯度余弦相似度 'c' 控制的动态加权平均：W* = α(c) * W_A + (1-α(c)) * W_B。
# 4. 当梯度冲突时 (c -> -1)，α(c) -> 1，自动保护基础模型 W_A。
#    当梯度协同时 (c -> 1)，α(c) -> 0.5，执行近似等权平均。
# ------------------------------------------------------------------------------------------

import os
import sys
import json
import torch
import safetensors.torch
import argparse
from tqdm import tqdm
import gc
import shutil
# 从 transformers 导入正确的模型类
from transformers import AutoTokenizer, AutoModelForVision2Seq
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

try:
    from datasets import load_dataset
except ImportError:
    print("="*80, file=sys.stderr)
    print("错误：无法导入 'datasets' 库。请运行 `pip install datasets`。", file=sys.stderr)
    print("这个库是获取探针数据集所必需的。", file=sys.stderr)
    print("="*80, file=sys.stderr)
    sys.exit(1)

# --- 模型与路径配置 (请在使用前修改) ---
CKPT_PATH = {
    "original_model": "/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-7B-Instruct",
    "qwen2_vl": "/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct",
    "llava-onevision-qwen": "/home/user/xieqiuhao/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si-hf"
}
INDEX_FILENAME = {
    "original_model": "model.safetensors.index.json",
    "qwen2_vl": "model.safetensors.index.json",
    "llava-onevision-qwen": "model.safetensors.index.json"
}

# --- 权重加载与辅助函数 (遵循您的模板) ---
def load_weights(base_path, index_filename):
    weights = {}
    index_path = os.path.join(base_path, index_filename)
    if not os.path.exists(index_path):
        single_file_path = os.path.join(base_path, "model.safetensors")
        if os.path.exists(single_file_path):
            print(f"Loading single weight file: {single_file_path}")
            return safetensors.torch.load_file(single_file_path)
        else:
            raise FileNotFoundError(f"Neither {index_filename} nor model.safetensors found in {base_path}")
    with open(index_path, 'r') as f:
        index = json.load(f)
    file_list = sorted(list(set(index["weight_map"].values())))
    for file in tqdm(file_list, desc=f"Loading weights from {os.path.basename(base_path)}"):
        weights.update(safetensors.torch.load_file(os.path.join(base_path, file)))
    return weights

def normalize_donor_keys(weights: dict) -> dict:
    """专门用于标准化 donor 模型（llava-onevision-qwen）的 key。"""
    prefix_to_remove = "language_model."
    normalized_weights = {}
    for key, value in weights.items():
        if key.startswith(prefix_to_remove):
            normalized_weights[key[len(prefix_to_remove):]] = value
        else:
            # 对于非语言模型部分（如 vision_tower），保留原样
            normalized_weights[key] = value
    return normalized_weights

# --- Helper Functions ---
def need_merge(name:str) -> bool:
    # 保持与原脚本一致的合并层选择逻辑
    if name in ['model.norm.weight']:
        return False
    if name in ['lm_head.weight', 'model.embed_tokens.weight']:
        return False
    if name.startswith("model.layers."):
        if name.endswith(".self_attn.rotary_emb.inv_freq"):
            return False
        # 在这个方法中，我们合并所有可合并的线性层权重
        if name.endswith((".weight")):
             return True
    return False

def create_soft_link(source_path, link_path):
    if not os.path.exists(source_path):
        print(f"Error: Source path '{source_path}' does not exist.")
        return
    if not os.path.exists(link_path):
        os.makedirs(link_path)
    for item in os.listdir(source_path):
        source_item = os.path.join(source_path, item)
        link_item = os.path.join(link_path, item)
        if item.endswith('.bin'):
            continue
        if os.path.isfile(source_item):
            try:
                if os.path.exists(link_item):
                    os.remove(link_item)
                os.symlink(source_item, link_item)
            except OSError as e:
                print(f"Error creating soft link for '{item}': {e}")
        elif os.path.isdir(source_item):
            continue

# --- 梯度计算与合并逻辑 ---
def cosine_similarity(t1, t2):
    t1_flat = t1.flatten()
    t2_flat = t2.flatten()
    dot_product = torch.sum(t1_flat * t2_flat)
    norm_t1 = torch.linalg.norm(t1_flat)
    norm_t2 = torch.linalg.norm(t2_flat)
    
    if norm_t1 == 0 or norm_t2 == 0:
        return 0.0
    
    return (dot_product / (norm_t1 * norm_t2)).item()

@torch.enable_grad() # 确保开启梯度计算
def get_gradients(model, tokenizer, layer_names, probe_dataloader, model_name, device):
    model.train() # 设置为训练模式以计算梯度
    
    # 初始化一个字典来累积梯度
    accumulated_gradients = {name: torch.zeros_like(p.data) for name, p in model.named_parameters() if name in layer_names}
    
    total_batches = 0
    for batch in tqdm(probe_dataloader, desc=f"Computing gradients for {model_name}"):
        model.zero_grad()
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        
        # 计算损失，将 input_ids 作为 labels 是 Causal LM 的标准做法
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        # 如果损失有效，则进行反向传播
        if torch.isfinite(loss):
            loss.backward()
            total_batches += 1
            # 累积梯度
            for name, param in model.named_parameters():
                if name in layer_names and param.grad is not None:
                    accumulated_gradients[name] += param.grad.detach().cpu()
        
        # 清理显存
        del input_ids, attention_mask, outputs, loss
        gc.collect()
        torch.cuda.empty_cache()

    # 对累积的梯度求平均
    if total_batches > 0:
        for name in accumulated_gradients:
            accumulated_gradients[name] /= total_batches
            
    model.eval() # 恢复评估模式
    return accumulated_gradients

# --- 主转换函数 ---
def convert(args, device):
    output_dir = "merged_models"
    model_name = f"gradient-guided-merge-{args.mode}"
    OUTPUT_PATH = os.path.join(output_dir, model_name)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # --- 模型权重加载 (保持不变) ---
    print("Loading all model weights from disk...")
    base_weights = load_weights(args.base_model_path, INDEX_FILENAME["qwen2_vl"])
    donor_weights_raw = load_weights(args.donor_model_path, INDEX_FILENAME["llava-onevision-qwen"])
    print("Normalizing donor model keys...")
    donor_weights = normalize_donor_keys(donor_weights_raw)
    
    merged_weights = {}

    # --- 梯度引导的自适应合并逻辑 ---
    print("="*80); print("Applying 'Gradient-Guided Adaptive Merging' strategy."); print("="*80)
    
    # 1. 准备通用探针数据集
    print(f"Preparing general probe dataset from '{args.probe_dataset}'...")
    probe_dataset_raw = load_dataset(args.probe_dataset, "20220301.en" if "wikipedia" in args.probe_dataset else "en", split="train", streaming=True).take(args.probe_samples)
    probe_texts = [item['text'] for item in probe_dataset_raw if item['text']]

    # 2. 识别需要计算梯度的目标层 (共享的语言模型层)
    # 我们从 base_weights 中识别出所有需要合并的层
    target_layer_keys = {k for k in base_weights.keys() if need_merge(k)}
    print(f"Found {len(target_layer_keys)} common layers to be merged.")

    # 3. 加载模型A并计算其梯度
    print("Loading Base Model (A) to GPU for gradient probing...")
    model_a = AutoModelForVision2Seq.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer_a = AutoTokenizer.from_pretrained(args.base_model_path)
    
    probe_inputs_a = tokenizer_a(probe_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    probe_dataset_a = TensorDataset(probe_inputs_a['input_ids'], probe_inputs_a['attention_mask'])
    probe_dataloader_a = DataLoader(probe_dataset_a, batch_size=args.probe_batch_size)
    
    gradients_a = get_gradients(model_a, tokenizer_a, target_layer_keys, probe_dataloader_a, "Base Model (A)", device)
    del model_a, tokenizer_a, probe_inputs_a, probe_dataset_a, probe_dataloader_a; gc.collect(); torch.cuda.empty_cache()

    # 4. 加载模型B并计算其梯度
    print("\nLoading Donor Model (B) to GPU for gradient probing...")
    model_b = AutoModelForVision2Seq.from_pretrained(args.donor_model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer_b = AutoTokenizer.from_pretrained(args.donor_model_path)

    # 在模型B中，我们需要找到与 target_layer_keys 对应的键
    # 对于本项目，键名 'language_model.' + key 应该存在于 model_b.named_parameters() 中
    target_layer_keys_b = {"language_model." + k for k in target_layer_keys}

    probe_inputs_b = tokenizer_b(probe_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    probe_dataset_b = TensorDataset(probe_inputs_b['input_ids'], probe_inputs_b['attention_mask'])
    probe_dataloader_b = DataLoader(probe_dataset_b, batch_size=args.probe_batch_size)
    
    gradients_b_raw = get_gradients(model_b, tokenizer_b, target_layer_keys_b, probe_dataloader_b, "Donor Model (B)", device)
    # 将B的梯度键名标准化，以便与A匹配
    gradients_b = {k.replace("language_model.", ""): v for k, v in gradients_b_raw.items()}

    del model_b, tokenizer_b, probe_inputs_b, probe_dataset_b, probe_dataloader_b; gc.collect(); torch.cuda.empty_cache()

    # 5. 计算梯度相似度
    gradient_similarity_scores = {}
    for key in tqdm(target_layer_keys, desc="Calculating Gradient Cosine Similarity"):
        if key in gradients_a and key in gradients_b:
            grad_a = gradients_a[key]
            grad_b = gradients_b[key]
            if grad_a.shape == grad_b.shape:
                similarity = cosine_similarity(grad_a, grad_b)
                gradient_similarity_scores[key] = similarity
            else:
                print(f"Warning: Shape mismatch for gradient '{key}', skipping.")
    
    del gradients_a, gradients_b; gc.collect()

    if not gradient_similarity_scores:
        print("错误：未能计算任何层的梯度相似度，无法继续合并。", file=sys.stderr)
        sys.exit(1)

    # 6. 逐层自适应合并
    for key in tqdm(base_weights.keys(), desc="Applying Gradient-Guided Merging"):
        # 检查是否是需要合并的层
        if key in gradient_similarity_scores and key in donor_weights and base_weights[key].shape == donor_weights[key].shape:
            
            w_a = base_weights[key].float().to(device)
            w_b = donor_weights[key].float().to(device)
            
            # 获取该层的梯度相似度，如果找不到则默认为0（中性）
            similarity = gradient_similarity_scores.get(key, 0.0)
            
            # 计算合并权重 alpha = f(similarity)
            # 我们的目标函数 alpha(c) 应该在 c=-1 (冲突)时为1，在 c=1 (协同)时为0.5
            k = args.gradient_steepness
            c0 = args.gradient_center_point
            # 使用一个平滑的 sigmoid 函数实现 1 -> 0.5 的映射
            # alpha = 0.5 + 0.5 * (1 / (1 + exp(k*(c - c0))))
            alpha_val = 0.5 + 0.5 / (1 + torch.exp(torch.tensor(k * (similarity - c0))))
            alpha = alpha_val.to(device)
            
            # 应用加权平均
            w_star = alpha * w_a + (1.0 - alpha) * w_b
            
            merged_weights[key] = w_star.to(base_weights[key].dtype).cpu()
            gc.collect(); torch.cuda.empty_cache()
        else:
            # 对于不合并或在其他模型中不存在的层，直接使用基础模型的权重
            if key in base_weights:
                merged_weights[key] = base_weights[key]
            elif key in donor_weights: # 处理vision tower等基础模型没有的权重
                 merged_weights[key] = donor_weights[key]


    # --- 保存模型 (保持不变) ---
    print("\nSaving merged model...")
    index_path = os.path.join(args.base_model_path, INDEX_FILENAME["qwen2_vl"])
    with open(index_path, "r") as f: index_map = json.load(f)["weight_map"]
    
    # 将 vision tower 等非共享权重也加入 index_map
    donor_index_path = os.path.join(args.donor_model_path, INDEX_FILENAME["llava-onevision-qwen"])
    with open(donor_index_path, "r") as f: donor_index_map = json.load(f)["weight_map"]
    for k, v in donor_index_map.items():
        if k not in index_map:
            index_map[k] = v

    sharded_weights = {filename: {} for filename in set(index_map.values())}
    for key, value in merged_weights.items():
        if key in index_map: 
            sharded_weights[index_map[key]][key] = value
    
    for filename, weights_dict in sharded_weights.items():
        safetensors.torch.save_file(weights_dict, os.path.join(OUTPUT_PATH, filename))
    
    create_soft_link(source_path=args.base_model_path, link_path=OUTPUT_PATH)
    shutil.copy(os.path.join(args.donor_model_path, "model.safetensors.index.json"), os.path.join(OUTPUT_PATH, "model.safetensors.index.json"))
    print(f"Merged model saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptively merge models based on gradient cosine similarity.")
    
    # 设备参数
    parser.add_argument('--cuda_device', type=int, default=0, help="CUDA device to use (e.g., 0, 1, 2).")
    
    # 模型路径
    parser.add_argument('--base_model_path', type=str, default=CKPT_PATH["qwen2_vl"])
    parser.add_argument('--donor_model_path', type=str, default=CKPT_PATH["llava-onevision-qwen"])
    # 原始模型在此方法中不是必需的，但为保持框架完整性而保留
    parser.add_argument('--original_model_path', type=str, default=CKPT_PATH["original_model"])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--mode', type=str, default="default", help="A name for this merging configuration.")

    # 探针数据集配置
    parser.add_argument('--probe_dataset', type=str, default="wikipedia", help="General dataset for probing gradients ('wikipedia' or 'c4').")
    parser.add_argument('--probe_samples', type=int, default=128, help="Number of samples for probing.")
    parser.add_argument('--probe_batch_size', type=int, default=1, help="Batch size for probing. Reduce if OOM during gradient computation.")

    # 新增：梯度引导合并策略的超参数
    parser.add_argument('--gradient_steepness', type=float, default=10.0, help="Steepness 'k' of the sigmoid function for alpha calculation. Higher values mean a sharper transition.")
    parser.add_argument('--gradient_center_point', type=float, default=0.0, help="Center point 'c0' of the sigmoid function. Usually 0.")
    
    args = parser.parse_args()
    
    if torch.cuda.is_available() and args.cuda_device < torch.cuda.device_count():
        device = f"cuda:{args.cuda_device}"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    CKPT_PATH.update({
        "qwen2_vl": args.base_model_path,
        "llava-onevision-qwen": args.donor_model_path,
        "original_model": args.original_model_path
    })

    print("--- Configuration ---")
    # 打印相关参数
    print(f"  Mode: {args.mode}")
    print(f"  Base Model (A): {args.base_model_path}")
    print(f"  Donor Model (B): {args.donor_model_path}")
    print(f"  Probe Dataset: {args.probe_dataset} ({args.probe_samples} samples)")
    print(f"  Probe Batch Size: {args.probe_batch_size}")
    print(f"  Gradient Steepness (k): {args.gradient_steepness}")
    print(f"  Gradient Center (c0): {args.gradient_center_point}")
    print("--------------------")

    convert(args, device)