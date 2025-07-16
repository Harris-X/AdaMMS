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

if __name__ == "__main__":
    outputs_dir =  "./downloaded_models/Llama-2-7b-hf" # "./downloaded_models/Llama-2-7b-hf" or "./downloaded_models/Qwen2-7B-Instruct"
    tokenizer_merged, model_merged = load_complete_model(outputs_dir, COMPUTE_DEVICE)
    prompt = "The capital of France is"
    inputs = tokenizer_merged(prompt, return_tensors="pt").to(COMPUTE_DEVICE)
    with torch.no_grad():
        outputs = model_merged.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer_merged.eos_token_id)
    print("输入:", prompt)
    print("合并后模型输出:", tokenizer_merged.decode(outputs[0], skip_special_tokens=True))