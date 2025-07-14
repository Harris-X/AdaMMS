################################################################################
#
#       完整、独立的异构大语言模型合并脚本 (LLaMA2-7B & Qwen2-7B)
#       基于 "Training-free Heterogeneous Model Merging" 论文思想实现
#
################################################################################

import os
import gc
import torch
import torch.nn as nn
import random
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from inspect import getmembers, isfunction
from types import MethodType

# --- 关键依赖 ---
# 确保这些文件位于正确的目录结构中
try:
    from graphs.transformer_enc_graph import TransformerEncoderGraph, bert
    from model_merger import ModelMerge
    from matching_functions import match_tensors_zipit
    from metric_calculators import CovarianceMetric
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确认您的文件结构是否正确，并已将依赖的 .py 文件放置在相应位置。")
    exit()

# --- 1. 核心配置 ---
CKPT_PATH = {
    "llama2": "./downloaded_models/Llama-2-7b-hf",
    "qwen2": "./downloaded_models/Qwen2-7B-Instruct",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {DEVICE}")


# --- 2. 辅助函数 ---

def set_seed(seed: int):
    """为CPU和GPU设置随机种子以确保可复现性。"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_model_and_tokenizer(model_id: str):
    """从Hugging Face Hub加载模型和分词器。"""
    print(f"正在加载模型: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"模型 '{model_id.split('/')[-1]}' 加载完成。")
    return tokenizer, model

def create_transformer_graph(model: nn.Module, model_name: str):
    """为给定的Transformer模型创建一个图谱实例。"""
    print(f"正在为 {model_name} 创建模型图谱...")
    if "llama" in model_name.lower() or "qwen" in model_name.lower():
        module_map = {
            'emb': 'model.embed_tokens',
            'emb_ln': 'model.norm',
            'q': 'self_attn.q_proj',
            'k': 'self_attn.k_proj',
            'v': 'self_attn.v_proj',
            'lin_attn': 'self_attn.o_proj',
            'attn_ln': 'post_attention_layernorm',
            'fc1': 'mlp.gate_proj',
            'fc2': 'mlp.down_proj',
            'final_ln': 'input_layernorm',
            'classifier': 'lm_head',
        }
    else:
        raise ValueError(f"未知的模型: {model_name}。请在 `create_transformer_graph` 中为其添加模块映射。")

    return bert(
        model,
        merge_type='all',
        qk=True,
        classifier=True
    ).graphify()

class PairedModelDataset(Dataset):
    """包装器，为两个异构模型提供成对的、已分词的输入。"""
    def __init__(self, dataset, tokenizer_a, tokenizer_b, max_length=256):
        self.dataset = dataset.filter(lambda ex: ex.get('text') and ex['text'].strip())
        self.tokenizer_a = tokenizer_a
        self.tokenizer_b = tokenizer_b
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        inputs_a = self.tokenizer_a(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        inputs_b = self.tokenizer_b(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids_a": inputs_a.input_ids.squeeze(0), "attention_mask_a": inputs_a.attention_mask.squeeze(0),
            "input_ids_b": inputs_b.input_ids.squeeze(0), "attention_mask_b": inputs_b.attention_mask.squeeze(0),
        }

def prepare_dataloader(tokenizer_a, tokenizer_b, dataset_name="wikitext", max_samples=64, batch_size=2, max_length=256):
    """准备一个能同时为两个模型提供输入的 DataLoader。"""
    print(f"正在加载并准备数据集: {dataset_name}...")
    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split="test").select(range(max_samples))
    
    paired_dataset = PairedModelDataset(dataset, tokenizer_a, tokenizer_b, max_length)

    def collate_fn(batch):
        keys = batch[0].keys()
        return {key: torch.stack([item[key] for item in batch]) for key in keys}

    return DataLoader(paired_dataset, batch_size=batch_size, collate_fn=collate_fn)


def patch_merger_logic(merger_instance: ModelMerge):
    """动态重写 ModelMerge 实例的方法以适应异构输入。"""
    def custom_compute_intermediates(self, batch_data):
        if not batch_data: return [{}, {}]
        inputs_a = {"input_ids": batch_data["input_ids_a"], "attention_mask": batch_data["attention_mask_a"]}
        inputs_b = {"input_ids": batch_data["input_ids_b"], "attention_mask": batch_data["attention_mask_b"]}
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            self.graphs[0].intermediates.clear()
            self.graphs[0].model(**inputs_a)
            self.graphs[1].intermediates.clear()
            self.graphs[1].model(**inputs_b)
            return [self.graphs[0].intermediates, self.graphs[1].intermediates]
            
    merger_instance.compute_intermediates = MethodType(custom_compute_intermediates, merger_instance)

    def custom_compute_metrics(self, dataloader, metric_classes):
        self.metrics, self.is_activated, self.model_dims = None, None, None
        numel = 0
        nodes = None
        for batch_data in tqdm(dataloader, desc="计算合并指标"):
            if not batch_data: continue
            numel += batch_data["input_ids_a"].shape[0]
            intermediates = self.compute_intermediates(batch_data)
            
            if nodes is None:
                nodes = list(set(item for i in intermediates for item in i))
            if self.is_activated is None:
                self.is_activated = {n: [1 if n in i else 0 for i in intermediates] for n in nodes}
            if self.model_dims is None:
                self.model_dims = {n: [i[n].shape[0] for i in intermediates if n in i] for n in nodes}
            if self.metrics is None:
                self.metrics = {n: {k: v() for k, v in metric_classes.items()} for n in nodes}
            
            for node in self.metrics:
                if node in intermediates[0] and node in intermediates[1]:
                    for metric in self.metrics[node].values():
                        intermeds_float = [i[node].float() for i in intermediates]
                        metric.update(batch_data["input_ids_a"].shape[0], *intermeds_float)
        
        for node in self.metrics:
            for metric_name, metric in self.metrics[node].items():
                self.metrics[node][metric_name] = metric.finalize(numel)
        return self.metrics

    merger_instance.compute_metrics = MethodType(custom_compute_metrics, merger_instance)
    return merger_instance


# --- 3. 主执行流程 ---

def main():
    """主执行函数"""
    set_seed(42)

    # --- 加载模型 ---
    tokenizer_llama, model_llama = load_model_and_tokenizer(CKPT_PATH["llama2"])
    tokenizer_qwen, model_qwen = load_model_and_tokenizer(CKPT_PATH["qwen2"])

    # --- 准备数据 ---
    # **注意**: 减小 batch_size 和 max_samples 可以显著降低内存消耗
    dataloader = prepare_dataloader(
        tokenizer_llama, tokenizer_qwen, 
        batch_size=2,          # 内存不足时请减小
        max_samples=16,        # 内存不足时请减小
        max_length=128
    )

    # --- 开始合并 ---
    print("\n--- 开始异构模型合并流程 ---")

    # 1. 创建图谱
    model_a = deepcopy(model_llama)
    model_b = deepcopy(model_qwen)
    graph_a = create_transformer_graph(model_a, "llama2")
    graph_b = create_transformer_graph(model_b, "qwen2")
    
    # 2. 初始化并修改合并器
    merger = ModelMerge(graph_a, graph_b, device=DEVICE)
    merger = patch_merger_logic(merger)

    # 3. 执行核心的变换计算
    # fix_rate=0.5 表示合并后的神经元数量约为两者维度之和的一半
    merger.transform(
        model=deepcopy(model_a),
        dataloader=dataloader,
        transform_fn=match_tensors_zipit,
        metric_classes=(CovarianceMetric,),
        fix_rate=0.5  
    )
    
    # 4. 获取并加载合并后的权重
    print("正在平均化已对齐的权重并创建最终模型...")
    merged_state_dict = merger.get_merged_state_dict()
    
    # 我们选择LLaMA2作为合并后模型的基础架构
    merged_model = deepcopy(model_llama)
    merged_model.load_state_dict(merged_state_dict, strict=False)
    
    print("\n--- 模型合并完成 ---")

    # --- 5. 测试与验证 ---
    prompt = "The future of artificial intelligence is"
    print(f"\n用合并后的模型生成文本，输入: '{prompt}'")
    
    inputs = tokenizer_llama(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = merged_model.generate(**inputs, max_new_tokens=60, do_sample=True, top_k=20, top_p=0.95)
    
    generated_text = tokenizer_llama.decode(outputs[0], skip_special_tokens=True)
    
    print("\n" + "="*20 + " 合并后模型生成结果 " + "="*20)
    print(generated_text)
    print("="*60)

    # --- 清理内存 ---
    del model_llama, model_qwen, model_a, model_b, merged_model, merger, dataloader
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()