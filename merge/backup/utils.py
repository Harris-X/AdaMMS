import os
import torch
import torch.nn as nn
import random
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from inspect import getmembers, isfunction # <-- **ADDED THIS IMPORT**
from transformers import AutoModelForCausalLM
# 从项目中的其他文件中导入必要的类和函数
# 确保这些文件与您的主脚本位于正确的目录结构中
from graphs.transformer_enc_graph import TransformerEncoderGraph

# --- 1. 通用辅助函数 ---

def set_seed(seed: int):
    """为CPU和GPU设置随机种子以确保可复现性。"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保卷积操作的可复现性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    """获取可用的计算设备 (CUDA或CPU)。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 2. 模型加载与图谱构建 ---

def load_model_and_tokenizer(model_id: str, device: torch.device):
    """
    通用函数，用于从Hugging Face Hub加载模型和分词器。
    使用 bfloat16 以优化内存使用。
    """
    print(f"正在加载模型: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(device).eval()  # 加载后立即设为评估模式

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # 如果没有 pad token，则将其设置为 eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"模型 '{model_id.split('/')[-1]}' 加载完成。")
    return tokenizer, model

def create_transformer_graph(model: nn.Module, model_name: str):
    """
    为给定的Transformer模型创建一个图谱实例。
    这个函数定义了模型中关键模块的名称映射，以便图谱能够正确地挂载钩子。
    """
    print(f"正在为 {model_name} 创建模型图谱...")

    # 为不同模型架构定义模块名称映射
    # 这是连接模型具体实现和图谱抽象逻辑的关键
    if "llama" in model_name.lower():
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
    elif "qwen" in model_name.lower():
        module_map = {
            # Qwen2的模块命名与LLaMA类似，可以直接复用
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
        raise ValueError(f"未知的模型名称: {model_name}。请在 `create_transformer_graph` 中添加其模块映射。")

    # `bert` 函数在此作为一个通用的 Transformer 图谱构建器模板
    # 我们传入模型实例、模块映射以及其他配置参数
    from graphs.transformer_enc_graph import bert
    return bert(
        model,
        merge_type='all',  # 'all' 表示在注意力和前馈网络的所有关键点都进行对齐
        qk=True,           # 假设Q和K的相似度是一起计算的
        classifier=True    # 表示模型包含一个分类头（lm_head）
    ).graphify()


# --- 3. 数据集准备 ---

class PairedDataset(Dataset):
    """
    一个包装器，用于将两个不同分词器处理过的数据集成对。
    这对于需要同时向两个异构模型输入数据的场景至关重要。
    """
    def __init__(self, dataset, tokenizer_a, tokenizer_b, max_length=128):
        self.dataset = dataset
        self.tokenizer_a = tokenizer_a
        self.tokenizer_b = tokenizer_b
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        if not text or not text.strip():
            # 如果文本为空，则返回一个空字典或跳过
            return None

        # 分别为两个模型进行分词和处理
        inputs_a = self.tokenizer_a(
            text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        inputs_b = self.tokenizer_b(
            text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        # `squeeze()` 用于移除批次维度，因为 DataLoader 会自动添加
        return {
            "input_ids_a": inputs_a.input_ids.squeeze(),
            "attention_mask_a": inputs_a.attention_mask.squeeze(),
            "input_ids_b": inputs_b.input_ids.squeeze(),
            "attention_mask_b": inputs_b.attention_mask.squeeze(),
        }

def prepare_paired_dataloader(
    tokenizer_a, tokenizer_b, dataset_name="wikitext", split="test",
    max_samples=64, batch_size=4, max_length=128
):
    """
    加载原始文本数据集，并创建一个能为两个异构模型提供成对输入的 DataLoader。
    """
    print(f"正在加载数据集: {dataset_name}...")
    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)
    
    # 截取子集以便快速实验
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    # 过滤掉空文本行
    dataset = dataset.filter(lambda example: example.get('text') and example['text'].strip())

    paired_dataset = PairedDataset(dataset, tokenizer_a, tokenizer_b, max_length=max_length)

    # 自定义 collate_fn，以正确地将字典列表打包成批次
    def collate_fn(batch):
        # 过滤掉getitem返回的None值
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
            
        keys = batch[0].keys()
        return {key: torch.stack([item[key] for item in batch]) for key in keys}

    return DataLoader(paired_dataset, batch_size=batch_size, collate_fn=collate_fn)


# --- 4. 合并功能相关函数 ---

def get_merging_fn(name):
    """
    **ADDED THIS FUNCTION BACK**
    根据名称从 `matching_functions` 模块中动态获取对齐函数。
    """
    import matching_functions
    # 使用 inspect.getmembers 查找所有名称中包含 'match_tensors' 的函数
    matching_fns = dict([(k, v) for (k, v) in getmembers(matching_functions, isfunction) if 'match_tensors' in k])
    if name not in matching_fns:
        raise ValueError(f"Merging function '{name}' not found in matching_functions.py")
    return matching_fns[name]

def add_custom_merger_logic(merger_instance):
    """
    动态地向 ModelMerge 实例添加或重写方法，以处理异构输入。
    """

    # 1. 重写 `compute_intermediates` 方法
    def custom_compute_intermediates(self, batch_data):
        if batch_data is None: return [{}, {}] 

        inputs_a = {"input_ids": batch_data["input_ids_a"], "attention_mask": batch_data["attention_mask_a"]}
        inputs_b = {"input_ids": batch_data["input_ids_b"], "attention_mask": batch_data["attention_mask_b"]}
        
        self.graphs[0].model.eval()
        self.graphs[1].model.eval()
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            self.graphs[0].intermediates.clear()
            self.graphs[0].model(**inputs_a)
            self.graphs[1].intermediates.clear()
            self.graphs[1].model(**inputs_b)
            return [self.graphs[0].intermediates, self.graphs[1].intermediates]
            
    # 2. 将自定义方法绑定到 merger 实例上
    from types import MethodType
    merger_instance.compute_intermediates = MethodType(custom_compute_intermediates, merger_instance)
    
    # 3. 重写 `compute_metrics` 方法
    def custom_compute_metrics(self, dataloader, metric_classes):
        self.metrics = None
        self.is_activated = None
        self.model_dims = None
        
        numel = 0
        nodes = None
        for batch_data, _ in tqdm(dataloader, desc="正在计算合并指标 (协方差)"):
            if batch_data is None: continue

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
            
            for node, node_metrics in self.metrics.items():
                for metric in node_metrics.values():
                    intermeds_float = [i[node].float() for i in intermediates if node in i]
                    metric.update(batch_data["input_ids_a"].shape[0], *intermeds_float)
    
        for node, node_metrics in self.metrics.items():
            for metric_name, metric in node_metrics.items():
                self.metrics[node][metric_name] = metric.finalize(numel)

        return self.metrics

    merger_instance.compute_metrics = MethodType(custom_compute_metrics, merger_instance)

    return merger_instance


def prepare_decoder_models(config, device):
    """ 加载 Llama2 和 Qwen2 模型 """
    bases = []
    for model_id in config['bases']:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(device)
        bases.append(model)
    
    # 创建一个新模型作为合并的目标，这里我们以第一个模型作为模板
    new_model = AutoModelForCausalLM.from_config(bases[0].config).to(device)
    
    return {'bases': bases, 'new': new_model}


def prepare_models(config, device='cuda'):
    """ Load all pretrained models in config. """
    if config['name'].startswith('decoder'): # 新增的处理分支
        return prepare_decoder_models(config, device)
    else:
        raise NotImplementedError(config['name'])

def get_config_from_name(name, device=None):
    """ Load config based on its name. """
    out = deepcopy(getattr(__import__('configs.' + name), name).config)
    if device is None and 'device' not in out:
        out['device'] = 'cuda'
    elif device is not None:
        out['device'] = device
    return out