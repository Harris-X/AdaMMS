# 7B version with Functional Alignment (CKA) and Direct Merging
import os
import sys
import json
import torch
import safetensors.torch
import argparse
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
import numpy as np

# --- NEW IMPORTS for Functional Alignment ---
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor,AutoModelForVision2Seq
from datasets import load_dataset
from load_models import *

# --- Model Paths and Configs ---
# Updated to focus on the two chosen heterogeneous models
CKPT_PATH = {
    "llava": "./downloaded_models/llava-v1.5-7b", # image-text-to-text
    "qwen2": "./downloaded_models/Qwen2-7B-Instruct", # text-generation
    "llama2": "./downloaded_models/Llama-2-7b-hf", # text-generation
}

# 首先，对俩个模型进行加载，确定base模型
# 假设是往QWEN2:28层 , llava:32层
# 此处以QWEN2:28层为BASE.

# 确定运行设备 (如果可用，则使用 GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")


## 考虑加载数据集以得到网络特征图以计算cka

# 加载数据集
# Prepare probe dataset
def load_probe_dataset():
    print("Loading probe dataset...")
    # Use a small but diverse subset for efficiency
    probe_dataset = load_dataset("wikipedia", "20220301.en", split='train[:50]', trust_remote_code=True)
    probe_texts = [text for text in probe_dataset['text'] if len(text) > 200]
    return probe_texts

# 按模块名注册钩子，特别关注关键Transformer组件
def register_hooks_for_model(model):
    layer_reps = {}  # 使用普通字典而不是defaultdict
    hooks = []
    
    # 关键组件列表 - 这些是我们特别关注的模块
    key_components = [
        "input_layernorm",
        "post_attention_layernorm",
        "self_attn",
        "mlp"
    ]
    
    # 关键子组件列表 - 这些是我们想要深入捕获的子模块
    key_subcomponents = {
        "self_attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"]
    }
    
    def get_hook_fn(module_name):
        def hook_fn(module, input, output):
            # 保留完整输出
            if isinstance(output, tuple):
                processed_output = list()
                for tensor in output:
                    if isinstance(tensor, torch.Tensor):
                        processed_output.append(tensor.detach().cpu())
                    else:
                        processed_output.append(tensor)
                if module_name not in layer_reps:
                    layer_reps[module_name] = []
                layer_reps[module_name].append(processed_output)
            else:
                if isinstance(output, torch.Tensor):
                    if module_name not in layer_reps:
                        layer_reps[module_name] = []
                    layer_reps[module_name].append(output.detach().cpu())
                else:
                    if module_name not in layer_reps:
                        layer_reps[module_name] = []
                    layer_reps[module_name].append(output)
        return hook_fn
    
    # 递归注册所有模块的钩子，但只关注关键组件
    def register_hooks_recursive(module, name_prefix=""):
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            
            # 检查是否是层级结构 (model.layers.X)
            is_layer = "layers" in full_name and any(str(i) in full_name.split(".")[-1] for i in range(32))
            
            # 如果是层，检查它的子组件是否是我们关注的关键组件
            if is_layer:
                for component_name, component in child.named_children():
                    # 检查是否是主要组件
                    if component_name in key_components:
                        component_full_name = f"{full_name}.{component_name}"
                        print(f"注册钩子: {component_full_name}")
                        hook = component.register_forward_hook(get_hook_fn(component_full_name))
                        hooks.append(hook)
                        
                        # 深入注册子组件的钩子
                        if component_name in key_subcomponents:
                            for subcomp_name, subcomp in component.named_children():
                                if subcomp_name in key_subcomponents[component_name]:
                                    subcomp_full_name = f"{component_full_name}.{subcomp_name}"
                                    print(f"注册子组件钩子: {subcomp_full_name}")
                                    hook = subcomp.register_forward_hook(get_hook_fn(subcomp_full_name))
                                    hooks.append(hook)
            
            # 继续递归，寻找更深层的组件
            register_hooks_recursive(child, full_name)
    
    # 从模型的根开始递归注册
    register_hooks_recursive(model)
    
    return layer_reps, hooks

def test():
    ## 加载qwen2模型参数
    model_id = CKPT_PATH["qwen2"]
    tokenizer, model = load_complete_qwen2(model_id)
    
    # 检查模型结构
    print(f"模型类型: {type(model)}")
    
    print("识别模型层级结构...")
    # 探索模型结构以确定层的路径
    def explore_first_level(model):
        for name, child in model.named_children():
            print(f"一级组件: {name} ({type(child).__name__})")
            if "layer" in name:
                for subname, subchild in child.named_children():
                    if subname == "0":  # 查看第一层的结构
                        print(f"  第一层组件: {subname}")
                        for compname, _ in subchild.named_children():
                            print(f"    组件: {compname}")
                        break
    
    explore_first_level(model)
    
    # 注册钩子 (整个模型，但只关注关键组件)
    layer_reps, hooks = register_hooks_for_model(model)
    
    # 准备输入数据并执行推理以触发钩子
    print("准备输入数据并执行推理...")
    input_text = "Hello, this is a test input for Qwen2 model."
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 检查层表示
    print(f"收集了 {len(layer_reps)} 个不同模块的表示")
    
    # 打印前几个模块的信息 (按层索引排序)
    sorted_keys = sorted(layer_reps.keys(), 
                          key=lambda x: (int(x.split('.')[2]) if x.split('.')[2].isdigit() else 999, x))
    
    for key in sorted_keys[:10]:  # 只显示前10个模块
        reps = layer_reps[key]
        print(f"模块 '{key}': {len(reps)} 个表示")
        
        # 显示第一个表示的详细信息
        rep = reps[0]
        if isinstance(rep, list):
            print(f"  - 包含 {len(rep)} 个元素的列表")
            for j, tensor in enumerate(rep[:3]):  # 只显示前3个元素
                if isinstance(tensor, torch.Tensor):
                    print(f"    元素 {j} 形状: {tensor.shape}")
                else:
                    print(f"    元素 {j} 类型: {type(tensor)}")
        elif isinstance(rep, torch.Tensor):
            print(f"  - 张量形状: {rep.shape}")
        else:
            print(f"  - 类型: {type(rep)}")

    # 清理钩子，防止内存泄漏
    for hook in hooks:
        hook.remove()


if __name__ == "__main__":
    test()








