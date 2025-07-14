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
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")  # 强制使用 CPU 进行调试
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
        # "input_layernorm",
        # "post_attention_layernorm",
        "self_attn",
        # "mlp"
    ]
    
    # 关键子组件列表 - 这些是我们想要深入捕获的子模块
    key_subcomponents = {
        "self_attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"]
    }

    key_subcomponents = None  # 如果不需要子组件，可以设置为None
    
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

                        if key_subcomponents != None:
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
    # model_id = CKPT_PATH["qwen2"]
    # tokenizer, model = load_complete_qwen2(model_id)

    model_id = CKPT_PATH["llama2"]
    tokenizer, model = load_complete_llama2(model_id)
    
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

# 将俩个模型注册得到的特征图进行CKA计算,以便后面对俩个模型进行对齐,找到对应层之间的关系
# 这些特征图由 register_hooks_for_model 函数收集
def compute_cka(layer_reps1, layer_reps2, device='cuda'):
    """
    计算两个模型的层级特征之间的CKA相似度
    
    参数:
    layer_reps1: 第一个模型的层级特征表示，由register_hooks_for_model收集
    layer_reps2: 第二个模型的层级特征表示，由register_hooks_for_model收集
    device: 计算设备
    
    返回:
    cka_matrix: 两个模型层之间的CKA相似度矩阵
    layer_names1: 第一个模型的层名列表
    layer_names2: 第二个模型的层名列表
    """
    def _HSIC(K, L, device='cuda'):
        """
        计算HSIC (Hilbert-Schmidt Independence Criterion)
        参考: https://arxiv.org/pdf/2010.15327.pdf 公式(3)
        """
        # 将输入张量转换为float32类型并移动到相同设备
        K = K.float().to(device)
        L = L.float().to(device)
        
        N = K.shape[0]
        ones = torch.ones(N, 1).to(device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()
    
    # 筛选有效的层表示
    valid_layers1 = {}
    valid_layers2 = {}
    
    # 对第一个模型的层表示进行处理
    for name, features in layer_reps1.items():
        if len(features) > 0 and isinstance(features[0], torch.Tensor):
            valid_layers1[name] = features[0]
        elif len(features) > 0 and isinstance(features[0], list):
            # 如果是列表，取第一个张量
            if len(features[0]) > 0 and isinstance(features[0][0], torch.Tensor):
                valid_layers1[name] = features[0][0]
    
    # 对第二个模型的层表示进行处理
    for name, features in layer_reps2.items():
        if len(features) > 0 and isinstance(features[0], torch.Tensor):
            valid_layers2[name] = features[0]
        elif len(features) > 0 and isinstance(features[0], list):
            # 如果是列表，取第一个张量
            if len(features[0]) > 0 and isinstance(features[0][0], torch.Tensor):
                valid_layers2[name] = features[0][0]
    
    layer_names1 = list(valid_layers1.keys())
    layer_names2 = list(valid_layers2.keys())
    
    N = len(layer_names1)
    M = len(layer_names2)
    
    # 创建CKA矩阵
    cka_matrix = torch.zeros(N, M)
    
    # 计算每对层之间的CKA相似度
    for i, (name1, feat1) in enumerate(valid_layers1.items()):
        # 将特征展平为二维矩阵并转换为float32
        X = feat1.flatten(1).float()
        # 计算核矩阵
        K = X @ X.t()
        K.fill_diagonal_(0.0)
        hsic_K_K = _HSIC(K, K, device)
        
        for j, (name2, feat2) in enumerate(valid_layers2.items()):
            # 将特征展平为二维矩阵并转换为float32
            Y = feat2.flatten(1).float()
            # 计算核矩阵
            L = Y @ Y.t()
            L.fill_diagonal_(0.0)
            
            # 确保核矩阵大小相同，如有必要进行调整
            if K.shape != L.shape:
                print(f"形状不匹配: {name1} ({K.shape}) vs {name2} ({L.shape})，跳过...")
                continue
            
            hsic_L_L = _HSIC(L, L, device)
            hsic_K_L = _HSIC(K, L, device)
            
            # 计算CKA
            cka = hsic_K_L / (torch.sqrt(torch.tensor(hsic_K_K * hsic_L_L)) + 1e-10)
            cka_matrix[i, j] = cka
    
    return cka_matrix, layer_names1, layer_names2

def plot_cka_heatmap(cka_matrix, layer_names1, layer_names2, title="CKA相似度", save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 设置英文标题，避免中文渲染问题
    english_title = title.replace("相似度", "Similarity").replace("模型", "Model").replace("层", "Layers")
    
    # 简化层名以便显示
    def simplify_name(name):
        parts = name.split('.')
        if len(parts) > 3:
            return f"{parts[0][:3]}..{parts[-3][:3]}.{parts[-2][:3]}.{parts[-1][:3]}"
        return name
    
    simple_names1 = [simplify_name(name) for name in layer_names1]
    simple_names2 = [simplify_name(name) for name in layer_names2]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cka_matrix.cpu().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel('Model 2 Layers')
    plt.ylabel('Model 1 Layers')
    plt.title(english_title)
    
    # 如果层数不多，显示所有层名
    if len(simple_names2) <= 20:
        plt.xticks(np.arange(len(simple_names2)), simple_names2, rotation=90)
    else:
        # 否则只显示部分层名
        stride = max(1, len(simple_names2) // 20)
        plt.xticks(np.arange(0, len(simple_names2), stride), 
                  [simple_names2[i] for i in range(0, len(simple_names2), stride)], 
                  rotation=90)
    
    if len(simple_names1) <= 20:
        plt.yticks(np.arange(len(simple_names1)), simple_names1)
    else:
        stride = max(1, len(simple_names1) // 20)
        plt.yticks(np.arange(0, len(simple_names1), stride), 
                  [simple_names1[i] for i in range(0, len(simple_names1), stride)])
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def find_layer_mappings(cka_matrix, layer_names1, layer_names2, threshold=0.1):
    """
    根据CKA相似度找到两个模型之间的层级映射关系
    
    参数:
    cka_matrix: CKA相似度矩阵
    layer_names1: 第一个模型的层名列表
    layer_names2: 第二个模型的层名列表
    threshold: 相似度阈值，只有超过此阈值的才被视为有映射关系
    
    返回:
    mappings: 从模型1层到模型2层的映射字典
    """
    mappings = {}
    
    # 对于模型1中的每一层，找到模型2中相似度最高的层
    for i, name1 in enumerate(layer_names1):
        max_sim = 0
        max_idx = -1
        
        for j, name2 in enumerate(layer_names2):
            if cka_matrix[i, j] > max_sim:
                max_sim = cka_matrix[i, j]
                max_idx = j
        
        # 只有相似度超过阈值的才保留
        if max_sim >= threshold and max_idx != -1:
            mappings[name1] = (layer_names2[max_idx], max_sim.item())
    
    return mappings

# 在test函数的最后添加测试代码:
def test_cka_alignment():
    # 加载两个不同的模型
    model_id1 = CKPT_PATH["qwen2"]
    model_id2 = CKPT_PATH["llama2"]
    
    tokenizer1, model1 = load_complete_qwen2(model_id1)
    tokenizer2, model2 = load_complete_llama2(model_id2)
    
    # 注册钩子并收集特征
    print("为模型1注册钩子...")
    layer_reps1, hooks1 = register_hooks_for_model(model1)
    
    print("为模型2注册钩子...")
    layer_reps2, hooks2 = register_hooks_for_model(model2)
    
    # 准备输入数据
    print("准备输入数据并执行推理...")
    input_text = "Hello, this is a test input for comparing models."
    inputs1 = tokenizer1(input_text, return_tensors="pt").to(device)
    inputs2 = tokenizer2(input_text, return_tensors="pt").to(device)
    
    # 执行推理以触发钩子
    with torch.no_grad():
        outputs1 = model1(**inputs1)
        outputs2 = model2(**inputs2)
    
    # 计算CKA相似度
    print("计算CKA相似度...")
    cka_matrix, layer_names1, layer_names2 = compute_cka(layer_reps1, layer_reps2)
    
    # 绘制相似度热图
    plot_cka_heatmap(cka_matrix, layer_names1, layer_names2, 
                    title=f"CKA相似度: {model_id1.split('/')[-1]} vs {model_id2.split('/')[-1]}")
    
    print(f"CKA矩阵统计: 最小值={cka_matrix.min().item():.4f}, 最大值={cka_matrix.max().item():.4f}, 平均值={cka_matrix.mean().item():.4f}")
    
    # 打印前10个最高相似度的层对
    flat_indices = torch.argsort(cka_matrix.flatten(), descending=True)[:10]
    rows = flat_indices // cka_matrix.shape[1]
    cols = flat_indices % cka_matrix.shape[1]
    print("\n前10个相似度最高的层对:")
    for i, (r, c) in enumerate(zip(rows, cols)):
        print(f"{i+1}. {layer_names1[r]} - {layer_names2[c]}: {cka_matrix[r, c].item():.4f}")
    # 找到层级映射关系
    mappings = find_layer_mappings(cka_matrix, layer_names1, layer_names2)
    
    print("层级映射关系:")
    for layer1, (layer2, sim) in mappings.items():
        print(f"{layer1} -> {layer2} (相似度: {sim:.4f})")
    
    # 清理钩子
    for hook in hooks1:
        hook.remove()
    for hook in hooks2:
        hook.remove()
    
    return cka_matrix, layer_names1, layer_names2, mappings


def test_cka_alignment_sequential():
    """使用顺序处理方式避免GPU内存不足问题"""
    # 准备输入数据
    input_text = "Hello, this is a test input for comparing models."
    
    # 首先处理模型1 (Qwen2)
    print("\n=== 处理模型1 (Qwen2) ===")
    model_id1 = CKPT_PATH["qwen2"]
    print(f"加载模型: {model_id1}")
    tokenizer1, model1 = load_complete_qwen2(model_id1)
    
    # 注册钩子
    print("为模型1注册钩子...")
    layer_reps1, hooks1 = register_hooks_for_model(model1)
    
    # 准备输入并执行推理
    print("模型1执行推理...")
    inputs1 = tokenizer1(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs1 = model1(**inputs1)
    
    # 收集完特征后释放模型1
    print("清理模型1...")
    del model1
    torch.cuda.empty_cache()  # 清理GPU缓存
    
    # 然后处理模型2 (Llama2)
    print("\n=== 处理模型2 (Llama2) ===")
    model_id2 = CKPT_PATH["llama2"]
    print(f"加载模型: {model_id2}")
    tokenizer2, model2 = load_complete_llama2(model_id2)
    
    # 注册钩子
    print("为模型2注册钩子...")
    layer_reps2, hooks2 = register_hooks_for_model(model2)
    
    # 准备输入并执行推理
    print("模型2执行推理...")
    inputs2 = tokenizer2(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs2 = model2(**inputs2)
    
    # 收集完特征后清理模型2
    print("清理模型2...")
    for hook in hooks2:
        hook.remove()
    del model2
    torch.cuda.empty_cache()  # 清理GPU缓存
    
    # 现在进行CKA计算 (此时两个模型都不在GPU内存中)
    print("\n=== 计算CKA相似度 ===")
    cka_matrix, layer_names1, layer_names2 = compute_cka(layer_reps1, layer_reps2, device='cpu')
    
    # 绘制相似度热图
    plot_title = f"CKA相似度: {model_id1.split('/')[-1]} vs {model_id2.split('/')[-1]}"
    plot_cka_heatmap(cka_matrix, layer_names1, layer_names2, title=plot_title)
    
    # 找到层级映射关系
    mappings = find_layer_mappings(cka_matrix, layer_names1, layer_names2, threshold=0.1)
    
    print("\n层级映射关系:")
    for layer1, (layer2, sim) in mappings.items():
        print(f"{layer1} -> {layer2} (相似度: {sim:.4f})")
    
    # 清理剩余的钩子
    for hook in hooks1:
        hook.remove()
    
    return cka_matrix, layer_names1, layer_names2, mappings

# 修改主函数调用
if __name__ == "__main__":
    test_cka_alignment()  # 原始函数 - 会导致内存错误
    # test_cka_alignment_sequential()  # 使用顺序处理的新函数
    # test()







