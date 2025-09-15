import os
import torch
import torch.nn as nn
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from copy import deepcopy
import numpy as np
import scipy

# --- 1. 核心配置 ---
CKPT_PATH = {
    "llama2": "./downloaded_models/Llama-2-7b-hf",
    "qwen2": "./downloaded_models/Qwen2-7B-Instruct",
}
# 检查是否有可用的 CUDA 设备，否则使用 CPU
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# --- 2. 辅助函数 (模型加载, 数据集, CKA计算, 对齐) ---

def load_complete_model(model_id):
    """通用模型加载函数，使用 bfloat16 以节省内存"""
    print(f"正在加载模型: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"{os.path.basename(model_id)} 模型加载完成。")
    return tokenizer, model

def load_and_prepare_dataset(tokenizer_a, tokenizer_b, dataset_name="wikitext", split="test", max_samples=32, max_length=128):
    """加载并处理数据集以进行特征提取"""
    print(f"正在加载数据集: {dataset_name}...")
    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)
    
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    def tokenize_fn(examples):
        text = [t for t in examples["text"] if t and t.strip()]
        if not text: return {}
        
        inputs_a = tokenizer_a(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        inputs_b = tokenizer_b(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        
        return {
            "input_ids_a": inputs_a.input_ids, "attention_mask_a": inputs_a.attention_mask,
            "input_ids_b": inputs_b.input_ids, "attention_mask_b": inputs_b.attention_mask,
        }

    processed_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names, batch_size=4)
    processed_dataset.set_format(type='torch')
    return processed_dataset

class CKA:
    """
    来自 torch_cka/cka.py 的核心 CKA 计算逻辑。
    用于计算两个模型各层之间表征的相似度。
    """
    def __init__(self, model1, model2, model1_name='llama2', model2_name='qwen2', device='cpu'):
        self.model1 = model1
        self.model2 = model2
        self.device = device
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.model1_features = {}
        self.model2_features = {}

    def _register_hooks(self, model, model_name, features_dict):
        hooks = []
        # 我们只关注 self_attn 层的输出
        for name, module in model.named_modules():
            if 'self_attn' in name and isinstance(module, nn.Module) and not list(module.children()): # 确保是叶子模块
                 # 修改：只捕获 self_attn 块的整体输出
                if name.endswith("self_attn"):
                    hook = module.register_forward_hook(self._create_hook(name, features_dict))
                    hooks.append(hook)
        return hooks

    def _create_hook(self, name, features_dict):
        def hook(model, input, output):
             # output[0] 是注意力层的输出张量
            features_dict[name] = output[0].detach()
        return hook

    def _HSIC(self, K, L):
        """计算希尔伯特-施密特独立性准则 (HSIC) [cite: 117]"""
        N = K.shape[0]
        if N < 4: return 0.0 # HSIC 定义要求 N > 3
        ones = torch.ones(N, 1).to(self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()

    def compare_layers(self, dataloader):
        """比较并计算两个模型所有 self_attn 层的 CKA 相似度矩阵"""
        self.model1.eval()
        self.model2.eval()
        
        hooks1 = self._register_hooks(self.model1, self.model1_name, self.model1_features)
        hooks2 = self._register_hooks(self.model2, self.model2_name, self.model2_features)
        
        # 初始化 hsic 矩阵
        layer_names1 = sorted(self.model1_features.keys())
        layer_names2 = sorted(self.model2_features.keys())
        
        hsic_matrix = torch.zeros(len(layer_names1), len(layer_names2), 3)

        num_batches = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="CKA 计算中"):
                input_ids_a, attention_mask_a = batch['input_ids_a'].to(self.device), batch['attention_mask_a'].to(self.device)
                input_ids_b, attention_mask_b = batch['input_ids_b'].to(self.device), batch['attention_mask_b'].to(self.device)

                self.model1_features.clear()
                self.model2_features.clear()

                _ = self.model1(input_ids=input_ids_a, attention_mask=attention_mask_a)
                _ = self.model2(input_ids=input_ids_b, attention_mask=attention_mask_b)
                
                # 更新 hsichsic_matrix
                layer_names1 = sorted(self.model1_features.keys())
                layer_names2 = sorted(self.model2_features.keys())

                for i, name1 in enumerate(layer_names1):
                    feat1 = self.model1_features[name1].flatten(1)
                    K = feat1 @ feat1.t()
                    K.fill_diagonal_(0.0)
                    hsic_matrix[i, :, 0] += self._HSIC(K, K)

                    for j, name2 in enumerate(layer_names2):
                        feat2 = self.model2_features[name2].flatten(1)
                        L = feat2 @ feat2.t()
                        L.fill_diagonal_(0.0)
                        
                        hsic_matrix[i, j, 1] += self._HSIC(K, L)
                        hsic_matrix[i, j, 2] += self._HSIC(L, L)
                num_batches += 1

        # 移除钩子
        for hook in hooks1 + hooks2:
            hook.remove()
            
        hsic_matrix /= num_batches
        # 归一化 CKA
        cka_matrix = hsic_matrix[:, :, 1] / (torch.sqrt(hsic_matrix[:, :, 0]) * torch.sqrt(hsic_matrix[:, :, 2]))
        cka_matrix = torch.nan_to_num(cka_matrix) # 处理可能的 NaN
        
        return cka_matrix, layer_names1, layer_names2

def align_layers_dp(C):
    """
    使用动态规划计算最优层映射，来自 `resnet_fusion_merging_auto.py` [cite: 366]
    C: CKA 相似度矩阵 (深模型层数 x 浅模型层数)
    """
    m, n = C.shape
    assert m >= n, "模型 A 必须比模型 B 更深或一样深"

    F = torch.zeros((n + 1, m + 1))
    for k in range(1, n + 1):
        for l in range(k, m + 1):
            # 论文中的层对齐（LMA）算法 [cite: 118, 120]
            # 这里我们简化为匹配当前层和上一层，实际论文中的SMA/LMA更复杂
            F[k, l] = max(F[k, l - 1], F[k - 1, l - 1] + C[l - 1, k - 1])

    A = torch.zeros(n, dtype=torch.long)
    k, l = n, m
    while k > 0 and l > 0:
        if F[k, l] == F[k - 1, l - 1] + C[l - 1, k - 1]:
            A[k - 1] = l - 1
            k -= 1
            l -= 1
        else:
            l -= 1
    return A

def get_weight_remapping(layer_map_indices, layer_names_deep, layer_names_shallow):
    """根据层映射生成权重字典的重命名规则"""
    remapping = {}
    for i, j in enumerate(layer_map_indices):
        shallow_name = layer_names_shallow[i]
        deep_name = layer_names_deep[j]
        
        # 我们只关注 self_attn 模块内的权重
        for weight_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            remapping[f"{shallow_name}.{weight_type}.weight"] = f"{deep_name}.{weight_type}.weight"

    return remapping

def align_and_create_new_model(model_deep, model_shallow, weight_remapping):
    """
    创建一个新的、统一架构的模型，并将浅层模型的权重映射过去
    """
    print("正在创建统一架构的新模型...")
    # 新模型以深层模型为模板
    new_model_from_shallow = deepcopy(model_deep)
    
    # 获取浅层模型的 state_dict
    shallow_state_dict = model_shallow.state_dict()
    
    # 创建新的 state_dict
    new_state_dict = new_model_from_shallow.state_dict()
    for shallow_key, deep_key in weight_remapping.items():
        if shallow_key in shallow_state_dict:
            # 检查尺寸是否匹配，如果不匹配则需要更复杂的操作（例如填充或截断）
            if new_state_dict[deep_key].shape == shallow_state_dict[shallow_key].shape:
                 new_state_dict[deep_key] = shallow_state_dict[shallow_key]
            else:
                 print(f"警告: 权重维度不匹配，跳过 {shallow_key} -> {deep_key}")

    new_model_from_shallow.load_state_dict(new_state_dict)
    print("新模型创建并加载权重完成。")
    return new_model_from_shallow
    
def match_tensors_zipit(correlation_matrix, model_dims):
    """
    弹性神经元压缩算法, 来自 `matching_functions.py`。
    它通过贪心策略，逐步合并最相似的神经元。
    """
    sims = compute_correlation(correlation_matrix)
    O = sims.shape[0]
    remainder = max(model_dims) # 合并后的维度以较宽的模型为准
    permutation_matrix = torch.eye(O, O, device=sims.device)
    
    torch.diagonal(sims)[:] = -torch.inf
    
    while permutation_matrix.shape[1] > remainder:
        best_idx = sims.reshape(-1).argmax()
        row_idx, col_idx = best_idx // sims.shape[1], best_idx % sims.shape[1]

        if col_idx < row_idx:
            row_idx, col_idx = col_idx, row_idx
        
        # 合并列
        permutation_matrix[:, row_idx] += permutation_matrix[:, col_idx]
        permutation_matrix = torch.cat([permutation_matrix[:, :col_idx], permutation_matrix[:, col_idx + 1:]], dim=1)

        # 更新相似度矩阵
        sims[:, row_idx] = torch.minimum(sims[:, row_idx], sims[:, col_idx])
        sims = torch.cat([sims[:, :col_idx], sims[:, col_idx + 1:]], dim=1)
        sims[row_idx, :] = torch.minimum(sims[row_idx, :], sims[col_idx, :])
        sims = torch.cat([sims[:col_idx, :], sims[col_idx + 1:, :]], dim=0)

    unmerge_matrix = permutation_matrix
    merge_matrix = permutation_matrix / (permutation_matrix.sum(dim=0, keepdim=True) + 1e-8)
    
    return merge_matrix.T, unmerge_matrix

def compute_correlation(covariance, eps=1e-7):
    """来自 matching_functions.py"""
    std = torch.diagonal(covariance).sqrt()
    covariance = covariance / (torch.clamp(torch.outer(std, std), min=eps))
    return covariance

def merge_weights(model_a, model_b):
    """最终的权重平均步骤"""
    merged_state_dict = deepcopy(model_a.state_dict())
    state_dict_b = model_b.state_dict()
    for key in merged_state_dict.keys():
        if key in state_dict_b:
            # 简单平均
            merged_state_dict[key] = (merged_state_dict[key].float() + state_dict_b[key].float()) / 2.0
            merged_state_dict[key] = merged_state_dict[key].to(model_a.dtype) # 转换回 bfloat16
    
    merged_model = deepcopy(model_a)
    merged_model.load_state_dict(merged_state_dict)
    return merged_model
    
# --- 3. 主执行流程 ---

# 加载模型和分词器
tokenizer_llama2, model_llama2 = load_complete_model(CKPT_PATH["llama2"])
tokenizer_qwen2, model_qwen2 = load_complete_model(CKPT_PATH["qwen2"])

# 加载和预处理数据集
dataset = load_and_prepare_dataset(tokenizer_llama2, tokenizer_qwen2)
dataloader = DataLoader(dataset, batch_size=4) # 减小批量大小以适应内存

# 确定哪个模型更深
num_layers_llama2 = model_llama2.config.num_hidden_layers
num_layers_qwen2 = model_qwen2.config.num_hidden_layers

if num_layers_llama2 >= num_layers_qwen2:
    model_deep, model_shallow = model_llama2, model_qwen2
    tok_deep, tok_shallow = tokenizer_llama2, tokenizer_qwen2
    name_deep, name_shallow = "Llama2", "Qwen2"
else:
    model_deep, model_shallow = model_qwen2, model_llama2
    tok_deep, tok_shallow = tokenizer_qwen2, tokenizer_llama2
    name_deep, name_shallow = "Qwen2", "Llama2"

print(f"深模型: {name_deep} ({model_deep.config.num_hidden_layers} 层), 浅模型: {name_shallow} ({model_shallow.config.num_hidden_layers} 层)")

# --- 步骤 1: 深度对齐 ---
print("\n--- 步骤 1: 开始计算层间相似度 (CKA) ---")
cka_computer = CKA(model_deep, model_shallow, device=device)
cka_matrix, layer_names_deep, layer_names_shallow = cka_computer.compare_layers(dataloader)
print("CKA 矩阵计算完成。")
print(f"深模型层: {layer_names_deep}")
print(f"浅模型层: {layer_names_shallow}")


print("\n--- 步骤 2: 使用动态规划进行层对齐 ---")
layer_map_indices = align_layers_dp(cka_matrix)
print("层对齐完成。映射关系 (浅模型层索引 -> 深模型层索引):")
for i, j in enumerate(layer_map_indices):
    print(f"  {layer_names_shallow[i]} -> {layer_names_deep[j]}")

# 生成权重重映射字典
weight_remapping = get_weight_remapping(layer_map_indices, layer_names_deep, layer_names_shallow)

# 创建一个新的统一架构的模型
aligned_model_from_shallow = align_and_create_new_model(model_deep, model_shallow, weight_remapping)

# 现在我们有两个架构相同的模型: model_deep 和 aligned_model_from_shallow
models_to_merge = [model_deep, aligned_model_from_shallow]

# --- 步骤 3: 宽度对齐与合并 (神经元压缩) ---
print("\n--- 步骤 3: 开始宽度对齐与合并 ---")
# 遍历所有需要合并的层
final_merged_model = deepcopy(model_deep)
final_state_dict = final_merged_model.state_dict()

# 获取对齐后的层名称
aligned_layer_names = [layer_names_deep[j] for j in layer_map_indices]

for layer_name in tqdm(aligned_layer_names, desc="合并各层权重"):
    for weight_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        
        weight_key = f"{layer_name}.{weight_type}.weight"
        
        w_a = model_deep.state_dict()[weight_key]
        w_b = aligned_model_from_shallow.state_dict()[weight_key]
        
        # 维度不同，需要压缩
        if w_a.shape != w_b.shape:
            # 准备数据进行压缩
            model_dims = [w_a.shape[0], w_b.shape[0]]
            # 这里的 correlation_matrix 应该基于神经元激活值，但为简化，我们使用权重的协方差作为代理
            # 这是一个近似，论文中会用真实数据计算激活
            combined_weights = torch.cat([w_a.flatten(1), w_b.flatten(1)], dim=0).float()
            covariance_matrix = torch.cov(combined_weights)
            
            # 执行神经元压缩 [cite: 139]
            merge_matrix, unmerge_matrix = match_tensors_zipit(covariance_matrix, model_dims)
            
            # 应用变换
            merged_w = unmerge_matrix @ merge_matrix @ combined_weights
            
            # 恢复形状并赋值
            final_state_dict[weight_key] = merged_w[:w_a.shape[0]].view(w_a.shape).to(w_a.dtype)

        else: # 维度相同，直接平均
            final_state_dict[weight_key] = ((w_a.float() + w_b.float()) / 2.0).to(w_a.dtype)
            
final_merged_model.load_state_dict(final_state_dict)

print("\n--- 合并完成！---")

# 清理内存
del model_llama2, model_qwen2, model_deep, model_shallow, aligned_model_from_shallow
gc.collect()
torch.cuda.empty_cache()

# 你现在可以使用 final_merged_model 和一个分词器（例如 tokenizer_deep）进行后续任务
print("最终合并的模型已准备就绪: final_merged_model")