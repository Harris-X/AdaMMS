import os
import sys
import json
import torch
import safetensors.torch
import argparse
from tqdm import tqdm
import gc
import shutil
from collections import defaultdict
from typing import Dict

# 导入指定的模型和分词器类
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForVision2Seq
from torch.utils.data import DataLoader, TensorDataset

# 尝试导入 Hugging Face datasets 库
try:
    from datasets import load_dataset
except ImportError:
    print("="*80, file=sys.stderr)
    print("错误：无法导入 'datasets' 库。请运行 `pip install datasets`。", file=sys.stderr)
    print("这个库是获取探针数据集所必需的。", file=sys.stderr)
    print("="*80, file=sys.stderr)
    sys.exit(1)

# --- 权重加载与辅助函数 (保持不变) ---

def load_weights(base_path, index_filename):
    """根据索引文件或单个文件加载 safetensors 权重。"""
    weights = {}
    index_path = os.path.join(base_path, index_filename)
    if not os.path.exists(index_path):
        single_file_path = os.path.join(base_path, "model.safetensors")
        if os.path.exists(single_file_path):
            print(f"正在加载单个权重文件: {single_file_path}")
            return safetensors.torch.load_file(single_file_path)
        else:
            raise FileNotFoundError(f"在 {base_path} 中既未找到 {index_filename} 也未找到 model.safetensors")
            
    with open(index_path, 'r') as f:
        index = json.load(f)
        
    file_list = sorted(list(set(index["weight_map"].values())))
    for file in tqdm(file_list, desc=f"从 {os.path.basename(base_path)} 加载权重"):
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

# 【混合策略修改】新增一个辅助函数，用于判断哪些层使用分解，哪些使用插值
ATTN_PROJ_ENDINGS = (".self_attn.q_proj.weight", ".self_attn.q_proj.bias",
                     ".self_attn.k_proj.weight", ".self_attn.k_proj.bias",
                     ".self_attn.v_proj.weight", ".self_attn.v_proj.bias",
                     ".self_attn.o_proj.weight", ".self_attn.o_proj.bias")

def is_attn_proj(name: str) -> bool:
    """判断一个层是否是 self-attention 的 QKV O 投影层"""
    return name.endswith(ATTN_PROJ_ENDINGS)

def needs_decomposition_merge(name: str) -> bool:
    """判断是否对该层使用梯度分解合并（即非 attn 投影的 MLP 等层）"""
    layers_idx = name.rfind("layers.")
    if layers_idx == -1:
        return False

    # 如果是 attn 投影层，则不使用分解法
    if is_attn_proj(name):
        return False
        
    suffix = name[layers_idx:]
    if suffix.endswith(".self_attn.rotary_emb.inv_freq") or "layernorm" in suffix:
        return False
        
    return True

def create_soft_link(source_path, link_path):
    # Check if source path exists
    if not os.path.exists(source_path):
        print(f"Error: Source path '{source_path}' does not exist.")
        return

    # Check if link path exists, if not create it
    if not os.path.exists(link_path):
        os.makedirs(link_path)
        print(f"Created directory '{link_path}'")

    # Iterate through all files and directories in the source path
    for item in os.listdir(source_path):
        source_item = os.path.join(source_path, item)
        link_item = os.path.join(link_path, item)

        # Skip files that end with '.bin'
        if item.endswith('.bin'):
            print(f"Skipping '{item}' as it ends with '.bin'")
            continue

        # If it's a file, create a symbolic link
        if os.path.isfile(source_item):
            try:
                os.symlink(source_item, link_item)
                print(f"Created soft link '{link_item}' -> '{source_item}'")
            except OSError as e:
                print(f"Error creating soft link for '{item}': {e}")

        # If it's a directory, ignore it
        elif os.path.isdir(source_item):
            continue

# --- 核心实现类 ---

class LowMemoryGradientMerger:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.output_dir = os.path.join("merged_models", f"gradient-merge-{args.mode}")
        self.cache_dir = os.path.join(self.output_dir, "cache")
        self.grad_dir = os.path.join(self.cache_dir, "approx_grads")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.grad_dir, exist_ok=True)
        print(f"使用设备: {self.device}")
        print(f"输出将保存至: {self.output_dir}")

    # _get_target_modules, _cache_activations_for_model, stage1, stage2 保持不变...
    # ... 此处省略未作修改的函数以保持简洁 ...
    def _get_target_modules(self, model_to_hook):
        target_module_names = set()
        for name in model_to_hook.state_dict().keys():
            if needs_decomposition_merge(name): # 我们只为需要分解的层缓存激活
                module_name = ".".join(name.split('.')[:-1])
                target_module_names.add(module_name)
        return list(target_module_names)

    def _cache_activations_for_model(self, model_path, cache_path, capture_inputs=False):
        if os.path.exists(cache_path) and not self.args.force_recompute:
            print(f"激活缓存文件已存在: {cache_path}, 跳过。")
            return

        print(f"正在为 {os.path.basename(model_path)} 缓存激活...")
        is_vision_model = "VL" in model_path or "llava" in model_path.lower()
        ModelClass = AutoModelForVision2Seq if is_vision_model else AutoModelForCausalLM
        model = ModelClass.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model_to_hook = None
        if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
            model_to_hook = model.language_model.model
        elif hasattr(model, "language_model"):
            model_to_hook = model.language_model
        elif hasattr(model, "model"):
            model_to_hook = model.model
        else:
            model_to_hook = model

        print(f"将对模块 '{type(model_to_hook).__name__}' 注册钩子。")
        target_module_names = self._get_target_modules(model_to_hook)
        
        if not target_module_names:
            print(f"警告: 在 {os.path.basename(model_path)} 中没有找到需要分解合并的模块，跳过激活缓存。")
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            return
            
        print(f"在 {os.path.basename(model_path)} 中找到 {len(target_module_names)} 个目标模块。")

        hooks = []
        captured_activations = defaultdict(lambda: {"inputs": [], "outputs": []})

        def get_hook(name):
            def hook_fn(module, input, output):
                output_tensor = output[0] if isinstance(output, tuple) else output
                if isinstance(output_tensor, torch.Tensor):
                    captured_activations[name]["outputs"].append(output_tensor.detach().cpu())
                if capture_inputs:
                    input_tensor = input[0] if isinstance(input, tuple) else input
                    if isinstance(input_tensor, torch.Tensor):
                        captured_activations[name]["inputs"].append(input_tensor.detach().cpu())
            return hook_fn

        for name, module in model_to_hook.named_modules():
            if name in target_module_names:
                hooks.append(module.register_forward_hook(get_hook(name)))

        # 准备探针数据集 - 使用更可靠的 wikitext
        try:
            # 避免使用 wikipedia 以防止冲突
            probe_dataset_raw = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True).take(self.args.probe_samples)
            probe_texts = [item['text'] for item in probe_dataset_raw if item['text']]
        except Exception as e:
            print(f"加载 wikitext 数据集失败: {e}")
            # 备用文本
            probe_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning models are trained on large datasets.",
                "Neural networks process information in a hierarchical manner."
            ] * (self.args.probe_samples // 3 + 1)
            probe_texts = probe_texts[:self.args.probe_samples]
    
        probe_inputs = tokenizer(probe_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        probe_dataset = TensorDataset(probe_inputs['input_ids'], probe_inputs['attention_mask'])
        probe_dataloader = DataLoader(probe_dataset, batch_size=self.args.probe_batch_size)

        model.eval()
        with torch.no_grad():
            for batch in tqdm(probe_dataloader, desc=f"前向传播 {os.path.basename(model_path)}"):
                input_ids, attention_mask = batch[0].to(self.device), batch[1].to(self.device)
                model(input_ids=input_ids, attention_mask=attention_mask)

        for h in hooks: h.remove()
        
        averaged_activations = {}
        for name, data in captured_activations.items():
            averaged_activations[name] = {}
            if data["outputs"]:
                averaged_activations[name]["output"] = torch.mean(torch.cat([t.float() for t in data["outputs"]], dim=0), dim=0)
            if data["inputs"]:
                averaged_activations[name]["input"] = torch.mean(torch.cat([t.float() for t in data["inputs"]], dim=0), dim=0)

        torch.save(averaged_activations, cache_path)
        print(f"激活已缓存至: {cache_path}")
        
        del model, tokenizer, captured_activations, averaged_activations, probe_dataloader
        gc.collect()
        torch.cuda.empty_cache()

    def stage1_cache_activations(self):
        print("\n--- [阶段一: 缓存激活] ---")
        activations_a_path = os.path.join(self.cache_dir, "activations_A.pt")
        activations_c_path = os.path.join(self.cache_dir, "activations_C.pt")
        self._cache_activations_for_model(self.args.base_model_path, activations_a_path, capture_inputs=True)
        self._cache_activations_for_model(self.args.original_model_path, activations_c_path, capture_inputs=False)

    def stage2_calculate_approx_gradients(self):
        print("\n--- [阶段二: 计算近似梯度] ---")
        activations_a_path = os.path.join(self.cache_dir, "activations_A.pt")
        activations_c_path = os.path.join(self.cache_dir, "activations_C.pt")
        if not os.path.exists(activations_a_path) or not os.path.exists(activations_c_path):
            print("警告: 激活缓存文件不存在。阶段二将跳过，仅线性插值部分可执行。")
            return
            
        print("加载缓存的激活...")
        activations_A = torch.load(activations_a_path, map_location="cpu")
        activations_C = torch.load(activations_c_path, map_location="cpu")
        
        base_weights = load_weights(self.args.base_model_path, "model.safetensors.index.json")

        for key in tqdm(base_weights.keys(), desc="计算近似梯度"):
            if not needs_decomposition_merge(key):
                continue
                
            grad_path = os.path.join(self.grad_dir, f"{key.replace('/', '_')}.pt")
            if os.path.exists(grad_path) and not self.args.force_recompute:
                continue

            layers_idx = key.rfind("layers.")
            if layers_idx == -1: continue
            relative_key = key[layers_idx:]
            module_name = ".".join(relative_key.split('.')[:-1])

            if module_name not in activations_A or module_name not in activations_C:
                print(f"警告: 模块 {module_name} (来自键 {key}) 的激活未找到，跳过梯度计算。")
                continue
            
            # 确保输入激活存在
            if "input" not in activations_A[module_name] or "output" not in activations_A[module_name] or "output" not in activations_C[module_name]:
                 print(f"警告: 模块 {module_name} 的激活不完整，跳过梯度计算。")
                 continue

            X_A = activations_A[module_name]["input"].to(self.device)
            Y_A = activations_A[module_name]["output"].to(self.device)
            Y_C = activations_C[module_name]["output"].to(self.device)
            
            delta_Y = Y_A - Y_C
            
            g_approx = None
            if key.endswith(".weight"):
                g_approx = delta_Y.T @ X_A
            elif key.endswith(".bias"):
                g_approx = delta_Y
            
            if g_approx is not None:
                torch.save(g_approx.cpu(), grad_path)

        del base_weights, activations_A, activations_C
        gc.collect()
        print("所有近似梯度计算并保存完毕。")

    def stage3_merge_models(self):
        """
        【混合策略修改】执行阶段三：执行混合策略合并。
        """
        print("\n--- [阶段三: 混合策略合并] ---")

        print("正在从磁盘加载所有模型权重...")
        base_weights = load_weights(self.args.base_model_path, "model.safetensors.index.json")
        donor_weights_raw = load_weights(self.args.donor_model_path, "model.safetensors.index.json")
        original_weights = load_weights(self.args.original_model_path, "model.safetensors.index.json")

        print("正在标准化 Donor 模型的层名...")
        donor_weights = normalize_donor_keys(donor_weights_raw)
        del donor_weights_raw
        gc.collect()

        merged_weights = {}

        # 【新方法修改】梯度显著性加权门控合并的辅助函数
        def get_gated_saliency_weighted_vector(tau, g_approx, lambda_s, lambda_c, temperature):
            # 步骤1：计算梯度显著图 S
            saliency_map = torch.softmax(torch.abs(g_approx) / temperature, dim=None)
            
            # 步骤2：构建协同/冲突门控 M
            product = tau * g_approx
            synergy_mask = (product < 0).float()
            conflict_mask = (product > 0).float()
            
            # 步骤3：融合
            # τ* = λ_s * (τ ⊙ M_s ⊙ S) - λ_c * (τ ⊙ M_c ⊙ S)
            tau_star = lambda_s * (tau * synergy_mask * saliency_map) - \
                       lambda_c * (tau * conflict_mask * saliency_map)
            
            return tau_star


        for key in tqdm(base_weights.keys(), desc="逐层合并权重"):
            # 默认使用原始模型C的权重作为起点
            if key in original_weights:
                merged_weights[key] = original_weights[key].clone()
            else:
                merged_weights[key] = base_weights[key].clone()

            # 检查所有张量是否都存在，避免KeyError
            if not all(k in w for k in [key] for w in [base_weights, donor_weights, original_weights]):
                 merged_weights[key] = base_weights[key].clone()
                 continue
            
            # 【新方法修改】对所有可合并层使用新方法
            if needs_decomposition_merge(key):
                grad_path = os.path.join(self.grad_dir, f"{key.replace('/', '_')}.pt")
                if not os.path.exists(grad_path):
                    print(f"警告: 分解所需的梯度 {key} 未找到，将使用基础模型权重。")
                    merged_weights[key] = base_weights[key].clone()
                    continue

                W_A = base_weights[key].float().to(self.device)
                W_B = donor_weights[key].float().to(self.device)
                W_C = original_weights[key].float().to(self.device)
                g_approx = torch.load(grad_path, map_location=self.device).float()

                tau_A = W_A - W_C
                tau_B = W_B - W_C

                # 应用新的门控加权方法
                tau_star_A = get_gated_saliency_weighted_vector(
                    tau_A, g_approx, self.args.lambda_A_s, self.args.lambda_A_c, self.args.temp_A
                )
                tau_star_B = get_gated_saliency_weighted_vector(
                    tau_B, g_approx, self.args.lambda_B_s, self.args.lambda_B_c, self.args.temp_B
                )
                
                # W* = W_C + τ*_A + τ*_B
                w_star = W_C + tau_star_A + tau_star_B
                
                merged_weights[key] = w_star.to(base_weights[key].dtype).cpu()
            
            # 其他所有层，包括之前未处理的 attn_proj，都直接使用基础模型的权重
            else:
                 merged_weights[key] = base_weights[key].clone()

        # --- 保存模型 ---
        print("\n正在保存合并后的模型...")
        index_path = os.path.join(self.args.base_model_path, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index_map = json.load(f)["weight_map"]
        
        sharded_weights = {filename: {} for filename in set(index_map.values())}
        for key, value in merged_weights.items():
            if key in index_map:
                sharded_weights[index_map[key]][key] = value
        
        for filename, weights_dict in sharded_weights.items():
            safetensors.torch.save_file(weights_dict, os.path.join(self.output_dir, filename))
        
        shutil.copy(index_path, os.path.join(self.output_dir, os.path.basename(index_path)))
        create_soft_link(source_path=self.args.base_model_path, link_path=self.output_dir)
        print(f"模型成功合并并保存至: {self.output_dir}")

    def run_pipeline(self):
        self.stage1_cache_activations()
        self.stage2_calculate_approx_gradients()
        self.stage3_merge_models()

if __name__ == "__main__":
    # 【新方法修改】更新 ArgumentParser 以支持门控加权方法的所有超参数
    parser = argparse.ArgumentParser(description="使用梯度显著性加权的门控方法进行低显存模型合并。")
    
    # 基本配置
    parser.add_argument('--base_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct", help="基础模型A的路径。")
    parser.add_argument('--donor_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si-hf", help="贡献模型B的路径。")
    parser.add_argument('--original_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-7B-Instruct", help="原始共同祖先模型C的路径。")
    parser.add_argument('--mode', type=str, default="saliency-gated-merge", help="为本次合并配置命名。")
    parser.add_argument('--cuda_device', type=int, default=3, help="使用的 CUDA 设备编号。")

    # 探针数据集配置
    parser.add_argument('--probe_samples', type=int, default=200, help="用于探测的样本数量。")
    parser.add_argument('--probe_batch_size', type=int, default=2, help="探测时的批处理大小。")

    # --- 门控加权合并组 ---
    # 模型A (Base Model) 的系数
    parser.add_argument('--lambda_A_s', type=float, default=1.0, help="基础模型A的协同部分强度。")
    parser.add_argument('--lambda_A_c', type=float, default=1.0, help="基础模型A的冲突部分抑制强度 (越大抑制越强)。")
    parser.add_argument('--temp_A', type=float, default=0.01, help="基础模型A的Softmax温度系数 (越小越尖锐)。")
    
    # 模型B (Donor Model) 的系数
    parser.add_argument('--lambda_B_s', type=float, default=1.0, help="贡献模型B的协同部分强度。")
    parser.add_argument('--lambda_B_c', type=float, default=0.0, help="贡献模型B的冲突部分抑制强度 (越大抑制越强)。")
    parser.add_argument('--temp_B', type=float, default=0.01, help="贡献模型B的Softmax温度系数 (越小越尖锐)。")
    
    # 功能性参数
    parser.add_argument('--force_recompute', action='store_true', help="强制重新计算缓存的激活或梯度。")

    args = parser.parse_args()
    
    if torch.cuda.is_available() and args.cuda_device < torch.cuda.device_count():
        device = f"cuda:{args.cuda_device}"
    else:
        device = "cpu"

    print("--- 配置信息 ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("--------------------")

    merger = LowMemoryGradientMerger(args, device)
    merger.run_pipeline()