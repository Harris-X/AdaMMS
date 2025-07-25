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

# 导入指定的模型和分词器类
from transformers import AutoTokenizer, AutoModelForVision2Seq
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

# --- 权重加载与辅助函数 (来自您的模板) ---

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

def need_merge(name: str) -> bool:
    """
    根据层名判断是否需要合并。
    修正：使其能处理带任意前缀的层名。
    """
    # 找到 "layers" 在名字中的位置
    layers_idx = name.rfind("layers.")
    if layers_idx == -1:
        return False

    # 从 "layers." 开始截取，以进行统一判断
    suffix = name[layers_idx:]
    
    if suffix.endswith(".self_attn.rotary_emb.inv_freq"):
        return False
    
    # 排除归一化层
    if "layernorm" in suffix:
        return False
    
    # 注意：您的原始 need_merge 函数排除了 QKV O 投影，这里遵循该设定。
    if suffix.endswith((".self_attn.q_proj.weight", ".self_attn.q_proj.bias",
                      ".self_attn.k_proj.weight", ".self_attn.k_proj.bias",
                      ".self_attn.v_proj.weight", ".self_attn.v_proj.bias",
                      ".self_attn.o_proj.weight", ".self_attn.o_proj.bias")):
        return False
        
    # 只要是 layers 内部的其他参数，都进行合并
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

    def _get_target_modules(self, model_to_hook):
        """
        获取模型中所有需要被hook的目标模块的名称。
        返回的名称是相对于 model_to_hook 的。
        """
        target_module_names = set()
        # 遍历 state_dict 来确定需要合并的参数
        for name in model_to_hook.state_dict().keys():
            # 使用完整的、相对于顶层模型的键名来判断
            if need_merge(name):
                # 从参数名找到对应的模块名
                # e.g., "layers.0.mlp.gate_proj.weight" -> "layers.0.mlp.gate_proj"
                module_name = ".".join(name.split('.')[:-1])
                target_module_names.add(module_name)
        return list(target_module_names)

    def _cache_activations_for_model(self, model_path, cache_path, capture_inputs=False):
        """阶段一的核心函数：为单个模型执行前向传播并缓存激活。"""
        if os.path.exists(cache_path) and not self.args.force_recompute:
            print(f"激活缓存文件已存在: {cache_path}, 跳过。")
            return

        print(f"正在为 {os.path.basename(model_path)} 缓存激活...")
        
        # 根据模型路径判断应该使用哪种加载方式
        is_vision_model = "VL" in model_path or "llava" in model_path.lower()
        
        
        if is_vision_model:
            model = AutoModelForVision2Seq.from_pretrained(
                model_path, torch_dtype=torch.bfloat16
            ).to(self.device)
        else:
            # 对于纯文本模型使用 AutoModelForCausalLM
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16
            ).to(self.device)
    
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        
        # 修正：智能地定位到包含 "layers" 的语言模型部分
        model_to_hook = None
        if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
             # 适用于 Qwen2-VL-7B-Instruct 结构
            model_to_hook = model.language_model.model
        elif hasattr(model, "language_model"):
            # 适用于 llava-onevision-qwen2-7b-si-hf 结构
            model_to_hook = model.language_model
        elif hasattr(model, "model"):
            # 适用于 Qwen2-7B-Instruct 结构
            model_to_hook = model.model
        else:
            # 备用方案
            model_to_hook = model

        print(f"将对模块 '{type(model_to_hook).__name__}' 注册钩子。")
        
        target_module_names = self._get_target_modules(model_to_hook)
        
        if not target_module_names:
            print(f"警告: 在 {os.path.basename(model_path)} 中没有找到符合 `need_merge` 条件的模块。")
            return # 如果找不到目标，直接返回
            
        print(f"在 {os.path.basename(model_path)} 中找到 {len(target_module_names)} 个目标模块。")

        hooks = []
        captured_activations = defaultdict(lambda: {"inputs": [], "outputs": []})

        def get_hook(name):
            def hook_fn(module, input, output):
                # 处理不同类型的输出
                if isinstance(output, tuple):
                    # 确保元组不为空
                    if not output: return
                    output_tensor = output[0]
                else:
                    output_tensor = output
                
                # 确保 output_tensor 是一个张量
                if not isinstance(output_tensor, torch.Tensor): return
                captured_activations[name]["outputs"].append(output_tensor.detach().cpu())
                
                if capture_inputs:
                    # 健壮性检查：确保 input 不为空
                    if not input: return
                    # 处理不同类型的输入
                    if isinstance(input, tuple):
                        input_tensor = input[0]
                    else:
                        input_tensor = input
                    
                    if not isinstance(input_tensor, torch.Tensor): return
                    captured_activations[name]["inputs"].append(input_tensor.detach().cpu())
            return hook_fn

        # 注册钩子
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
        
        # 求平均并保存
        averaged_activations = {}
        for name, data in captured_activations.items():
            averaged_activations[name] = {}
            if data["outputs"] and len(data["outputs"]) > 0:
                try:
                    averaged_activations[name]["output"] = torch.mean(torch.cat(data["outputs"], dim=0).float(), dim=0)
                except Exception as e:
                    print(f"处理 {name} 的输出激活时出错: {e}")
                    continue
                    
            if data["inputs"] and len(data["inputs"]) > 0:
                try:
                    averaged_activations[name]["input"] = torch.mean(torch.cat(data["inputs"], dim=0).float(), dim=0)
                except Exception as e:
                    print(f"处理 {name} 的输入激活时出错: {e}")
                    continue

        torch.save(averaged_activations, cache_path)
        print(f"激活已缓存至: {cache_path}")
        
        del model, tokenizer, captured_activations, averaged_activations, probe_dataloader
        gc.collect()
        torch.cuda.empty_cache()

    def stage1_cache_activations(self):
        """执行阶段一：缓存模型A和模型C的激活。"""
        print("\n--- [阶段一: 缓存激活] ---")
        activations_a_path = os.path.join(self.cache_dir, "activations_A.pt")
        activations_c_path = os.path.join(self.cache_dir, "activations_C.pt")

        # 缓存模型A (qwen2_vl) 的输入和输出激活
        self._cache_activations_for_model(self.args.base_model_path, activations_a_path, capture_inputs=True)
        # 缓存模型C (original_model) 的输出激活
        self._cache_activations_for_model(self.args.original_model_path, activations_c_path, capture_inputs=False)

    def stage2_calculate_approx_gradients(self):
        """执行阶段二：逐层计算近似梯度。"""
        print("\n--- [阶段二: 计算近似梯度] ---")
        
        activations_a_path = os.path.join(self.cache_dir, "activations_A.pt")
        activations_c_path = os.path.join(self.cache_dir, "activations_C.pt")
        if not os.path.exists(activations_a_path) or not os.path.exists(activations_c_path):
            print("错误: 激活缓存文件不存在。请先运行阶段一。")
            return
            
        print("加载缓存的激活...")
        activations_A = torch.load(activations_a_path, map_location="cpu")
        activations_C = torch.load(activations_c_path, map_location="cpu")
        
        # 以模型A的参数作为遍历蓝本
        base_weights = load_weights(self.args.base_model_path, "model.safetensors.index.json")

        for key in tqdm(base_weights.keys(), desc="计算近似梯度"):
            
            if not need_merge(key):
                continue
                
            grad_path = os.path.join(self.grad_dir, f"{key.replace('/', '_')}.pt") # 替换/防止路径问题
            if os.path.exists(grad_path) and not self.args.force_recompute:
                continue

            # 修正：从绝对路径key中提取出相对于语言模型部分的模块名
            layers_idx = key.rfind("layers.")
            if layers_idx == -1: continue
            relative_key = key[layers_idx:]
            module_name = ".".join(relative_key.split('.')[:-1])

            if module_name not in activations_A or module_name not in activations_C:
                print(f"警告: 模块 {module_name} (来自键 {key}) 的激活未找到，跳过。")
                continue

            # 加载激活到设备
            X_A = activations_A[module_name]["input"].to(self.device)
            Y_A = activations_A[module_name]["output"].to(self.device)
            Y_C = activations_C[module_name]["output"].to(self.device)
            
            # 计算期望变化方向 (近似的梯度信号)
            delta_Y = Y_A - Y_C #
            
            g_approx = None
            if key.endswith(".weight"):
                # 权重梯度: g' ≈ ΔY^T @ X_A
                g_approx = delta_Y.T @ X_A #
            elif key.endswith(".bias"):
                # 偏置梯度: g' ≈ sum(ΔY) over batch/seq dim
                g_approx = delta_Y.sum(dim=0) #
            
            if g_approx is not None:
                torch.save(g_approx.cpu(), grad_path)

        del base_weights, activations_A, activations_C
        gc.collect()
        print("所有近似梯度计算并保存完毕。")

    def stage3_merge_models(self):
        """执行阶段三：逐层加载权重，执行梯度引导的分解与合并。"""
        print("\n--- [阶段三: 分解与合并] ---")

        print("正在从磁盘加载所有模型权重...")
        base_weights = load_weights(self.args.base_model_path, "model.safetensors.index.json") #
        donor_weights_raw = load_weights(self.args.donor_model_path, "model.safetensors.index.json") #
        original_weights = load_weights(self.args.original_model_path, "model.safetensors.index.json") #

        print("正在标准化 Donor 模型的层名...")
        donor_weights = normalize_donor_keys(donor_weights_raw) # language_model.model.layers.0.mlp.down_proj.weight
        del donor_weights_raw
        gc.collect()

        merged_weights = {}

        for key in tqdm(base_weights.keys(), desc="逐层合并权重"):
            # 默认直接使用基础模型的权重
            merged_weights[key] = base_weights[key] 

            # 检查是否满足合并条件,额外需要注意此处的检查代码
            # 使用原始的、未经标准化的 donor_weights_raw 进行检查，以匹配前缀
            if need_merge(key) and key in donor_weights and key in original_weights and base_weights[key].shape == donor_weights[key].shape:
                grad_path = os.path.join(self.grad_dir, f"{key.replace('/', '_')}.pt")
                if not os.path.exists(grad_path):
                    print(f"警告: 参数 {key} 的近似梯度未找到，将使用基础模型权重。")
                    continue

                # 加载张量到设备
                W_A = base_weights[key].float().to(self.device)
                W_B = donor_weights[key].float().to(self.device)
                W_C = original_weights[key].float().to(self.device)
                g_approx = torch.load(grad_path, map_location=self.device).float()

                # 计算任务向量 τ_B
                tau_B = W_B - W_C #

                # --- 梯度引导的向量分解 (方法论来自核心文档) ---
                g_norm_sq = torch.sum(g_approx * g_approx)
                if g_norm_sq < 1e-9:
                    print(f"警告: 参数 {key} 的梯度范数过小，跳过分解。")
                    tau_B_synergy, tau_B_conflict, tau_B_ortho = torch.zeros_like(tau_B), torch.zeros_like(tau_B), tau_B
                else:
                    proj_scalar = torch.sum(g_approx * tau_B) / g_norm_sq
                    # 协同分量 (与 -g_approx 方向一致)
                    tau_B_synergy = torch.clamp_min(-proj_scalar, 0) * g_approx #
                    # 冲突分量 (与 +g_approx 方向一致)
                    tau_B_conflict = torch.clamp_min(proj_scalar, 0) * g_approx #
                    # 正交分量
                    tau_B_ortho = tau_B - tau_B_conflict - tau_B_synergy #

                # --- 最终合并公式 ---
                # W* = W_A + λ_s * τ_B_synergy - λ_c * τ_B_conflict + λ_o * τ_B_ortho
                w_star = W_A + (self.args.lambda_s * tau_B_synergy) - \
                               (self.args.lambda_c * tau_B_conflict) + \
                               (self.args.lambda_o * tau_B_ortho)
                
                merged_weights[key] = w_star.to(base_weights[key].dtype).cpu()

        # --- 保存模型 ---
        print("\n正在保存合并后的模型...")
        # 遵循您的分片保存逻辑
        index_path = os.path.join(self.args.base_model_path, "model.safetensors.index.json") #
        with open(index_path, "r") as f:
            index_map = json.load(f)["weight_map"] #
        
        sharded_weights = {filename: {} for filename in set(index_map.values())} #
        for key, value in merged_weights.items():
            if key in index_map:
                sharded_weights[index_map[key]][key] = value #
        
        for filename, weights_dict in sharded_weights.items():
            safetensors.torch.save_file(weights_dict, os.path.join(self.output_dir, filename)) #
        
        # 复制/链接其他必要文件
        shutil.copy(index_path, os.path.join(self.output_dir, os.path.basename(index_path)))
        create_soft_link(source_path=self.args.base_model_path, link_path=self.output_dir)
        print(f"模型成功合并并保存至: {self.output_dir}")

    def run_pipeline(self):
        """按顺序执行所有阶段。"""
        self.stage1_cache_activations()
        self.stage2_calculate_approx_gradients()
        self.stage3_merge_models()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用局部梯度近似进行低显存模型合并。")
    
    # 基本配置
    parser.add_argument('--base_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct", help="基础模型A的路径。")
    parser.add_argument('--donor_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si-hf", help="贡献模型B的路径。")
    parser.add_argument('--original_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-7B-Instruct", help="原始共同祖先模型C的路径。")
    parser.add_argument('--mode', type=str, default="default", help="为本次合并配置命名。")
    parser.add_argument('--cuda_device', type=int, default=2, help="使用的 CUDA 设备编号。")

    # 探针数据集配置
    parser.add_argument('--probe_dataset', type=str, default="wikitext", help="用于探测激活的数据集 ('wikipedia' 或 'c4')。")
    parser.add_argument('--probe_samples', type=int, default=128, help="用于探测的样本数量。")
    parser.add_argument('--probe_batch_size', type=int, default=2, help="探测时的批处理大小，如果显存不足请减小。")

    # 合并超参数
    parser.add_argument('--lambda_s', type=float, default=1.0, help="协同分量的系数。")
    parser.add_argument('--lambda_c', type=float, default=1.0, help="冲突分量的系数。")
    parser.add_argument('--lambda_o', type=float, default=1.0, help="正交分量的系数。")
    
    # 功能性参数
    parser.add_argument('--force_recompute', action='store_true', help="强制重新计算缓存的激活或梯度。")

    args = parser.parse_args()
    
    # 设置设备
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