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
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoModelForVision2Seq
from torch.utils.data import DataLoader, Dataset, TensorDataset

# 尝试导入 Hugging Face datasets 库
try:
    from datasets import load_dataset
except ImportError:
    print("="*80, file=sys.stderr)
    print("错误：无法导入 'datasets' 库。请运行 `pip install datasets`。", file=sys.stderr)
    print("这个库是获取探针数据集所必需的。", file=sys.stderr)
    print("="*80, file=sys.stderr)
    sys.exit(1)

# --- 权重加载与辅助函数 ---

def load_weights(base_path, index_filename="model.safetensors.index.json"):
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

def normalize_llm_keys(weights_to_norm: dict, reference_keys: list) -> dict:
    """通用函数，用于将任何模型的LLM部分键名与参考键名对齐。"""
    ref_prefix = ""
    for key in reference_keys:
        if "layers" in key:
            ref_prefix = key.split("layers")[0]
            break
            
    norm_prefix = ""
    for key in weights_to_norm.keys():
        if "layers" in key:
            norm_prefix = key.split("layers")[0]
            break
            
    if not ref_prefix or not norm_prefix:
        print("警告：无法在模型中定位到 'layers'，键名标准化可能失败。")
        return weights_to_norm

    normalized_weights = {}
    for key, value in weights_to_norm.items():
        if key.startswith(norm_prefix):
            new_key = ref_prefix + key[len(norm_prefix):]
            normalized_weights[new_key] = value
        else:
            normalized_weights[key] = value
            
    return normalized_weights

def need_merge(name: str) -> bool:
    """根据层名判断是否需要合并 - 仅合并LLM的权重参数。"""
    is_in_llm_layers = "language_model.layers" in name or "model.layers" in name
    if not is_in_llm_layers: return False
    if not name.endswith(".weight"): return False
    if "layernorm" in name or "embed_tokens" in name or "norm" in name or ".inv_freq" in name: return False
    return True

# --- 核心实现类 ---
class FAPISAMerger:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.output_dir = os.path.join("merged_models", f"fapisam-{args.mode}")
        self.cache_dir = os.path.join(self.output_dir, "cache")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"使用设备: {self.device}")
        print(f"输出将保存至: {self.output_dir}")
        self.EPS = 1e-9

    def _get_target_module_map(self, model):
        """获取需要hook的模块名到模块实例的映射。"""
        module_map = {}
        llm_module = None
        if hasattr(model, 'language_model'): llm_module = model.language_model
        elif hasattr(model, 'model'): llm_module = model.model
        else: llm_module = model

        for name, module in llm_module.named_modules():
            # 兼容不同模型结构
            base_prefix = ""
            if hasattr(model, 'language_model'): base_prefix = "language_model."
            elif hasattr(model, 'model'): base_prefix = "model."
            
            full_module_name = f"{base_prefix}{name}"
            
            if any(need_merge(f"{full_module_name}.{param_name}") for param_name, _ in module.named_parameters()):
                module_map[name] = module
        return module_map
    
    def _cache_activations_raw(self, model_info, model_path, required_activations, dataset_raw):
        """为每个模型从原始数据集处理数据并缓存激活（内存优化版）。"""
        cache_path = os.path.join(self.cache_dir, f"activations_{model_info}.pt")
        if os.path.exists(cache_path) and not self.args.force_recompute:
            print(f"激活缓存文件 {cache_path} 已存在, 跳过。")
            return

        print(f"正在为 {model_info} ({os.path.basename(model_path)}) 缓存激活...")
        is_vision_model = "VL" in model_path or "llava" in model_path.lower()
        is_llava = "llava" in model_path.lower()
        
        ModelClass = AutoModelForVision2Seq if is_vision_model else AutoModelForCausalLM
        model = ModelClass.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(self.device)
        processor = AutoProcessor.from_pretrained(model_path)
        model.eval()

        target_modules = self._get_target_module_map(model)
        
        activation_stats = defaultdict(lambda: {
            "input_sum": None, "input_tokens": 0,
            "output_sum": None, "output_tokens": 0
        })
    
        def get_hook_with_kwargs(name, req_act):
            def hook_fn(module, args, kwargs, output):
                if "output" in req_act:
                    out_tensor = output[0] if isinstance(output, tuple) else output
                    if isinstance(out_tensor, torch.Tensor):
                        t_float = out_tensor.detach().cpu().float()
                        t_reshaped = t_float.reshape(-1, t_float.shape[-1])
                        current_sum = torch.sum(t_reshaped, dim=0)
                        
                        if activation_stats[name]["output_sum"] is None:
                            activation_stats[name]["output_sum"] = current_sum
                        else:
                            activation_stats[name]["output_sum"] += current_sum
                        activation_stats[name]["output_tokens"] += t_reshaped.shape[0]
                
                if "input" in req_act:
                    in_tensor = kwargs.get("hidden_states", args[0] if args and isinstance(args[0], torch.Tensor) else None)
                    if isinstance(in_tensor, torch.Tensor):
                        t_float = in_tensor.detach().cpu().float()
                        t_reshaped = t_float.reshape(-1, t_float.shape[-1])
                        current_sum = torch.sum(t_reshaped, dim=0)

                        if activation_stats[name]["input_sum"] is None:
                            activation_stats[name]["input_sum"] = current_sum
                        else:
                            activation_stats[name]["input_sum"] += current_sum
                        activation_stats[name]["input_tokens"] += t_reshaped.shape[0]
            return hook_fn
    
        hooks = [
            module.register_forward_hook(get_hook_with_kwargs(name, required_activations), with_kwargs=True)
            for name, module in target_modules.items()
        ]
    
        original_samples = []
        dataset_iterator = iter(dataset_raw)
        for item in dataset_iterator:
            if len(original_samples) >= self.args.probe_samples: break
            image = item["image"]
            if image.mode == 'RGBA': image = image.convert('RGB')
            original_samples.append({"image": image, "text": item["question"]})
        
        with torch.no_grad():
            num_batches = (len(original_samples) + self.args.probe_batch_size - 1) // self.args.probe_batch_size
            pbar = tqdm(range(0, len(original_samples), self.args.probe_batch_size), total=num_batches, desc=f"前向传播 {model_info}")
            for i in pbar:
                batch_data = original_samples[i:i+self.args.probe_batch_size]
                images = [item["image"] for item in batch_data]
                texts = [item["text"] for item in batch_data]
                
                if is_llava:
                    conversations = [{"role": "user", "content": [{"type": "text", "text": t}, {"type": "image"}]} for t in texts]
                    prompts = [processor.apply_chat_template([conv], tokenize=False, add_generation_prompt=True) for conv in conversations]
                    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
                elif is_vision_model:
                    messages_batch = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}] for text in texts]
                    prompt_batch = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
                    inputs = processor(text=prompt_batch, images=images, return_tensors="pt", padding=True)
                else:
                    tokenizer = processor
                    formatted_texts = [tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True) for text in texts]
                    inputs = tokenizer(text=formatted_texts, return_tensors="pt", padding=True, truncation=True)
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                model(**inputs)

        for h in hooks: h.remove()
    
        averaged_activations = {}
        for name, stats in activation_stats.items():
            averaged_activations[name] = {}
            if stats["input_sum"] is not None and stats["input_tokens"] > 0:
                averaged_activations[name]["input"] = stats["input_sum"] / stats["input_tokens"]
            if stats["output_sum"] is not None and stats["output_tokens"] > 0:
                averaged_activations[name]["output"] = stats["output_sum"] / stats["output_tokens"]

        torch.save(averaged_activations, cache_path)
        del model, processor, hooks, activation_stats, averaged_activations
        gc.collect(); torch.cuda.empty_cache()


    def stage1_cache_all_activations(self):
        """阶段一：为所有模型分别缓存所需的激活。"""
        print("\n--- [阶段一: 缓存所有激活] ---")
        probe_dataset_raw = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
        self._cache_activations_raw("A", self.args.base_model_path, ["input", "output"], probe_dataset_raw)
        self._cache_activations_raw("B", self.args.donor_model_path, ["input", "output"], probe_dataset_raw)
        self._cache_activations_raw("C", self.args.original_model_path, ["input", "output"], probe_dataset_raw)

    # ########################################################################## #
    # #                           关键代码修改区域                             # #
    # ########################################################################## #

    def _min_max_normalize(self, tensor):
        """对张量进行min-max归一化。"""
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val + self.EPS)

    def stage2_regularized_saliency_analysis(self):
        """阶段二：【FA-PISAM】执行正则化本征显著性评分。"""
        print("\n--- [阶段二: FA-PISAM 正则化重要性评分] ---")
        mask_cache_path = os.path.join(self.cache_dir, f"fapisam_mask_r{self.args.top_k_ratio}_alpha{self.args.alpha}.pt")
        if os.path.exists(mask_cache_path) and not self.args.force_recompute:
            print("FA-PISAM 统一掩码缓存文件已存在, 跳过。")
            return

        print("加载所有权重和缓存的激活...")
        weights_A = load_weights(self.args.base_model_path)
        weights_B_raw = load_weights(self.args.donor_model_path)
        weights_C_raw = load_weights(self.args.original_model_path)
        
        weights_B = normalize_llm_keys(weights_B_raw, list(weights_A.keys())); del weights_B_raw
        weights_C = normalize_llm_keys(weights_C_raw, list(weights_A.keys())); del weights_C_raw

        activations = {
            'A': torch.load(os.path.join(self.cache_dir, "activations_A.pt")),
            'B': torch.load(os.path.join(self.cache_dir, "activations_B.pt")),
            'C': torch.load(os.path.join(self.cache_dir, "activations_C.pt"))
        }

        unified_masks = {}
        pbar = tqdm(weights_A.keys(), desc="【FA-PISAM】分析神经元")
        for key in pbar:
            if not need_merge(key): continue
            
            key_in_c = key.replace("model.language_model.", "model.")
            if not (key_in_c in weights_B and key_in_c in weights_C): continue
                
            module_name = ".".join(key.split('.')[1:-1]) # e.g., layers.0.mlp
            
            try:
                # 步骤 1: 计算目标域显著性 S_spec (来自TAG-M)
                W_A, W_B, W_C = weights_A[key].float(), weights_B[key].float(), weights_C[key].float()
                tau_A, tau_B = W_A - W_C, W_B - W_C
                
                # 使用完整的激活数据计算伪梯度
                g_approx_A = torch.outer(activations['A'][module_name]['output'], activations['A'][module_name]['input'])
                g_approx_B = torch.outer(activations['B'][module_name]['output'], activations['B'][module_name]['input'])
                
                s_spec_A = (-g_approx_A * tau_A + (self.args.alpha / 2) * (g_approx_A**2) * (tau_A**2)).abs()
                s_spec_B = (-g_approx_B * tau_B + (self.args.alpha / 2) * (g_approx_B**2) * (tau_B**2)).abs()

                # 步骤 2: 计算本征重要性 S_int
                s_int_A = tau_A.abs()
                s_int_B = tau_B.abs()

                # 步骤 3: 分数归一化
                s_spec_A_norm = self._min_max_normalize(s_spec_A)
                s_int_A_norm = self._min_max_normalize(s_int_A)
                s_spec_B_norm = self._min_max_normalize(s_spec_B)
                s_int_B_norm = self._min_max_normalize(s_int_B)

                # 步骤 4: 谐波平均计算最终分数 S_final
                s_final_A = (2 * s_spec_A_norm * s_int_A_norm) / (s_spec_A_norm + s_int_A_norm + self.EPS)
                s_final_B = (2 * s_spec_B_norm * s_int_B_norm) / (s_spec_B_norm + s_int_B_norm + self.EPS)

                # 步骤 5: 生成统一重要性掩码 M
                k = int(s_final_A.numel() * self.args.top_k_ratio)
                if k == 0: continue
                
                mask_A = s_final_A >= torch.topk(s_final_A.flatten(), k=k, sorted=False)[0].min()
                mask_B = s_final_B >= torch.topk(s_final_B.flatten(), k=k, sorted=False)[0].min()
                
                unified_masks[key] = (mask_A | mask_B).cpu()

            except KeyError:
                pbar.set_description(f"警告: 模块 {module_name} 激活数据不完整，跳过 {key}")
                continue

        torch.save(unified_masks, mask_cache_path)
        print(f"FA-PISAM 统一掩码计算完成并缓存至: {mask_cache_path}")
        
    def stage3_fast_procrustes_fusion(self):
        """阶段三：【FA-PISAM】执行快速普氏引导的对齐与融合。"""
        print("\n--- [阶段三: FA-PISAM 快速对齐融合] ---")
        
        print("加载所有权重和统一掩码...")
        weights_A = load_weights(self.args.base_model_path)
        weights_B_raw = load_weights(self.args.donor_model_path)
        weights_C_raw = load_weights(self.args.original_model_path)
        
        weights_B = normalize_llm_keys(weights_B_raw, list(weights_A.keys())); del weights_B_raw
        weights_C = normalize_llm_keys(weights_C_raw, list(weights_A.keys())); del weights_C_raw

        mask_cache_path = os.path.join(self.cache_dir, f"fapisam_mask_r{self.args.top_k_ratio}_alpha{self.args.alpha}.pt")
        unified_masks = torch.load(mask_cache_path)

        final_merged_weights = weights_A.copy()
        pbar = tqdm(unified_masks.items(), desc="【FA-PISAM】执行快速对齐")
        for key, M_i in pbar:
            if not (key in weights_B and key in weights_C): continue
                
            W_A, W_B, W_C = weights_A[key].float().to(self.device), weights_B[key].float().to(self.device), weights_C[key].float().to(self.device)
            M_i = M_i.to(self.device)

            # 步骤 1: 提取活跃子空间权重
            W_prime_A = (W_A * M_i).to(self.device)
            W_prime_B = (W_B * M_i).to(self.device)

            # 步骤 2: 通过迭代法快速求解旋转矩阵 R
            R = torch.eye(W_prime_A.shape[1], device=self.device, dtype=torch.float32)
            try:
                C = W_prime_A.T @ W_prime_B
                # Newton-Schulz 迭代, 使用伪逆增强鲁棒性
                for _ in range(self.args.fast_align_iters):
                    R_inv_t = torch.linalg.pinv(R).T
                    R = 0.5 * (R + R_inv_t)

                # 步骤 3: 计算缩放因子 s
                tr_W_prime_B_sq = torch.trace(W_prime_B.T @ W_prime_B)
                s = torch.trace(R.T @ C) / (tr_W_prime_B_sq + self.EPS)
            
            except torch.linalg.LinAlgError:
                pbar.set_description(f"警告: 快速对齐在层 {key} 上失败，使用单位阵")
                R = torch.eye(W_A.shape[1], device=self.device)
                s = torch.tensor(1.0, device=self.device)
            
            # 步骤 4: 对齐完整任务向量
            tau_B = W_B - W_C
            tau_B_aligned = s * (tau_B.to(self.device) @ R)

            # 步骤 5: 原则性平均融合
            tau_A = W_A - W_C
            tau_merged = (1 - self.args.beta) * tau_A.to(self.device) + self.args.beta * tau_B_aligned

            # 步骤 6: 计算最终合并权重
            W_star = W_C.to(self.device) + tau_merged
            final_merged_weights[key] = W_star.cpu().to(weights_A[key].dtype)
        
        self._save_model(final_merged_weights)

    # ########################################################################## #
    # #                         关键代码修改区域结束                             # #
    # ########################################################################## #
    
    def _save_model(self, merged_weights):
        """保存模型权重。"""
        print("\n正在保存合并后的模型...")
        # 保存逻辑与模板相同
        index_path = os.path.join(self.args.base_model_path, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index_map = json.load(f)["weight_map"]
        
        sharded_weights = defaultdict(dict)
        for key, value in merged_weights.items():
            if key in index_map:
                sharded_weights[index_map[key]][key] = value
        
        for filename, weights_dict in sharded_weights.items():
            safetensors.torch.save_file(weights_dict, os.path.join(self.output_dir, filename))
        
        shutil.copy(index_path, os.path.join(self.output_dir, os.path.basename(index_path)))
        for filename in os.listdir(self.args.base_model_path):
            if filename.endswith(('.json', '.model', '.py', '.md')):
                source_file = os.path.join(self.args.base_model_path, filename)
                dest_file = os.path.join(self.output_dir, filename)
                if not os.path.exists(dest_file):
                       shutil.copy(source_file, dest_file)
        print(f"模型成功合并并保存至: {self.output_dir}")

    def run_pipeline(self):
        """按顺序执行所有阶段。"""
        self.stage1_cache_all_activations()
        self.stage2_regularized_saliency_analysis()
        self.stage3_fast_procrustes_fusion()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用FA-PISAM进行超高效、鲁棒且兼顾泛化性的模型合并。")
    
    # 基本配置
    parser.add_argument('--base_model_path', type=str, default="./downloaded_models/Qwen2-VL-7B-Instruct", help="基础模型A的路径。")
    parser.add_argument('--donor_model_path', type=str, default="./downloaded_models/llava-onevision-qwen2-7b-si-hf", help="贡献模型B的路径。")
    parser.add_argument('--original_model_path', type=str, default="./downloaded_models/Qwen2-7B-Instruct", help="原始共同祖先模型C的路径。")
    parser.add_argument('--mode', type=str, default="fapisam-default", help="为本次合并配置命名。")
    parser.add_argument('--cuda_device', type=int, default=5, help="使用的 CUDA 设备编号。")

    # 数据集配置
    parser.add_argument('--probe_samples', type=int, default=100, help="用于引导合并的目标域样本数量。")
    parser.add_argument('--probe_batch_size', type=int, default=2, help="处理引导数据时的批处理大小。")

    # FA-PISAM 合并超参数
    parser.add_argument('--top_k_ratio', type=float, default=0.2, help="【阶段二】用于选举关键神经元的Top-K比率。")
    parser.add_argument('--alpha', type=float, default=1.0, help="【阶段二】平衡泰勒展开一阶和二阶项的超参数。")
    parser.add_argument('--fast_align_iters', type=int, default=8, help="【阶段三】快速对齐迭代次数。")
    parser.add_argument('--beta', type=float, default=0.5, help="【阶段三】最终融合时模型B的贡献权重，唯一核心可调超参数。")
    
    # 功能性参数
    parser.add_argument('--force_recompute', action='store_true', help="强制重新计算缓存的数据。")

    args = parser.parse_args()
    
    device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"

    print("--- 配置信息 ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("--------------------")

    merger = FAPISAMerger(args, device)
    merger.run_pipeline()