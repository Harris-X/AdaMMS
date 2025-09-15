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

# MODIFIED: 键名标准化，以适应异构模型LLM部分的对齐
def normalize_llm_keys(weights_to_norm: dict, reference_keys: list) -> dict:
    """通用函数，用于将任何模型的LLM部分键名与参考键名对齐。"""
    key_map = {}
    
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
            normalized_weights[key] = value # 保留非LLM部分的权重
            
    return normalized_weights

def need_merge(name: str) -> bool:
    """根据层名判断是否需要合并 - 仅合并LLM的权重参数。"""
    if ("language_model.layers" not in name) or ("model.layers" not in name):
        
        if not name.endswith(".weight"): # 只处理权重，忽略偏置等
            return False

        if "layernorm" in name or "embed_tokens" in name or "norm" in name or ".inv_freq" in name:
            return False
            
        return True
    else:
        return False

# --- 数据集处理函数 ---
class VQAv2ProbeDataset(Dataset):
    def __init__(self, hf_dataset, max_samples=100):
        self.samples = []
        for item in hf_dataset:
            self.samples.append({"image": item["image"], "text": item["question"]})
            if len(self.samples) >= max_samples: break
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        item = self.samples[idx]
        image = item["image"]
        if image.mode == 'RGBA': image = image.convert('RGB')
        return {"image": image, "text": item['text']}

def collate_fn_factory(processor):
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        texts = [item['text'] for item in batch]
        messages_batch = [[{"role": "user", "content": [{"type": "text", "text": text},{"type": "image"}]}] for text in texts]
        prompt_batch = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
        inputs = processor(text=prompt_batch, images=images, return_tensors="pt", padding=True)
        return inputs
    return collate_fn

# --- 核心实现类 ---
# NEW: 引入全新的 AGIDPMMerger 类来实现新方法
class AGIDPMMerger:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.output_dir = os.path.join("merged_models", f"agidpm-{args.mode}")
        self.cache_dir = os.path.join(self.output_dir, "cache")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"使用设备: {self.device}")
        print(f"输出将保存至: {self.output_dir}")

    def _get_target_module_map(self, model):
        """获取需要hook的模块名到模块实例的映射。"""
        module_map = {}
        for name, module in model.named_modules():
            # 检查是否有任何权重参数需要合并
            if any(need_merge(f"{name}.{param_name}") for param_name, _ in module.named_parameters()):
                module_map[name] = module
        return module_map
    
    def _cache_activations_raw(self, model_info, model_path, required_activations, dataset_raw):
        """为每个模型从原始数据集处理数据并缓存激活（内存优化版）。"""
        cache_path = os.path.join(self.cache_dir, f"activations_{model_info}.pt")
        if os.path.exists(cache_path) and not self.args.force_recompute:
            print(f"激活缓存文件 {cache_path} 已存在, 跳过。")
            return torch.load(cache_path, map_location="cpu")

        print(f"正在为 {model_info} ({os.path.basename(model_path)}) 缓存激活...")
        is_vision_model = "VL" in model_path or "llava" in model_path.lower()
        is_llava = "llava" in model_path.lower()
        
        ModelClass = AutoModelForVision2Seq if is_vision_model else AutoModelForCausalLM
        model = ModelClass.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(self.device)
        processor = AutoProcessor.from_pretrained(model_path)
        model.eval()

        if is_vision_model:
            if is_llava:
                model_to_hook = model.model.language_model
            else:
                model_to_hook = model.model.language_model
        else:
            model_to_hook = model.model
        target_modules = self._get_target_module_map(model_to_hook)
        
        # 内存优化：不再存储所有张量，而是存储运行总和和计数
        activation_stats = defaultdict(lambda: {
            "input_sum": None, "input_tokens": 0,
            "output_sum": None, "output_tokens": 0
        })
    
        # 内存优化：钩子函数直接更新总和，而不是追加到列表
        def get_hook_with_kwargs(name, req_act):
            def hook_fn(module, args, kwargs, output):
                # 处理输出
                if "output" in req_act:
                    out_tensor = None
                    if isinstance(output, tuple) and len(output) > 0:
                        out_tensor = output[0]
                    else:
                        out_tensor = output
                    
                    if isinstance(out_tensor, torch.Tensor):
                        t_float = out_tensor.detach().cpu().float()
                        t_reshaped = t_float.reshape(-1, t_float.shape[-1])
                        current_sum = torch.sum(t_reshaped, dim=0)
                        
                        if activation_stats[name]["output_sum"] is None:
                            activation_stats[name]["output_sum"] = current_sum
                        else:
                            activation_stats[name]["output_sum"] += current_sum
                        activation_stats[name]["output_tokens"] += t_reshaped.shape[0]
            
                # 处理输入
                if "input" in req_act:
                    in_tensor = None
                    if "hidden_states" in kwargs:
                        in_tensor = kwargs["hidden_states"]
                    elif isinstance(args, tuple) and len(args) > 0 and isinstance(args[0], torch.Tensor):
                        in_tensor = args[0]
                        
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
    
        hooks = []
        for name, module in target_modules.items():
            hooks.append(module.register_forward_hook(
                get_hook_with_kwargs(name, required_activations), 
                with_kwargs=True
            ))
    
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
    
        # 内存优化：从运行总和计算最终平均值
        averaged_activations = {}
        for name, stats in activation_stats.items():
            averaged_activations[name] = {}
            if stats["input_sum"] is not None and stats["input_tokens"] > 0:
                averaged_activations[name]["input"] = stats["input_sum"] / stats["input_tokens"]
            if stats["output_sum"] is not None and stats["output_tokens"] > 0:
                averaged_activations[name]["output"] = stats["output_sum"] / stats["output_tokens"]

        torch.save(averaged_activations, cache_path)
        del model, processor, hooks, activation_stats, averaged_activations
        gc.collect() 
        torch.cuda.empty_cache()
        # 返回值是可选的，因为结果已经保存到文件了
        # return averaged_activations

    def stage1_cache_all_activations(self):
        """阶段一：为所有模型分别缓存所需的激活。"""
        print("\n--- [阶段一: 缓存所有激活] ---")
        
        # 直接加载原始数据集，不预先处理
        probe_dataset_raw = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
        
        # 为每个模型单独处理数据并缓存激活
        self._cache_activations_raw("A", self.args.base_model_path, "input_output", probe_dataset_raw)
        self._cache_activations_raw("B", self.args.donor_model_path, "input_output", probe_dataset_raw)
        self._cache_activations_raw("C", self.args.original_model_path, "output", probe_dataset_raw)

    def stage2_analyze_neurons_and_get_masks(self):
        """阶段二：计算近似SNIP并生成非冲突掩码。"""
        print("\n--- [阶段二: 近似SNIP神经元分析] ---")
        mask_cache_path = os.path.join(self.cache_dir, f"non_conflict_masks_r{self.args.top_k_ratio}.pt")
        if os.path.exists(mask_cache_path) and not self.args.force_recompute:
            print("非冲突掩码缓存文件已存在, 跳过。")
            return

        print("加载所有权重和缓存的激活...")
        weights_A = load_weights(self.args.base_model_path)
        weights_B_raw = load_weights(self.args.donor_model_path)
        weights_C = load_weights(self.args.original_model_path)
        weights_B = normalize_llm_keys(weights_B_raw, list(weights_C.keys())); del weights_B_raw

        activations = {
            'A': torch.load(os.path.join(self.cache_dir, "activations_A.pt")),
            'B': torch.load(os.path.join(self.cache_dir, "activations_B.pt")),
            'C': torch.load(os.path.join(self.cache_dir, "activations_C.pt"))
        }

        non_conflict_masks = {}
        if os.path.exists(mask_cache_path) and not self.args.force_recompute:
            print(f"激活缓存文件 {mask_cache_path} 已存在, 跳过。")
            return torch.load(mask_cache_path, map_location="cpu")
        
        for key in tqdm(weights_A.keys(), desc="分析神经元"):
            if not need_merge(key): continue
            
            key_in_c = key.replace("model.language_model.", "model.")
            if not (key_in_c in weights_B and key_in_c in weights_C): continue
                
            module_name = ".".join(key.split('.')[1:-1]) # e.g., layers.0.mlp
            try:
                # 计算近似梯度
                delta_Y_A = activations['A'][module_name]['output'] - activations['C'][module_name]['output']
                g_approx_A = torch.outer(delta_Y_A, activations['A'][module_name]['input'])
                
                delta_Y_B = activations['B'][module_name]['output'] - activations['C'][module_name]['output']
                g_approx_B = torch.outer(delta_Y_B, activations['B'][module_name]['input'])
                
                # 计算近似SNIP分数
                W_A, W_B, W_C = weights_A[key], weights_B[key_in_c], weights_C[key_in_c]
                snip_A = (W_A.float() * g_approx_A).abs()
                snip_B = (W_B.float() * g_approx_B).abs()

                # 选举与分离
                k = int(snip_A.numel() * self.args.top_k_ratio)
                mask_A = snip_A >= torch.topk(snip_A.flatten(), k=k)[0].min()
                mask_B = snip_B >= torch.topk(snip_B.flatten(), k=k)[0].min()
                
                tau_A, tau_B = W_A - W_C, W_B - W_C
                conflict_mask = (mask_A & mask_B) & (torch.sign(tau_A) != torch.sign(tau_B))
                
                final_mask_B = mask_B & (~conflict_mask)
                non_conflict_masks[key] = final_mask_B
            except KeyError:
                print(f"警告: 模块 {module_name} 的激活数据不完整，跳过参数 {key}。")
                continue

        torch.save(non_conflict_masks, mask_cache_path)
        print(f"非冲突掩码计算完成并缓存至: {mask_cache_path}")

    def stage3_project_and_merge(self):
        """阶段三：执行分离投影合并。"""
        print("\n--- [阶段三: 分离投影合并] ---")
        
        print("加载所有权重、掩码和方向向量...")
        weights_A = load_weights(self.args.base_model_path)
        weights_B_raw = load_weights(self.args.donor_model_path)
        weights_C = load_weights(self.args.original_model_path)
        weights_B = normalize_llm_keys(weights_B_raw, list(weights_C.keys())); del weights_B_raw

        mask_cache_path = os.path.join(self.cache_dir, f"non_conflict_masks_r{self.args.top_k_ratio}.pt")
        non_conflict_masks = torch.load(mask_cache_path)
        activations_A = torch.load(os.path.join(self.cache_dir, "activations_A.pt"))

        final_merged_weights = weights_A.copy()
        for key, mask_B in tqdm(non_conflict_masks.items(), desc="执行合并"):
            key_in_c = key.replace("model.language_model.", "model.")
            module_name = ".".join(key.split('.')[1:-1])
            # module_name_A = "language_model." + module_name
            # module_name_B = "language_model." + module_name
            # module_name_C = module_name

            W_A, W_B, W_C = weights_A[key], weights_B[key_in_c], weights_C[key_in_c]
            d_i = activations_A[module_name]['input'].to(self.device) # 投影方向
            
            tau_B = (W_B - W_C).to(self.device)
            tau_B_disjoint = tau_B * mask_B.to(self.device)
            
            d_i_norm_sq = torch.sum(d_i * d_i)
            if d_i_norm_sq > 1e-9:
                # 投影操作
                proj_scalar = (tau_B_disjoint.float() @ d_i.float()) / d_i_norm_sq
                tau_B_proj = torch.outer(proj_scalar, d_i.float())
                tau_B_ortho = tau_B_disjoint - tau_B_proj
            else:
                tau_B_proj = torch.zeros_like(tau_B)
                tau_B_ortho = tau_B_disjoint

            # 最终合并
            W_star = W_A.to(self.device).float() + self.args.lambda_proj * tau_B_proj + self.args.lambda_ortho * tau_B_ortho
            final_merged_weights[key] = W_star.cpu().to(W_A.dtype)
        
        # 保存模型
        self._save_model(final_merged_weights)

    def _save_model(self, merged_weights):
        """保存模型权重。"""
        print("\n正在保存合并后的模型...")
        # ... (此处省略与模板相同的保存逻辑)
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
        self.stage2_analyze_neurons_and_get_masks()
        self.stage3_project_and_merge()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用AGIDPM进行高效、无需反向传播的模型合并。")
    
    # 基本配置
    parser.add_argument('--base_model_path', type=str, default="./downloaded_models/Qwen2-VL-7B-Instruct", help="基础模型A的路径。")
    parser.add_argument('--donor_model_path', type=str, default="./downloaded_models/llava-onevision-qwen2-7b-si-hf", help="贡献模型B的路径。")
    parser.add_argument('--original_model_path', type=str, default="./downloaded_models/Qwen2-7B-Instruct", help="原始共同祖先模型C的路径。")
    parser.add_argument('--mode', type=str, default="agidpm-default", help="为本次合并配置命名。")
    parser.add_argument('--cuda_device', type=int, default=2, help="使用的 CUDA 设备编号。")

    # 数据集配置
    parser.add_argument('--probe_samples', type=int, default=100, help="用于引导合并的目标域样本数量。")
    parser.add_argument('--probe_batch_size', type=int, default=2, help="处理引导数据时的批处理大小。")

    # AGIDPM 合并超参数
    parser.add_argument('--top_k_ratio', type=float, default=0.2, help="用于选举关键神经元的Top-K比率。")
    parser.add_argument('--lambda_proj', type=float, default=1.0, help="投影（相关）分量的系数。")
    parser.add_argument('--lambda_ortho', type=float, default=0.4, help="正交（无关）分量的系数。")
    
    # 功能性参数
    parser.add_argument('--force_recompute', action='store_true', help="强制重新计算缓存的激活或掩码。")

    args = parser.parse_args()
    
    device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"

    print("--- 配置信息 ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("--------------------")

    merger = AGIDPMMerger(args, device)
    merger.run_pipeline()