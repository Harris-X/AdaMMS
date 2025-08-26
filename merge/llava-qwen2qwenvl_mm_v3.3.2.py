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
            answer = item["answers"][0]["answer"] if item.get("answers") else "yes"  # 假设有answers，取第一个
            self.samples.append({"image": item["image"], "text": item["question"], "answer": answer})
            if len(self.samples) >= max_samples: break
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        item = self.samples[idx]
        image = item["image"]
        if image.mode == 'RGBA': image = image.convert('RGB')
        return {"image": image, "text": item['text'], "answer": item['answer']}

def collate_fn_factory(processor):
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        texts = [item['text'] for item in batch]
        answers = [item['answer'] for item in batch]
        return {"images": images, "texts": texts, "answers": answers}
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
        cache_path = os.path.join(self.cache_dir, f"importance_{model_info}.pt")
        if os.path.exists(cache_path) and not self.args.force_recompute:
            print(f"重要性缓存文件 {cache_path} 已存在, 跳过。")
            return torch.load(cache_path, map_location="cpu")

        print(f"正在为 {model_info} ({os.path.basename(model_path)}) 缓存重要性...")
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
        
        # 内存优化：存储重要性总和和计数
        importance_stats = defaultdict(lambda: {"sum": None, "count": 0})
    
        original_samples = []
        dataset_iterator = iter(dataset_raw)
        for item in dataset_iterator:
            if len(original_samples) >= self.args.probe_samples: break
            image = item["image"]
            if image.mode == 'RGBA': image = image.convert('RGB')
            answer = item["answers"][0]["answer"] if item.get("answers") else "yes"
            original_samples.append({"image": image, "text": item["question"], "answer": answer})
        
        with torch.no_grad():
            num_batches = (len(original_samples) + self.args.probe_batch_size - 1) // self.args.probe_batch_size
            pbar = tqdm(range(0, len(original_samples), self.args.probe_batch_size), total=num_batches, desc=f"前向传播 {model_info}")
            for i in pbar:
                batch_data = original_samples[i:i+self.args.probe_batch_size]
                images = [item["image"] for item in batch_data] if images else None
                texts = [item["text"] for item in batch_data]
                answers = [item["answer"] for item in batch_data]
                
                if is_llava or is_vision_model:
                    user_messages = [[{"role": "user", "content": [{"type": "text", "text": t}, {"type": "image"}]}] for t in texts]
                    user_prompt = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in user_messages]
                    user_inputs = processor(text=user_prompt, images=images, return_tensors="pt", padding=True)
                    prompt_length = user_inputs["input_ids"].shape[1]
                    
                    full_messages = [user_messages[j] + [{"role": "assistant", "content": a}] for j, a in enumerate(answers)]
                    full_prompt = [processor.apply_chat_template(m, tokenize=False) for m in full_messages]
                    inputs = processor(text=full_prompt, images=images, return_tensors="pt", padding=True)
                    
                    labels = inputs["input_ids"].clone()
                    labels[:, :prompt_length] = -100
                    inputs["labels"] = labels
                else:
                    tokenizer = processor
                    user_formatted = [tokenizer.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True) for t in texts]
                    user_inputs = tokenizer(user_formatted, return_tensors="pt", padding=True, truncation=True)
                    prompt_length = user_inputs["input_ids"].shape[1]
                    
                    full_formatted = [tokenizer.apply_chat_template([{"role": "user", "content": t}, {"role": "assistant", "content": a}], tokenize=False) for t, a in zip(texts, answers)]
                    inputs = tokenizer(text=full_formatted, return_tensors="pt", padding=True, truncation=True)
                    
                    labels = inputs["input_ids"].clone()
                    labels[:, :prompt_length] = -100
                    inputs["labels"] = labels
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                output = model(**inputs)
                loss = output.loss
                loss.backward()
                
                for name, module in target_modules.items():
                    if hasattr(module, 'weight') and module.weight.grad is not None:
                        current_import = torch.abs(module.weight * module.weight.grad).detach().cpu()
                        if importance_stats[name]["sum"] is None:
                            importance_stats[name]["sum"] = current_import
                        else:
                            importance_stats[name]["sum"] += current_import
                        importance_stats[name]["count"] += 1
                
                model.zero_grad()
    
        # 计算平均重要性
        averaged_importance = {}
        for name, stats in importance_stats.items():
            if stats["sum"] is not None and stats["count"] > 0:
                averaged_importance[name] = stats["sum"] / stats["count"]

        torch.save(averaged_importance, cache_path)
        del model, processor, importance_stats, averaged_importance
        gc.collect() 
        torch.cuda.empty_cache()

    def stage1_cache_all_activations(self):
        """阶段一：为所有模型分别缓存所需的激活。"""
        print("\n--- [阶段一: 缓存所有重要性] ---")
        
        # 直接加载原始数据集，不预先处理
        probe_dataset_raw = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
        
        # 为每个模型单独处理数据并缓存激活
        self._cache_activations_raw("A", self.args.base_model_path, "input_output", probe_dataset_raw)
        self._cache_activations_raw("B", self.args.donor_model_path, "input_output", probe_dataset_raw)
        self._cache_activations_raw("C", self.args.original_model_path, "output", probe_dataset_raw)

    def stage2_analyze_neurons_and_get_masks(self):
        """阶段二：计算近似SNIP并生成非冲突掩码。"""
        print("\n--- [阶段二: 神经元分析与选举] ---")
        mask_cache_path = os.path.join(self.cache_dir, f"non_conflict_masks_r{self.args.top_k_ratio}.pt")
        if os.path.exists(mask_cache_path) and not self.args.force_recompute:
            print("非冲突掩码缓存文件已存在, 跳过。")
            return

        print("加载所有权重和缓存的重要性...")
        weights_A = load_weights(self.args.base_model_path)
        weights_B_raw = load_weights(self.args.donor_model_path)
        weights_C = load_weights(self.args.original_model_path)
        weights_B = normalize_llm_keys(weights_B_raw, list(weights_C.keys())); del weights_B_raw

        importance = {
            'A': torch.load(os.path.join(self.cache_dir, "importance_A.pt")),
            'B': torch.load(os.path.join(self.cache_dir, "importance_B.pt")),
            'C': torch.load(os.path.join(self.cache_dir, "importance_C.pt"))
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
                I_A = importance['A'][module_name]
                I_B = importance['B'][module_name]
                I_C = importance['C'][module_name]
                
                r = self.args.top_k_ratio
                k = int(I_A.numel() * r)
                thresh_A = torch.topk(I_A.flatten(), k=k)[0].min()
                thresh_B = torch.topk(I_B.flatten(), k=k)[0].min()
                thresh_C = torch.topk(I_C.flatten(), k=k)[0].min()
                
                mask_A = I_A >= thresh_A
                mask_B = I_B >= thresh_B
                mask_base = I_C >= thresh_C
                
                T_A = mask_A & mask_base
                T_B = mask_B & mask_base
                
                intersect_AB = T_A & T_B
                disjoint_T_A = T_A & ~intersect_AB
                disjoint_T_B = T_B & ~intersect_AB
                
                non_conflict_masks[key + '_A'] = disjoint_T_A
                non_conflict_masks[key + '_B'] = disjoint_T_B
            except KeyError:
                print(f"警告: 模块 {module_name} 的重要性数据不完整，跳过参数 {key}。")
                continue

        torch.save(non_conflict_masks, mask_cache_path)
        print(f"非冲突掩码计算完成并缓存至: {mask_cache_path}")

    def stage3_project_and_merge(self):
        """阶段三：执行分离投影合并。"""
        print("\n--- [阶段三: 非冲突合并] ---")
        
        print("加载所有权重、掩码...")
        weights_A = load_weights(self.args.base_model_path)
        weights_B_raw = load_weights(self.args.donor_model_path)
        weights_C = load_weights(self.args.original_model_path)
        weights_B = normalize_llm_keys(weights_B_raw, list(weights_C.keys())); del weights_B_raw

        mask_cache_path = os.path.join(self.cache_dir, f"non_conflict_masks_r{self.args.top_k_ratio}.pt")
        non_conflict_masks = torch.load(mask_cache_path)

        final_merged_weights = weights_C.copy()  # 以C作为基模型
        for key in tqdm(weights_A.keys(), desc="执行合并"):
            if not need_merge(key): 
                final_merged_weights[key] = weights_A[key]  # 非合并参数使用A
                continue
            
            key_in_c = key.replace("model.language_model.", "model.")
            if not (key_in_c in weights_B and key_in_c in weights_C): continue
            
            W_A, W_B, W_C = weights_A[key], weights_B[key_in_c], weights_C[key_in_c]
            tau_A = (W_A - W_C).to(self.device)
            tau_B = (W_B - W_C).to(self.device)
            
            mask_A = non_conflict_masks.get(key + '_A')
            mask_B = non_conflict_masks.get(key + '_B')
            
            if mask_A is not None and mask_B is not None:
                W_star = W_C.to(self.device).float() + 1.0 * (tau_A * mask_A.to(self.device)) + 1.0 * (tau_B * mask_B.to(self.device))
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
    parser.add_argument('--cuda_device', type=int, default=7, help="使用的 CUDA 设备编号。")

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