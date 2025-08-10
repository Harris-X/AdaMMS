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

# --- 权重加载与辅助函数 (无修改) ---
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
            
    if not ref_prefix and not norm_prefix:
       print("警告：在模型中未找到 'layers' 结构，将假定键名兼容。")
       return weights_to_norm
       
    if not norm_prefix and ref_prefix:
        print(f"警告: 贡献模型中未找到 'layers'，将尝试使用参考前缀 '{ref_prefix}' 进行对齐。")
        norm_prefix = "" 

    if not ref_prefix and norm_prefix:
       print(f"警告: 参考模型中未找到 'layers'，将尝试移除贡献模型前缀 '{norm_prefix}'。")
       ref_prefix = ""

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
    if not is_in_llm_layers:
        return False
    if not name.endswith(".weight"):
        return False
    if "layernorm" in name or "embed_tokens" in name or "norm" in name or ".inv_freq" in name:
        return False
    return True

# --- 核心实现类 ---
class DAVTMMerger:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        # 【修改】使用新的方法名命名输出目录
        self.output_dir = os.path.join("merged_models", f"davtm-{args.mode}")
        self.cache_dir = os.path.join(self.output_dir, "cache")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"使用设备: {self.device}")
        print(f"输出将保存至: {self.output_dir}")

    # --- 阶段一：激活缓存 (无修改) ---
    def _get_target_module_map(self, model):
        module_map = {}
        llm_module = None
        if hasattr(model, 'language_model'):
            llm_module = model.language_model
        elif hasattr(model, 'model'):
             llm_module = model.model
        else:
            llm_module = model

        for name, module in llm_module.named_modules():
            full_prefix = ""
            if hasattr(model, 'language_model'):
                full_prefix = f"language_model.{name}" 
            elif hasattr(model, 'model'):
                full_prefix = f"model.{name}"
            else:
                full_prefix = name

            if any(need_merge(f"{full_prefix}.{param_name}") for param_name, _ in module.named_parameters()):
                module_map[name] = module
        return module_map

    def _cache_activations_raw(self, model_info, model_path, required_activations, dataset_raw):
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
        self._cache_activations_raw("C", self.args.original_model_path, ["output"], probe_dataset_raw)

    # --- 阶段二：泰勒重要性分数计算 (逻辑不变, 仅修改名称和缓存) ---
    def stage2_importance_analysis(self):
        """阶段二：【DA-VTM】计算并缓存泰勒重要性分数 S_i。"""
        print("\n--- [阶段二: DA-VTM 重要性分数计算] ---")
        # 【修改】缓存文件名，现在只缓存分数
        score_cache_path = os.path.join(self.cache_dir, f"davtm_importance_scores_alpha{self.args.alpha}.pt")
        if os.path.exists(score_cache_path) and not self.args.force_recompute:
            print("DA-VTM 重要性分数缓存文件已存在, 跳过。")
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

        # 【修改】只为模型 B 计算重要性分数
        importance_scores = {}
        
        pbar = tqdm(weights_A.keys(), desc="【DA-VTM】计算重要性分数")
        for key in pbar:
            if not need_merge(key): continue
            if not (key in weights_B and key in weights_C): continue
                
            module_name = ".".join(key.split('.')[1:-1]) if key.startswith("model.") else ".".join(key.split('.')[2:-1])
            
            try:
                # 只需计算模型 B 的伪梯度
                delta_Y_B = activations['B'][module_name]['output'] - activations['C'][module_name]['output']
                g_approx_B = torch.outer(delta_Y_B, activations['B'][module_name]['input'])

                W_B, W_C = weights_B[key], weights_C[key]
                tau_B = W_B.float() - W_C.float()

                # 计算泰勒重要性分数 I_TAG
                itag_B = -g_approx_B * tau_B + (self.args.alpha / 2) * (g_approx_B**2) * (tau_B**2)

                # 【修改】存储分数的绝对值 S_i，并移至CPU
                importance_scores[key] = itag_B.abs().cpu()

            except KeyError:
                pbar.set_description(f"警告: 模块 {module_name} 激活数据不完整，跳过")
                continue

        torch.save(importance_scores, score_cache_path)
        print(f"DA-VTM 重要性分数计算完成并缓存至: {score_cache_path}")
        
    # ########################################################################## #
    # #                           核心代码修改区域                             # #
    # ########################################################################## #

    def stage3_damped_aligned_merge(self):
        """阶段三：【DA-VTM】执行阻尼对齐融合。"""
        print("\n--- [阶段三: DA-VTM 阻尼对齐融合] ---")
        
        print("加载所有权重和重要性分数...")
        weights_A = load_weights(self.args.base_model_path)
        weights_B_raw = load_weights(self.args.donor_model_path)
        weights_C_raw = load_weights(self.args.original_model_path)
        
        weights_B = normalize_llm_keys(weights_B_raw, list(weights_A.keys())); del weights_B_raw
        weights_C = normalize_llm_keys(weights_C_raw, list(weights_A.keys())); del weights_C_raw

        score_cache_path = os.path.join(self.cache_dir, f"davtm_importance_scores_alpha{self.args.alpha}.pt")
        importance_scores = torch.load(score_cache_path, map_location="cpu")

        final_merged_weights = weights_A.copy()
        
        pbar = tqdm(importance_scores.keys(), desc="【DA-VTM】分解并融合")
        for key in pbar:
            # 步骤 1: 加载权重和分数
            W_A, W_B, W_C = weights_A[key], weights_B[key], weights_C[key]
            S_i = importance_scores[key].to(self.device)

            # 步骤 2: 计算任务向量
            tau_A = (W_A - W_C).to(self.device).float()
            tau_B = (W_B - W_C).to(self.device).float()

            # 步骤 3: 正交投影分解
            # 3a. 计算对齐分量 tau_align
            tau_A_flat = tau_A.flatten()
            tau_B_flat = tau_B.flatten()
            
            # 计算内积
            dot_product = torch.dot(tau_B_flat, tau_A_flat)
            norm_sq_A = torch.dot(tau_A_flat, tau_A_flat) + 1e-9 # 加上epsilon防止除零
            
            # 投影操作
            proj_scalar = dot_product / norm_sq_A
            tau_align = proj_scalar * tau_A
            
            # 3b. 计算新颖分量 tau_novel
            tau_novel = tau_B - tau_align

            # 步骤 4: 对新颖分量进行阻尼
            # 4a. 创建阻尼门控 d_i
            k = int(S_i.numel() * self.args.damping_ratio)
            if k == 0:
                d_i = torch.zeros_like(S_i, dtype=torch.bool)
            else:
                threshold = torch.topk(S_i.flatten(), k=k, sorted=False)[0].min()
                d_i = S_i >= threshold
            
            # 4b. 应用阻尼
            tau_novel_damped = tau_novel * d_i

            # 步骤 5: 最终加权合并
            W_star = W_A.to(self.device).float() + self.args.lambda_align * tau_align + self.args.lambda_novel * tau_novel_damped
            final_merged_weights[key] = W_star.cpu().to(W_A.dtype)
        
        self._save_model(final_merged_weights)

    # ########################################################################## #
    # #                         核心代码修改区域结束                             # #
    # ########################################################################## #

    def _save_model(self, merged_weights):
        """保存模型权重。"""
        print("\n正在保存合并后的模型...")
        # 保存逻辑无修改
        index_path = os.path.join(self.args.base_model_path, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index_map = json.load(f)["weight_map"]
        
        sharded_weights = defaultdict(dict)
        for key, value in merged_weights.items():
            if key in index_map:
                sharded_weights[index_map[key]][key] = value
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        for filename, weights_dict in sharded_weights.items():
            safetensors.torch.save_file(weights_dict, os.path.join(self.output_dir, filename))
        
        shutil.copy(index_path, os.path.join(self.output_dir, os.path.basename(index_path)))
        for filename in os.listdir(self.args.base_model_path):
            if not filename.startswith('.') and filename.endswith(('.json', '.model', '.py', '.md')):
                source_file = os.path.join(self.args.base_model_path, filename)
                dest_file = os.path.join(self.output_dir, filename)
                if not os.path.exists(dest_file):
                       shutil.copy(source_file, dest_file)
        print(f"模型成功合并并保存至: {self.output_dir}")

    def run_pipeline(self):
        """按顺序执行所有阶段。"""
        self.stage1_cache_all_activations()
        self.stage2_importance_analysis()
        self.stage3_damped_aligned_merge()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用DA-VTM进行高效、高泛化性、易于调参的模型合并。")
    
    # 基本配置
    parser.add_argument('--base_model_path', type=str, default="./downloaded_models/Qwen2-VL-7B-Instruct", help="基础模型A的路径。")
    parser.add_argument('--donor_model_path', type=str, default="./downloaded_models/llava-onevision-qwen2-7b-si-hf", help="贡献模型B的路径。")
    parser.add_argument('--original_model_path', type=str, default="./downloaded_models/Qwen2-7B-Instruct", help="原始共同祖先模型C的路径。")
    parser.add_argument('--mode', type=str, default="davtm-default", help="为本次合并配置命名。")
    parser.add_argument('--cuda_device', type=int, default=5, help="使用的 CUDA 设备编号。")

    # 数据集配置
    parser.add_argument('--probe_samples', type=int, default=100, help="用于引导合并的目标域样本数量。")
    parser.add_argument('--probe_batch_size', type=int, default=2, help="处理引导数据时的批处理大小。")

    # DA-VTM 合并超参数
    parser.add_argument('--alpha', type=float, default=1.0, help="【阶段二】平衡泰勒展开一阶和二阶项的信任域参数。")
    parser.add_argument('--damping_ratio', type=float, default=0.2, help="【阶段三】用于创建阻尼门控的Top-K比率 (即r)。")
    parser.add_argument('--lambda_align', type=float, default=1.0, help="【阶段三】对齐(Aligned)知识分量的融合系数 (建议固定为1.0)。")
    parser.add_argument('--lambda_novel', type=float, default=0.5, help="【阶段三】阻尼后新颖(Novel)知识分量的融合系数 (核心可调参数)。")
    
    # 功能性参数
    parser.add_argument('--force_recompute', action='store_true', help="强制重新计算缓存的激活或分数。")

    args = parser.parse_args()
    
    device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"

    print("--- 配置信息 ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("--------------------")

    merger = DAVTMMerger(args, device)
    merger.run_pipeline()