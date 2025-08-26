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
    # 确定参考前缀 (例如 'model.layers' 或 'language_model.model.layers')
    for key in reference_keys:
        if "layers" in key and "language_model" in key:
            ref_prefix = key.split("layers")[0]
            break
    
    # 如果没找到特定的VL前缀，就用通用的
    if not ref_prefix:
        for key in reference_keys:
            if "layers" in key:
                ref_prefix = key.split("layers")[0]
                break

    norm_prefix = ""
    # 确定待标准化模型的前缀
    for key in weights_to_norm.keys():
        if "layers" in key and "language_model" in key:
            norm_prefix = key.split("layers")[0]
            break
            
    if not norm_prefix:
        for key in weights_to_norm.keys():
            if "layers" in key:
                norm_prefix = key.split("layers")[0]
                break
                
    if not ref_prefix or not norm_prefix:
        print("警告：无法在模型中定位到 'layers'，键名标准化可能失败。")
        return weights_to_norm
        
    if ref_prefix == norm_prefix:
        print("模型键名前缀已对齐，无需标准化。")
        return weights_to_norm

    print(f"检测到键名前缀不匹配。参考: '{ref_prefix}', 待标准化: '{norm_prefix}'. 正在进行标准化...")

    normalized_weights = {}
    for key, value in weights_to_norm.items():
        if key.startswith(norm_prefix):
            new_key = ref_prefix + key[len(norm_prefix):]
            normalized_weights[new_key] = value
        else:
            normalized_weights[key] = value # 保留非LLM部分的权重
    return normalized_weights

def get_llm_layer_prefix(model_keys):
    """从模型权重键集合中动态推断LLM层的前缀。"""
    for key in model_keys:
        if key.endswith("layers.0.self_attn.q_proj.weight"):
            return key.rsplit("layers.0.self_attn.q_proj.weight", 1)[0]
    # 备用查找
    for key in model_keys:
        if "layers.0" in key:
            return key.split("layers.0")[0]
    return "" # 未找到

def need_merge(name: str, llm_prefix: str) -> bool:
    """根据层名判断是否需要合并 - 仅合并LLM的权重参数。"""
    if not name.startswith(llm_prefix):
        return False
    
    if not name.endswith(".weight"): # 只处理权重，忽略偏置等
        return False
        
    # 排除各种归一化层和嵌入层
    if "layernorm" in name or "embed_tokens" in name or "norm" in name or ".inv_freq" in name or "lm_head" in name:
        return False
        
    return True

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
# NEW: 类名更新为 DSDAGIDPMMerger 以反映新方法
class DSDAGIDPMMerger:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        # NEW: 输出目录反映 DSD 方法
        self.output_dir = os.path.join("merged_models", f"dsd-agidpm-{args.mode}")
        self.cache_dir = os.path.join(self.output_dir, "cache")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"使用设备: {self.device}")
        print(f"输出将保存至: {self.output_dir}")

    def _get_target_module_map(self, model, llm_prefix):
        """获取需要hook的模块名到模块实例的映射。"""
        module_map = {}
        # 移除末尾的点，以正确匹配模块名
        prefix_to_match = llm_prefix.rstrip('.')
        
        for name, module in model.named_modules():
            # 我们需要hook的是包含可合并参数的模块
            # 例如，如果 `language_model.model.layers.0.self_attn.q_proj.weight` 需要合并,
            # 那么我们需要 hook `language_model.model.layers.0.self_attn.q_proj`
            if name.startswith(prefix_to_match) and not list(module.children()):
                if any(p.requires_grad for p in module.parameters()):
                     module_map[name] = module
        return module_map

    def _cache_activations_raw(self, model_info, model_path, required_activations, dataset_raw):
        """为每个模型从原始数据集处理数据并缓存激活（内存优化版）。"""
        cache_path = os.path.join(self.cache_dir, f"activations_{model_info}.pt")
        if os.path.exists(cache_path) and not self.args.force_recompute:
            print(f"激活缓存文件 {cache_path} 已存在, 跳过。")
            return torch.load(cache_path, map_location="cpu")

        print(f"正在为 {model_info} ({os.path.basename(model_path)}) 缓存激活...")
        # 动态判断模型类型
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        is_vision_model = 'vision_config' in config or any("llava" in arch for arch in config.get("architectures", []))

        ModelClass = AutoModelForVision2Seq if is_vision_model else AutoModelForCausalLM
        model = ModelClass.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(self.device)
        
        if is_vision_model:
            processor = AutoProcessor.from_pretrained(model_path)
        else:
            processor = AutoTokenizer.from_pretrained(model_path)
        
        model.eval()
        
        # 动态获取LLM部分
        llm_prefix = get_llm_layer_prefix(model.state_dict().keys())
        model_to_hook = model

        target_modules = self._get_target_module_map(model_to_hook, llm_prefix)

        activation_stats = defaultdict(lambda: {
            "input_sum": None, "input_count": 0,
            "output_sum": None, "output_count": 0
        })

        def get_hook_with_kwargs(name, req_act):
            def hook_fn(module, args, kwargs, output):
                with torch.no_grad():
                    # 处理输出
                    if "output" in req_act:
                        out_tensor = output[0] if isinstance(output, tuple) else output
                        if isinstance(out_tensor, torch.Tensor):
                            t_float = out_tensor.cpu().float()
                            current_sum = torch.sum(t_float, dim=tuple(range(t_float.ndim - 1)))
                            if activation_stats[name]["output_sum"] is None:
                                activation_stats[name]["output_sum"] = current_sum
                            else:
                                activation_stats[name]["output_sum"] += current_sum
                            activation_stats[name]["output_count"] += t_float.shape[0] * (t_float.nelement() // t_float.shape[-1] // t_float.shape[0]) if t_float.ndim > 2 else t_float.shape[0]
                    # 处理输入
                    if "input" in req_act:
                        in_tensor = kwargs.get("hidden_states", args[0] if args else None)
                        if isinstance(in_tensor, torch.Tensor):
                            t_float = in_tensor.cpu().float()
                            current_sum = torch.sum(t_float, dim=tuple(range(t_float.ndim - 1)))
                            if activation_stats[name]["input_sum"] is None:
                                activation_stats[name]["input_sum"] = current_sum
                            else:
                                activation_stats[name]["input_sum"] += current_sum
                            activation_stats[name]["input_count"] += t_float.shape[0] * (t_float.nelement() // t_float.shape[-1] // t_float.shape[0]) if t_float.ndim > 2 else t_float.shape[0]

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
                if is_vision_model:
                    images = [item["image"] for item in batch_data]
                    texts = [item["text"] for item in batch_data]
                    # This template is generic for Qwen-VL style
                    messages_batch = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}] for text in texts]
                    prompt_batch = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
                    inputs = processor(text=prompt_batch, images=images, return_tensors="pt", padding=True)
                else: # LLM only
                    texts = [item["text"] for item in batch_data]
                    # Use a generic chat template
                    formatted_texts = [processor.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True) for text in texts]
                    inputs = processor(text=formatted_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                model(**inputs)

        for h in hooks: h.remove()

        averaged_activations = {}
        for name, stats in activation_stats.items():
            averaged_activations[name] = {}
            if stats["input_sum"] is not None and stats["input_count"] > 0:
                averaged_activations[name]["input"] = stats["input_sum"] / stats["input_count"]
            if stats["output_sum"] is not None and stats["output_count"] > 0:
                averaged_activations[name]["output"] = stats["output_sum"] / stats["output_count"]
        
        torch.save(averaged_activations, cache_path)
        del model, processor, hooks, activation_stats, averaged_activations
        gc.collect() 
        torch.cuda.empty_cache()

    def stage1_cache_all_activations(self):
        """阶段一：为所有模型分别缓存所需的激活。"""
        print("\n--- [阶段一: 缓存所有激活] ---")
        # 直接加载原始数据集，不预先处理
        probe_dataset_raw = load_dataset("HuggingFaceM4/VQAv2", split="validation", streaming=True)
        # 为每个模型单独处理数据并缓存激活
        self._cache_activations_raw("A", self.args.base_model_path, "input_output", probe_dataset_raw)
        self._cache_activations_raw("B", self.args.donor_model_path, "input_output", probe_dataset_raw)
        self._cache_activations_raw("C", self.args.original_model_path, "output", probe_dataset_raw)

    def stage2_analyze_neurons_and_get_masks(self):
        """阶段二：计算近似SNIP并生成包含方向敏感分析的掩码。"""
        print("\n--- [阶段二: DSD神经元分析] ---")
        # NEW: 缓存文件名更具体
        analysis_cache_path = os.path.join(self.cache_dir, f"dsd_analysis_r{self.args.top_k_ratio}.pt")
        if os.path.exists(analysis_cache_path) and not self.args.force_recompute:
            print("DSD分析缓存文件已存在, 跳过。")
            return

        print("加载所有权重和缓存的激活...")
        weights_A = load_weights(self.args.base_model_path)
        weights_B_raw = load_weights(self.args.donor_model_path)
        weights_C = load_weights(self.args.original_model_path)
        
        # 键名标准化
        weights_B = normalize_llm_keys(weights_B_raw, list(weights_A.keys()))
        del weights_B_raw
        
        llm_prefix = get_llm_layer_prefix(weights_A.keys())
        print(f"推断出的LLM层前缀为: '{llm_prefix}'")

        activations = {
            'A': torch.load(os.path.join(self.cache_dir, "activations_A.pt")),
            'B': torch.load(os.path.join(self.cache_dir, "activations_B.pt")),
            'C': torch.load(os.path.join(self.cache_dir, "activations_C.pt"))
        }

        # NEW: 存储更丰富的分析结果
        analysis_results = {}
        for key in tqdm(weights_A.keys(), desc="DSD分析神经元"):
            if not need_merge(key, llm_prefix): continue

            module_name = ".".join(key.split('.')[:-1])
            try:
                # 加载权重
                W_A, W_B, W_C = weights_A[key], weights_B[key], weights_C[key]

                # 计算近似梯度 (g' ≈ (ΔY)ᵀ @ X)
                # 由于我们处理的是平均激活（向量），所以外积(outer product)是合适的
                delta_Y_A = activations['A'][module_name]['output'] - activations['C'][module_name]['output']
                g_approx_A = torch.outer(delta_Y_A, activations['A'][module_name]['input'])
                
                delta_Y_B = activations['B'][module_name]['output'] - activations['C'][module_name]['output']
                g_approx_B = torch.outer(delta_Y_B, activations['B'][module_name]['input'])

                # 计算近似SNIP分数
                snip_A = (W_A.float() * g_approx_A).abs()
                snip_B = (W_B.float() * g_approx_B).abs()
                
                # 选举与方向敏感分离
                k = int(snip_A.numel() * self.args.top_k_ratio)
                if k == 0: continue # 避免topk k=0的错误

                mask_A = snip_A >= torch.kthvalue(snip_A.flatten(), snip_A.numel() - k + 1).values
                mask_B = snip_B >= torch.kthvalue(snip_B.flatten(), snip_B.numel() - k + 1).values
                
                tau_A, tau_B = W_A - W_C, W_B - W_C

                # NEW: 显式计算重叠、冲突和一致性掩码
                overlap_mask = mask_A & mask_B
                conflict_mask = overlap_mask & (torch.sign(tau_A) != torch.sign(tau_B))
                consistent_mask = overlap_mask & (torch.sign(tau_A) == torch.sign(tau_B))

                # 模型B的最终掩码：保留其所有被选举的神经元，除了那些与A冲突的
                final_mask_B = mask_B & (~conflict_mask)
                
                analysis_results[key] = {
                    'mask_B': final_mask_B.cpu(),
                    'consistent_mask': consistent_mask.cpu()
                }

            except KeyError as e:
                # print(f"警告: 模块 {module_name} (对应参数 {key}) 的激活数据缺失: {e}，跳过。")
                continue
            except RuntimeError as e:
                print(f"运行时错误于 {key}: {e}")
                continue

        torch.save(analysis_results, analysis_cache_path)
        print(f"DSD分析完成并缓存至: {analysis_cache_path}")

    def stage3_project_and_merge(self):
        """阶段三：执行包含加权融合的分离投影合并。"""
        print("\n--- [阶段三: DSD 分离投影合并] ---")
        print("加载所有权重、分析结果和方向向量...")
        weights_A = load_weights(self.args.base_model_path)
        weights_B_raw = load_weights(self.args.donor_model_path)
        weights_C = load_weights(self.args.original_model_path)
        
        weights_B = normalize_llm_keys(weights_B_raw, list(weights_A.keys()))
        del weights_B_raw

        analysis_cache_path = os.path.join(self.cache_dir, f"dsd_analysis_r{self.args.top_k_ratio}.pt")
        analysis_results = torch.load(analysis_cache_path)
        
        activations_A = torch.load(os.path.join(self.cache_dir, "activations_A.pt"))
        
        final_merged_weights = weights_A.copy()

        for key, analysis_data in tqdm(analysis_results.items(), desc="执行DSD合并"):
            module_name = ".".join(key.split('.')[:-1])
            
            # 解包分析数据
            mask_B = analysis_data['mask_B'].to(self.device)
            consistent_mask = analysis_data['consistent_mask'].to(self.device)

            W_A, W_B, W_C = weights_A[key], weights_B[key], weights_C[key]
            
            # 计算任务向量
            tau_A = (W_A - W_C).to(self.device).float()
            tau_B = (W_B - W_C).to(self.device).float()
            
            # 应用分离掩码，得到初步的任务向量
            tau_B_disjoint = tau_B * mask_B

            # NEW: 对一致性神经元执行加权融合
            if torch.any(consistent_mask):
                # 计算幅度加权的融合任务向量
                tau_A_abs = tau_A.abs()
                tau_B_abs = tau_B.abs()
                
                numerator = tau_A_abs * tau_A + tau_B_abs * tau_B
                denominator = tau_A_abs + tau_B_abs
                
                # 避免除以零
                tau_fused = numerator / (denominator + 1e-9)

                # 将融合后的值更新到任务向量中
                tau_B_disjoint = torch.where(consistent_mask, tau_fused, tau_B_disjoint)

            # --- 投影对齐部分保持不变 ---
            try:
                d_i = activations_A[module_name]['input'].to(self.device) # 投影方向
                d_i_norm_sq = torch.sum(d_i * d_i)

                if d_i_norm_sq > 1e-9:
                    # 投影操作: 将任务向量的每一行投影到方向向量d_i上
                    # tau_B_disjoint @ d_i 计算了每个输出神经元对应权重行与d_i的点积
                    proj_scalar = (tau_B_disjoint @ d_i) / d_i_norm_sq
                    tau_B_proj = torch.outer(proj_scalar, d_i)
                    tau_B_ortho = tau_B_disjoint - tau_B_proj
                else:
                    tau_B_proj = torch.zeros_like(tau_B_disjoint)
                    tau_B_ortho = tau_B_disjoint
            except KeyError:
                # 如果没有激活（例如对于某些非标准层），则不进行投影
                tau_B_proj = torch.zeros_like(tau_B_disjoint)
                tau_B_ortho = tau_B_disjoint

            # 最终合并
            W_star = W_A.to(self.device).float() + self.args.lambda_proj * tau_B_proj + self.args.lambda_ortho * tau_B_ortho
            final_merged_weights[key] = W_star.cpu().to(W_A.dtype)

        # 保存模型
        self._save_model(final_merged_weights)
        
    def _save_model(self, merged_weights):
        """保存模型权重及配置文件。"""
        print("\n正在保存合并后的模型...")
        # 尝试从基础模型加载索引
        try:
            index_path = os.path.join(self.args.base_model_path, "model.safetensors.index.json")
            with open(index_path, "r") as f:
                index_map = json.load(f)["weight_map"]
            
            sharded_weights = defaultdict(dict)
            for key, value in merged_weights.items():
                if key in index_map:
                    sharded_weights[index_map[key]][key] = value
                else:
                    # 如果某个键（例如新加的）不在索引中，放入第一个分片
                    first_shard_file = next(iter(index_map.values()))
                    sharded_weights[first_shard_file][key] = value

            # 保存分片
            for filename, weights_dict in tqdm(sharded_weights.items(), desc="保存分片"):
                safetensors.torch.save_file(weights_dict, os.path.join(self.output_dir, filename))
            
            # 拷贝索引文件
            shutil.copy(index_path, os.path.join(self.output_dir, os.path.basename(index_path)))

        except FileNotFoundError:
            # 如果没有索引文件，说明是单个文件模型
            print("未找到 model.safetensors.index.json，将以单个文件形式保存。")
            safetensors.torch.save_file(merged_weights, os.path.join(self.output_dir, "model.safetensors"))

        # 拷贝所有配置文件
        for filename in os.listdir(self.args.base_model_path):
            source_file = os.path.join(self.args.base_model_path, filename)
            dest_file = os.path.join(self.output_dir, filename)
            if filename.endswith(('.json', '.model', '.py', '.md')) and not os.path.isdir(source_file):
                if not os.path.exists(dest_file):
                     shutil.copy(source_file, dest_file)
                     
        print(f"模型成功合并并保存至: {self.output_dir}")

    def run_pipeline(self):
        """按顺序执行所有阶段。"""
        self.stage1_cache_all_activations()
        self.stage2_analyze_neurons_and_get_masks()
        self.stage3_project_and_merge()

if __name__ == "__main__":
    # MODIFIED: 更新描述以反映 DSD-AGIDPM 方法
    parser = argparse.ArgumentParser(description="使用DSD-AGIDPM进行高效、方向敏感的模型合并。")
    # 基本配置
    parser.add_argument('--base_model_path', type=str, default="./downloaded_models/Qwen2-VL-7B-Instruct", help="基础模型A的路径。")
    parser.add_argument('--donor_model_path', type=str, default="./downloaded_models/llava-onevision-qwen2-7b-si-hf", help="贡献模型B的路径。")
    parser.add_argument('--original_model_path', type=str, default="./downloaded_models/Qwen2-7B-Instruct", help="原始共同祖先模型C的路径。")
    parser.add_argument('--mode', type=str, default="default-run", help="为本次合并配置命名，用于生成输出目录。")
    parser.add_argument('--cuda_device', type=int, default=7, help="使用的 CUDA 设备编号。")
    # 数据集配置
    parser.add_argument('--probe_samples', type=int, default=100, help="用于引导合并的目标域样本数量。")
    parser.add_argument('--probe_batch_size', type=int, default=2, help="处理引导数据时的批处理大小。")
    # DSD-AGIDPM 合并超参数
    parser.add_argument('--top_k_ratio', type=float, default=0.2, help="用于选举关键神经元的Top-K比率。")
    parser.add_argument('--lambda_proj', type=float, default=1.0, help="投影（相关）分量的系数。")
    parser.add_argument('--lambda_ortho', type=float, default=0.4, help="正交（无关）分量的系数。")
    # 功能性参数
    parser.add_argument('--force_recompute', action='store_true', help="强制重新计算缓存的激活或分析结果。")
    
    args = parser.parse_args()
    
    device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    
    print("--- 配置信息 ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("--------------------")
    
    # MODIFIED: 实例化新的 Merger 类
    merger = DSDAGIDPMMerger(args, device)
    merger.run_pipeline()