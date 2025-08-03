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
from torch.utils.data import DataLoader, TensorDataset, Dataset

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
    """
    通用函数，用于将任何模型的LLM部分键名与参考键名对齐。
    """
    # 动态检测两个模型LLM部分的共同父模块
    # 例如, A是 "model.language_model.layers...", B是 "model.language_model.layers..."
    # C是 "model.layers..."
    key_map = {}
    
    # 找到参考模型（通常是模型C）的layers前缀
    ref_prefix = ""
    for key in reference_keys:
        if "layers" in key:
            ref_prefix = key.split("layers")[0]
            break
            
    # 找到待标准化模型的layers前缀
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
            # 将 B 的前缀替换为 C 的前缀
            new_key = ref_prefix + key[len(norm_prefix):]
            normalized_weights[new_key] = value
        else:
            # 保留非LLM部分的权重
            normalized_weights[key] = value
            
    return normalized_weights

def need_merge(name: str) -> bool:
    """根据层名判断是否需要合并 - 仅合并LLM的特定层"""
    # MODIFIED: 更精确地定位到可合并的LLM层
    if "language_model.layers" not in name:
        return False
    
    # 排除所有 layernorm 和 embedding
    if "layernorm" in name or "embed_tokens" in name or "norm" in name:
        return False
        
    if name.endswith(".inv_freq"):
        return False
        
    # 我们将合并所有线性层和MLP层
    return True

# --- 数据集处理函数 ---
class VQAv2ProbeDataset(Dataset):
    def __init__(self, hf_dataset, max_samples=100):
        self.samples = []
        for item in hf_dataset:
            self.samples.append({
                "image": item["image"],
                "text": item["question"]
            })
            if len(self.samples) >= max_samples:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = item["image"]
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        return {"image": image, "text": item['text']}

def collate_fn_factory(processor):
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        texts = [f"Question: {item['text']} Answer:" for item in batch]
        
        # 使用 apply_chat_template 来为每个样本构建正确的输入格式
        messages_batch = [
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
            for text in texts
        ]
        
        prompt_batch = [
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_batch
        ]

        inputs = processor(text=prompt_batch, images=images, return_tensors="pt", padding=True)
        return inputs
    return collate_fn

# --- 核心实现类 ---
class LEDTTDPMerger:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.output_dir = os.path.join("merged_models", f"led-ttdpm-{args.mode}")
        self.cache_dir = os.path.join(self.output_dir, "cache")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        print(f"使用设备: {self.device}")
        print(f"输出将保存至: {self.output_dir}")
        
        self.conflict_masks = {}

    # NEW: 准备阶段 - 识别并缓存冲突神经元
    def prepare_conflict_sets(self):
        """
        离线准备阶段：计算任务向量并识别冲突神经元。
        """
        conflict_cache_path = os.path.join(self.cache_dir, f"conflict_masks_r{self.args.conflict_ratio}.pt")
        if os.path.exists(conflict_cache_path) and not self.args.force_recompute:
            print(f"冲突掩码缓存文件已存在: {conflict_cache_path}, 正在加载。")
            self.conflict_masks = torch.load(conflict_cache_path, map_location="cpu")
            return

        print("\n--- [准备阶段: 识别冲突神经元] ---")
        print("正在加载所有模型权重用于冲突分析...")
        weights_A = load_weights(self.args.base_model_path)
        weights_B_raw = load_weights(self.args.donor_model_path)
        weights_C = load_weights(self.args.original_model_path)
        
        # 标准化模型B的键名以匹配模型C
        weights_B = normalize_llm_keys(weights_B_raw, list(weights_C.keys()))
        del weights_B_raw

        print("正在计算任务向量并识别冲突集...")
        for key in tqdm(weights_A.keys(), desc="分析冲突"):
            if not need_merge(key):
                continue
                
            # 从模型A的键名构造模型C中对应的键名
            key_in_c = key.replace("model.language_model.", "model.")

            if key_in_c in weights_B and key_in_c in weights_C:
                W_A, W_B, W_C = weights_A[key], weights_B[key_in_c], weights_C[key_in_c]
                
                if W_A.shape != W_B.shape or W_A.shape != W_C.shape:
                    continue

                tau_A = W_A - W_C
                tau_B = W_B - W_C
                
                # Location & Election (Heuristic)
                k_A = int(tau_A.numel() * self.args.conflict_ratio)
                topk_A_vals, _ = torch.topk(tau_A.abs().flatten(), k=k_A)
                mask_A = tau_A.abs() >= topk_A_vals.min()

                k_B = int(tau_B.numel() * self.args.conflict_ratio)
                topk_B_vals, _ = torch.topk(tau_B.abs().flatten(), k=k_B)
                mask_B = tau_B.abs() >= topk_B_vals.min()
                
                # Disjoint: 识别冲突
                conflict_mask = (mask_A & mask_B) & (torch.sign(tau_A) != torch.sign(tau_B))
                self.conflict_masks[key] = conflict_mask.cpu()

        print(f"冲突分析完成，共为 {len(self.conflict_masks)} 个参数层计算了掩码。")
        torch.save(self.conflict_masks, conflict_cache_path)
        print(f"冲突掩码已缓存至: {conflict_cache_path}")
        
        del weights_A, weights_B, weights_C
        gc.collect()

    # MODIFIED: 核心逻辑 - 动态合并与平均
    def dynamic_merge_and_average(self):
        """
        在线模拟阶段：在探针数据上模拟动态合并过程，并对结果进行平均。
        """
        print("\n--- [核心阶段: 动态合并与平均] ---")
        
        # --- 1. 加载所有权重到CPU ---
        print("正在加载所有模型权重到CPU...")
        weights_A = load_weights(self.args.base_model_path)
        weights_B_raw = load_weights(self.args.donor_model_path)
        weights_C = load_weights(self.args.original_model_path)
        weights_B = normalize_llm_keys(weights_B_raw, list(weights_C.keys()))
        del weights_B_raw
        
        # --- 2. 加载模型A到GPU用于前向传播 ---
        print(f"正在加载基础模型 {self.args.base_model_path} 到 {self.device}...")
        model_A = AutoModelForVision2Seq.from_pretrained(
            self.args.base_model_path, torch_dtype=torch.bfloat16
        ).to(self.device)
        processor = AutoProcessor.from_pretrained(self.args.base_model_path)
        model_A.eval()

        # --- 3. 准备探针数据集 ---
        print("正在准备探针数据集 (VQAv2)...")
        probe_dataset_raw = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
        probe_dataset = VQAv2ProbeDataset(probe_dataset_raw, max_samples=self.args.probe_samples)
        collate_function = collate_fn_factory(processor)
        probe_dataloader = DataLoader(probe_dataset, batch_size=self.args.probe_batch_size, collate_fn=collate_function)

        # --- 4. 初始化合并结果和累加器 ---
        final_merged_weights = weights_A.copy()
        # 用于累加每个批次产生的权重 *变化量*
        delta_accumulators = {key: torch.zeros_like(weights_A[key], dtype=torch.float32) for key in self.conflict_masks.keys()}
        batch_count = 0

        # --- 5. 遍历探针数据，模拟动态合并 ---
        for batch in tqdm(probe_dataloader, desc="模拟动态合并"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 捕获当前批次的动态方向向量 (输入激活)
            captured_activations = {}
            hooks = []
            
            def get_hook(name):
                def hook_fn(module, input_tensor, output):
                    # 输入可能是元组，取第一个张量
                    act = input_tensor[0] if isinstance(input_tensor, tuple) else input_tensor
                    captured_activations[name] = act.detach()
                return hook_fn

            for name, module in model_A.model.language_model.named_modules():
                # 确保只 hook 需要合并的层
                full_name = f"model.language_model.{name}.weight"
                if any(k.startswith(f"model.language_model.{name}") for k in self.conflict_masks.keys()):
                     hooks.append(module.register_forward_hook(get_hook(name)))

            with torch.no_grad():
                model_A(**batch)
            
            for h in hooks: h.remove()
            
            # 对每个层执行动态分离投影合并
            for key, conflict_mask in self.conflict_masks.items():
                module_name = ".".join(key.split('.')[2:-1]) # e.g., layers.0.mlp
                if module_name not in captured_activations:
                    continue

                direction_vector = captured_activations[module_name]
                # 将方向向量从 [batch, seq, dim] 扁平化为 [batch*seq, dim]
                # 并取平均得到一个方向 [dim]
                d_i = torch.mean(direction_vector.float().view(-1, direction_vector.shape[-1]), dim=0)
                
                # 加载权重 (CPU -> GPU)
                key_in_c = key.replace("model.language_model.", "model.")
                W_A = weights_A[key].float().to(self.device)
                W_B = weights_B[key_in_c].float().to(self.device)
                W_C = weights_C[key_in_c].float().to(self.device)
                
                # LED-TTDPM 计算
                tau_B = W_B - W_C
                conflict_mask_gpu = conflict_mask.to(self.device)
                
                tau_B_nonconflict = tau_B.where(~conflict_mask_gpu, torch.tensor(0.0, device=self.device))

                # 投影
                d_i_norm_sq = torch.sum(d_i * d_i)
                if d_i_norm_sq > 1e-9:
                    # 这里的投影需要在参数空间进行，但方向来自激活空间
                    # 我们需要将参数张量视为向量集合进行操作
                    # 对于FC层权重 [out_features, in_features], 我们将其视为 out_features 个 in_features 维的向量
                    # 方向向量 d_i 也是 in_features 维
                    proj_scalar = (tau_B_nonconflict @ d_i) / d_i_norm_sq
                    tau_B_proj = torch.outer(proj_scalar, d_i)
                    tau_B_ortho = tau_B_nonconflict - tau_B_proj
                else:
                    tau_B_proj = torch.zeros_like(tau_B)
                    tau_B_ortho = tau_B_nonconflict

                # 削弱A中的冲突知识
                tau_A = W_A - W_C
                conflict_reduction = tau_A * conflict_mask_gpu

                # 合并
                W_star = (W_A + 
                          self.args.lambda_proj * tau_B_proj + 
                          self.args.lambda_ortho * tau_B_ortho -
                          self.args.lambda_conflict * conflict_reduction)
                
                # 累加变化量
                delta = W_star - W_A
                delta_accumulators[key] += delta.cpu()

            batch_count += 1
            
        del model_A, processor
        gc.collect()
        torch.cuda.empty_cache()

        # --- 6. 计算平均权重并更新 ---
        print("正在计算最终平均权重...")
        for key, acc_delta in delta_accumulators.items():
            if batch_count > 0:
                avg_delta = acc_delta / batch_count
                final_merged_weights[key] = (weights_A[key].float() + avg_delta).to(weights_A[key].dtype)
        
        # --- 7. 保存最终模型 ---
        self.save_model(final_merged_weights)
        
    def save_model(self, merged_weights):
        """保存模型权重。"""
        print("\n正在保存合并后的模型...")
        index_path = os.path.join(self.args.base_model_path, "model.safetensors.index.json")
        try:
            with open(index_path, "r") as f:
                index_map = json.load(f)["weight_map"]
        except FileNotFoundError:
            print("未找到索引文件，将保存为单个 model.safetensors 文件。")
            safetensors.torch.save_file(merged_weights, os.path.join(self.output_dir, "model.safetensors"))
            return

        sharded_weights = defaultdict(dict)
        for key, value in merged_weights.items():
            if key in index_map:
                sharded_weights[index_map[key]][key] = value
        
        for filename, weights_dict in sharded_weights.items():
            safetensors.torch.save_file(weights_dict, os.path.join(self.output_dir, filename))
        
        shutil.copy(index_path, os.path.join(self.output_dir, os.path.basename(index_path)))
        # MODIFIED: create_soft_link is not defined in the template, assuming it's a helper
        # to copy tokenizer and config files. A robust copy is better.
        for filename in os.listdir(self.args.base_model_path):
            if filename.endswith(('.json', '.model', '.py')):
                source_file = os.path.join(self.args.base_model_path, filename)
                dest_file = os.path.join(self.output_dir, filename)
                if not os.path.exists(dest_file):
                     shutil.copy(source_file, dest_file)

        print(f"模型成功合并并保存至: {self.output_dir}")

    def run_pipeline(self):
        """按顺序执行所有阶段。"""
        self.prepare_conflict_sets()
        self.dynamic_merge_and_average()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 LED-TTDPM 进行无需微调的测试时自适应模型合并。")
    
    # 基本配置
    parser.add_argument('--base_model_path', type=str, default="./downloaded_models/Qwen2-VL-7B-Instruct", help="基础模型A (Qwen2-VL) 的路径。")
    parser.add_argument('--donor_model_path', type=str, default="./downloaded_models/llava-onevision-qwen2-7b-si-hf", help="贡献模型B (Llava-Onevision) 的路径。")
    parser.add_argument('--original_model_path', type=str, default="./downloaded_models/Qwen2-7B-Instruct", help="原始共同祖先模型C (Qwen2) 的路径。")
    parser.add_argument('--mode', type=str, default="default", help="为本次合并配置命名。")
    parser.add_argument('--cuda_device', type=int, default=0, help="使用的 CUDA 设备编号。")

    # 探针数据集配置
    parser.add_argument('--probe_samples', type=int, default=100, help="用于模拟动态合并的样本数量。")
    parser.add_argument('--probe_batch_size', type=int, default=2, help="模拟时的批处理大小。")

    # NEW: LED-TTDPM 合并超参数
    parser.add_argument('--conflict_ratio', type=float, default=0.1, help="用于识别潜在冲突神经元的Top-k比率。")
    parser.add_argument('--lambda_proj', type=float, default=1.0, help="相关分量的系数。")
    parser.add_argument('--lambda_ortho', type=float, default=0.5, help="正交分量的系数。")
    parser.add_argument('--lambda_conflict', type=float, default=0.1, help="冲突削弱的系数。")
    
    # 功能性参数
    parser.add_argument('--force_recompute', action='store_true', help="强制重新计算冲突掩码。")

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

    merger = LEDTTDPMerger(args, device)
    merger.run_pipeline()