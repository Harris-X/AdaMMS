import os
import sys
import json
import safetensors
import torch
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
            normalized_weights[key] = value
    return normalized_weights

def need_merge(name: str) -> bool:
    """根据层名判断是否需要合并 - 仅合并LLM的特定层"""
    if "language_model.layers" not in name:
        return False
    if "layernorm" in name or "embed_tokens" in name or "norm" in name:
        return False
    if name.endswith(".inv_freq"):
        return False
    return True

# --- 数据集处理函数 (保持不变) ---
class VQAv2TargetDataset(Dataset):
    def __init__(self, hf_dataset, max_samples=100):
        self.samples = []
        # MODIFIED: 确保加载的是训练集以获取标签
        for item in hf_dataset:
            # 过滤掉没有有效答案的样本
            if 'multiple_choice_answer' in item:
                self.samples.append({
                    "image": item["image"],
                    "text": item["question"],
                    "label": item["multiple_choice_answer"] 
                })
            if len(self.samples) >= max_samples:
                break
        print(f"已加载 {len(self.samples)} 个目标域样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = item["image"]
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        return {"image": image, "text": item['text'], "label": item['label']}

def collate_fn_factory(processor):
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        texts = [f"Question: {item['text']} Answer:" for item in batch]
        labels = [item["label"] for item in batch] # 标签也一并处理
        
        messages_batch = [
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
            for text in texts
        ]
        
        prompt_batch = [
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_batch
        ]

        inputs = processor(text=prompt_batch, images=images, return_tensors="pt", padding=True)
        
        # 将标签也处理并加入batch
        label_inputs = processor.tokenizer(labels, padding=True, return_tensors='pt')
        inputs['labels'] = label_inputs.input_ids
        
        return inputs
    return collate_fn


# --- 核心实现类 ---
# MODIFIED: 类名更改以反映新方法
class SampleGuidedMerger:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.output_dir = os.path.join("merged_models", f"sg-dpm-{args.mode}")
        self.cache_dir = os.path.join(self.output_dir, "cache")
        # MODIFIED: 存储的是“方向”，而不是“梯度”
        self.direction_dir = os.path.join(self.cache_dir, "target_directions")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.direction_dir, exist_ok=True)
        
        print(f"使用设备: {self.device}")
        print(f"输出将保存至: {self.output_dir}")

    def _get_target_modules(self, model):
        """获取模型中所有需要被hook的目标模块的名称。"""
        target_module_names = set()
        # MODIFIED: 模块名从参数键名中推断
        for name in model.state_dict().keys():
            if need_merge(name):
                module_name = ".".join(name.split('.')[:-1])
                target_module_names.add(module_name)
        return list(target_module_names)

    # MODIFIED: 核心函数之一，用于缓存激活
    def _cache_activations_on_target_data(self, model_path, cache_path, is_vision_model, capture_inputs=False):
        if os.path.exists(cache_path) and not self.args.force_recompute:
            print(f"激活缓存文件已存在: {cache_path}, 跳过。")
            return

        print(f"正在为 {os.path.basename(model_path)} 在目标数据上缓存激活...")
        
        if is_vision_model:
            model = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(self.device)
            processor = AutoProcessor.from_pretrained(model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        # --- 准备目标域数据集 ---
        print(f"正在准备目标域数据集 ({self.args.target_dataset})...")
        # MODIFIED: 加载目标训练集
        target_dataset_raw = load_dataset(self.args.target_dataset, split="validation", streaming=True)
        
        if is_vision_model:
            target_dataset = VQAv2TargetDataset(target_dataset_raw, max_samples=self.args.target_samples)
            collate_function = collate_fn_factory(processor)
            target_dataloader = DataLoader(target_dataset, batch_size=self.args.probe_batch_size, collate_fn=collate_function)
        else:
            target_texts = [item['question'] for item in target_dataset_raw.take(self.args.target_samples)]
            formatted_texts = [tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True) for text in target_texts]
            inputs = tokenizer(formatted_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            target_dataloader = DataLoader(TensorDataset(inputs['input_ids'], inputs['attention_mask']), batch_size=self.args.probe_batch_size)

        model_to_hook = model.model
        target_module_names = self._get_target_modules(model_to_hook)

        hooks = []
        captured_activations = defaultdict(lambda: {"inputs": [], "outputs": []})

        def get_hook(name):
            def hook_fn(module, input_tensor, output):
                output_tensor = output[0] if isinstance(output, tuple) else output
                if isinstance(output_tensor, torch.Tensor):
                    captured_activations[name]["outputs"].append(output_tensor.detach().cpu())
                if capture_inputs:
                    input_t = input_tensor[0] if isinstance(input_tensor, tuple) else input_tensor
                    if isinstance(input_t, torch.Tensor):
                        captured_activations[name]["inputs"].append(input_t.detach().cpu())
            return hook_fn

        for name, module in model_to_hook.named_modules():
            if name in target_module_names:
                hooks.append(module.register_forward_hook(get_hook(name)))

        model.eval()
        with torch.no_grad():
            for batch in tqdm(target_dataloader, desc=f"前向传播 {os.path.basename(model_path)}"):
                if is_vision_model:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    model(**batch)
                else:
                    input_ids, attention_mask = batch[0].to(self.device), batch[1].to(self.device)
                    model(input_ids=input_ids, attention_mask=attention_mask)

        for h in hooks: h.remove()
        
        averaged_activations = {}
        for name, data in captured_activations.items():
            averaged_activations[name] = {}
            if data["outputs"]:
                all_tokens = torch.cat([t.float().view(-1, t.shape[-1]) for t in data["outputs"]], dim=0)
                averaged_activations[name]["output"] = torch.mean(all_tokens, dim=0)
            if data["inputs"]:
                all_tokens = torch.cat([t.float().view(-1, t.shape[-1]) for t in data["inputs"]], dim=0)
                averaged_activations[name]["input"] = torch.mean(all_tokens, dim=0)

        torch.save(averaged_activations, cache_path)
        print(f"激活已缓存至: {cache_path}")
        
        del model, captured_activations, averaged_activations, target_dataloader
        gc.collect()
        torch.cuda.empty_cache()

    def stage1_cache_activations(self):
        print("\n--- [阶段一: 缓存目标域激活] ---")
        activations_a_path = os.path.join(self.cache_dir, "activations_A_target.pt")
        activations_c_path = os.path.join(self.cache_dir, "activations_C_target.pt")
        self._cache_activations_on_target_data(self.args.base_model_path, activations_a_path, is_vision_model=True, capture_inputs=True)
        self._cache_activations_on_target_data(self.args.original_model_path, activations_c_path, is_vision_model=False, capture_inputs=False)

    def stage2_calculate_guided_directions(self):
        print("\n--- [阶段二: 计算目标指导方向] ---")
        activations_a_path = os.path.join(self.cache_dir, "activations_A_target.pt")
        activations_c_path = os.path.join(self.cache_dir, "activations_C_target.pt")
        if not os.path.exists(activations_a_path) or not os.path.exists(activations_c_path):
            print("错误: 激活缓存不存在。")
            sys.exit(1)
            
        activations_A = torch.load(activations_a_path, map_location="cpu")
        activations_C = torch.load(activations_c_path, map_location="cpu")
        
        base_weights = load_weights(self.args.base_model_path)

        for key in tqdm(base_weights.keys(), desc="计算指导方向"):
            if not need_merge(key):
                continue
            
            direction_path = os.path.join(self.direction_dir, f"{key.replace('/', '_')}.pt")
            if os.path.exists(direction_path) and not self.args.force_recompute:
                continue

            module_name = ".".join(key.split('.')[2:-1])
            if module_name not in activations_A or "input" not in activations_A[module_name] or \
               module_name not in activations_C or "output" not in activations_C[module_name]:
                continue
            
            X_A_avg = activations_A[module_name]["input"].to(self.device)
            Y_A_avg = activations_A[module_name]["output"].to(self.device)
            Y_C_avg = activations_C[module_name]["output"].to(self.device)

            delta_Y_avg = Y_A_avg - Y_C_avg
            
            g_prime = None
            if key.endswith(".weight"):
                g_prime = torch.outer(delta_Y_avg, X_A_avg)
            elif key.endswith(".bias"):
                g_prime = delta_Y_avg
            
            if g_prime is not None:
                torch.save(g_prime.cpu(), direction_path)

        del base_weights, activations_A, activations_C
        gc.collect()
        print("所有指导方向计算并保存完毕。")

    def stage3_merge_models(self):
        print("\n--- [阶段三: 指导性分解与合并] ---")
        print("正在从磁盘加载所有模型权重...")
        weights_A = load_weights(self.args.base_model_path)
        weights_B_raw = load_weights(self.args.donor_model_path)
        weights_C = load_weights(self.args.original_model_path)

        weights_B = normalize_llm_keys(weights_B_raw, list(weights_C.keys()))
        del weights_B_raw
        
        final_merged_weights = weights_A.copy()

        for key in tqdm(final_merged_weights.keys(), desc="逐层合并权重"):
            if not need_merge(key):
                continue

            key_in_c = key.replace("model.language_model.", "model.")
            if key_in_c not in weights_B or key_in_c not in weights_C:
                continue

            direction_path = os.path.join(self.direction_dir, f"{key.replace('/', '_')}.pt")
            if not os.path.exists(direction_path):
                continue

            W_A = weights_A[key].float().to(self.device)
            W_B = weights_B[key_in_c].float().to(self.device)
            W_C = weights_C[key_in_c].float().to(self.device)
            g_prime = torch.load(direction_path, map_location=self.device).float()

            if W_A.shape != g_prime.shape:
                # 兼容偏置项和权重项的不同处理
                if W_A.dim() == 1 and g_prime.dim() > 1: # Bias vs Weight grad
                    g_prime = g_prime.sum(dim=1) if g_prime.shape[0] == W_A.shape[0] else g_prime.sum(dim=0)
                elif W_A.shape != g_prime.shape:
                    print(f"警告：{key} 的指导方向形状 {g_prime.shape} 与权重形状 {W_A.shape} 不匹配，跳过。")
                    continue

            tau_B = W_B - W_C
            
            g_norm_sq = torch.sum(g_prime * g_prime)
            if g_norm_sq > 1e-9:
                proj_scalar = torch.sum(g_prime * tau_B) / g_norm_sq
                
                tau_B_synergy = torch.clamp_min(-proj_scalar, 0) * g_prime
                tau_B_conflict = torch.clamp_min(proj_scalar, 0) * g_prime
                tau_B_ortho = tau_B - tau_B_conflict - tau_B_synergy
            else:
                tau_B_synergy, tau_B_conflict, tau_B_ortho = torch.zeros_like(tau_B), torch.zeros_like(tau_B), tau_B

            w_star = W_A + (self.args.lambda_s * tau_B_synergy) - \
                           (self.args.lambda_c * tau_B_conflict) + \
                           (self.args.lambda_o * tau_B_ortho)
            
            final_merged_weights[key] = w_star.to(weights_A[key].dtype).cpu()
        
        self.save_model(final_merged_weights)

    def save_model(self, merged_weights):
        """保存模型权重。"""
        # (此函数无需修改，保持原样)
        print("\n正在保存合并后的模型...")
        index_path = os.path.join(self.args.base_model_path, "model.safetensors.index.json")
        try:
            with open(index_path, "r") as f:
                index_map = json.load(f)["weight_map"]
        except FileNotFoundError:
            safetensors.torch.save_file(merged_weights, os.path.join(self.output_dir, "model.safetensors"))
            return

        sharded_weights = defaultdict(dict)
        for key, value in merged_weights.items():
            if key in index_map:
                sharded_weights[index_map[key]][key] = value
        
        for filename, weights_dict in sharded_weights.items():
            safetensors.torch.save_file(weights_dict, os.path.join(self.output_dir, filename))
        
        shutil.copy(index_path, os.path.join(self.output_dir, os.path.basename(index_path)))
        for filename in os.listdir(self.args.base_model_path):
            if filename.endswith(('.json', '.model', '.py')):
                source_file = os.path.join(self.args.base_model_path, filename)
                dest_file = os.path.join(self.output_dir, filename)
                if not os.path.exists(dest_file):
                     shutil.copy(source_file, dest_file)
        print(f"模型成功合并并保存至: {self.output_dir}")

    def run_pipeline(self):
        """按顺序执行所有阶段。"""
        self.stage1_cache_activations()
        self.stage2_calculate_guided_directions()
        self.stage3_merge_models()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用少量目标域样本指导的离线模型合并。")
    
    # 基本配置
    parser.add_argument('--base_model_path', type=str, default="./downloaded_models/Qwen2-VL-7B-Instruct", help="基础模型A的路径。")
    parser.add_argument('--donor_model_path', type=str, default="./downloaded_models/llava-onevision-qwen2-7b-si-hf", help="贡献模型B的路径。")
    parser.add_argument('--original_model_path', type=str, default="./downloaded_models/Qwen2-7B-Instruct", help="原始共同祖先模型C的路径。")
    parser.add_argument('--mode', type=str, default="sg-dpm", help="为本次合并配置命名。")
    parser.add_argument('--cuda_device', type=int, default=0, help="使用的 CUDA 设备编号。")

    # NEW: 目标域数据集配置
    parser.add_argument('--target_dataset', type=str, default="lmms-lab/VQAv2", help="用于指导合并的目标域数据集名称 (来自Hugging Face Hub)。")
    parser.add_argument('--target_samples', type=int, default=100, help="用于指导合并的样本数量。")
    parser.add_argument('--probe_batch_size', type=int, default=2, help="计算激活时的批处理大小。")

    # 合并超参数
    parser.add_argument('--lambda_s', type=float, default=1.0, help="协同分量的系数。")
    parser.add_argument('--lambda_c', type=float, default=0.5, help="冲突分量的系数。")
    parser.add_argument('--lambda_o', type=float, default=0.8, help="正交分量的系数。")
    
    # 功能性参数
    parser.add_argument('--force_recompute', action='store_true', help="强制重新计算缓存的激活或指导方向。")

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

    merger = SampleGuidedMerger(args, device)
    merger.run_pipeline()