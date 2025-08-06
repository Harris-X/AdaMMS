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
    """根据层名判断是否需要合并 - 仅合并LLM的权重参数。"""
    if "language_model.layers" not in name:
        return False
    
    if not name.endswith(".weight"):
        return False

    if "layernorm" in name or "embed_tokens" in name or "norm" in name or ".inv_freq" in name:
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
        texts = [f"Question: {item['text']} Answer:" for item in batch]
        messages_batch = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}] for text in texts]
        prompt_batch = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
        inputs = processor(text=prompt_batch, images=images, return_tensors="pt", padding=True)
        return inputs
    return collate_fn

# --- 核心实现类 ---
# NEW: 引入全新的 FAPMMerger 类来实现新方法
class FAPMMerger:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.output_dir = os.path.join("merged_models", f"fapm-{args.mode}")
        self.cache_dir = os.path.join(self.output_dir, "cache")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"使用设备: {self.device}")
        print(f"输出将保存至: {self.output_dir}")

    def _get_target_module_map(self, model):
        module_map = {}
        for name, module in model.named_modules():
            if any(need_merge(f"{name}.{param_name}") for param_name, _ in module.named_parameters()):
                module_map[name] = module
        return module_map
    
    def _cache_activations(self, model_info, model_path, required_activations, dataloader):
        cache_path = os.path.join(self.cache_dir, f"activations_{model_info}.pt")
        if os.path.exists(cache_path) and not self.args.force_recompute:
            print(f"激活缓存文件 {cache_path} 已存在, 跳过。")
            return
        print(f"正在为 {model_info} ({os.path.basename(model_path)}) 缓存激活...")
        is_vision_model = "VL" in model_path or "llava" in model_path.lower()
        ModelClass = AutoModelForVision2Seq if is_vision_model else AutoModelForCausalLM
        model = ModelClass.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(self.device)
        model.eval()
        model_to_hook = model.model
        target_modules = self._get_target_module_map(model_to_hook)
        
        hooks, captured_activations = [], defaultdict(lambda: {"inputs": [], "outputs": []})
        def get_hook(name, req_act):
            def hook_fn(module, input, output):
                if "output" in req_act:
                    out_tensor = output[0] if isinstance(output, tuple) else output
                    if isinstance(out_tensor, torch.Tensor):
                        captured_activations[name]["outputs"].append(out_tensor.detach().cpu())
                if "input" in req_act:
                    in_tensor = input[0] if isinstance(input, tuple) else input
                    if isinstance(in_tensor, torch.Tensor):
                        captured_activations[name]["inputs"].append(in_tensor.detach().cpu())
            return hook_fn

        for name, module in target_modules.items():
            hooks.append(module.register_forward_hook(get_hook(name, required_activations)))
            
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"前向传播 {model_info}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                model(**batch)
        for h in hooks: h.remove()
        
        averaged_activations = {}
        for name, data in captured_activations.items():
            averaged_activations[name] = {}
            for act_type in ["inputs", "outputs"]:
                if data[act_type]:
                    all_tokens = torch.cat([t.float().view(-1, t.shape[-1]) for t in data[act_type]], dim=0)
                    averaged_activations[name][act_type[:-1]] = torch.mean(all_tokens, dim=0)

        torch.save(averaged_activations, cache_path)
        del model, hooks, captured_activations
        gc.collect()
        torch.cuda.empty_cache()

    def stage1_cache_activations(self):
        """阶段一：为所有模型缓存所需的激活。"""
        print("\n--- [阶段一: 缓存激活] ---")
        processor = AutoProcessor.from_pretrained(self.args.base_model_path)
        probe_dataset_raw = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
        probe_dataset = VQAv2ProbeDataset(probe_dataset_raw, max_samples=self.args.probe_samples)
        collate_function = collate_fn_factory(processor)
        dataloader = DataLoader(probe_dataset, batch_size=self.args.probe_batch_size, collate_fn=collate_function)
        
        self._cache_activations("A", self.args.base_model_path, "input_output", dataloader)
        self._cache_activations("B", self.args.donor_model_path, "input_output", dataloader)
        self._cache_activations("C", self.args.original_model_path, "output", dataloader)

    # MODIFIED: 阶段二现在是自适应频域滤波
    def stage2_adaptive_frequency_filtering(self):
        """阶段二：在频域中进行自适应滤波，并缓存过滤后的任务向量。"""
        print("\n--- [阶段二: 自适应频域滤波] ---")
        filtered_tau_cache_path = os.path.join(self.cache_dir, "filtered_tau_B.pt")
        if os.path.exists(filtered_tau_cache_path) and not self.args.force_recompute:
            print("已过滤的任务向量缓存文件存在, 跳过。")
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

        filtered_tau_B_all = {}
        for key in tqdm(weights_A.keys(), desc="自适应频域滤波"):
            if not need_merge(key): continue
            
            key_in_c = key.replace("model.language_model.", "model.")
            if not (key_in_c in weights_B and key_in_c in weights_C): continue
                
            module_name = ".".join(key.split('.')[2:-1]) # e.g., layers.0.mlp
            
            try:
                # 1. 计算任务向量和近似梯度
                W_A, W_B, W_C = weights_A[key], weights_B[key_in_c], weights_C[key_in_c]
                tau_A, tau_B = (W_A - W_C).float(), (W_B - W_C).float()
                
                delta_Y_A = activations['A'][module_name]['output'] - activations['C'][module_name]['output']
                g_approx_A = torch.outer(delta_Y_A, activations['A'][module_name]['input'])
                delta_Y_B = activations['B'][module_name]['output'] - activations['C'][module_name]['output']
                g_approx_B = torch.outer(delta_Y_B, activations['B'][module_name]['input'])
                
                # 2. 进入频域
                with torch.cuda.amp.autocast(enabled=False): # FFT需要fp32
                    F_tau_A = torch.fft.fftn(tau_A)
                    F_tau_B = torch.fft.fftn(tau_B)
                    F_g_A = torch.fft.fftn(g_approx_A.T)
                    F_g_B = torch.fft.fftn(g_approx_B.T)
                
                # 3. 构建自适应滤波器
                # 有益掩码: B的任务向量与B的近似梯度相位相似
                # 使用cos(angle_diff)作为相似度度量
                phase_sim = torch.cos(F_tau_B.angle() - F_g_B.angle())
                m_benefit = phase_sim > self.args.phase_similarity_threshold

                # 冲突掩码: A和B的任务向量频谱幅度都很大，且相位相反
                abs_F_tau_A_norm = F_tau_A.abs() / F_tau_A.abs().max()
                abs_F_tau_B_norm = F_tau_B.abs() / F_tau_B.abs().max()
                
                phase_diff = torch.cos(F_tau_A.angle() - F_tau_B.angle())
                m_conflict = (abs_F_tau_A_norm > self.args.freq_conflict_threshold) & \
                             (abs_F_tau_B_norm > self.args.freq_conflict_threshold) & \
                             (phase_diff < -0.9) # cos(pi) = -1
                
                # 4. 应用滤波器
                F_tau_B_filtered = F_tau_B * m_benefit * (~m_conflict)

                # 5. 返回参数空间
                with torch.cuda.amp.autocast(enabled=False):
                    tau_B_filtered = torch.fft.ifftn(F_tau_B_filtered).real
                
                filtered_tau_B_all[key] = tau_B_filtered.cpu()

            except KeyError:
                print(f"警告: 模块 {module_name} 的激活数据不完整，跳过参数 {key}。")
                continue

        torch.save(filtered_tau_B_all, filtered_tau_cache_path)
        print(f"已过滤的任务向量计算完成并缓存至: {filtered_tau_cache_path}")

    # MODIFIED: 阶段三现在使用过滤后的任务向量进行投影合并
    def stage3_project_and_merge(self):
        """阶段三：使用过滤后的任务向量执行投影合并。"""
        print("\n--- [阶段三: 投影合并] ---")
        
        print("加载权重、过滤后的任务向量和投影方向...")
        weights_A = load_weights(self.args.base_model_path)
        filtered_tau_cache_path = os.path.join(self.cache_dir, "filtered_tau_B.pt")
        filtered_tau_B_all = torch.load(filtered_tau_cache_path)
        activations_A = torch.load(os.path.join(self.cache_dir, "activations_A.pt"))

        final_merged_weights = weights_A.copy()
        for key, tau_B_filtered in tqdm(filtered_tau_B_all.items(), desc="执行投影合并"):
            module_name = ".".join(key.split('.')[2:-1])
            d_i = activations_A[module_name]['input'].to(self.device).float() # 投影方向
            
            tau_B_filtered_gpu = tau_B_filtered.to(self.device).float()
            
            d_i_norm_sq = torch.sum(d_i * d_i)
            if d_i_norm_sq > 1e-9:
                # 投影操作
                proj_scalar = (tau_B_filtered_gpu @ d_i) / d_i_norm_sq
                tau_B_proj = torch.outer(proj_scalar, d_i)
                tau_B_ortho = tau_B_filtered_gpu - tau_B_proj
            else:
                tau_B_proj = torch.zeros_like(tau_B_filtered_gpu)
                tau_B_ortho = tau_B_filtered_gpu

            # 最终合并
            W_A = weights_A[key].to(self.device).float()
            W_star = W_A + self.args.lambda_proj * tau_B_proj + self.args.lambda_ortho * tau_B_ortho
            final_merged_weights[key] = W_star.cpu().to(weights_A[key].dtype)
        
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
        self.stage1_cache_activations()
        self.stage2_adaptive_frequency_filtering()
        self.stage3_project_and_merge()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用FAPM进行频域感知下的自适应投影合并。")
    
    # 基本配置
    parser.add_argument('--base_model_path', type=str, default="./models/Qwen2-VL-7B-Instruct", help="基础模型A的路径。")
    parser.add_argument('--donor_model_path', type=str, default="./models/llava-onevision-qwen2-7b-si-hf", help="贡献模型B的路径。")
    parser.add_argument('--original_model_path', type=str, default="./models/Qwen2-7B-Instruct", help="原始共同祖先模型C的路径。")
    parser.add_argument('--mode', type=str, default="fapm-default", help="为本次合并配置命名。")
    parser.add_argument('--cuda_device', type=int, default=0, help="使用的 CUDA 设备编号。")

    # 数据集配置
    parser.add_argument('--probe_samples', type=int, default=100, help="用于引导合并的目标域样本数量。")
    parser.add_argument('--probe_batch_size', type=int, default=2, help="处理引导数据时的批处理大小。")

    # NEW: FAPM 合并超参数
    parser.add_argument('--phase_similarity_threshold', type=float, default=0.0, help="频域有益性判断的相位相似度(cosine)阈值。")
    parser.add_argument('--freq_conflict_threshold', type=float, default=0.1, help="频域冲突判断的归一化幅度阈值。")
    parser.add_argument('--lambda_proj', type=float, default=1.0, help="投影（相关）分量的系数。")
    parser.add_argument('--lambda_ortho', type=float, default=0.5, help="正交（无关）分量的系数。")
    
    # 功能性参数
    parser.add_argument('--force_recompute', action='store_true', help="强制重新计算缓存的激活或过滤后的任务向量。")

    args = parser.parse_args()
    
    device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"

    print("--- 配置信息 ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("--------------------")

    merger = FAPMMerger(args, device)
    merger.run_pipeline()