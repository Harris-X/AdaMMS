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
            
    if not ref_prefix and not norm_prefix:
         # 如果两个模型都没有 "layers" (例如，它们都是纯编码器/解码器)，则直接返回
        print("警告：在模型中未找到 'layers' 结构，将假定键名兼容。")
        return weights_to_norm
        
    if not norm_prefix and ref_prefix:
        print(f"警告: 贡献模型中未找到 'layers'，将尝试使用参考前缀 '{ref_prefix}' 进行对齐。")
        norm_prefix = "" # 假设贡献模型没有前缀

    if not ref_prefix and norm_prefix:
         print(f"警告: 参考模型中未找到 'layers'，将尝试移除贡献模型前缀 '{norm_prefix}'。")
         ref_prefix = "" # 假设参考模型没有前缀

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

# --- 数据集处理函数 (无修改) ---
# ... (此处省略与前一版本完全相同的数据集处理代码)

# --- 核心实现类 ---
class VDTMMerger:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.output_dir = os.path.join("merged_models", f"vdtm-{args.mode}")
        self.cache_dir = os.path.join(self.output_dir, "cache")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"使用设备: {self.device}")
        print(f"输出将保存至: {self.output_dir}")

    # 阶段一：激活缓存 (无修改)
    # ... (此处省略与前一版本完全相同的 stage1_cache_all_activations 及其辅助函数)
    def _get_target_module_map(self, model):
        """获取需要hook的模块名到模块实例的映射。"""
        module_map = {}
        # 寻找语言模型的顶层模块
        llm_module = None
        if hasattr(model, 'language_model'):
            llm_module = model.language_model
        elif hasattr(model, 'model'):
             llm_module = model.model
        else:
            llm_module = model

        for name, module in llm_module.named_modules():
            # 检查是否有任何权重参数需要合并
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
    
        class ProbeIterableDataset(Dataset):
            def __init__(self, raw_dataset, max_samples):
                self.raw_dataset = raw_dataset
                self.max_samples = max_samples
            def __iter__(self):
                count = 0
                for item in self.raw_dataset:
                    if count >= self.max_samples:
                        break
                    image = item["image"]
                    if image.mode == 'RGBA':
                        image = image.convert('RGB')
                    yield {"image": image, "text": item["question"]}
                    count += 1
        
        probe_dataset = ProbeIterableDataset(dataset_raw, self.args.probe_samples)

        with torch.no_grad():
            num_batches = (self.args.probe_samples + self.args.probe_batch_size - 1) // self.args.probe_batch_size
            pbar = tqdm(DataLoader(probe_dataset, batch_size=self.args.probe_batch_size), total=num_batches, desc=f"前向传播 {model_info}")
            for batch in pbar:
                images = batch["image"]
                texts = batch["text"]

                if is_llava:
                    conversations = [{"role": "user", "content": [{"type": "text", "text": t}, {"type": "image"}]} for t in texts]
                    prompts = [processor.apply_chat_template([conv], tokenize=False, add_generation_prompt=True) for conv in conversations]
                    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
                elif is_vision_model:
                    messages_batch = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}] for text in texts]
                    prompt_batch = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
                    inputs = processor(text=prompt_batch, images=images, return_tensors="pt", padding=True)
                else: # 纯文本模型
                    tokenizer = processor
                    formatted_texts = [tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True) for text in texts]
                    inputs = tokenizer(text=formatted_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                try:
                    model(**inputs)
                except Exception as e:
                    print(f"模型前向传播时出错: {e}")
                    continue

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
        probe_dataset_raw = load_dataset("HuggingFaceM4/VQAv2", "default", split="validation", streaming=True)
        self._cache_activations_raw("A", self.args.base_model_path, ["input", "output"], probe_dataset_raw)
        self._cache_activations_raw("B", self.args.donor_model_path, ["input", "output"], probe_dataset_raw)
        self._cache_activations_raw("C", self.args.original_model_path, ["output"], probe_dataset_raw)


    # 阶段二：基于泰勒近似的重要性定位 (逻辑不变, 仅修改了缓存保存方式)
    def stage2_importance_analysis(self):
        """阶段二：【TAG-M】计算重要性分数并缓存A和B的重要性掩码。"""
        print("\n--- [阶段二: TAG-M 重要性分析] ---")
        mask_cache_path = os.path.join(self.cache_dir, f"tagm_importance_masks_r{self.args.top_k_ratio}_alpha{self.args.alpha}.pt")
        if os.path.exists(mask_cache_path) and not self.args.force_recompute:
            print("TAG-M 重要性掩码缓存文件已存在, 跳过。")
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

        # 【修改】为A和B分别创建掩码字典
        masks_A = {}
        masks_B = {}
        
        pbar = tqdm(weights_A.keys(), desc="【TAG-M】分析神经元")
        for key in pbar:
            if not need_merge(key): continue
            if not (key in weights_B and key in weights_C): continue
                
            module_name = ".".join(key.split('.')[1:-1]) if key.startswith("model.") else ".".join(key.split('.')[2:-1])
            
            try:
                g_approx_A = torch.outer(activations['A'][module_name]['output'], activations['A'][module_name]['input'])
                delta_Y_B = activations['B'][module_name]['output'] - activations['C'][module_name]['output']
                g_approx_B = torch.outer(delta_Y_B, activations['B'][module_name]['input'])

                W_A, W_B, W_C = weights_A[key], weights_B[key], weights_C[key]
                tau_A = W_A.float() - W_C.float()
                tau_B = W_B.float() - W_C.float()

                itag_A = -g_approx_A * tau_A + (self.args.alpha / 2) * (g_approx_A**2) * (tau_A**2)
                itag_B = -g_approx_B * tau_B + (self.args.alpha / 2) * (g_approx_B**2) * (tau_B**2)

                importance_A = itag_A.abs()
                importance_B = itag_B.abs()

                k = int(importance_A.numel() * self.args.top_k_ratio)
                if k == 0: continue
                
                # 【修改】分别计算并存储A和B的重要性掩码
                masks_A[key] = (importance_A >= torch.topk(importance_A.flatten(), k=k, sorted=False)[0].min()).cpu()
                masks_B[key] = (importance_B >= torch.topk(importance_B.flatten(), k=k, sorted=False)[0].min()).cpu()

            except KeyError:
                pbar.set_description(f"警告: 模块 {module_name} 激活数据不完整，跳过")
                continue

        # 【修改】将两个掩码字典保存在一个文件中
        torch.save({'mask_A': masks_A, 'mask_B': masks_B}, mask_cache_path)
        print(f"TAG-M 重要性掩码计算完成并缓存至: {mask_cache_path}")
        
    # ########################################################################## #
    # #                           核心代码修改区域                             # #
    # ########################################################################## #

    def stage3_vector_decomposition_and_merge(self):
        """阶段三：【VDT-M】执行矢量分解与加权融合。"""
        print("\n--- [阶段三: VDT-M 矢量分解与融合] ---")
        
        print("加载所有权重和重要性掩码...")
        weights_A = load_weights(self.args.base_model_path)
        weights_B_raw = load_weights(self.args.donor_model_path)
        weights_C_raw = load_weights(self.args.original_model_path)
        
        weights_B = normalize_llm_keys(weights_B_raw, list(weights_A.keys())); del weights_B_raw
        weights_C = normalize_llm_keys(weights_C_raw, list(weights_A.keys())); del weights_C_raw

        mask_cache_path = os.path.join(self.cache_dir, f"tagm_importance_masks_r{self.args.top_k_ratio}_alpha{self.args.alpha}.pt")
        # 【修改】加载包含两个掩码的字典
        all_masks = torch.load(mask_cache_path, map_location="cpu")
        masks_A = all_masks['mask_A']
        masks_B = all_masks['mask_B']

        final_merged_weights = weights_A.copy()
        
        pbar = tqdm(masks_B.keys(), desc="【VDT-M】分解并融合")
        for key in pbar:
            # 步骤 1: 加载权重和对应掩码
            W_A, W_B, W_C = weights_A[key], weights_B[key], weights_C[key]
            mask_A = masks_A[key].to(self.device)
            mask_B = masks_B[key].to(self.device)

            # 步骤 2: 计算任务向量
            tau_A = (W_A - W_C).to(self.device).float()
            tau_B = (W_B - W_C).to(self.device).float()

            # 步骤 3: 计算分解掩码 (Synergy, Donor-Specific)
            # 协同掩码 m_syn: A和B都重要且方向一致
            sign_A = torch.sign(tau_A)
            sign_B = torch.sign(tau_B)
            m_syn = mask_A & mask_B & (sign_A == sign_B)
            
            # 贡献者专属掩码 m_don: B重要但A不重要
            m_don = mask_B & (~mask_A)

            # 步骤 4: 分解任务向量 tau_B
            tau_B_syn = tau_B * m_syn
            tau_B_don = tau_B * m_don
            # 冲突部分被隐式地丢弃了

            # 步骤 5: 最终加权合并
            W_star = W_A.to(self.device).float() + self.args.lambda_syn * tau_B_syn + self.args.lambda_don * tau_B_don
            final_merged_weights[key] = W_star.cpu().to(W_A.dtype)
        
        self._save_model(final_merged_weights)

    # _save_model (无修改)
    def _save_model(self, merged_weights):
        """保存模型权重。"""
        print("\n正在保存合并后的模型...")
        index_path = os.path.join(self.args.base_model_path, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index_map = json.load(f)["weight_map"]
        
        sharded_weights = defaultdict(dict)
        for key, value in merged_weights.items():
            if key in index_map:
                sharded_weights[index_map[key]][key] = value
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        for filename, weights_dict in sharded_weights.items():
            safetensors.torch.save_file(weights_dict, os.path.join(self.output_dir, filename))
        
        # 复制配置文件
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
        self.stage3_vector_decomposition_and_merge()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用VDT-M进行高效、精细、理论完备的模型合并。")
    
    # 基本配置
    parser.add_argument('--base_model_path', type=str, required=True, help="基础模型A的路径。")
    parser.add_argument('--donor_model_path', type=str, required=True, help="贡献模型B的路径。")
    parser.add_argument('--original_model_path', type=str, required=True, help="原始共同祖先模型C的路径。")
    parser.add_argument('--mode', type=str, default="vdtm-default", help="为本次合并配置命名。")
    parser.add_argument('--cuda_device', type=int, default=0, help="使用的 CUDA 设备编号。")

    # 数据集配置
    parser.add_argument('--probe_samples', type=int, default=128, help="用于引导合并的目标域样本数量。")
    parser.add_argument('--probe_batch_size', type=int, default=4, help="处理引导数据时的批处理大小。")

    # VDT-M 合并超参数
    parser.add_argument('--top_k_ratio', type=float, default=0.1, help="【阶段二】用于选举关键神经元的Top-K比率。")
    parser.add_argument('--alpha', type=float, default=1.0, help="【阶段二】平衡泰勒展开一阶和二阶项的信任域参数。")
    parser.add_argument('--lambda_syn', type=float, default=0.5, help="【阶段三】协同(Synergy)知识分量的融合系数。")
    parser.add_argument('--lambda_don', type=float, default=1.0, help="【阶段三】贡献者专属(Donor-Specific)知识分量的融合系数。")
    
    # 功能性参数
    parser.add_argument('--force_recompute', action='store_true', help="强制重新计算缓存的激活或掩码。")

    args = parser.parse_args()
    
    device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"

    print("--- 配置信息 ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("--------------------")

    merger = VDTMMerger(args, device)
    merger.run_pipeline()