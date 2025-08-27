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
import random

# 导入指定的模型和分词器类
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoModelForVision2Seq
from torch.utils.data import DataLoader, Dataset, TensorDataset

# 尝试导入 Hugging Face datasets 库
try:
    from datasets import load_dataset, concatenate_datasets
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
    """
    SAM-S-DREAM的复杂合并目标：
    - 仅处理 transformer layers 内部的线性权重与 bias
    - 显式排除所有 norm 与 rotary_emb
    """
    is_in_layers = name.startswith("model.layers.") or name.startswith("language_model.layers.")
    if not is_in_layers:
        return False

    # 显式排除 norm 和 rotary
    if 'layernorm' in name or 'norm' in name or 'rotary_emb' in name:
        return False

    # 线性层的 .weight/.bias 进入复杂合并
    if name.endswith('.weight') or name.endswith('.bias'):
        return True

    return False

# --- 核心实现类 ---
class SAMSDREAMMerger:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.output_dir = os.path.join("merged_models", f"idream-{args.mode}")
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
            # 确保我们处理的是完整的模块名
            base_prefix = ""
            if hasattr(model, 'language_model'): base_prefix = "language_model."
            elif hasattr(model, 'model'): base_prefix = "model."
            
            full_module_name_prefix = f"{base_prefix}{name}"

            # 仅为需要复杂合并的模块挂钩
            if any(need_merge(f"{full_module_name_prefix}.{param_name}") for param_name, _ in module.named_parameters()):
                 module_map[name] = module

        return module_map
    
    # ########################################################################## #
    # #                           关键代码修改区域 (1/4)                         # #
    # ########################################################################## #
    
    # def _create_meta_probe_dataset(self):
    #     """
    #     构建并返回一个由多个数据源组成的元探测数据集。
    #     """
    #     print("--- [元探测数据集构建] ---")
    #     meta_probe_samples = []
        
    #     # 1. 加载并处理 VQA v2 (综合能力)
    #     if self.args.n_vqa > 0:
    #         print(f"从 VQA v2 加载 {self.args.n_vqa} 个样本...")
    #         vqa_dataset = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True).shuffle(seed=42).take(self.args.n_vqa)
    #         for item in vqa_dataset:
    #             meta_probe_samples.append({"image": item["image"], "question": item["question"]})

    #     # 2. 加载并处理 ScienceQA (认知与推理)
    #     if self.args.n_scienceqa > 0:
    #         print(f"从 ScienceQA 加载 {self.args.n_scienceqa} 个样本...")
    #         # ScienceQA non-streaming for easier filtering
    #         scienceqa_dataset = load_dataset("derek-thomas/ScienceQA", split="train").shuffle(seed=42)
    #         # 筛选出包含图像的样本
    #         scienceqa_with_images = scienceqa_dataset.filter(lambda x: x['image'] is not None)
    #         count = 0
    #         for item in scienceqa_with_images:
    #             if count >= self.args.n_scienceqa: break
    #             question = f"{item['hint']} {item['question']}" if item['hint'] else item['question']
    #             meta_probe_samples.append({"image": item["image"], "question": question})
    #             count += 1

    #     # 3. 加载并处理 ST-VQA (富文本VQA)
    #     if self.args.n_stvqa > 0:
    #         print(f"从 ST-VQA 加载 {self.args.n_stvqa} 个样本...")
    #         # ST-VQA 字段名为 'question', 'image'
    #         stvqa_dataset = load_dataset("danjacobellis/stvqa_task1", split="train", streaming=True).shuffle(seed=42).take(self.args.n_stvqa)
    #         for item in stvqa_dataset:
    #              meta_probe_samples.append({"image": item["image"], "question": item["question"]})

    #     # 打乱最终的数据集
    #     random.shuffle(meta_probe_samples)
    #     print(f"元探测数据集构建完成，总样本数: {len(meta_probe_samples)}")
    #     print("--------------------------")
    #     return meta_probe_samples

    def _create_meta_probe_dataset(self):
        """
        构建并返回一个由多个数据源组成的元探测数据集。
        此版本包含了更具挑战性的数据集。
        """
        print("--- [元探测数据集构建] ---")
        meta_probe_samples = []
        
        # ======================================================================
        # 类别 1: 综合评估 (Comprehensive-Evaluation)
        # ======================================================================

        # 推荐数据集: MMBench (替代 MME, SEED-Bench)
        if hasattr(self.args, 'n_mmbench') and self.args.n_mmbench > 0:
            print(f"从 MMBench 加载 {self.args.n_mmbench} 个样本...")
            # MMBench 的问题是选择题，需要格式化
            mmbench_dataset = load_dataset("lmms-lab/MMBench", 'en',split="test", streaming=True).shuffle(seed=42).take(self.args.n_mmbench)
            for item in mmbench_dataset:
                # 组合问题和选项
                question = item['question']
                options = f"A. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}"
                # 如果有提示，可以加在问题前面
                if item.get('hint'):
                    full_question = f"{item['hint']}\n{question}\n{options}"
                else:
                    full_question = f"{question}\n{options}"
                
                meta_probe_samples.append({"image": item["image"], "question": full_question, "answer": item["answer"]})
            del mmbench_dataset  # 释放内存
        # ======================================================================
        # 类别 2: 认知与推理 (Cognition and Reasoning)
        # ======================================================================

        # 推荐数据集: VCR (Visual Commonsense Reasoning) (替代 GQA, MMMU)
        if hasattr(self.args, 'n_vcr') and self.args.n_vcr > 0:
            print(f"从 VCR 加载 {self.args.n_vcr} 个样本...")
            # VCR 包含问题->答案(Q->A)和答案->理由(A->R)两个子任务。这里我们使用 Q->A。
            # 注意: VCR 的 'image' 字段是字符串路径，需要特殊处理，但 huggingface 会自动加载。
            vcr_dataset = load_dataset("pingzhili/vcr-qa", split="validation", streaming=True).shuffle(seed=42).take(self.args.n_vcr)
            for item in vcr_dataset:
                # 格式化问题和答案选项
                question = item['question']
                choices = "\n".join([f"- {c}" for c in item['answer_choices']])
                full_question = f"{question}\n\nChoices:\n{choices}"
                
                # 记录正确答案的文本以供参考
                correct_answer_text = item['answer_choices'][item['answer_label']]
                meta_probe_samples.append({"image": item["image"], "question": full_question, "answer": correct_answer_text})
            del vcr_dataset  # 释放内存
        # ======================================================================
        # 类别 3: 富文本视觉问答 (Text-rich VQA)
        # ======================================================================

        # 推荐数据集: DocVQA (替代 TextVQA, OCRBench)
        if hasattr(self.args, 'n_docvqa') and self.args.n_docvqa > 0:
            print(f"从 DocVQA 加载 {self.args.n_docvqa} 个样本...")
            # DocVQA 的问题字段名为 'question'
            docvqa_dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation", streaming=True).shuffle(seed=42).take(self.args.n_docvqa)
            for item in docvqa_dataset:
                # DocVQA 的 'answers' 是一个列表，这里我们只用问题
                meta_probe_samples.append({"image": item["image"], "question": item["question"], "answers": item["answers"]})
            del docvqa_dataset  # 释放内存

        # (此处可以保留或添加您原来的数据集加载逻辑)
        # 1. 加载并处理 VQA v2 (综合能力)
        if self.args.n_vqa > 0:
            print(f"从 VQA v2 加载 {self.args.n_vqa} 个样本...")
            vqa_dataset = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True).shuffle(seed=42).take(self.args.n_vqa)
            for item in vqa_dataset:
                meta_probe_samples.append({"image": item["image"], "question": item["question"]})
        
            del vqa_dataset  # 释放内存

        # 2. 加载并处理 ScienceQA (认知与推理)
        if self.args.n_scienceqa > 0:
            print(f"从 ScienceQA 加载 {self.args.n_scienceqa} 个样本...")
            # ScienceQA non-streaming for easier filtering
            scienceqa_dataset = load_dataset("derek-thomas/ScienceQA", split="validation").shuffle(seed=42)
            # 筛选出包含图像的样本
            scienceqa_with_images = scienceqa_dataset.filter(lambda x: x['image'] is not None)
            count = 0
            for item in scienceqa_with_images:
                if count >= self.args.n_scienceqa: break
                question = f"{item['hint']} {item['question']}" if item['hint'] else item['question']
                meta_probe_samples.append({"image": item["image"], "question": question})
                count += 1
            del scienceqa_dataset  # 释放内存

        # 3. 加载并处理 ST-VQA (富文本VQA)
        if self.args.n_stvqa > 0:
            print(f"从 ST-VQA 加载 {self.args.n_stvqa} 个样本...")
            # ST-VQA 字段名为 'question', 'image'
            stvqa_dataset = load_dataset("danjacobellis/stvqa_task1", split="test", streaming=True).shuffle(seed=42).take(self.args.n_stvqa)
            for item in stvqa_dataset:
                 meta_probe_samples.append({"image": item["image"], "question": item["question"]})
            del stvqa_dataset  # 释放内存
        # 打乱最终的数据集
        random.shuffle(meta_probe_samples)
        print(f"元探测数据集构建完成，总样本数: {len(meta_probe_samples)}")
        print("--------------------------")
        return meta_probe_samples



    # ########################################################################## #
    # #                           关键代码修改区域 (2/4)                         # #
    # ########################################################################## #

    def _cache_activations_raw(self, model_info, model_path, required_activations, probe_dataset_list):
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
            "output_sum": None, "output_tokens": 0,
            "input_samples": [],   # 新增：少量输入方向样本
            "output_samples": []   # 新增：少量输出方向样本
        })
    
        def get_hook_with_kwargs(name, req_act):
            def hook_fn(module, args, kwargs, output):
                max_dirs = getattr(self.args, "probe_directions", 8)

                if "output" in req_act:
                    out_tensor = output[0] if isinstance(output, tuple) else output
                    if isinstance(out_tensor, torch.Tensor):
                        t_float = out_tensor.detach().cpu().float()
                        t_reshaped = t_float.reshape(-1, t_float.shape[-1])
                        current_sum = torch.sum(t_reshaped, dim=0)
                        # 累加平均
                        if activation_stats[name]["output_sum"] is None:
                            activation_stats[name]["output_sum"] = current_sum
                        else:
                            activation_stats[name]["output_sum"] += current_sum
                        activation_stats[name]["output_tokens"] += t_reshaped.shape[0]
                        # 采样一个方向（该批的均值作为一个方向）
                        if len(activation_stats[name]["output_samples"]) < max_dirs:
                            activation_stats[name]["output_samples"].append(t_reshaped.mean(dim=0))
                        else:
                            # 水塘抽样，保持多样性
                            j = random.randint(0, activation_stats[name]["output_tokens"])
                            if j < max_dirs:
                                activation_stats[name]["output_samples"][j] = t_reshaped.mean(dim=0)

                if "input" in req_act:
                    in_tensor = kwargs.get("hidden_states", args[0] if args and isinstance(args[0], torch.Tensor) else None)
                    if isinstance(in_tensor, torch.Tensor):
                        t_float = in_tensor.detach().cpu().float()
                        t_reshaped = t_float.reshape(-1, t_float.shape[-1])
                        current_sum = torch.sum(t_reshaped, dim=0)
                        # 累加平均
                        if activation_stats[name]["input_sum"] is None:
                            activation_stats[name]["input_sum"] = current_sum
                        else:
                            activation_stats[name]["input_sum"] += current_sum
                        activation_stats[name]["input_tokens"] += t_reshaped.shape[0]
                        # 采样一个方向（该批的均值作为一个方向）
                        if len(activation_stats[name]["input_samples"]) < max_dirs:
                            activation_stats[name]["input_samples"].append(t_reshaped.mean(dim=0))
                        else:
                            j = random.randint(0, activation_stats[name]["input_tokens"])
                            if j < max_dirs:
                                activation_stats[name]["input_samples"][j] = t_reshaped.mean(dim=0)
            return hook_fn
    
        hooks = [
            module.register_forward_hook(get_hook_with_kwargs(name, required_activations), with_kwargs=True)
            for name, module in target_modules.items()
        ]
    
        # 直接使用传入的样本列表
        original_samples = []
        for item in probe_dataset_list:
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
            # 保存样本方向（若存在）
            if len(stats["input_samples"]) > 0:
                averaged_activations[name]["input_samples"] = torch.stack(stats["input_samples"], dim=0)
            if len(stats["output_samples"]) > 0:
                averaged_activations[name]["output_samples"] = torch.stack(stats["output_samples"], dim=0)

        torch.save(averaged_activations, cache_path)
        del model, processor, hooks, activation_stats, averaged_activations
        gc.collect(); torch.cuda.empty_cache()

    # ########################################################################## #
    # #                           关键代码修改区域 (3/4)                         # #
    # ########################################################################## #

    def stage1_cache_all_activations(self):
        """阶段一：构建元探测数据集并为所有模型缓存激活。"""
        print("\n--- [阶段一: 缓存所有激活] ---")
        
        # 调用新函数来构建元探测数据集
        meta_probe_dataset = self._create_meta_probe_dataset()
        
        # 为每个模型使用同一个元探测集来缓存激活
        self._cache_activations_raw("A", self.args.base_model_path, ["input", "output"], meta_probe_dataset)
        self._cache_activations_raw("B", self.args.donor_model_path, ["output"], meta_probe_dataset)
        self._cache_activations_raw("C", self.args.original_model_path, ["output"], meta_probe_dataset)

    # stage2 和 stage3 保持不变
    def _min_max_normalize(self, tensor):
        """对张量进行min-max归一化。"""
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val + self.EPS)

    def stage2_regularized_disjoint_mask_generation(self):
        """阶段二：【SAM-S-DREAM】生成夏普斯感知的非冲突更新掩码。"""
        print("\n--- [阶段二: SAM-S-DREAM 夏普斯感知评分与掩码生成] ---")
        mask_cache_path = os.path.join(self.cache_dir, f"sams-dream_disjoint_mask_r{self.args.top_k_ratio}_alpha{self.args.alpha}.pt")
        if os.path.exists(mask_cache_path) and not self.args.force_recompute:
            print("SAM-S-DREAM 非冲突掩码缓存文件已存在, 跳过。")
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

        disjoint_masks = {}
        pbar = tqdm(weights_A.keys(), desc="【SAM-S-DREAM】分析神经元")
        for key in pbar:
            if not need_merge(key): 
                continue
            if not (key in weights_B and key in weights_C): 
                continue

            module_name = ".".join(key.replace("model.language_model.", "model.").split('.')[1:-1])
            if module_name not in activations['A'] or 'output' not in activations['A'][module_name]:
                pbar.set_description(f"警告: 模块 {module_name} 激活缺失，跳过 {key}")
                continue

            try:
                W_A, W_B, W_C = weights_A[key].float(), weights_B[key].float(), weights_C[key].float()
                # 1) 构造近似“梯度方向”
                if W_A.ndim == 2 and key.endswith(".weight"):
                    # 2D: outer(output_A, input_A) / outer((output_B-output_C), input_A)
                    if 'input' not in activations['A'][module_name]:
                        pbar.set_description(f"警告: {module_name} 无 input 激活，跳过 {key}")
                        continue
                    out_A = activations['A'][module_name]['output']
                    in_A  = activations['A'][module_name]['input']
                    out_B = activations['B'][module_name].get('output', None)
                    out_C = activations['C'][module_name].get('output', None)
                    if out_B is None or out_C is None:
                        pbar.set_description(f"警告: {module_name} 无 B/C 输出，跳过 {key}")
                        continue

                    g_approx_A = torch.outer(out_A, in_A)
                    g_approx_B = torch.outer(out_B - out_C, in_A)

                elif W_A.ndim == 1 and key.endswith(".bias"):
                    # 1D bias: 用输出差向量作为方向
                    out_A = activations['A'][module_name]['output']
                    out_B = activations['B'][module_name].get('output', None)
                    out_C = activations['C'][module_name].get('output', None)
                    if out_B is None or out_C is None:
                        pbar.set_description(f"警告: {module_name} 无 B/C 输出，跳过 {key}")
                        continue
                    g_approx_A = out_A        # A 的输出强度作为显著性参考
                    g_approx_B = (out_B - out_C)  # B 相对 C 的输出变化决定注入方向
                else:
                    # 非预期形状，跳过
                    continue

                # 2) 夏普斯感知的显著性评分与掩码
                saliency_A = (W_A * g_approx_A).abs()
                sharpness_penalty_A = 1 + self.args.alpha * (g_approx_A**2)
                s_sas_A = saliency_A / sharpness_penalty_A

                saliency_B = (W_B * g_approx_B).abs()
                sharpness_penalty_B = 1 + self.args.alpha * (g_approx_B**2)
                s_sas_B = saliency_B / sharpness_penalty_B

                k = max(1, int(s_sas_A.numel() * self.args.top_k_ratio))
                if k <= 0: 
                    continue

                # 维度自适应 top-k
                th_A = torch.topk(s_sas_A.flatten(), k=k, sorted=False).values.min()
                th_B = torch.topk(s_sas_B.flatten(), k=k, sorted=False).values.min()
                mask_A = (s_sas_A >= th_A)
                mask_B = (s_sas_B >= th_B)

                tau_A = W_A - W_C
                tau_B = W_B - W_C

                # 冲突：同一位置两侧入选但方向相反
                conflict_mask = mask_A & mask_B & (torch.sign(tau_A) != torch.sign(tau_B))
                disjoint_mask_B = mask_B & (~conflict_mask)

                disjoint_masks[key] = disjoint_mask_B.cpu()

            except KeyError:
                pbar.set_description(f"警告: 模块 {module_name} 激活数据不完整，跳过 {key}")
                continue

        torch.save(disjoint_masks, mask_cache_path)
        print(f"I-DREAM 非冲突掩码计算完成并缓存至: {mask_cache_path}")
        
    def stage3_disentangled_reprojection_fusion(self):
        """阶段三：【SAM-S-DREAM】执行解耦重投影融合。"""
        print("\n--- [阶段三: SAM-S-DREAM 解耦重投影融合] ---")
        
        print("加载所有权重、掩码和激活...")
        weights_A = load_weights(self.args.base_model_path)
        weights_B_raw = load_weights(self.args.donor_model_path)
        weights_C_raw = load_weights(self.args.original_model_path)
        
        weights_B = normalize_llm_keys(weights_B_raw, list(weights_A.keys())); del weights_B_raw
        weights_C = normalize_llm_keys(weights_C_raw, list(weights_A.keys())); del weights_C_raw

        mask_cache_path = os.path.join(self.cache_dir, f"sams-dream_disjoint_mask_r{self.args.top_k_ratio}_alpha{self.args.alpha}.pt")
        disjoint_masks = torch.load(mask_cache_path)
        
        activations_A = torch.load(os.path.join(self.cache_dir, "activations_A.pt"))
        # 新增：bias 投影需要 B/C 的输出方向
        activations_B = torch.load(os.path.join(self.cache_dir, "activations_B.pt"))
        activations_C = torch.load(os.path.join(self.cache_dir, "activations_C.pt"))

        final_merged_weights = weights_A.copy()
        
        # 新增：安全合并的超参数
        beta_safety = getattr(self.args, "beta_safety", 2.0)
        rho_max = getattr(self.args, "rho_ortho_max_ratio", 0.5)
        delta_mode = getattr(self.args, "delta_mode", "delta")
        two_sided = getattr(self.args, "two_sided_weights", True)  # 新增：权重双侧投影
        lambda_bias = getattr(self.args, "lambda_bias", 0.1)       # 新增：偏置合并主系数

        pbar = tqdm(disjoint_masks.items(), desc="【SAM-S-DREAM】执行重投影融合")
        processed_keys = set()
        for key, M_prime_B in pbar:
            if not (key in weights_B and key in weights_C): 
                continue
            
            processed_keys.add(key)
            W_A, W_B, W_C = weights_A[key].float(), weights_B[key].float(), weights_C[key].float()
            M_prime_B = M_prime_B.to(self.device)

            module_name = ".".join(key.replace("model.language_model.", "model.").split('.')[1:-1])
            tau_B = (W_B - W_C).to(self.device) if delta_mode == "delta" else W_B.to(self.device)
            tau_B_update = tau_B * M_prime_B.to(self.device)

            if W_A.ndim == 2 and key.endswith(".weight"):
                # --- 构造输入子空间 P_in ---
                D_in = None
                if module_name in activations_A and "input_samples" in activations_A[module_name]:
                    D_in = activations_A[module_name]["input_samples"].to(self.device).float()  # [m_in, in_dim]
                elif module_name in activations_A and "input" in activations_A[module_name]:
                    D_in = activations_A[module_name]["input"].unsqueeze(0).to(self.device).float()  # [1, in_dim]

                # --- 构造输出子空间 P_out（优先使用 B-C 的输出变化子空间）---
                D_out = None
                if module_name in activations_B and "output_samples" in activations_B[module_name] and \
                   module_name in activations_C and "output_samples" in activations_C[module_name]:
                    D_out = (activations_B[module_name]["output_samples"] -
                             activations_C[module_name]["output_samples"]).to(self.device).float()  # [m_out, out_dim]
                elif module_name in activations_B and "output" in activations_B[module_name] and \
                     module_name in activations_C and "output" in activations_C[module_name]:
                    D_out = (activations_B[module_name]["output"] -
                             activations_C[module_name]["output"]).unsqueeze(0).to(self.device).float()  # [1, out_dim]

                if D_in is not None:
                    # P_in = D_in^T (D_in D_in^T)^+ D_in
                    G_in = D_in @ D_in.T + self.EPS * torch.eye(D_in.shape[0], device=self.device)
                    P_in = D_in.T @ torch.linalg.pinv(G_in) @ D_in      # [in_dim, in_dim]
                    if two_sided and (D_out is not None):
                        # P_out = D_out^T (D_out D_out^T)^+ D_out
                        G_out = D_out @ D_out.T + self.EPS * torch.eye(D_out.shape[0], device=self.device)
                        P_out = D_out.T @ torch.linalg.pinv(G_out) @ D_out  # [out_dim, out_dim]
                        # 双侧投影
                        tau_proj = P_out @ tau_B_update @ P_in
                    else:
                        # 仅输入侧投影（与原实现兼容）
                        tau_proj = tau_B_update @ P_in
                    tau_ortho = tau_B_update - tau_proj
                else:
                    # 回退到单方向（平均输入向量）
                    d_i = activations_A[module_name]['input'].to(self.device).float()
                    d_i_norm_sq = torch.sum(d_i * d_i)
                    if d_i_norm_sq > self.EPS:
                        proj_scalar = (tau_B_update @ d_i) / d_i_norm_sq
                        tau_proj = torch.outer(proj_scalar, d_i)
                        tau_ortho = tau_B_update - tau_proj
                    else:
                        tau_proj = torch.zeros_like(tau_B_update)
                        tau_ortho = tau_B_update

            elif W_A.ndim == 1 and key.endswith(".bias"):
                # --- 多方向投影：基于输出子空间 D_out ---
                D_out = None
                if module_name in activations_B and "output_samples" in activations_B[module_name] and \
                   module_name in activations_C and "output_samples" in activations_C[module_name]:
                    D_out = (activations_B[module_name]["output_samples"] -
                             activations_C[module_name]["output_samples"]).to(self.device).float()  # [m, out_dim]
                elif module_name in activations_B and "output" in activations_B[module_name] and \
                     module_name in activations_C and "output" in activations_C[module_name]:
                    D_out = (activations_B[module_name]["output"] -
                             activations_C[module_name]["output"]).unsqueeze(0).to(self.device).float()  # [1, out_dim]
                
                if D_out is not None:
                    G = D_out @ D_out.T + self.EPS * torch.eye(D_out.shape[0], device=self.device)
                    P_rows = D_out.T @ torch.linalg.pinv(G) @ D_out      # [out_dim, out_dim]
                    tau_proj = P_rows @ tau_B_update                     # 只保留 B 相对 C 的输出变化方向
                    tau_ortho = tau_B_update - tau_proj
                else:
                    # 没有可靠的输出子空间时，对偏置合并极度保守
                    tau_proj = torch.zeros_like(tau_B_update)
                    tau_ortho = torch.zeros_like(tau_B_update)
            else:
                continue

            # --- 信任域：限制正交增量规模 ---
            proj_norm = torch.linalg.norm(tau_proj)
            ortho_norm = torch.linalg.norm(tau_ortho)
            if proj_norm > self.EPS and ortho_norm > self.EPS:
                scale_cap = (rho_max * proj_norm / (ortho_norm + self.EPS)).clamp(max=1.0)
                tau_ortho = tau_ortho * scale_cap

            # --- 功能性安全因子（多样本） ---
            W_A_device = W_A.to(self.device)
            if W_A.ndim == 2 and key.endswith(".weight") and D_in is not None:
                # 基函数：输出扰动矩阵
                Y_A_mat = W_A_device @ D_in.T                   # [out, m]
                dY_mat  = tau_ortho @ D_in.T                   # [out, m]
                den = torch.linalg.norm(Y_A_mat)
                rel = (torch.linalg.norm(dY_mat) / (den + self.EPS)).clamp(0.0, 1.0)
                safety_factor = (1 - rel) ** beta_safety
            elif W_A.ndim == 1 and key.endswith(".bias"):
                # 偏置：对每个样本都加同一向量，等价于 sqrt(m) 放大
                m = D_out.shape[0] if (D_out is not None) else 1
                dY_norm = (torch.linalg.norm(tau_ortho) * (m ** 0.5))
                # 近似基准：用 A 的输出均值向量
                if module_name in activations_A and "output" in activations_A[module_name]:
                    Y_A = activations_A[module_name]["output"].to(self.device).float()
                    den = (torch.linalg.norm(Y_A) * (m ** 0.5))
                else:
                    den = torch.tensor(1.0, device=self.device)
                rel = (dY_norm / (den + self.EPS)).clamp(0.0, 1.0)
                safety_factor = (1 - rel) ** beta_safety
            else:
                # 回退：不使用参数余弦，直接保守缩放
                safety_factor = 0.0

            adaptive_lambda_ortho = self.args.lambda_ortho * safety_factor

            # --- 偏置的主系数更小，且默认不引入正交项 ---
            if W_A.ndim == 1 and key.endswith(".bias"):
                self_lambda_proj = min(self.args.lambda_proj, lambda_bias)
                self_lambda_ortho = 0.0  # 偏置正交项默认禁用
            else:
                self_lambda_proj = self.args.lambda_proj
                # 正交项安全系数计算（已有功能扰动比），保留原逻辑
                # adaptive_lambda_ortho 已在下方计算

            W_star = W_A_device + self_lambda_proj * tau_proj + adaptive_lambda_ortho * tau_ortho
            final_merged_weights[key] = W_star.cpu().to(weights_A[key].dtype)

        # Part 2: 其余参数 (含 norm) 的保守加权平均
        print("\n正在使用简单加权平均处理其余参数 (norm, 其他未入选的参数)...")
        lam_default = self.args.lambda_proj
        lam_norm = getattr(self.args, "lambda_norm", 0.0)  # 新增: norm 的保守合并系数

        other_keys_pbar = tqdm(weights_A.keys(), desc="简单平均合并")
        for key in other_keys_pbar:
            # print(f"keys: {key}")
            if key in processed_keys:
                # print(f"跳过已处理的键: {key}")
                continue
            if key not in weights_B:
                continue
            if "lm_head" in key or "model.embed_tokens.weight" in key:
                # print(f"特殊键: {key}")
                continue

            W_A = weights_A[key].float()
            W_B = weights_B[key].float()

            lam = lam_default
            key_l = key.lower()
            if ('norm' in key_l) : # or ('layernorm' in key_l)
                # print(f"norm层: {key}")
                lam = lam_norm  # norm 更保守

            final_merged_weights[key] = ((1 - lam) * W_A + lam * W_B).to(W_A.dtype)
        
        self._save_model(final_merged_weights)

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
        self.stage2_regularized_disjoint_mask_generation()
        self.stage3_disentangled_reprojection_fusion()

# ########################################################################## #
# #                           关键代码修改区域 (4/4)                         # #
# ########################################################################## #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用I-DREAM进行最终的、兼顾性能与泛化性的模型合并。")
    
    # 基本配置
    parser.add_argument('--base_model_path', type=str, default="./downloaded_models/Qwen2-VL-7B-Instruct", help="基础模型A的路径。")
    parser.add_argument('--donor_model_path', type=str, default="./downloaded_models/llava-onevision-qwen2-7b-si-hf", help="贡献模型B的路径。")
    parser.add_argument('--original_model_path', type=str, default="./downloaded_models/Qwen2-7B-Instruct", help="原始共同祖先模型C的路径。")
    parser.add_argument('--mode', type=str, default="sams-dream-0.1-0.8-safety", help="为本次合并配置命名。")
    parser.add_argument('--cuda_device', type=int, default=2, help="使用的 CUDA 设备编号。")

    # 数据集配置 (修改为元探测数据集)
    parser.add_argument('--n_mmbench', type=int, default=40, help="用于元探测集的MMBench样本数。")
    parser.add_argument('--n_vcr', type=int, default=0, help="用于元探测集的VCR样本数。")
    parser.add_argument('--n_docvqa', type=int, default=10, help="用于元探测集的DocVQA样本数。")
    parser.add_argument('--n_vqa', type=int, default=50, help="用于元探测集的VQA v2样本数。")
    parser.add_argument('--n_scienceqa', type=int, default=50, help="用于元探测集的ScienceQA样本数。")
    parser.add_argument('--n_stvqa', type=int, default=50, help="用于元探测集的ST-VQA样本数。")
    parser.add_argument('--probe_batch_size', type=int, default=1, help="处理引导数据时的批处理大小。")

    # I-DREAM 合并超参数
    parser.add_argument('--top_k_ratio', type=float, default=0.1, help="【阶段二】用于选举关键神经元的Top-K比率。")
    parser.add_argument('--alpha', type=float, default=0.1, help="【阶段二】夏普斯惩罚系数，控制对高曲率区域的惩罚力度。")
    parser.add_argument('--lambda_proj', type=float, default=1.0, help="【阶段三】投影（相关）分量的合并系数。")
    parser.add_argument('--lambda_ortho', type=float, default=0.8, help="【阶段三】正交（无关）分量的基础合并系数。")
    parser.add_argument('--beta_safety', type=float, default=2.0, help="【阶段三】正交分量安全合并的敏感度系数。")
    parser.add_argument('--lambda_norm', type=float, default=0.0, help="norm 参数的加权平均系数（不走梯度合并）。")
    parser.add_argument('--probe_directions', type=int, default=8, help="每层缓存的输入/输出方向样本数，用于多方向投影。")  # 新增
    parser.add_argument('--rho_ortho_max_ratio', type=float, default=0.5, help="正交增量相对投影增量的最大规模比。")        # 新增
    parser.add_argument('--delta_mode', type=str, default='delta', choices=['delta','direct'], help="delta=使用W_B-W_C；direct=使用W_B。")  # 新增
    parser.add_argument('--two_sided_weights', type=bool, default=True, help="是否启用权重双侧投影。")  # 新增
    parser.add_argument('--lambda_bias', type=float, default=0.1, help="偏置合并主系数。")  # 新增
    
    # 功能性参数
    parser.add_argument('--force_recompute', action='store_true', help="强制重新计算缓存的数据。")

    args = parser.parse_args()
    
    device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"

    print("--- 配置信息 ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("--------------------")

    merger = SAMSDREAMMerger(args, device)
    merger.run_pipeline()