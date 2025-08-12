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
    """根据层名判断是否需要合并 - 仅合并LLM的权重参数。"""
    is_in_llm_layers = "language_model.layers" in name or "model.layers" in name
    if not is_in_llm_layers: return False
    if not name.endswith(".weight"): return False
    if "layernorm" in name or "embed_tokens" in name or "norm" in name or ".inv_freq" in name: return False
    return True

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
            base_prefix = ""
            if hasattr(model, 'language_model'): base_prefix = "language_model."
            elif hasattr(model, 'model'): base_prefix = "model."
            
            full_module_name = f"{base_prefix}{name}"
            
            if any(need_merge(f"{full_module_name}.{param_name}") for param_name, _ in module.named_parameters()):
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
        
        # 为每个模型使用同一个元探测数据集来缓存激活
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
            if not need_merge(key): continue
            if not (key in weights_B and key in weights_C): continue

            module_name = ".".join(key.replace("model.language_model.", "model.").split('.')[1:-1])
            
            try:
                W_A, W_B, W_C = weights_A[key].float(), weights_B[key].float(), weights_C[key].float()
                tau_A, tau_B = W_A - W_C, W_B - W_C
                
                g_approx_A = torch.outer(activations['A'][module_name]['output'], activations['A'][module_name]['input'])
                g_approx_B = torch.outer(activations['B'][module_name]['output'] - activations['C'][module_name]['output'], activations['A'][module_name]['input'])
                
                # 步骤 2: 计算夏普斯感知显著性分数 S_SAS
                saliency_A = (W_A * g_approx_A).abs()
                sharpness_penalty_A = 1 + self.args.alpha * (g_approx_A**2)
                s_sas_A = saliency_A / sharpness_penalty_A

                saliency_B = (W_B * g_approx_B).abs()
                sharpness_penalty_B = 1 + self.args.alpha * (g_approx_B**2)
                s_sas_B = saliency_B / sharpness_penalty_B

                # 步骤 3: 选举与冲突消解
                k = int(s_sas_A.numel() * self.args.top_k_ratio)
                if k == 0: continue
                
                mask_A = s_sas_A >= torch.topk(s_sas_A.flatten(), k=k, sorted=False)[0].min()
                mask_B = s_sas_B >= torch.topk(s_sas_B.flatten(), k=k, sorted=False)[0].min()
                
                tau_A, tau_B = W_A - W_C, W_B - W_C
                
                # 识别冲突集 $\mathcal{T}_{conflict, i}$，即在 $\mathcal{T}_{A,i}$ 和 $\mathcal{T}_{B,i}$ 中都存在，但任务向量符号相反的神经元。
                conflict_mask = mask_A & mask_B & (torch.sign(tau_A) != torch.sign(tau_B))
                
                # 生成最终用于模型B的非冲突掩码 $m_{B,i}$，其对应的神经元集合为 $\mathcal{T}_{B,i} - \mathcal{T}_{conflict, i}$。
                disjoint_mask_B = mask_B & (~conflict_mask)
                
                disjoint_masks[key] = disjoint_mask_B.cpu()

                # s_spec_A = (-g_approx_A * tau_A + (self.args.alpha / 2) * (g_approx_A**2) * (tau_A**2)).abs()
                # s_spec_B = (-g_approx_B * tau_B + (self.args.alpha / 2) * (g_approx_B**2) * (tau_B**2)).abs()
                # s_int_A, s_int_B = tau_A.abs(), tau_B.abs()

                # s_final_A = (2*self._min_max_normalize(s_spec_A)*self._min_max_normalize(s_int_A))/(self._min_max_normalize(s_spec_A)+self._min_max_normalize(s_int_A)+self.EPS)
                # s_final_B = (2*self._min_max_normalize(s_spec_B)*self._min_max_normalize(s_int_B))/(self._min_max_normalize(s_spec_B)+self._min_max_normalize(s_int_B)+self.EPS)

                # k = int(s_final_A.numel() * self.args.top_k_ratio)
                # if k == 0: continue
                
                # mask_A = s_final_A >= torch.topk(s_final_A.flatten(), k=k, sorted=False)[0].min()
                # mask_B = s_final_B >= torch.topk(s_final_B.flatten(), k=k, sorted=False)[0].min()
                
                # # 识别冲突集 $\mathcal{T}_{conflict, i}$，即在 $\mathcal{T}_{A,i}$ 和 $\mathcal{T}_{B,i}$ 中都存在，但任务向量符号相反的神经元。
                # conflict_mask = mask_A & mask_B & (torch.sign(tau_A) != torch.sign(tau_B))
                
                # # 生成最终用于模型B的非冲突掩码 $m_{B,i}$，其对应的神经元集合为 $\mathcal{T}_{B,i} - \mathcal{T}_{conflict, i}$。
                # disjoint_mask_B = mask_B & (~conflict_mask)
                
                # disjoint_masks[key] = disjoint_mask_B.cpu()

            except KeyError:
                pbar.set_description(f"警告: 模块 {module_name} 激活数据不完整，跳过 {key}")
                continue

        torch.save(disjoint_masks, mask_cache_path)
        print(f"I-DREAM 非冲突掩码计算完成并缓存至: {mask_cache_path}")
        
    def stage3_disentangled_reprojection_fusion(self):
        """阶段三：【SAM-S-DREAM】执行解耦重投影融合 (与I-DREAM相同)。"""
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

        final_merged_weights = weights_A.copy()
        pbar = tqdm(disjoint_masks.items(), desc="【SAM-S-DREAM】执行重投影融合")
        for key, M_prime_B in pbar:
            if not (key in weights_B and key in weights_C): continue
                
            W_A, W_B, W_C = weights_A[key].float(), weights_B[key].float(), weights_C[key].float()
            M_prime_B = M_prime_B.to(self.device)

            module_name = ".".join(key.replace("model.language_model.", "model.").split('.')[1:-1])
            
            # 准备高质量的更新向量
            tau_B = W_B - W_C
            tau_B_update = tau_B.to(self.device) * M_prime_B
            
            # 执行知识解耦 (投影分解)
            d_i = activations_A[module_name]['input'].to(self.device).float()
            d_i_norm_sq = torch.sum(d_i * d_i)

            if d_i_norm_sq > self.EPS:
                if tau_B_update.ndim > 1 and d_i.ndim == 1:
                    proj_scalar = (tau_B_update @ d_i) / d_i_norm_sq
                    tau_proj = torch.outer(proj_scalar, d_i)
                else:
                    proj_scalar = torch.sum(tau_B_update * d_i) / d_i_norm_sq
                    tau_proj = proj_scalar * d_i
                tau_ortho = tau_B_update - tau_proj
            else:
                tau_proj = torch.zeros_like(tau_B_update)
                tau_ortho = tau_B_update
            
            # 最终增广合并
            W_star = W_A.to(self.device) + self.args.lambda_proj * tau_proj + self.args.lambda_ortho * tau_ortho
            final_merged_weights[key] = W_star.cpu().to(weights_A[key].dtype)
        
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
    parser.add_argument('--mode', type=str, default="sams-dream-final", help="为本次合并配置命名。")
    parser.add_argument('--cuda_device', type=int, default=5, help="使用的 CUDA 设备编号。")

    # 数据集配置 (修改为元探测数据集)
    parser.add_argument('--n_mmbench', type=int, default=40, help="用于元探测集的MMBench样本数。")
    parser.add_argument('--n_vcr', type=int, default=0, help="用于元探测集的VCR样本数。")
    parser.add_argument('--n_docvqa', type=int, default=10, help="用于元探测集的DocVQA样本数。")
    parser.add_argument('--n_vqa', type=int, default=50, help="用于元探测集的VQA v2样本数。")
    parser.add_argument('--n_scienceqa', type=int, default=50, help="用于元探测集的ScienceQA样本数。")
    parser.add_argument('--n_stvqa', type=int, default=50, help="用于元探测集的ST-VQA样本数。")
    parser.add_argument('--probe_batch_size', type=int, default=2, help="处理引导数据时的批处理大小。")

    # I-DREAM 合并超参数
    parser.add_argument('--top_k_ratio', type=float, default=0.1, help="【阶段二】用于选举关键神经元的Top-K比率。")
    parser.add_argument('--alpha', type=float, default=0.3, help="【阶段二】夏普斯惩罚系数，控制对高曲率区域的惩罚力度。")
    parser.add_argument('--lambda_proj', type=float, default=1.0, help="【阶段三】投影（相关）分量的合并系数。")
    parser.add_argument('--lambda_ortho', type=float, default=0.7, help="【阶段三】正交（无关）分量的合并系数，保护泛化性。")
    
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