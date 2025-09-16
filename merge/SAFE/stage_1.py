import argparse
from collections import defaultdict
import gc
import os
import random
from seaborn import load_dataset
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from transformers import TextStreamer
# 注意注释
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

# 注意注释
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model


#liuhaotian/llava-v1.5-7b  
def load_model(args, device):
    """加载指定的模型和处理器。"""
    
    processor = AutoProcessor.from_pretrained(args.target_model_path)
    if "vision" in args.target_model_path:
        model = AutoModelForVision2Seq.from_pretrained(args.target_model_path).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.target_model_path).to(device)
    return model, processor

def load_llava_v1_5(args):
    """加载 liuhaotian/llava-v1.5-7b 模型的权重。"""
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name)
    print(model)

    return tokenizer, model, image_processor, context_len

def load_mplugowl2(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")
    return tokenizer, model, image_processor, context_len



def inference_mplugowl2(args, image_file, query):
    tokenizer, model, image_processor, context_len = load_mplugowl2(args)
    conv = conv_templates["mplug_owl2"].copy()
    roles = conv.roles

    image = Image.open(image_file).convert('RGB')
    max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
    image = image.resize((max_edge, max_edge))

    image_tensor = process_images([image], image_processor)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    inp = DEFAULT_IMAGE_TOKEN + query
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    temperature = 0.7
    max_new_tokens = 512

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(outputs)


def inference_llava_v1_5(args, image_file, query):
    from llava.eval.run_llava import eval_model

    model_path = args.model_path
    prompt = query

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    eval_model(args)





def check_model_load(args):
    if "mplug-owl2" in args.model_path.lower():
        return load_mplugowl2(args)
    elif "llava-v1.5-7b" in args.model_path.lower():
        return load_llava_v1_5(args)
    return "unknown"

class CacheActivations:
    def __init__(self, args):
        self.args = args

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

    def need_merge(self, name: str) -> bool:
        """
        SAM-S-DREAM的复杂合并目标：
        - 仅处理 transformer layers 内部的线性权重与 bias
        - 显式排除所有 norm 与 rotary_emb
        """
        is_in_layers = name.startswith("model.layers.") or name.startswith("language_model.layers.") or name.startswith("language_model.model.layers.")
        if not is_in_layers:
            return False

        # 显式排除 norm 和 rotary
        if 'layernorm' in name or 'norm' in name or 'rotary_emb' in name:
            return False

        # 线性层的 .weight/.bias 进入复杂合并
        if name.endswith('.weight') or name.endswith('.bias'):
            return True

        return False

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
            if any(self.need_merge(f"{full_module_name_prefix}.{param_name}") for param_name, _ in module.named_parameters()):
                 module_map[name] = module

        return module_map
    

    def _cache_activations_raw(self, model_info, model_path, required_activations, probe_dataset_list):
        """为每个模型从原始数据集处理数据并缓存激活（内存优化版）。"""
        cache_path = os.path.join(self.cache_dir, f"activations_{model_info}.pt")
        if os.path.exists(cache_path) and not self.args.force_recompute:
            print(f"激活缓存文件 {cache_path} 已存在, 跳过。")
            return

        print(f"正在为 {model_info} ({os.path.basename(model_path)}) 缓存激活...")
        is_vision_model = "VL" in model_path or "llava" in model_path.lower()
        is_llava = "llava" in model_path.lower()
        is_mplugowl2 = "mplug-owl2" in model_path.lower()
        
        model, processor = load_model(self.args, self.device)
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

    def cache(self):
                # 调用新函数来构建元探测数据集
        meta_probe_dataset = self._create_meta_probe_dataset()
        
        # 为每个模型使用同一个元探测集来缓存激活
        self._cache_activations_raw("A", self.args.base_model_path, ["input", "output"], meta_probe_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="缓存激活。")
    
    # 基本配置
    parser.add_argument('--target_model_path', type=str, default="./downloaded_models/Qwen2-VL-7B-Instruct", help="基础模型A的路径。")
    parser.add_argument('--mode', type=str, default="sams-dream-0.1-0.8-norm", help="用于缓存的地址")
    parser.add_argument('--cuda_device', type=int, default=6, help="使用的 CUDA 设备编号。")

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
    parser.add_argument('--lambda_ortho', type=float, default=0.8, help="【阶段三】正交（无关）分量的合并系数，保护泛化性。")
    parser.add_argument('--lambda_norm', type=float, default=0.0, help="norm 参数的加权平均系数（不走梯度合并）。")
    
    # 功能性参数
    parser.add_argument('--force_recompute', action='store_true', help="强制重新计算缓存的数据。")

    args = parser.parse_args()
    
    device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"

    print("--- 配置信息 ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("--------------------")

    CacheActivations(args=args)