# -*- coding: utf-8 -*-
"""
ASAM v2.0: 统一、自适应与结构感知的低显存模型合并框架
本代码基于用户提供的原始方案1代码进行升级，实现了ASAM v2.0框架。
核心升级点:
1.  扩展激活缓存: 为Attention层缓存Q, K, V向量。
2.  结构感知梯度: 为Q,K,V,O投影实现基于ΔS的、理论更精确的梯度近似。
3.  自适应合并: 引入基于内部几何变化与外部功能变化比率的动态λ系数。
4.  闭环验证: 新增阶段四，用于性能评估和迭代优化。
"""
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
import torch.nn.functional as F

# 导入指定的模型和分词器类
from transformers import AutoTokenizer, AutoModel, AutoConfig
# 针对不同的模型结构，可能需要不同的AutoModel类
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq
from torch.utils.data import DataLoader, TensorDataset

# 尝试导入 Hugging Face datasets 库
try:
    from datasets import load_dataset
    from PIL import Image
    import requests
except ImportError:
    print("="*80, file=sys.stderr)
    print("错误：无法导入 'datasets' 或 'Pillow'。请运行 `pip install datasets Pillow requests`。", file=sys.stderr)
    print("这些库是获取和处理多模态探针数据集所必需的。", file=sys.stderr)
    print("="*80, file=sys.stderr)
    sys.exit(1)

# --- 权重加载与辅助函数 (大部分来自您的模板，稍作修改) ---

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
            # 尝试查找 .bin 文件，虽然不直接加载，但可以提示用户
            bin_files = [f for f in os.listdir(base_path) if f.endswith('.bin')]
            if bin_files:
                raise FileNotFoundError(f"在 {base_path} 中找到 .bin 文件但未找到 {index_filename} 或 model.safetensors。请先将模型转换为safetensors格式。")
            else:
                 raise FileNotFoundError(f"在 {base_path} 中既未找到 {index_filename} 也未找到 model.safetensors")

    with open(index_path, 'r') as f:
        index = json.load(f)

    file_list = sorted(list(set(index["weight_map"].values())))
    for file in tqdm(file_list, desc=f"从 {os.path.basename(base_path)} 加载权重"):
        weights.update(safetensors.torch.load_file(os.path.join(base_path, file)))
    return weights

def normalize_donor_keys(weights: dict, model_config) -> dict:
    """
    根据模型配置智能地标准化donor模型的key。
    常见的donor模型可能将语言模型部分包裹在 "language_model." 前缀下。
    """
    # 这是一个启发式方法，实际可能需要根据具体模型调整
    if any("language_model" in key for key in weights.keys()):
        prefix_to_remove = "language_model."
        print(f"检测到 '{prefix_to_remove}' 前缀，将进行标准化...")
        return {
            key[len(prefix_to_remove):] if key.startswith(prefix_to_remove) else key: value
            for key, value in weights.items()
        }
    return weights

def get_layer_type(name: str) -> str:
    """根据层名判断其类型 (mlp, q_proj, k_proj, v_proj, o_proj, other)。"""
    # 排除 embedding 和 norm 层
    if "embed" in name or "layernorm" in name or name.endswith(".inv_freq"):
        return "ignore"
    
    # 找到 "layers." 在名字中的位置
    layers_idx = name.rfind("layers.")
    if layers_idx == -1:
        return "ignore"
    
    suffix = name[layers_idx:]
    
    if ".mlp." in suffix:
        return "mlp"
    elif ".self_attn.q_proj" in suffix:
        return "q_proj"
    elif ".self_attn.k_proj" in suffix:
        return "k_proj"
    elif ".self_attn.v_proj" in suffix:
        return "v_proj"
    elif ".self_attn.o_proj" in suffix:
        return "o_proj"
    else:
        return "ignore"

def create_soft_link(source_path, link_path):
    """创建软链接以避免复制非模型文件。"""
    if not os.path.exists(source_path):
        print(f"错误: 源路径 '{source_path}' 不存在。")
        return
    
    os.makedirs(link_path, exist_ok=True)

    for item in os.listdir(source_path):
        source_item = os.path.join(source_path, item)
        link_item = os.path.join(link_path, item)
        
        # 跳过safetensors和bin文件，因为它们将被新生成的模型替代
        if item.endswith(('.safetensors', '.bin', '.pt')):
            continue
        
        # 如果链接已存在，跳过
        if os.path.lexists(link_item):
            continue

        try:
            os.symlink(os.path.abspath(source_item), link_item)
        except OSError as e:
            print(f"创建软链接 '{link_item}' 时出错: {e}")

# --- 核心实现类 ---

class ASAMerger:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.output_dir = os.path.join("merged_models", f"ASAM-{args.mode}")
        self.cache_dir = os.path.join(self.output_dir, "cache")
        self.grad_dir = os.path.join(self.cache_dir, "approx_grads")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.grad_dir, exist_ok=True)
        print(f"使用设备: {self.device}")
        print(f"输出将保存至: {self.output_dir}")
        
    def _get_model_and_tokenizer(self, model_path):
        """智能加载模型和分词器。"""
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # 这是一个常见的判断多模态模型的启发式方法
        is_vision_model = "vision" in config.model_type.lower() or any("Vision" in arch for arch in config.architectures)

        if is_vision_model:
            model_class = AutoModelForVision2Seq
        else:
            model_class = AutoModelForCausalLM
            
        print(f"正在以 {model_class.__name__} 格式加载模型: {model_path}")
        model = model_class.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer, config
        
    def _find_target_submodule(self, model):
        """智能地定位到包含 "layers" 的语言模型部分。"""
        # 常见的多模态模型结构
        if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
            return model.language_model.model
        elif hasattr(model, "language_model"):
            return model.language_model
        # 常见的纯语言模型结构
        elif hasattr(model, "model"):
            return model.model
        # 备用方案，直接返回模型本身
        else:
            return model

    def _get_multimodal_probe_dataloader(self, tokenizer):
        """准备多模态探针数据集。"""
        print("正在准备多模态探针数据集 (LAION)...")
        try:
            # 使用一个公开且较小的数据集
            dataset = load_dataset("laion/laion400m_experiments", "laion400m_alpha_1_1_1", split="train", streaming=True).take(self.args.probe_samples)
        except Exception as e:
            print(f"加载 LAION 数据集失败: {e}。将使用备用文本。")
            return self._get_text_probe_dataloader(tokenizer)

        image_processor = None
        # 尝试从分词器加载图像处理器，这是常见的多模态做法
        if hasattr(tokenizer, 'image_processor'):
             image_processor = tokenizer.image_processor
        else:
             from transformers import AutoImageProcessor
             # 这是一个通用的备用方案
             image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")


        def preprocess(examples):
            images = []
            texts = []
            for example in examples:
                try:
                    # 'URL' 和 'TEXT' 是 LAION 数据集中的常见字段
                    url = example.get('URL')
                    text = example.get('TEXT')
                    if not url or not text: continue
                    
                    image = Image.open(requests.get(url, stream=True, timeout=5).raw).convert("RGB")
                    images.append(image)
                    texts.append(text)
                except Exception:
                    continue # 跳过无法加载的样本
            
            if not images:
                return None

            pixel_values = image_processor(images=images, return_tensors="pt")['pixel_values']
            tokenized_text = tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            return {
                'pixel_values': pixel_values,
                'input_ids': tokenized_text['input_ids'],
                'attention_mask': tokenized_text['attention_mask']
            }

        all_processed = []
        for batch in dataset.iter(batch_size=self.args.probe_batch_size):
            processed_batch = preprocess(batch)
            if processed_batch:
                all_processed.append(processed_batch)

        if not all_processed:
            print("警告: 未能从 LAION 加载任何有效样本。将使用备用文本。")
            return self._get_text_probe_dataloader(tokenizer)

        # 将所有批次的数据合并
        pixel_values = torch.cat([b['pixel_values'] for b in all_processed])
        input_ids = torch.cat([b['input_ids'] for b in all_processed])
        attention_mask = torch.cat([b['attention_mask'] for b in all_processed])
        
        probe_dataset = TensorDataset(pixel_values, input_ids, attention_mask)
        return DataLoader(probe_dataset, batch_size=self.args.probe_batch_size)

    def _get_text_probe_dataloader(self, tokenizer):
        """备用的纯文本探针数据集。"""
        print("正在准备纯文本探针数据集 (wikitext)...")
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True).take(self.args.probe_samples)
            texts = [item['text'] for item in dataset if item['text'].strip()]
        except Exception:
            texts = ["The quick brown fox jumps over the lazy dog."] * self.args.probe_samples
        
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        probe_dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
        return DataLoader(probe_dataset, batch_size=self.args.probe_batch_size)

    def stage1_cache_activations(self):
        """执行阶段一：缓存模型A和模型C的激活。"""
        print("\n--- [阶段一: 缓存多层级功能性激活] ---")
        
        activations_paths = {
            "A": os.path.join(self.cache_dir, "activations_A.pt"),
            "C": os.path.join(self.cache_dir, "activations_C.pt")
        }
        
        model_paths = {
            "A": self.args.base_model_path,
            "C": self.args.original_model_path
        }
        
        for model_key in ["A", "C"]:
            cache_path = activations_paths[model_key]
            if os.path.exists(cache_path) and not self.args.force_recompute:
                print(f"激活缓存文件 {cache_path} 已存在，跳过。")
                continue

            model, tokenizer, _ = self._get_model_and_tokenizer(model_paths[model_key])
            is_multimodal = "vision" in model.config.model_type.lower() or any("Vision" in arch for arch in model.config.architectures)
            
            # 智能地选择dataloader
            probe_dataloader = self._get_multimodal_probe_dataloader(tokenizer) if is_multimodal else self._get_text_probe_dataloader(tokenizer)

            target_submodule = self._find_target_submodule(model)
            captured_activations = defaultdict(dict)
            hooks = []

            def get_hook(name):
                def hook_fn(module, input_tensor, output_tensor):
                    # 缓存 MLP 的输出 Y
                    if get_layer_type(name) == "mlp":
                        captured_activations[name]['Y'] = output_tensor[0].detach().cpu()
                        
                    # 缓存 Attention 模块的 Q, K, V 和最终输出 Y
                    elif get_layer_type(name) in ["q_proj", "k_proj", "v_proj"]:
                        key = get_layer_type(name)[0].upper() # Q, K, or V
                        parent_name = ".".join(name.split('.')[:-1])
                        captured_activations[parent_name][key] = output_tensor.detach().cpu()
                    elif get_layer_type(name) == "o_proj":
                        parent_name = ".".join(name.split('.')[:-1])
                        captured_activations[parent_name]['Y_attn'] = output_tensor[0].detach().cpu()

                    # 统一缓存所有模块的输入 X
                    if 'X' not in captured_activations[name]:
                         captured_activations[name]['X'] = input_tensor[0].detach().cpu()
                return hook_fn

            for name, module in target_submodule.named_modules():
                if get_layer_type(name) != "ignore":
                    hooks.append(module.register_forward_hook(get_hook(name)))
            
            model.eval()
            with torch.no_grad():
                for batch in tqdm(probe_dataloader, desc=f"前向传播 {os.path.basename(model.name_or_path)}"):
                    inputs = {k: v.to(self.device) for k, v in batch.items()} if isinstance(batch, dict) else \
                             {'input_ids': batch[0].to(self.device), 'attention_mask': batch[1].to(self.device)}
                    if is_multimodal and 'pixel_values' in inputs:
                        model(**inputs)
                    else:
                        model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            
            for h in hooks: h.remove()
            torch.save(captured_activations, cache_path)
            print(f"激活已缓存至: {cache_path}")
            
            del model, tokenizer, captured_activations, probe_dataloader
            gc.collect()
            torch.cuda.empty_cache()

    def stage2_calculate_approx_gradients(self):
        """执行阶段二：计算区分化的结构感知近似梯度。"""
        print("\n--- [阶段二: 计算结构感知近似梯度] ---")

        activations_A = torch.load(os.path.join(self.cache_dir, "activations_A.pt"), map_location="cpu")
        activations_C = torch.load(os.path.join(self.cache_dir, "activations_C.pt"), map_location="cpu")
        base_weights = load_weights(self.args.base_model_path)
        
        for key, _ in tqdm(base_weights.items(), desc="计算近似梯度"):
            layer_type = get_layer_type(key)
            if layer_type == "ignore": continue
            
            grad_path = os.path.join(self.grad_dir, f"{key.replace('/', '_')}.pt")
            if os.path.exists(grad_path) and not self.args.force_recompute: continue

            module_name = ".".join(key.split('.')[:-1])
            parent_module_name = ".".join(key.split('.')[:-2]) # For attention parts
            
            g_approx = None
            
            try:
                if layer_type == "mlp":
                    X_A = activations_A[module_name]['X'].to(self.device)
                    Y_A = activations_A[module_name]['Y'].to(self.device)
                    Y_C = activations_C[module_name]['Y'].to(self.device)
                    delta_Y_mlp = Y_A - Y_C
                    
                    if key.endswith(".weight"): g_approx = delta_Y_mlp.T @ X_A
                    else: g_approx = delta_Y_mlp.sum(dim=0)
                
                elif layer_type in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    # 加载通用激活
                    X_A = activations_A[module_name]['X'].to(self.device)
                    Y_A_attn = activations_A[parent_module_name]['Y_attn'].to(self.device)
                    Y_C_attn = activations_C[parent_module_name]['Y_attn'].to(self.device)
                    delta_Y_attn = Y_A_attn - Y_C_attn

                    # 计算ΔS
                    Q_A = activations_A[parent_module_name]['Q'].to(self.device)
                    K_A = activations_A[parent_module_name]['K'].to(self.device)
                    Q_C = activations_C[parent_module_name]['Q'].to(self.device)
                    K_C = activations_C[parent_module_name]['K'].to(self.device)
                    
                    # 逐头或平均计算
                    # 为了简化，我们在这里直接进行批次计算，实际可实现head-wise
                    S_A = Q_A @ K_A.transpose(-1, -2)
                    S_C = Q_C @ K_C.transpose(-1, -2)
                    delta_S = S_A - S_C

                    if layer_type == "q_proj":
                        g_q_space = delta_S @ K_A
                        if key.endswith(".weight"): g_approx = g_q_space.transpose(-1,-2) @ X_A
                        else: g_approx = g_q_space.sum(dim=0)
                    elif layer_type == "k_proj":
                        g_k_space = delta_S.transpose(-1,-2) @ Q_A
                        if key.endswith(".weight"): g_approx = g_k_space.transpose(-1,-2) @ X_A
                        else: g_approx = g_k_space.sum(dim=0)
                    elif layer_type == "v_proj":
                        V_A = activations_A[parent_module_name]['V'].to(self.device)
                        A_A = F.softmax(S_A, dim=-1)
                        g_v_space = A_A.transpose(-1, -2) @ delta_Y_attn
                        if key.endswith(".weight"): g_approx = g_v_space.transpose(-1, -2) @ X_A
                        else: g_approx = g_v_space.sum(dim=0)
                    elif layer_type == "o_proj":
                        V_A = activations_A[parent_module_name]['V'].to(self.device)
                        A_A = F.softmax(S_A, dim=-1)
                        Z_out_A = A_A @ V_A
                        if key.endswith(".weight"): g_approx = Z_out_A.transpose(-1, -2) @ delta_Y_attn
                        else: g_approx = Z_out_A.sum(dim=0)
            except KeyError as e:
                 print(f"警告: 计算梯度时缺少激活 {e} (来自键 {key})，跳过。")
                 continue

            if g_approx is not None:
                torch.save(g_approx.cpu(), grad_path)
                # 保存ΔS和ΔY的范数，用于阶段三
                if layer_type in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    norm_path = os.path.join(self.grad_dir, f"{parent_module_name.replace('/', '_')}_norms.pt")
                    if not os.path.exists(norm_path):
                         torch.save({
                             'delta_s_norm': torch.linalg.norm(delta_S).cpu(),
                             'delta_y_norm': torch.linalg.norm(delta_Y_attn).cpu()
                         }, norm_path)
        
        del base_weights, activations_A, activations_C
        gc.collect()
        print("所有近似梯度计算并保存完毕。")

    def stage3_merge_models(self):
        """执行阶段三：执行梯度引导的自适应合并。"""
        print("\n--- [阶段三: 梯度引导的自适应合并] ---")

        # 加载所有权重
        base_weights = load_weights(self.args.base_model_path)
        donor_raw_weights = load_weights(self.args.donor_model_path)
        original_weights = load_weights(self.args.original_model_path)
        
        # 标准化donor的key
        # 注意：这里假设base和original的结构是兼容的
        donor_config = AutoConfig.from_pretrained(self.args.donor_model_path, trust_remote_code=True)
        donor_weights = normalize_donor_keys(donor_raw_weights, donor_config)
        
        merged_weights = {}

        for key in tqdm(base_weights.keys(), desc="逐层自适应合并"):
            layer_type = get_layer_type(key)
            if layer_type == "ignore" or key not in donor_weights or key not in original_weights:
                merged_weights[key] = base_weights[key]
                continue

            grad_path = os.path.join(self.grad_dir, f"{key.replace('/', '_')}.pt")
            if not os.path.exists(grad_path):
                merged_weights[key] = base_weights[key]
                continue
            
            W_A = base_weights[key].float().to(self.device)
            W_B = donor_weights[key].float().to(self.device)
            W_C = original_weights[key].float().to(self.device)
            g_approx = torch.load(grad_path, map_location=self.device).float()

            tau_B = W_B - W_C
            
            g_norm_sq = torch.sum(g_approx * g_approx)
            if g_norm_sq < 1e-12:
                merged_weights[key] = W_A.to(base_weights[key].dtype).cpu()
                continue
                
            proj_scalar = torch.sum(g_approx * tau_B) / g_norm_sq
            tau_B_synergy = torch.clamp_min(-proj_scalar, 0) * g_approx
            tau_B_conflict = torch.clamp_min(proj_scalar, 0) * g_approx
            tau_B_ortho = tau_B - tau_B_conflict - tau_B_synergy
            
            lambda_s, lambda_c = self.args.lambda_s, self.args.lambda_c
            
            # --- 自适应赋权 ---
            if layer_type in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                parent_module_name = ".".join(key.split('.')[:-2])
                norm_path = os.path.join(self.grad_dir, f"{parent_module_name.replace('/', '_')}_norms.pt")
                if os.path.exists(norm_path):
                    norms = torch.load(norm_path)
                    delta_s_norm = norms['delta_s_norm']
                    delta_y_norm = norms['delta_y_norm']
                    if delta_y_norm > 1e-6:
                        ratio = delta_s_norm / delta_y_norm
                        # 动态调整λ
                        lambda_s = self.args.lambda_s * torch.clamp(1 - self.args.alpha * ratio, 0.5, 1.5)
                        lambda_c = self.args.lambda_c * torch.clamp(self.args.alpha * ratio, 0.1, 1.0)

            w_star = W_A + (lambda_s * tau_B_synergy) - \
                           (lambda_c * tau_B_conflict) + \
                           (self.args.lambda_o * tau_B_ortho)

            merged_weights[key] = w_star.to(base_weights[key].dtype).cpu()
            
        # 保存模型
        print("\n正在保存合并后的模型...")
        index_path = os.path.join(self.args.base_model_path, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index_map = json.load(f)["weight_map"]
        
        sharded_weights = {filename: {} for filename in set(index_map.values())}
        for key, value in merged_weights.items():
            if key in index_map:
                sharded_weights[index_map[key]][key] = value
        
        for filename, weights_dict in sharded_weights.items():
            safetensors.torch.save_file(weights_dict, os.path.join(self.output_dir, filename))
            
        shutil.copy(index_path, os.path.join(self.output_dir, os.path.basename(index_path)))
        create_soft_link(source_path=self.args.base_model_path, link_path=self.output_dir)
        print(f"模型已初步合并并保存至: {self.output_dir}")

    def stage4_validate_and_finetune(self):
        """执行阶段四：全局验证与迭代微调。"""
        print("\n--- [阶段四: 全局验证与迭代微调] ---")
        
        print("加载合并后的模型进行评估...")
        merged_model, tokenizer, _ = self._get_model_and_tokenizer(self.output_dir)
        is_multimodal = "vision" in merged_model.config.model_type.lower() or any("Vision" in arch for arch in merged_model.config.architectures)
        
        # 使用与阶段一相同的逻辑准备验证数据
        eval_dataloader = self._get_multimodal_probe_dataloader(tokenizer) if is_multimodal else self._get_text_probe_dataloader(tokenizer)

        print("正在计算评估指标 (例如：伪困惑度)...")
        total_loss = 0
        total_tokens = 0
        merged_model.eval()
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="评估中"):
                 inputs = {k: v.to(self.device) for k, v in batch.items()} if isinstance(batch, dict) else \
                         {'input_ids': batch[0].to(self.device), 'attention_mask': batch[1].to(self.device)}
                 
                 if is_multimodal and 'pixel_values' in inputs:
                     outputs = merged_model(**inputs, labels=inputs['input_ids'])
                 else:
                     outputs = merged_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['input_ids'])
                 
                 loss = outputs.loss
                 total_loss += loss.item() * inputs['input_ids'].size(0)
                 total_tokens += inputs['input_ids'].size(0)

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss != float('inf') else float('inf')
        
        print(f"合并模型评估完成 - 平均损失: {avg_loss:.4f}, 伪困惑度: {perplexity:.4f}")

        # 迭代决策逻辑
        if perplexity > self.args.perf_threshold:
            print(f"警告: 性能 ({perplexity:.4f}) 未达到阈值 ({self.args.perf_threshold}).")
            print("建议调整超参数 (例如 --alpha) 并重新运行阶段三和阶段四。")
            print("当前实现为单次评估，完整的迭代循环需要您根据此输出来调整启动参数。")
        else:
            print("性能达标！最终模型已验证。")

    def run_pipeline(self):
        """按顺序执行所有阶段。"""
        self.stage1_cache_activations()
        self.stage2_calculate_approx_gradients()
        self.stage3_merge_models()
        self.stage4_validate_and_finetune()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASAM v2.0: 统一、自适应与结构感知的低显存模型合并框架。")
    
    # 基本配置
    parser.add_argument('--base_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct", help="基础模型A的路径。")
    parser.add_argument('--donor_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si-hf", help="贡献模型B的路径。")
    parser.add_argument('--original_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-7B-Instruct", help="原始共同祖先模型C的路径。")
    parser.add_argument('--mode', type=str, default="default", help="为本次合并配置命名。")
    parser.add_argument('--cuda_device', type=int, default=3, help="使用的 CUDA 设备编号。")

    # 探针数据集配置
    parser.add_argument('--probe_samples', type=int, default=200, help="用于探测的样本数量。")
    parser.add_argument('--probe_batch_size', type=int, default=1, help="探测时的批处理大小，如果显存不足请减小。")

    # 自适应合并超参数
    parser.add_argument('--lambda_s', type=float, default=1.4, help="协同分量的基础系数。")
    parser.add_argument('--lambda_c', type=float, default=0.7, help="冲突分量的基础系数。")
    parser.add_argument('--lambda_o', type=float, default=1.0, help="正交分量的系数。")
    parser.add_argument('--alpha', type=float, default=0.5, help="自适应赋权机制的敏感度系数。")
    
    # 验证阶段配置
    parser.add_argument('--perf_threshold', type=float, default=20.0, help="可接受的最大困惑度或其他性能指标阈值。")
    
    # 功能性参数
    parser.add_argument('--force_recompute', action='store_true', help="强制重新计算缓存的激活或梯度。")

    args = parser.parse_args()
    
    # 设置设备
    if torch.cuda.is_available() and args.cuda_device < torch.cuda.device_count():
        device = f"cuda:{args.cuda_device}"
    else:
        device = "cpu"

    print("--- ASAM v2.0 配置信息 ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("--------------------")

    merger = ASAMerger(args, device)
    merger.run_pipeline()