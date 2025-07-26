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
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoModelForCausalLM, AutoProcessor
from torch.utils.data import DataLoader, TensorDataset

# 尝试导入 Hugging Face datasets 库
try:
    from datasets import load_dataset
    from PIL import Image
    import requests
except ImportError:
    print("="*80, file=sys.stderr)
    print("错误：无法导入 'datasets', 'Pillow', 或 'requests'。请运行 `pip install datasets Pillow requests`。", file=sys.stderr)
    print("这些库是获取多模态探针数据集所必需的。", file=sys.stderr)
    print("="*80, file=sys.stderr)
    sys.exit(1)

# --- 权重加载与辅助函数 (大部分保持不变) ---

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

def normalize_donor_keys(weights: dict) -> dict:
    """专门用于标准化可能带有不同前缀的 donor 模型。"""
    # 增加更多可能的语言模型前缀
    prefixes_to_remove = ["language_model.", "model."]
    normalized_weights = {}
    for key, value in weights.items():
        original_key = key
        for prefix in prefixes_to_remove:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break # 找到一个匹配的前缀就停止
        normalized_weights[key] = value
    return normalized_weights

def create_soft_link(source_path, link_path):
    """创建必要的符号链接来构成一个完整的模型目录。"""
    if not os.path.exists(link_path):
        os.makedirs(link_path)

    for item in os.listdir(source_path):
        source_item = os.path.join(os.path.abspath(source_path), item)
        link_item = os.path.join(link_path, item)
        
        # 避免创建指向自身的链接或覆盖已存在的文件
        if os.path.exists(link_item):
            continue
        
        # 忽略二进制权重文件，因为我们会自己生成它们
        if item.endswith('.safetensors') or item.endswith('.bin'):
            continue
        
        try:
            os.symlink(source_item, link_item)
        except OSError as e:
            print(f"创建符号链接失败 '{link_item}': {e}", file=sys.stderr)


# --- 核心实现类: ASAMerger ---

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
        
        # 存储计算出的范数，用于阶段三的自适应计算
        self.norm_cache = {}

    # =================================================================================
    # ASAM v2.0 关键修改: 新的 `need_merge` 逻辑
    # 目的：现在我们需要合并所有的 Attention 层 (Q,K,V,O) 和 MLP 层
    # =================================================================================
    def _is_merge_target(self, param_name: str) -> (bool, str):
        """
        判断一个参数是否属于合并目标，并返回其类型（'mlp' or 'attn'）。
        """
        # 标准化，去除顶层模型前缀，如 'model.' 或 'language_model.'
        for prefix in ["language_model.model.", "language_model.", "model."]:
            if param_name.startswith(prefix):
                param_name = param_name[len(prefix):]
                break

        if not param_name.startswith("layers."):
            return False, None

        if "layernorm" in param_name or "norm" in param_name:
            return False, None
        
        if "rotary_emb" in param_name:
            return False, None

        if "mlp" in param_name:
            return True, "mlp"
        
        if "self_attn" in param_name:
            return True, "attn"

        return False, None

    def _get_model_and_tokenizer(self, model_path):
        """
        根据模型类型加载模型和分词器。
        修正：始终使用 AutoTokenizer 加载分词器，以避免在纯文本输入时触发图像处理器。
        """
        is_vision_model = "VL" in model_path or "llava" in model_path.lower()
        
        # 统一加载分词器，这是关键修正点
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # 根据模型类型加载模型
        if is_vision_model:
            try:
                model = AutoModelForVision2Seq.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
                )
                # 对于多模态模型，返回的是模型和分词器
                return model, tokenizer
            except Exception as e:
                print(f"警告: 尝试作为多模态模型加载 {model_path} 失败: {e}。将回退到纯语言模型加载。")

        # 对于纯语言模型或多模态模型加载失败的情况
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        return model, tokenizer

    def _find_target_submodule(self, model):
        """智能地定位到包含 'layers' 的子模块。"""
        if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
            return model.language_model.model
        if hasattr(model, "language_model"):
            return model.language_model
        if hasattr(model, "model"):
            return model.model
        return model

    # =================================================================================
    # ASAM v2.0 阶段一: 增强的激活缓存
    # 目的：为 Attention 层额外缓存 Q, K, V 向量。
    # =================================================================================
    def stage1_cache_activations(self):
        """执行阶段一：缓存模型A和模型C的多层级功能性激活。"""
        print("\n--- [阶段一: 功能性激活缓存] ---")
        
        # 缓存模型A (基础模型)
        self._cache_activations_for_model(
            model_name="Model A (Base)",
            model_path=self.args.base_model_path,
            cache_path=os.path.join(self.cache_dir, "activations_A.pt"),
            capture_qkv=True, # 捕获 QKV
            capture_inputs=True, # 捕获输入
            capture_outputs=True, # 捕获输出
        )
        
        # 缓存模型C (共同祖先)
        self._cache_activations_for_model(
            model_name="Model C (Ancestor)",
            model_path=self.args.original_model_path,
            cache_path=os.path.join(self.cache_dir, "activations_C.pt"),
            capture_qkv=True, # 捕获 QKV
            capture_inputs=False, # 无需捕获输入
            capture_outputs=True, # 捕获输出
        )

    def _cache_activations_for_model(self, model_name, model_path, cache_path, capture_qkv, capture_inputs, capture_outputs):
        """阶段一的核心函数：为单个模型执行前向传播并缓存激活。"""
        if os.path.exists(cache_path) and not self.args.force_recompute:
            print(f"{model_name} 的激活缓存文件已存在，跳过。")
            return

        print(f"正在为 {model_name} ({os.path.basename(model_path)}) 缓存激活...")
        
        model, tokenizer = self._get_model_and_tokenizer(model_path)
        model.to(self.device).eval()
        
        model_to_hook = self._find_target_submodule(model)

        # 修正 1: 初始化用于收集所有批次激活的列表
        captured_activations = defaultdict(lambda: defaultdict(list))
        hooks = []

        def get_hook(module_name, module_type):
            def hook_fn(module, input, output):
                if not input:
                    return
                
                # 修正 2: 在钩子函数中直接向列表追加(append)激活，而不是覆盖
                if module_type == 'mlp':
                    if capture_inputs and len(input) > 0: 
                        captured_activations[module_name]['input'].append(input[0].detach().cpu())
                    if capture_outputs: 
                        if isinstance(output, tuple):
                            if not output: return
                            output_tensor = output[0]
                        else:
                            output_tensor = output
                        
                        if not isinstance(output_tensor, torch.Tensor): return
                        captured_activations[module_name]['output'].append(output_tensor.detach().cpu())
                
                elif module_type == 'attn':
                    if capture_inputs and len(input) > 0: 
                        captured_activations[module_name]['input'].append(input[0].detach().cpu())
                    if capture_outputs: 
                        if isinstance(output, tuple):
                            if not output: return
                            output_tensor = output[0]
                        else:
                            output_tensor = output
                        
                        if not isinstance(output_tensor, torch.Tensor): return
                        captured_activations[module_name]['output'].append(output_tensor.detach().cpu())

                    if capture_qkv and len(input) > 0:
                        try:
                            if hasattr(module, 'q_proj'):
                                q = module.q_proj(input[0])
                                captured_activations[module_name]['q'].append(q.detach().cpu())
                            if hasattr(module, 'k_proj'):
                                k = module.k_proj(input[0])
                                captured_activations[module_name]['k'].append(k.detach().cpu())
                            if hasattr(module, 'v_proj'):
                                v = module.v_proj(input[0])
                                captured_activations[module_name]['v'].append(v.detach().cpu())
                        except Exception as e:
                            print(f"提取 Q/K/V 时出错: {e}")
            return hook_fn

        for name, module in model_to_hook.named_modules():
            is_target, module_type = self._is_merge_target(name + ".weight")
            if is_target:
                if module_type == 'attn':
                    if name.endswith("self_attn"):
                         hooks.append(module.register_forward_hook(get_hook(name, 'attn')))
                elif module_type == 'mlp':
                     if name.endswith("mlp"):
                        hooks.append(module.register_forward_hook(get_hook(name, 'mlp')))

        print(f"在 {model_name} 中注册了 {len(hooks)} 个钩子。")

        is_vision_model = "VL" in model_path or "llava" in model_path.lower()
        
        # 修正 3: 简化前向传播循环，钩子函数会直接填充好所有数据
        with torch.no_grad():
            try:
                probe_dataset_raw = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True).take(self.args.probe_samples)
                probe_texts = [item['text'] for item in probe_dataset_raw if item['text']]
            except Exception as e:
                print(f"加载 wikitext 数据集失败: {e}")
                probe_texts = [
                    "The quick brown fox jumps over the lazy dog.",
                    "Machine learning models are trained on large datasets.",
                    "Neural networks process information in a hierarchical manner."
                ] * (self.args.probe_samples // 3 + 1)
                probe_texts = probe_texts[:self.args.probe_samples]
    
            if is_vision_model:
                formatted_texts = [f"<|user|>\n{text}<|endoftext|><|assistant|>" for text in probe_texts]
                probe_inputs = tokenizer(formatted_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            else:
                probe_inputs = tokenizer(probe_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            
            probe_dataset = TensorDataset(probe_inputs['input_ids'], probe_inputs['attention_mask'])
            probe_dataloader = DataLoader(probe_dataset, batch_size=8)

            for batch in tqdm(probe_dataloader, desc=f"前向传播 {model_name}"):
                input_ids, attention_mask = batch[0].to(self.device), batch[1].to(self.device)
                model(input_ids=input_ids, attention_mask=attention_mask)

        for h in hooks: h.remove()

        # 修正 4: 使用 torch.cat 处理不同大小的批次，然后计算平均值
        averaged_activations = {}
        for module_name, activations_dict in captured_activations.items():
            averaged_activations[module_name] = {}
            for key, tensor_list in activations_dict.items():
                if tensor_list:
                    concatenated_tensor = torch.cat(tensor_list, dim=0)
                    averaged_activations[module_name][key] = torch.mean(concatenated_tensor.float(), dim=0)
        
        torch.save(averaged_activations, cache_path)
        print(f"{model_name} 的激活已缓存至: {cache_path}")

        del model, tokenizer, captured_activations, averaged_activations
        gc.collect()
        torch.cuda.empty_cache()

    # =================================================================================
    # ASAM v2.0 阶段二: 区分化的结构感知近似梯度计算
    # 目的：为 MLP 和 Attention 层使用不同的、更精确的梯度近似公式。
    # =================================================================================
    def stage2_calculate_approx_gradients(self):
        """执行阶段二：计算区分化的、结构感知的近似梯度。"""
        print("\n--- [阶段二: 结构感知近似梯度计算] ---")
        
        activations_A = torch.load(os.path.join(self.cache_dir, "activations_A.pt"), map_location="cpu")
        activations_C = torch.load(os.path.join(self.cache_dir, "activations_C.pt"), map_location="cpu")
        base_weights = load_weights(self.args.base_model_path)
        
        # 获取模型配置以得到多头注意力的头数
        config = AutoModelForVision2Seq.from_pretrained(self.args.base_model_path).config
        # 兼容纯语言模型
        lang_config = getattr(config, "language_config", config)
        num_heads = lang_config.num_attention_heads
        head_dim = lang_config.hidden_size // num_heads

        for key in tqdm(base_weights.keys(), desc="计算近似梯度"):
            is_target, module_type = self._is_merge_target(key)
            if not is_target:
                continue
            
            grad_path = os.path.join(self.grad_dir, f"{key.replace('/', '_')}.pt")
            if os.path.exists(grad_path) and not self.args.force_recompute:
                continue

            # 修正：从绝对路径key中提取出相对于语言模型部分的模块名
            layers_idx = key.rfind("layers.")
            if layers_idx == -1: continue
            relative_key = key[layers_idx:]
            module_name = ".".join(relative_key.split('.')[:-1])
            
            # 调试信息
            if module_name not in activations_A:
                print(f"警告: 键 '{module_name}' 不在激活缓存中。可用的键: {list(activations_A.keys())[:5]}...")
                continue
                
            if 'input' not in activations_A[module_name]:
                print(f"警告: 键 '{module_name}' 存在，但没有'input'。可用的子键: {list(activations_A[module_name].keys())}")
                continue

            # 加载激活到设备
            X_A = activations_A[module_name]['input'].to(self.device)
            Y_A = activations_A[module_name]['output'].to(self.device)
            Y_C = activations_C[module_name]['output'].to(self.device)
            
            delta_Y = Y_A - Y_C
            
            g_approx = None
            if key.endswith(".weight"):
                g_approx = delta_Y.T @ X_A
            elif key.endswith(".bias"):
                g_approx = delta_Y.sum(dim=0)
        
            # --- Attention 参数梯度计算 ---
            elif module_type == "attn":
                X_A = activations_A[module_name]['input'].to(self.device)
                Q_A = activations_A[module_name]['q'].to(self.device)
                K_A = activations_A[module_name]['k'].to(self.device)
                V_A = activations_A[module_name]['v'].to(self.device)
                Y_A_attn = activations_A[module_name]['output'].to(self.device)

                Q_C = activations_C[module_name]['q'].to(self.device)
                K_C = activations_C[module_name]['k'].to(self.device)
                Y_C_attn = activations_C[module_name]['output'].to(self.device)
                
                # Reshape for multi-head attention
                # (seq_len, hidden_size) -> (seq_len, num_heads, head_dim) -> (num_heads, seq_len, head_dim)
                def reshape_for_heads(tensor):
                    return tensor.view(tensor.shape[0], num_heads, head_dim).transpose(0, 1)

                Q_A_h, K_A_h, V_A_h = map(reshape_for_heads, (Q_A, K_A, V_A))
                Q_C_h, K_C_h = map(reshape_for_heads, (Q_C, K_C))

                # a. 计算核心结构变化信号 ΔS (逐头)
                S_A = (Q_A_h @ K_A_h.transpose(-1, -2)) / (head_dim ** 0.5)
                S_C = (Q_C_h @ K_C_h.transpose(-1, -2)) / (head_dim ** 0.5)
                delta_S = S_A - S_C

                # b. 缓存范数用于阶段三
                delta_Y_attn = Y_A_attn - Y_C_attn
                self.norm_cache[module_name] = {
                    "delta_S_norm": torch.linalg.norm(delta_S).cpu().item(),
                    "delta_Y_attn_norm": torch.linalg.norm(delta_Y_attn).cpu().item()
                }

                # c. 根据参数类型，使用链式法则近似梯度
                g_approx = None
                if key.endswith("q_proj.weight"):
                    g_Q_space = delta_S @ K_A_h
                    g_Q_space = g_Q_space.transpose(0, 1).reshape(Q_A.shape) # Back to (seq_len, hidden_size)
                    g_approx = g_Q_space.T @ X_A
                
                elif key.endswith("k_proj.weight"):
                    g_K_space = delta_S.transpose(-1, -2) @ Q_A_h
                    g_K_space = g_K_space.transpose(0, 1).reshape(K_A.shape)
                    g_approx = g_K_space.T @ X_A

                elif key.endswith("v_proj.weight"):
                    A_A = F.softmax(S_A, dim=-1)
                    g_V_space = A_A.transpose(-1, -2) @ reshape_for_heads(delta_Y_attn)
                    g_V_space = g_V_space.transpose(0, 1).reshape(V_A.shape)
                    g_approx = g_V_space.T @ X_A

                elif key.endswith("o_proj.weight"):
                    A_A = F.softmax(S_A, dim=-1)
                    Z_out_A = A_A @ V_A_h
                    Z_out_A = Z_out_A.transpose(0, 1).reshape(V_A.shape[0], -1) # (seq_len, hidden_size)
                    g_approx = Z_out_A.T @ delta_Y_attn

            if g_approx is not None:
                torch.save(g_approx.cpu(), grad_path)

        # 保存范数缓存
        torch.save(self.norm_cache, os.path.join(self.cache_dir, "norm_cache.pt"))
        print("所有近似梯度计算并保存完毕。")
        del base_weights, activations_A, activations_C
        gc.collect()

    # =================================================================================
    # ASAM v2.0 阶段三: 自适应梯度引导合并
    # 目的：为 Attention 层实现基于 ||ΔS||/||ΔY|| 的动态 λ 系数调整。
    # =================================================================================
    def stage3_merge_models(self):
        """执行阶段三：使用自适应权重进行梯度引导的分解与合并。"""
        print("\n--- [阶段三: 自适应分解与合并] ---")

        base_weights = load_weights(self.args.base_model_path)
        donor_weights_raw = load_weights(self.args.donor_model_path)
        original_weights = load_weights(self.args.original_model_path)
        self.norm_cache = torch.load(os.path.join(self.cache_dir, "norm_cache.pt"))

        donor_weights = normalize_donor_keys(donor_weights_raw)
        merged_weights = {}

        for key in tqdm(base_weights.keys(), desc="逐层自适应合并权重"):
            merged_weights[key] = base_weights[key]
            
            is_target, module_type = self._is_merge_target(key)
            
            if is_target and key in donor_weights and key in original_weights and base_weights[key].shape == donor_weights[key].shape:
                grad_path = os.path.join(self.grad_dir, f"{key.replace('/', '_')}.pt")
                if not os.path.exists(grad_path): continue

                W_A = base_weights[key].float().to(self.device)
                W_B = donor_weights[key].float().to(self.device)
                W_C = original_weights[key].float().to(self.device)
                g_approx = torch.load(grad_path, map_location=self.device).float()

                tau_B = W_B - W_C
                g_norm_sq = torch.sum(g_approx * g_approx)

                if g_norm_sq < 1e-9:
                    w_star = W_A + self.args.lambda_o * tau_B
                else:
                    proj_scalar = torch.sum(g_approx * tau_B) / g_norm_sq
                    tau_B_synergy = torch.clamp_min(-proj_scalar, 0) * g_approx
                    tau_B_conflict = torch.clamp_min(proj_scalar, 0) * g_approx
                    tau_B_ortho = tau_B - tau_B_conflict - tau_B_synergy

                    # --- 自适应赋权 ---
                    lambda_s, lambda_c = self.args.lambda_s, self.args.lambda_c
                    if module_type == "attn":
                        module_name = ".".join(key.split('.')[:-2])
                        if module_name in self.norm_cache:
                            norms = self.norm_cache[module_name]
                            if norms["delta_Y_attn_norm"] > 1e-6:
                                ratio = norms["delta_S_norm"] / norms["delta_Y_attn_norm"]
                                alpha = self.args.alpha
                                # 动态调整 λs 和 λc
                                lambda_s = self.args.lambda_s * max(0.5, 1 - alpha * ratio)
                                lambda_c = self.args.lambda_c * min(1.5, 0.1 + alpha * ratio)

                    w_star = W_A + (lambda_s * tau_B_synergy) - \
                                 (lambda_c * tau_B_conflict) + \
                                 (self.args.lambda_o * tau_B_ortho)
                
                merged_weights[key] = w_star.to(base_weights[key].dtype).cpu()
        
        # --- 保存模型 ---
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
        print(f"模型成功合并并保存至: {self.output_dir}")
        
    # =================================================================================
    # ASAM v2.0 阶段四: 全局验证与微调
    # 目的：提供一个量化的性能反馈，为迭代优化提供基础。
    # =================================================================================
    def stage4_validate_and_finalize(self):
        """执行阶段四：加载合并模型并进行快速验证。"""
        print("\n--- [阶段四: 全局验证与最终确认] ---")
        
        if not self.args.run_validation:
            print("跳过验证阶段。合并流程完成！")
            return

        print("正在加载合并后的模型进行验证...")
        try:
            model, processor = self._get_model_and_tokenizer(self.output_dir)
            model.to(self.device).eval()
        except Exception as e:
            print(f"加载合并模型失败: {e}. 无法进行验证。", file=sys.stderr)
            return

        # 使用与阶段一相同的探针数据集进行一个简单的困惑度或损失计算
        try:
            probe_dataset_raw = load_dataset("laion/laion400m", split="train", streaming=True).take(50) # 使用少量样本验证
            probe_data = list(probe_dataset_raw)
        except:
             probe_data = [{"TEXT": "a cat sitting on a couch", "URL": "http://images.cocodataset.org/val2017/000000039769.jpg"}] * 50

        total_loss = 0
        num_samples = 0
        with torch.no_grad():
            for item in tqdm(probe_data, desc="正在验证模型"):
                text = item.get("TEXT", "")
                image_url = item.get("URL", "")
                try:
                    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
                    inputs = processor(text=[f"<|user|>\n<|image_1|>\n{text}<|endoftext|><|assistant|>"], images=[image], return_tensors="pt").to(self.device)
                    outputs = model(**inputs, labels=inputs.input_ids)
                    total_loss += outputs.loss.item()
                    num_samples += 1
                except Exception:
                    continue
        
        if num_samples > 0:
            avg_loss = total_loss / num_samples
            print(f"\n验证完成。平均损失: {avg_loss:.4f}")
            # 这里可以加入迭代逻辑，例如：
            # if avg_loss > TARGET_LOSS:
            #     print("性能未达标，建议调整 alpha 或其他超参数并重新运行阶段三。")
            # else:
            #     print("性能达标！")
        else:
            print("未能成功处理任何验证样本。")
            
        print("ASAM v2.0 框架所有阶段执行完毕！")

    def run_pipeline(self):
        """按顺序执行所有阶段。"""
        self.stage1_cache_activations()
        self.stage2_calculate_approx_gradients()
        self.stage3_merge_models()
        self.stage4_validate_and_finalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASAM v2.0: 统一、自适应与结构感知的模型合并框架。")
    
    # 基本配置
    parser.add_argument('--base_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct", help="基础模型A的路径。")
    parser.add_argument('--donor_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si-hf", help="贡献模型B的路径。")
    parser.add_argument('--original_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-7B-Instruct", help="原始共同祖先模型C的路径。")
    parser.add_argument('--mode', type=str, default="default", help="为本次合并配置命名。")
    parser.add_argument('--cuda_device', type=int, default=6, help="使用的 CUDA 设备编号。")

    # 探针数据集配置
    parser.add_argument('--probe_samples', type=int, default=200, help="用于探测的样本数量。")

    # 合并超参数
    parser.add_argument('--lambda_s', type=float, default=1.4, help="协同分量的基础系数。")
    parser.add_argument('--lambda_c', type=float, default=0.7, help="冲突分量的基础系数（在公式中为减去）。")
    parser.add_argument('--lambda_o', type=float, default=1.0, help="正交分量的系数。")
    parser.add_argument('--alpha', type=float, default=0.5, help="自适应赋权机制的敏感度系数。")
    
    # 功能性参数
    parser.add_argument('--force_recompute', action='store_true', help="强制重新计算缓存的激活或梯度。")
    parser.add_argument('--run_validation', action='store_true', help="在合并后执行阶段四的验证。")

    args = parser.parse_args()
    
    # 设置设备
    if torch.cuda.is_available() and args.cuda_device < torch.cuda.device_count():
        device = f"cuda:{args.cuda_device}"
    else:
        device = "cpu"

    print("--- ASAM v2.0 合并框架配置 ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("--------------------------------")

    merger = ASAMerger(args, device)
    merger.run_pipeline()