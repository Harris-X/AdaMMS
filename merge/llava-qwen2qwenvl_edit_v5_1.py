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
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import DataLoader, TensorDataset

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

def normalize_keys(weights: dict, prefixes_to_remove=["language_model.model.", "model."]) -> dict:
    """
    通用key标准化函数。
    修正：确保能处理多种前缀，并按最长前缀优先的顺序移除。
    """
    # 按长度降序排序，以优先匹配更具体的前缀
    # 例如，优先匹配 "model.language_model." 而不是 "model."
    sorted_prefixes = sorted(prefixes_to_remove, key=len, reverse=True)
    
    normalized_weights = {}
    for key, value in weights.items():
        processed_key = key
        for prefix in sorted_prefixes:
            if processed_key.startswith(prefix):
                processed_key = processed_key[len(prefix):]
                break # 找到第一个匹配项后就停止
        normalized_weights[processed_key] = value
    return normalized_weights


def create_soft_link(source_path, link_path):
    """创建必要的符号链接来构成一个完整的模型目录。"""
    if not os.path.exists(link_path):
        os.makedirs(link_path)
    
    # 链接非权重文件
    for item in os.listdir(source_path):
        if item.endswith(('.safetensors', '.bin')):
            continue
        
        source_item = os.path.join(os.path.abspath(source_path), item)
        link_item = os.path.join(link_path, item)
        
        if not os.path.exists(link_item):
            try:
                os.symlink(source_item, link_item)
            except OSError as e:
                print(f"创建符号链接失败 '{link_item}': {e}", file=sys.stderr)

# --- 核心实现类: ASAMerger 3.0 ---

class ASAMerger:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.output_dir = os.path.join("merged_models", f"ASAM3-{args.mode}")
        self.cache_dir = os.path.join(self.output_dir, "cache")
        
        # ASAM 3.0: 为对称梯度创建独立的目录
        self.grad_dir_A = os.path.join(self.cache_dir, "approx_grads_A")
        self.grad_dir_B = os.path.join(self.cache_dir, "approx_grads_B")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.grad_dir_A, exist_ok=True)
        os.makedirs(self.grad_dir_B, exist_ok=True)
        print(f"使用设备: {self.device}")
        print(f"输出将保存至: {self.output_dir}")
        
        self.merge_scope = {}

    def _is_llm_param(self, param_name: str) -> bool:
        """
        判断一个参数是否属于语言模型部分。
        修正：通过检查 'layers' 关键字并排除视觉关键字来更灵活地识别LLM参数。
        """
        # 视觉模型的参数通常包含 'visual' 或 'vision_tower'
        if 'visual' in param_name or 'vision_tower' in param_name:
            return False
        # 语言模型的 Transformer 层通常包含 'layers'
        if 'layers' in param_name:
            return True
        return False

    def _should_merge(self, param_name: str) -> bool:
        """判断一个参数是否在合并范围内。"""
        if not self._is_llm_param(param_name):
            return False
        
        # 排除列表
        if any(excluded in param_name for excluded in ["embed_tokens", "lm_head", "layernorm"]):
            return False
        
        return True

    def stage0_define_scope(self):
        """阶段零：定义合并范围，区分语言模型和视觉/连接器组件。"""
        print("\n--- [阶段零: 定义合并范围] ---")
        base_weights = load_weights(self.args.base_model_path)
        
        self.merge_scope['language_model'] = {k for k in base_weights if self._should_merge(k)}
        self.merge_scope['preserve_from_A'] = {k for k in base_weights if not self._should_merge(k)}
        
        print(f"识别到 {len(self.merge_scope['language_model'])} 个可合并的语言模型参数。")
        print(f"将保留来自模型A的 {len(self.merge_scope['preserve_from_A'])} 个非语言模型或被排除的参数。")
        del base_weights
        gc.collect()

    def _find_target_submodule(self, model):
        """智能地定位到包含 'layers' 的子模块。"""
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            return model.model.language_model
        # 为纯语言模型添加路径
        if hasattr(model, "model"):
            return model.model
        return model

    def _get_model_and_tokenizer(self, model_path):
        """
        根据模型类型智能加载模型和分词器。
        参考 v4.3.2 的实现。
        """
        is_vision_model = "VL" in model_path or "llava" in model_path.lower()
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if is_vision_model:
            try:
                model = AutoModelForVision2Seq.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
                )
                return model, tokenizer
            except Exception as e:
                print(f"警告: 尝试作为多模态模型加载 {model_path} 失败: {e}。将回退到纯语言模型加载。")

        # 对于纯语言模型或多模态模型加载失败的情况
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        return model, tokenizer

    def _cache_activations_for_model(self, model_name, model_path, cache_path, capture_inputs):
        """阶段一的核心函数：为单个模型执行前向传播并缓存激活。"""
        if os.path.exists(cache_path) and not self.args.force_recompute:
            print(f"{model_name} 的激活缓存文件已存在，跳过。")
            return

        print(f"正在为 {model_name} ({os.path.basename(model_path)}) 缓存激活...")
        model, tokenizer = self._get_model_and_tokenizer(model_path)
        model.to(self.device)
        
        model_to_hook = self._find_target_submodule(model)

        captured_activations = defaultdict(lambda: defaultdict(list))
        hooks = []

        # 修正1：修改钩子函数签名以接收 kwargs
        def get_hook(module_name, module_type):
            def hook_fn(module, args, kwargs, output):
                # 修正2：健壮地获取输入张量
                input_tensor = None
                if args:
                    input_tensor = args[0]
                elif "hidden_states" in kwargs:
                    input_tensor = kwargs["hidden_states"]
                
                if input_tensor is None:
                    # print(f"警告: 无法在模块 {module_name} 中找到输入张量。")
                    return

                output_tensor = output[0] if isinstance(output, tuple) else output

                if capture_inputs:
                    captured_activations[module_name]['input'].append(input_tensor.detach().cpu())
                
                captured_activations[module_name]['output'].append(output_tensor.detach().cpu())
                
                # 修正3：对于 attn_block，直接计算并缓存 Q、K、V
                if module_type == "attn_block":
                    try:
                        with torch.no_grad():
                            q = module.q_proj(input_tensor)
                            k = module.k_proj(input_tensor)
                            v = module.v_proj(input_tensor)
                            captured_activations[module_name]['q'].append(q.detach().cpu())
                            captured_activations[module_name]['k'].append(k.detach().cpu())
                            captured_activations[module_name]['v'].append(v.detach().cpu())
                    except Exception as e:
                        print(f"警告: 提取 Q/K/V 时出错在模块 {module_name}: {e}")

            # PyTorch 2.1+ 推荐使用新的签名
            return hook_fn

        # 修正：严格按照 v4.7.md 实现钩子注册逻辑
        registered_hooks = set() # 防止对同一模块重复挂钩
        for name, module in model_to_hook.named_modules():
            # 检查模块的参数是否需要合并，避免重复挂钩
            if name in registered_hooks:
                continue

            # 构造一个虚拟的完整键名来检查是否需要合并
            # 我们检查 .weight 后缀，因为 _should_merge 是基于参数名设计的
            key_for_check = "model." + name + ".weight" # 假设所有模块都有weight

            if self._should_merge(key_for_check):
                # 挂钩到 Attention 和 MLP 的叶子模块
                if any(name.endswith(p) for p in [".q_proj", ".k_proj", ".v_proj", ".o_proj", ".gate_proj", ".up_proj", ".down_proj"]):
                    hooks.append(module.register_forward_hook(get_hook(name, "leaf_module"), with_kwargs=True))
                    registered_hooks.add(name)

            # 额外：为 self_attn 整体挂钩，以捕获 QKV 和最终输出
            if name.endswith(".self_attn"):
                if name not in registered_hooks:
                    hooks.append(module.register_forward_hook(get_hook(name, "attn_block"), with_kwargs=True))
                    registered_hooks.add(name)


        print(f"在 {model_name} 中注册了 {len(hooks)} 个钩子。")

        # 准备探针数据集
        probe_dataset_raw = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True).take(self.args.probe_samples)
        probe_texts = [item['text'] for item in probe_dataset_raw if item['text'].strip()]
        
        # 使用 apply_chat_template 来格式化文本
        formatted_texts = [tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True) for text in probe_texts]
        inputs = tokenizer(formatted_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        dataloader = DataLoader(TensorDataset(inputs['input_ids'], inputs['attention_mask']), batch_size=self.args.probe_batch_size)

        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"前向传播 {model_name}"):
                input_ids, attention_mask = batch[0].to(self.device), batch[1].to(self.device)
                # 移除 output_attentions=True，直接进行前向传播
                model(input_ids=input_ids, attention_mask=attention_mask)

        for h in hooks: h.remove()
        
        # 平均并保存
        averaged_activations = {}
        for name, data in captured_activations.items():
            averaged_activations[name] = {k: torch.mean(torch.cat(v, dim=0).float(), dim=0) for k, v in data.items() if v}

        torch.save(averaged_activations, cache_path)
        print(f"{model_name} 的激活已缓存至: {cache_path}")

        del model, tokenizer, captured_activations, averaged_activations
        gc.collect()
        torch.cuda.empty_cache()

    def stage1_cache_activations(self):
        """阶段一（对称）：为模型 A, B, C 缓存激活。"""
        print("\n--- [阶段一 (ASAM 3.0): 对称激活缓存] ---")
        self._cache_activations_for_model("Model A (Base)", self.args.base_model_path, os.path.join(self.cache_dir, "activations_A.pt"), capture_inputs=True)
        self._cache_activations_for_model("Model B (Donor)", self.args.donor_model_path, os.path.join(self.cache_dir, "activations_B.pt"), capture_inputs=True)
        self._cache_activations_for_model("Model C (Ancestor)", self.args.original_model_path, os.path.join(self.cache_dir, "activations_C.pt"), capture_inputs=False)

    def _calculate_and_save_gradients(self, model_label, grad_dir):
        """阶段二的内部函数：为指定模型计算并保存所有梯度。"""
        print(f"正在为 {model_label} 计算近似梯度...")
        activations_model = torch.load(os.path.join(self.cache_dir, f"activations_{model_label}.pt"), map_location="cpu")
        activations_C = torch.load(os.path.join(self.cache_dir, "activations_C.pt"), map_location="cpu")

        # 假设所有模型的语言部分key是一致的
        if model_label == "A":
            # 修正：加载模型A的权重
            base_weights = load_weights(self.args.base_model_path)
            config = AutoConfig.from_pretrained(self.args.base_model_path, trust_remote_code=True)
        elif model_label == "B":
            # 修正：加载模型B的权重
            base_weights = load_weights(self.args.donor_model_path)
            config = AutoConfig.from_pretrained(self.args.donor_model_path, trust_remote_code=True)
        
        # 修正：使用 AutoConfig 智能加载配置，并处理嵌套结构以兼容多模态模型
        lang_config = getattr(config, "text_config", config)
        num_heads = lang_config.num_attention_heads
        head_dim = lang_config.hidden_size // num_heads

        for key in tqdm(self.merge_scope['language_model'], desc=f"计算 {model_label} 的梯度"):
            grad_path = os.path.join(grad_dir, f"{key.replace('/', '_')}.pt")
            if os.path.exists(grad_path) and not self.args.force_recompute: continue

            # 修正：提供所有可能的前缀以进行正确的标准化
            all_prefixes = ["model.language_model.", "language_model.", "model."]
            norm_key = normalize_keys({key:0}, prefixes_to_remove=all_prefixes).popitem()[0]
            
            # 定位模块名
            module_name = ".".join(norm_key.split('.')[:-1])
            attn_block_name = ".".join(norm_key.split('.')[:-2]) if "self_attn" in norm_key else ""
            
            g_approx = None
            try:
                # --- MLP 梯度计算 ---
                if "mlp" in norm_key:
                    X = activations_model[module_name]['input'].to(self.device)
                    Y_model = activations_model[module_name]['output'].to(self.device)
                    Y_C = activations_C[module_name]['output'].to(self.device)
                    delta_Y = Y_model - Y_C
                    if key.endswith(".weight"): g_approx = delta_Y.T @ X
                    else: g_approx = delta_Y
                
                # --- Attention 梯度计算 ---
                elif "self_attn" in norm_key:
                    # 修正：为GQA正确处理多头注意力
                    num_kv_heads = lang_config.num_key_value_heads
                    num_q_heads = lang_config.num_attention_heads
                    num_repetition_groups = num_q_heads // num_kv_heads

                    X_attn = activations_model[attn_block_name]['input'].to(self.device)
                    Y_attn_model = activations_model[attn_block_name]['output'].to(self.device)
                    Y_attn_C = activations_C[attn_block_name]['output'].to(self.device)

                    # --- Reshape and Repeat K/V for GQA ---
                    def reshape_and_repeat(tensor, num_heads):
                        batch_size, seq_len, _ = tensor.shape
                        tensor = tensor.view(batch_size, seq_len, num_heads, head_dim)
                        if num_heads == num_kv_heads: # Only repeat K and V
                            tensor = tensor.unsqueeze(3).expand(batch_size, seq_len, num_heads, num_repetition_groups, head_dim)
                            tensor = tensor.reshape(batch_size, seq_len, num_q_heads, head_dim)
                        return tensor.transpose(1, 2) # (batch, num_heads, seq_len, head_dim)

                    # 加载并处理模型A的激活
                    Q_model_raw = activations_model[attn_block_name]['q'].to(self.device)
                    K_model_raw = activations_model[attn_block_name]['k'].to(self.device)
                    V_model_raw = activations_model[attn_block_name]['v'].to(self.device)
                    
                    Q_model = reshape_and_repeat(Q_model_raw.unsqueeze(0), num_q_heads)
                    K_model = reshape_and_repeat(K_model_raw.unsqueeze(0), num_kv_heads)
                    V_model = reshape_and_repeat(V_model_raw.unsqueeze(0), num_kv_heads)

                    # 加载并处理模型C的激活
                    Q_C_raw = activations_C[attn_block_name]['q'].to(self.device)
                    K_C_raw = activations_C[attn_block_name]['k'].to(self.device)
                    V_C_raw = activations_C[attn_block_name]['v'].to(self.device)

                    Q_C = reshape_and_repeat(Q_C_raw.unsqueeze(0), num_q_heads)
                    K_C = reshape_and_repeat(K_C_raw.unsqueeze(0), num_kv_heads)
                    V_C = reshape_and_repeat(V_C_raw.unsqueeze(0), num_kv_heads)

                    # 计算注意力分数的差值
                    S_model = Q_model @ K_model.transpose(-2, -1) / (head_dim ** 0.5)
                    S_C = Q_C @ K_C.transpose(-2, -1) / (head_dim ** 0.5)
                    delta_S = (S_model - S_C).squeeze(0) # (num_q_heads, seq_len, seq_len)
                    delta_Y_attn = Y_attn_model - Y_attn_C
                    
                    # 计算注意力权重用于V投影梯度
                    A_model = torch.softmax(S_model.squeeze(0), dim=-1)

                    # --- 梯度计算（保持不变，但输入维度已正确）---
                    if norm_key.endswith("q_proj.weight"):
                        g_space = delta_S @ K_model.squeeze(0)
                        g_space = g_space.transpose(0, 1).reshape(Q_model_raw.shape)
                        g_approx = g_space.T @ X_attn
                    elif norm_key.endswith("q_proj.bias"):
                        g_space = delta_S @ K_model.squeeze(0)
                        g_approx = g_space.transpose(0, 1).reshape(Q_model_raw.shape).sum(dim=0)
                    elif norm_key.endswith("k_proj.weight"):
                        # 修正：对于K，需要从num_q_heads汇总到num_kv_heads
                        g_space = delta_S.transpose(-2, -1) @ Q_model.squeeze(0)
                        # 将g_space从(seq_len, num_q_heads, head_dim)重塑，然后按num_repetition_groups汇总
                        g_space = g_space.transpose(0, 1).reshape(K_model_raw.shape[0], num_q_heads, head_dim)
                        # 重塑为(seq_len, num_kv_heads, num_repetition_groups, head_dim)，然后在repetition维度上求和
                        g_space = g_space.reshape(K_model_raw.shape[0], num_kv_heads, num_repetition_groups, head_dim).sum(dim=2)
                        g_approx = g_space.reshape(K_model_raw.shape).T @ X_attn
                    elif norm_key.endswith("k_proj.bias"):
                        g_space = delta_S.transpose(-2, -1) @ Q_model.squeeze(0)
                        g_space = g_space.transpose(0, 1).reshape(K_model_raw.shape[0], num_q_heads, head_dim)
                        g_space = g_space.reshape(K_model_raw.shape[0], num_kv_heads, num_repetition_groups, head_dim).sum(dim=2)
                        g_approx = g_space.reshape(K_model_raw.shape).sum(dim=0)
                    elif norm_key.endswith("v_proj.weight"):
                        # 修正：处理GQA中V投影的梯度计算
                        # 首先，将delta_Y_attn重塑为每头的形式
                        delta_Y_reshaped = delta_Y_attn.view(delta_Y_attn.shape[0], num_q_heads, head_dim)
                        
                        # 将注意力权重应用到delta_Y上
                        # A_model形状: (num_q_heads, seq_len, seq_len)
                        # delta_Y_reshaped形状: (seq_len, num_q_heads, head_dim)
                        g_space = A_model.transpose(-2, -1) @ delta_Y_reshaped.transpose(0, 1)  # (num_q_heads, seq_len, head_dim)
                        
                        # 转置回来并重塑为(seq_len, num_q_heads, head_dim)
                        g_space = g_space.transpose(0, 1)  # (seq_len, num_q_heads, head_dim)
                        
                        # 将g_space重塑并在重复维度上汇总，以匹配V的原始形状
                        g_space = g_space.reshape(V_model_raw.shape[0], num_kv_heads, num_repetition_groups, head_dim).sum(dim=2)
                        
                        # 计算最终梯度
                        g_approx = g_space.reshape(V_model_raw.shape).T @ X_attn
                    elif norm_key.endswith("v_proj.bias"):
                        # 类似地修复v_proj.bias的梯度计算
                        delta_Y_reshaped = delta_Y_attn.view(delta_Y_attn.shape[0], num_q_heads, head_dim)
                        g_space = A_model.transpose(-2, -1) @ delta_Y_reshaped.transpose(0, 1)
                        g_space = g_space.transpose(0, 1)
                        g_space = g_space.reshape(V_model_raw.shape[0], num_kv_heads, num_repetition_groups, head_dim).sum(dim=2)
                        g_approx = g_space.reshape(V_model_raw.shape).sum(dim=0)
                    elif norm_key.endswith("o_proj.weight"):
                        Z_out = A_model @ V_model.squeeze(0)
                        Z_out = Z_out.transpose(0, 1).reshape(Y_attn_model.shape)
                        g_approx = Z_out.T @ delta_Y_attn
                    elif norm_key.endswith("o_proj.bias"):
                        g_approx = delta_Y_attn.sum(dim=0)
            
            except KeyError as e:
                print(f"警告: 计算 {key} 梯度时激活未找到: {e}。跳过。")
                continue
            
            if g_approx is not None:
                torch.save(g_approx.cpu(), grad_path)
        
        del activations_model, activations_C, base_weights
        gc.collect()

    def stage2_calculate_approx_gradients(self):
        """阶段二（对称与共识）：计算 g'_A 和 g'_B，然后融合成共识梯度。"""
        print("\n--- [阶段二 (ASAM 3.0): 对称与共识梯度计算] ---")
        
        # 分别计算 g'_A 和 g'_B
        self._calculate_and_save_gradients("A", self.grad_dir_A)
        self._calculate_and_save_gradients("B", self.grad_dir_B)

        print("正在融合为共识梯度...")
        consensus_grad_dir = os.path.join(self.cache_dir, "consensus_grads")
        os.makedirs(consensus_grad_dir, exist_ok=True)
        
        # 保存一个日志，记录处理的层和跳过的层
        fusion_log = {"processed": [], "skipped": [], "dimension_mismatch": []}
        
        for key in tqdm(self.merge_scope['language_model'], desc="融合共识梯度"):
            path_A = os.path.join(self.grad_dir_A, f"{key.replace('/', '_')}.pt")
            path_B = os.path.join(self.grad_dir_B, f"{key.replace('/', '_')}.pt")
            path_consensus = os.path.join(consensus_grad_dir, f"{key.replace('/', '_')}.pt")

            if os.path.exists(path_consensus) and not self.args.force_recompute: 
                fusion_log["processed"].append(key)
                continue
                
            if not os.path.exists(path_A) or not os.path.exists(path_B): 
                fusion_log["skipped"].append(key)
                continue

            g_A = torch.load(path_A)
            g_B = torch.load(path_B)
            
            # 检查形状是否匹配
            if g_A.shape != g_B.shape:
                fusion_log["dimension_mismatch"].append(f"{key}: {g_A.shape} vs {g_B.shape}")
                # 模型结构不同时，优先保留基础模型A的梯度
                torch.save(g_A, path_consensus)
                continue
                
            # 符号选举法 - 只对形状匹配的层进行
            g_consensus = torch.zeros_like(g_A)
            mask_same_sign = (torch.sign(g_A) == torch.sign(g_B))
            g_consensus[mask_same_sign] = (g_A[mask_same_sign] + g_B[mask_same_sign]) / 2.0
            
            torch.save(g_consensus, path_consensus)
            fusion_log["processed"].append(key)
        
        # 保存处理日志
        log_path = os.path.join(self.cache_dir, "fusion_log.json")
        with open(log_path, 'w') as f:
            json.dump({
                "processed_count": len(fusion_log["processed"]),
                "skipped_count": len(fusion_log["skipped"]),
                "dimension_mismatch_count": len(fusion_log["dimension_mismatch"]),
                "dimension_mismatch": fusion_log["dimension_mismatch"]
            }, f, indent=2)
        
        print(f"共识梯度计算完毕。处理了 {len(fusion_log['processed'])} 个参数，跳过了 {len(fusion_log['skipped'])} 个参数。")
        print(f"发现 {len(fusion_log['dimension_mismatch'])} 个参数因维度不匹配而使用了模型A的梯度。")
        print(f"详细日志已保存至 {log_path}")

    def stage3_merge_models(self):
        """阶段三（对称）：从共同祖先 W_C 出发，对称地融合 τ_A 和 τ_B。"""
        print("\n--- [阶段三 (ASAM 3.0): 对称分解与融合] ---")

        W_A_all = load_weights(self.args.base_model_path)
        W_B_all = load_weights(self.args.donor_model_path)
        W_C_all = load_weights(self.args.original_model_path)

        consensus_grad_dir = os.path.join(self.cache_dir, "consensus_grads")
        merged_weights = {}

        # 跟踪不同情况的统计数据
        stats = {
            "直接复制": 0,
            "直接合并": 0,
            "分解合并": 0,
            "跳过_找不到梯度": 0,
            "跳过_权重形状不匹配": 0,
            "跳过_梯度形状不匹配": 0
        }

        # 1. 直接复制保留的权重
        for key in self.merge_scope['preserve_from_A']:
            merged_weights[key] = W_A_all[key]
            stats["直接复制"] += 1
        
        # 2. 对称合并语言模型权重
        for key in tqdm(self.merge_scope['language_model'], desc="对称合并语言模型层"):
            try:
                grad_path = os.path.join(consensus_grad_dir, f"{key.replace('/', '_')}.pt")
                
                # 尝试获取对应的权重键
                try:
                    norm_key_A = normalize_keys({key:0}, prefixes_to_remove=["model.language_model."]).popitem()[0]
                    norm_key_B = normalize_keys({key:0}, prefixes_to_remove=["model.language_model."]).popitem()[0]
                    
                    W_A = W_A_all[key].float().to(self.device)
                    
                    # 安全地找到模型B中的对应键
                    matching_B_keys = [k for k in W_B_all if k.endswith(norm_key_B)]
                    if not matching_B_keys:
                        print(f"[警告] 在模型B中找不到对应的键 {norm_key_B}，直接保留A模型参数。")
                        merged_weights[key] = W_A_all[key]
                        stats["跳过_找不到梯度"] += 1
                        continue
                        
                    W_B_key = matching_B_keys[0]
                    W_B = W_B_all[W_B_key].float().to(self.device)
                    
                    # 安全地找到模型C中的对应键
                    if norm_key_A not in W_C_all:
                        print(f"[警告] 在模型C中找不到对应的键 {norm_key_A}，直接保留A模型参数。")
                        merged_weights[key] = W_A_all[key]
                        stats["跳过_找不到梯度"] += 1
                        continue
                        
                    W_C = W_C_all[norm_key_A].float().to(self.device)
                except Exception as e:
                    print(f"[错误] 加载权重时出错 {key}: {e}，直接保留A模型参数。")
                    merged_weights[key] = W_A_all[key]
                    stats["跳过_找不到梯度"] += 1
                    continue

                # 检查shape一致性
                if W_A.shape != W_B.shape or W_A.shape != W_C.shape:
                    print(f"[警告] 参数 {key} 维度不一致: A={W_A.shape}, B={W_B.shape}, C={W_C.shape}，直接保留A模型参数。")
                    merged_weights[key] = W_A_all[key]
                    stats["跳过_权重形状不匹配"] += 1
                    continue

                tau_A = W_A - W_C
                tau_B = W_B - W_C
                
                if not os.path.exists(grad_path):
                    # 简单线性合并
                    w_star = W_C + self.args.lambda_o_A * tau_A + self.args.lambda_o_B * tau_B
                    merged_weights[key] = w_star.to(W_A_all[key].dtype).cpu()
                    stats["直接合并"] += 1
                    continue
                    
                # 安全地加载共识梯度
                try:
                    g_consensus = torch.load(grad_path, map_location=self.device).float()
                except Exception as e:
                    print(f"[错误] 加载共识梯度时出错 {key}: {e}，直接保留A模型参数。")
                    merged_weights[key] = W_A_all[key]
                    stats["跳过_找不到梯度"] += 1
                    continue
                    
                # 检查梯度和tau_A的形状是否匹配
                if g_consensus.shape != tau_A.shape:
                    print(f"[警告] 共识梯度与tau_A维度不一致: {g_consensus.shape} vs {tau_A.shape}，直接保留A模型参数。")
                    merged_weights[key] = W_A_all[key]
                    stats["跳过_梯度形状不匹配"] += 1
                    continue
                    
                g_norm_sq = torch.sum(g_consensus * g_consensus)
                
                if g_norm_sq < 1e-9:
                    # 梯度过小，使用简单线性合并
                    w_star = W_C + self.args.lambda_o_A * tau_A + self.args.lambda_o_B * tau_B
                    merged_weights[key] = w_star.to(W_A_all[key].dtype).cpu()
                    stats["直接合并"] += 1
                else:
                    # 安全地计算投影和分解
                    try:
                        # 分解 tau_A
                        proj_A = torch.sum(g_consensus * tau_A) / g_norm_sq
                        tau_A_syn = torch.clamp_min(-proj_A, 0) * g_consensus
                        tau_A_con = torch.clamp_min(proj_A, 0) * g_consensus
                        tau_A_ortho = tau_A - tau_A_syn - tau_A_con

                        # 分解 tau_B
                        proj_B = torch.sum(g_consensus * tau_B) / g_norm_sq
                        tau_B_syn = torch.clamp_min(-proj_B, 0) * g_consensus
                        tau_B_con = torch.clamp_min(proj_B, 0) * g_consensus
                        tau_B_ortho = tau_B - tau_B_syn - tau_B_con
                        
                        # 合并
                        w_star = W_C + \
                                self.args.lambda_s_A * tau_A_syn - self.args.lambda_c_A * tau_A_con + self.args.lambda_o_A * tau_A_ortho + \
                                self.args.lambda_s_B * tau_B_syn - self.args.lambda_c_B * tau_B_con + self.args.lambda_o_B * tau_B_ortho
                                
                        merged_weights[key] = w_star.to(W_A_all[key].dtype).cpu()
                        stats["分解合并"] += 1
                    except Exception as e:
                        print(f"[错误] 计算分解时出错 {key}: {e}，直接保留A模型参数。")
                        merged_weights[key] = W_A_all[key]
                        stats["跳过_梯度形状不匹配"] += 1
            except Exception as e:
                print(f"[错误] 处理参数 {key} 时发生未捕获的异常: {e}，直接保留A模型参数。")
                merged_weights[key] = W_A_all[key]
                stats["跳过_找不到梯度"] += 1
    
        # 打印统计信息
        print("\n合并统计:")
        for stat_name, count in stats.items():
            print(f"  {stat_name}: {count} 参数")
        
        # --- 保存模型 ---
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
        create_soft_link(source_path=self.args.base_model_path, link_path=self.output_dir)
        print(f"模型成功合并并保存至: {self.output_dir}")

    def stage4_validate_and_finalize(self):
        """阶段四：全局验证。"""
        print("\n--- [阶段四 (ASAM 3.0): 全局验证] ---")
        if not self.args.run_validation:
            print("跳过验证阶段。合并流程完成！")
            return
        
        print("加载合并后的模型进行快速验证... (此部分需要根据最终模型类型进行调整)")
        # 由于是多模态，验证逻辑会更复杂，这里仅为占位符
        print("验证逻辑需要手动实现，以匹配多模态模型的输入格式。")
        print("ASAM 3.0 框架所有阶段执行完毕！")

    def run_pipeline(self):
        """按顺序执行所有阶段。"""
        self.stage0_define_scope()
        self.stage1_cache_activations()
        self.stage2_calculate_approx_gradients()
        self.stage3_merge_models()
        self.stage4_validate_and_finalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASAM v3.0: 对称与组件感知的模型合并框架。")
    
    # 基本配置
    parser.add_argument('--base_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct", help="基础模型A的路径 (e.g., qwen2-vl-7b-instruct)。")
    parser.add_argument('--donor_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si-hf", help="贡献模型B的路径 (e.g., llava-onevision-qwen2-7b-si-hf)。")
    parser.add_argument('--original_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-7B-Instruct", help="原始共同祖先模型C的路径 (e.g., qwen2-7b-instruct)。")
    parser.add_argument('--mode', type=str, default="symmetric_merge", help="为本次合并配置命名。")
    parser.add_argument('--cuda_device', type=int, default=2, help="使用的 CUDA 设备编号。")

    # 探针数据集配置
    parser.add_argument('--probe_samples', type=int, default=200, help="用于探测的样本数量。")
    parser.add_argument('--probe_batch_size', type=int, default=1, help="探测时的批处理大小。")

    # 对称合并超参数
    parser.add_argument('--lambda_s_A', type=float, default=1.0, help="模型A协同分量的系数。")
    parser.add_argument('--lambda_c_A', type=float, default=1.0, help="模型A冲突分量的系数（减去）。")
    parser.add_argument('--lambda_o_A', type=float, default=1.0, help="模型A正交分量的系数。")
    parser.add_argument('--lambda_s_B', type=float, default=1.0, help="模型B协同分量的系数。")
    parser.add_argument('--lambda_c_B', type=float, default=1.0, help="模型B冲突分量的系数（减去）。")
    parser.add_argument('--lambda_o_B', type=float, default=1.0, help="模型B正交分量的系数。")
    
    # 功能性参数
    parser.add_argument('--force_recompute', action='store_true', help="强制重新计算缓存的激活或梯度。")
    parser.add_argument('--run_validation', action='store_true', help="在合并后执行阶段四的验证。")

    args = parser.parse_args()
    
    if torch.cuda.is_available() and args.cuda_device < torch.cuda.device_count():
        device = f"cuda:{args.cuda_device}"
    else:
        device = "cpu"

    print("--- ASAM v3.0 合并框架配置 ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("--------------------------------")

    merger = ASAMerger(args, device)
    merger.run_pipeline()