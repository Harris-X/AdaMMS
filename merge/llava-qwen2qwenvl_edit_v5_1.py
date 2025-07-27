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
                
                # 修正3：仅在 self_attn 级别捕获 QKV
                if module_type == "attn_block":
                    with torch.no_grad():
                        q = module.q_proj(input_tensor)
                        k = module.k_proj(input_tensor)
                        v = module.v_proj(input_tensor)
                        captured_activations[module_name]['q'].append(q.detach().cpu())
                        captured_activations[module_name]['k'].append(k.detach().cpu())
                        captured_activations[module_name]['v'].append(v.detach().cpu())

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
        base_weights = load_weights(self.args.base_model_path)
        
        # 修正：使用 AutoConfig 智能加载配置，并处理嵌套结构以兼容多模态模型
        config = AutoConfig.from_pretrained(self.args.base_model_path, trust_remote_code=True)
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
                    X_attn = activations_model[attn_block_name]['input'].to(self.device)
                    Q_model = activations_model[attn_block_name]['q'].to(self.device)
                    K_model = activations_model[attn_block_name]['k'].to(self.device)
                    V_model = activations_model[attn_block_name]['v'].to(self.device)
                    Y_attn_model = activations_model[attn_block_name]['output'].to(self.device)
                    
                    Q_C = activations_C[attn_block_name]['q'].to(self.device)
                    K_C = activations_C[attn_block_name]['k'].to(self.device)
                    V_C = activations_C[attn_block_name]['v'].to(self.device)
                    Y_attn_C = activations_C[attn_block_name]['output'].to(self.device)
                    
                    S_model = Q_model @ K_model.T
                    S_C = Q_C @ K_C.T
                    delta_S = S_model - S_C
                    delta_Y_attn = Y_attn_model - Y_attn_C
                    A_model = torch.softmax(S_model / (head_dim**0.5), dim=-1)

                    if norm_key.endswith("q_proj.weight"):
                        g_space = delta_S @ K_model
                        g_approx = g_space.T @ X_attn
                    elif norm_key.endswith("q_proj.bias"):
                        g_approx = delta_S @ K_model
                    elif norm_key.endswith("k_proj.weight"):
                        g_space = delta_S.T @ Q_model
                        g_approx = g_space.T @ X_attn
                    elif norm_key.endswith("k_proj.bias"):
                        g_approx = delta_S.T @ Q_model
                    elif norm_key.endswith("v_proj.weight"):
                        g_space = A_model.T @ delta_Y_attn
                        g_approx = g_space.T @ X_attn
                    elif norm_key.endswith("v_proj.bias"):
                        g_approx = A_model.T @ delta_Y_attn
                    elif norm_key.endswith("o_proj.weight"):
                        Z_out = A_model @ V_model
                        g_approx = Z_out.T @ delta_Y_attn
                    elif norm_key.endswith("o_proj.bias"):
                        g_approx = delta_Y_attn
            
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
        # 此处简化，实际可以在阶段三加载时动态融合
        # 为了演示，可以创建一个 consensus_grad 目录
        consensus_grad_dir = os.path.join(self.cache_dir, "consensus_grads")
        os.makedirs(consensus_grad_dir, exist_ok=True)
        
        for key in tqdm(self.merge_scope['language_model'], desc="融合共识梯度"):
            path_A = os.path.join(self.grad_dir_A, f"{key.replace('/', '_')}.pt")
            path_B = os.path.join(self.grad_dir_B, f"{key.replace('/', '_')}.pt")
            path_consensus = os.path.join(consensus_grad_dir, f"{key.replace('/', '_')}.pt")

            if os.path.exists(path_consensus) and not self.args.force_recompute: continue
            if not os.path.exists(path_A) or not os.path.exists(path_B): continue

            g_A = torch.load(path_A)
            g_B = torch.load(path_B)
            
            # 符号选举法
            g_consensus = torch.zeros_like(g_A)
            mask_same_sign = (torch.sign(g_A) == torch.sign(g_B))
            g_consensus[mask_same_sign] = (g_A[mask_same_sign] + g_B[mask_same_sign]) / 2.0
            
            torch.save(g_consensus, path_consensus)
        
        print("共识梯度计算完毕。")

    def stage3_merge_models(self):
        """阶段三（对称）：从共同祖先 W_C 出发，对称地融合 τ_A 和 τ_B。"""
        print("\n--- [阶段三 (ASAM 3.0): 对称分解与融合] ---")

        W_A_all = load_weights(self.args.base_model_path)
        W_B_all = load_weights(self.args.donor_model_path)
        W_C_all = load_weights(self.args.original_model_path)

        consensus_grad_dir = os.path.join(self.cache_dir, "consensus_grads")
        merged_weights = {}

        # 1. 直接复制保留的权重
        for key in self.merge_scope['preserve_from_A']:
            merged_weights[key] = W_A_all[key]
        
        # 2. 对称合并语言模型权重
        for key in tqdm(self.merge_scope['language_model'], desc="对称合并语言模型层"):
            grad_path = os.path.join(consensus_grad_dir, f"{key.replace('/', '_')}.pt")
            
            # 标准化key以匹配非多模态模型C
            norm_key_A = normalize_keys({key:0}, prefixes_to_remove=["model.language_model."]).popitem()[0]
            norm_key_B = normalize_keys({key:0}, prefixes_to_remove=["model.language_model."]).popitem()[0]

            W_A = W_A_all[key].float().to(self.device)
            # 在llava-onevision-qwen2-7b-si-hf中，语言模型没有前缀
            W_B_key = [k for k in W_B_all if k.endswith(norm_key_B)][0]
            W_B = W_B_all[W_B_key].float().to(self.device)
            W_C = W_C_all[norm_key_A].float().to(self.device)

            tau_A = W_A - W_C
            tau_B = W_B - W_C
            
            if not os.path.exists(grad_path):
                # 如果没有共识梯度，执行简单平均
                w_star = W_C + self.args.lambda_A * tau_A + self.args.lambda_B * tau_B
            else:
                g_consensus = torch.load(grad_path, map_location=self.device).float()
                g_norm_sq = torch.sum(g_consensus * g_consensus)
                
                if g_norm_sq < 1e-9:
                    w_star = W_C + self.args.lambda_o_A * tau_A + self.args.lambda_o_B * tau_B
                else:
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
                    
                    # 对称合并公式
                    w_star = W_C + \
                             self.args.lambda_s_A * tau_A_syn - self.args.lambda_c_A * tau_A_con + self.args.lambda_o_A * tau_A_ortho + \
                             self.args.lambda_s_B * tau_B_syn - self.args.lambda_c_B * tau_B_con + self.args.lambda_o_B * tau_B_ortho

            merged_weights[key] = w_star.to(W_A_all[key].dtype).cpu()
        
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
    parser.add_argument('--cuda_device', type=int, default=3, help="使用的 CUDA 设备编号。")

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