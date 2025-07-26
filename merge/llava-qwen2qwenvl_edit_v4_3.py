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
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoModelForCausalLM

# 尝试导入 Hugging Face datasets 库
try:
    from datasets import load_dataset
except ImportError:
    print("="*80, file=sys.stderr)
    print("错误：无法导入 'datasets' 库。请运行 `pip install datasets`。", file=sys.stderr)
    print("这个库是获取探针数据集所必需的。", file=sys.stderr)
    print("="*80, file=sys.stderr)
    sys.exit(1)

# --- 权重加载与辅助函数 (保持不变) ---

def load_weights(base_path, index_filename):
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
    """专门用于标准化 donor 模型（可能带有不同前缀）的 key。"""
    # 动态检测常见的前缀
    prefixes_to_try = ["language_model.model.", "language_model.", "model."]
    detected_prefix = ""
    for p in prefixes_to_try:
        if any(key.startswith(p) for key in weights.keys()):
            detected_prefix = p
            print(f"检测到并移除前缀: '{detected_prefix}'")
            break
            
    if not detected_prefix:
        return weights # 如果没有常见前缀，直接返回

    normalized_weights = {}
    for key, value in weights.items():
        if key.startswith(detected_prefix):
            normalized_weights[key[len(detected_prefix):]] = value
        else:
            # 对于非语言模型部分（如 vision_tower），保留原样
            normalized_weights[key] = value
    return normalized_weights

# --- ASAM v2.0 核心逻辑 ---

def get_module_type(name: str) -> str:
    """
    根据层名判断模块类型，用于ASAM框架。
    返回 'mlp', 'attn_q', 'attn_k', 'attn_v', 'attn_o', 'ignore' 或 'other'。
    """
    layers_idx = name.rfind("layers.")
    if layers_idx == -1:
        return 'ignore'

    suffix = name[layers_idx:]
    
    if suffix.endswith(".self_attn.rotary_emb.inv_freq") or "layernorm" in suffix:
        return 'ignore'
    
    if ".mlp." in suffix:
        return 'mlp'
    elif ".self_attn.q_proj." in suffix:
        return 'attn_q'
    elif ".self_attn.k_proj." in suffix:
        return 'attn_k'
    elif ".self_attn.v_proj." in suffix:
        return 'attn_v'
    elif ".self_attn.o_proj." in suffix:
        return 'attn_o'
    else:
        # 其他在layers内但未明确分类的参数，可以选择忽略或默认处理
        return 'ignore'

def create_soft_link(source_path, link_path):
    """创建必要的符号链接以构成完整的模型目录。"""
    if not os.path.exists(source_path):
        print(f"错误: 源路径 '{source_path}' 不存在。")
        return

    os.makedirs(link_path, exist_ok=True)

    for item in os.listdir(source_path):
        source_item = os.path.abspath(os.path.join(source_path, item))
        link_item = os.path.join(link_path, item)

        # 跳过权重文件和已存在的链接
        if item.endswith('.safetensors') or os.path.exists(link_item):
            continue
        
        try:
            os.symlink(source_item, link_item)
        except OSError as e:
            print(f"为 '{item}' 创建软链接时出错: {e}")


class ASAMerger:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.output_dir = os.path.join("merged_models", f"ASAM-{args.config_name}")
        self.cache_dir = os.path.join(self.output_dir, "cache")
        self.grad_dir = os.path.join(self.cache_dir, "approx_grads_ASAM")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.grad_dir, exist_ok=True)
        print(f"使用设备: {self.device}")
        print(f"输出将保存至: {self.output_dir}")

    def _get_target_modules(self, model_to_hook):
        """获取所有需要被hook的目标模块（MLP和Attention）。"""
        target_modules = {'mlp': set(), 'attn': set()}
        # 遍历所有模块来识别
        for name, module in model_to_hook.named_modules():
            if ".mlp" in name and not any(sub in name for sub in ['.q_proj', '.k_proj', '.v_proj', '.o_proj']):
                target_modules['mlp'].add(name)
            elif ".self_attn" in name and not "layernorm" in name:
                # 注意力模块我们hook其父模块以捕获所有QKV
                parent_name = ".".join(name.split('.')[:-1])
                if "self_attn" in parent_name:
                    target_modules['attn'].add(parent_name)
        
        # 将set转为list
        for key in target_modules:
            target_modules[key] = list(target_modules[key])
        return target_modules

    def _cache_activations_for_model(self, model_path, cache_path, is_base_model=False):
        """阶段一：为单个模型执行前向传播并缓存ASAM所需的激活。"""
        if os.path.exists(cache_path) and not self.args.force_recompute:
            print(f"激活缓存文件已存在: {cache_path}, 跳过。")
            return

        print(f"正在为 {os.path.basename(model_path)} 缓存激活 (ASAM)...")
        
        is_vision_model = "VL" in model_path or "llava" in model_path.lower()
        ModelClass = AutoModelForVision2Seq if is_vision_model else AutoModelForCausalLM
        
        model = ModelClass.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 智能定位到包含 "layers" 的语言模型部分
        model_to_hook = getattr(model, 'model', getattr(model, 'language_model', model))
        if hasattr(model_to_hook, 'model'): # 适配更深层的嵌套
             model_to_hook = model_to_hook.model
             
        print(f"将对模块 '{type(model_to_hook).__name__}' 注册钩子。")
        
        target_modules = self._get_target_modules(model_to_hook)
        print(f"找到 {len(target_modules['mlp'])} 个MLP模块和 {len(target_modules['attn'])} 个Attention模块。")

        hooks = []
        captured_activations = defaultdict(dict)

        def get_mlp_hook(name):
            def hook_fn(module, input_tensor, output_tensor):
                if is_base_model:
                    captured_activations[name]['input'] = input_tensor[0].detach().cpu()
                captured_activations[name]['output'] = output_tensor[0].detach().cpu()
            return hook_fn

        def get_attn_hook(name):
            def hook_fn(module, input_tensor, output_tensor):
                # output_tensor for attention is (hidden_states, past_key_value)
                if is_base_model:
                    captured_activations[name]['input'] = input_tensor[0].detach().cpu()
                    # 捕获Q,K,V (在投影之后)
                    captured_activations[name]['Q'] = module.q_proj(input_tensor[0]).detach().cpu()
                    captured_activations[name]['K'] = module.k_proj(input_tensor[0]).detach().cpu()
                    captured_activations[name]['V'] = module.v_proj(input_tensor[0]).detach().cpu()
                else: # 对于模型C, 只需要Q, K
                    captured_activations[name]['Q'] = module.q_proj(input_tensor[0]).detach().cpu()
                    captured_activations[name]['K'] = module.k_proj(input_tensor[0]).detach().cpu()
                
                captured_activations[name]['output'] = output_tensor[0].detach().cpu()
            return hook_fn

        # 注册钩子
        for name, module in model_to_hook.named_modules():
            if name in target_modules['mlp']:
                hooks.append(module.register_forward_hook(get_mlp_hook(name)))
            elif name in target_modules['attn']:
                hooks.append(module.register_forward_hook(get_attn_hook(name)))

        # 准备多模态探针数据集 (如果适用)
        if is_vision_model:
            # 这是一个占位符 - 实际的多模态数据集加载会更复杂
            # 您需要一个能同时提供图像和文本的数据加载器
            print("警告: 检测到多模态模型，但使用纯文本探针。为获得最佳效果，请实现多模态数据加载器。")
        
        try:
            probe_dataset_raw = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True).take(self.args.probe_samples)
            probe_texts = [item['text'] for item in probe_dataset_raw if item['text']]
        except Exception as e:
            print(f"加载 wikitext 数据集失败: {e}. 使用备用文本。")
            probe_texts = ["The quick brown fox jumps over the lazy dog."] * self.args.probe_samples
        
        probe_inputs = tokenizer(probe_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        probe_dataset = TensorDataset(probe_inputs['input_ids'], probe_inputs['attention_mask'])
        probe_dataloader = DataLoader(probe_dataset, batch_size=self.args.probe_batch_size)

        # 累积激活
        accumulated_activations = defaultdict(lambda: defaultdict(list))
        model.eval()
        with torch.no_grad():
            for batch in tqdm(probe_dataloader, desc=f"前向传播 {os.path.basename(model_path)}"):
                input_ids, attention_mask = batch[0].to(self.device), batch[1].to(self.device)
                model(input_ids=input_ids, attention_mask=attention_mask)
                for name, data in captured_activations.items():
                    for key, tensor in data.items():
                        accumulated_activations[name][key].append(tensor)
                captured_activations.clear() # 清空本批次的缓存

        for h in hooks: h.remove()
        
        # 求平均
        averaged_activations = {}
        for name, data in accumulated_activations.items():
            averaged_activations[name] = {}
            for key, tensor_list in data.items():
                if tensor_list:
                    averaged_activations[name][key] = torch.mean(torch.cat(tensor_list, dim=0).float(), dim=0)

        torch.save(averaged_activations, cache_path)
        print(f"激活已缓存至: {cache_path}")
        
        del model, tokenizer, probe_dataloader, accumulated_activations, averaged_activations
        gc.collect()
        torch.cuda.empty_cache()

    def stage1_cache_activations(self):
        """执行阶段一：缓存ASAM所需的激活。"""
        print("\n--- [阶段一: 缓存激活 (ASAM)] ---")
        activations_a_path = os.path.join(self.cache_dir, "activations_A_asam.pt")
        activations_c_path = os.path.join(self.cache_dir, "activations_C_asam.pt")

        self._cache_activations_for_model(self.args.base_model_path, activations_a_path, is_base_model=True)
        self._cache_activations_for_model(self.args.original_model_path, activations_c_path, is_base_model=False)

    def stage2_calculate_approx_gradients(self):
        """阶段二：计算区分化的、结构感知的近似梯度。"""
        print("\n--- [阶段二: 计算近似梯度 (ASAM)] ---")
        
        activations_A = torch.load(os.path.join(self.cache_dir, "activations_A_asam.pt"), map_location="cpu")
        activations_C = torch.load(os.path.join(self.cache_dir, "activations_C_asam.pt"), map_location="cpu")
        
        base_weights = load_weights(self.args.base_model_path, "model.safetensors.index.json")

        for key in tqdm(base_weights.keys(), desc="计算ASAM近似梯度"):
            module_type = get_module_type(key)
            if module_type == 'ignore':
                continue
            
            grad_path = os.path.join(self.grad_dir, f"{key.replace('/', '_')}.pt")
            if os.path.exists(grad_path) and not self.args.force_recompute:
                continue

            # 从参数键找到对应的模块名
            # e.g., language_model.model.layers.0.mlp.gate_proj.weight -> layers.0.mlp.gate_proj
            layers_idx = key.rfind("layers.")
            relative_key_prefix = key[layers_idx:]

            g_approx = None
            try:
                if module_type == 'mlp':
                    module_name = ".".join(relative_key_prefix.split('.')[:-1])
                    X_A = activations_A[module_name]['input'].to(self.device)
                    Y_A_mlp = activations_A[module_name]['output'].to(self.device)
                    Y_C_mlp = activations_C[module_name]['output'].to(self.device)
                    
                    delta_Y_mlp = Y_A_mlp - Y_C_mlp
                    
                    if key.endswith(".weight"):
                        g_approx = delta_Y_mlp.T @ X_A
                    elif key.endswith(".bias"):
                        g_approx = delta_Y_mlp.sum(dim=0)

                elif module_type in ['attn_q', 'attn_k', 'attn_v', 'attn_o']:
                    attn_module_name = ".".join(relative_key_prefix.split('.')[:-2]) # e.g. layers.0.self_attn
                    
                    X_A = activations_A[attn_module_name]['input'].to(self.device)
                    Y_A_attn = activations_A[attn_module_name]['output'].to(self.device)
                    Y_C_attn = activations_C[attn_module_name]['output'].to(self.device)
                    delta_Y_attn = Y_A_attn - Y_C_attn

                    Q_A = activations_A[attn_module_name]['Q'].to(self.device)
                    K_A = activations_A[attn_module_name]['K'].to(self.device)
                    V_A = activations_A[attn_module_name]['V'].to(self.device)
                    Q_C = activations_C[attn_module_name]['Q'].to(self.device)
                    K_C = activations_C[attn_module_name]['K'].to(self.device)

                    S_A = Q_A @ K_A.transpose(-1, -2)
                    S_C = Q_C @ K_C.transpose(-1, -2)
                    delta_S = S_A - S_C
                    
                    if module_type == 'attn_q':
                        g_q_space = delta_S @ K_A
                        g_approx = g_q_space.transpose(-1, -2) @ X_A
                    elif module_type == 'attn_k':
                        g_k_space = delta_S.transpose(-1, -2) @ Q_A
                        g_approx = g_k_space.transpose(-1, -2) @ X_A
                    elif module_type == 'attn_v':
                        A_A = F.softmax(S_A, dim=-1)
                        g_v_space = A_A.transpose(-1, -2) @ delta_Y_attn
                        g_approx = g_v_space.transpose(-1, -2) @ X_A
                    elif module_type == 'attn_o':
                        A_A = F.softmax(S_A, dim=-1)
                        Z_out_A = A_A @ V_A
                        g_approx = Z_out_A.transpose(-1, -2) @ delta_Y_attn
                
                if g_approx is not None:
                    # 将梯度中的NaN或inf替换为0，以增加稳定性
                    g_approx = torch.nan_to_num(g_approx)
                    torch.save(g_approx.cpu(), grad_path)

            except KeyError as e:
                print(f"警告: 计算 {key} 的梯度时缺少激活: {e}，跳过。")
            except Exception as e:
                print(f"警告: 计算 {key} 的梯度时发生未知错误: {e}，跳过。")

        del base_weights, activations_A, activations_C
        gc.collect()
        print("所有ASAM近似梯度计算并保存完毕。")

    def stage3_merge_models(self):
        """阶段三：执行梯度引导的自适应合并。"""
        print("\n--- [阶段三: 自适应分解与合并 (ASAM)] ---")

        print("正在从磁盘加载所有模型权重...")
        base_weights = load_weights(self.args.base_model_path, "model.safetensors.index.json")
        donor_weights_raw = load_weights(self.args.donor_model_path, "model.safetensors.index.json")
        original_weights = load_weights(self.args.original_model_path, "model.safetensors.index.json")
        
        # 在合并前，预加载一次激活，用于自适应赋权
        activations_A = torch.load(os.path.join(self.cache_dir, "activations_A_asam.pt"), map_location="cpu")
        activations_C = torch.load(os.path.join(self.cache_dir, "activations_C_asam.pt"), map_location="cpu")

        print("正在标准化 Donor 模型的层名...")
        donor_weights = normalize_donor_keys(donor_weights_raw)
        del donor_weights_raw
        gc.collect()

        merged_weights = {}

        for key in tqdm(base_weights.keys(), desc="逐层自适应合并权重"):
            merged_weights[key] = base_weights[key]
            
            module_type = get_module_type(key)
            if module_type == 'ignore':
                continue
                
            if key in donor_weights and key in original_weights and base_weights[key].shape == donor_weights[key].shape:
                grad_path = os.path.join(self.grad_dir, f"{key.replace('/', '_')}.pt")
                if not os.path.exists(grad_path):
                    continue

                W_A = base_weights[key].float().to(self.device)
                W_B = donor_weights[key].float().to(self.device)
                W_C = original_weights[key].float().to(self.device)
                g_approx = torch.load(grad_path, map_location=self.device).float()

                tau_B = W_B - W_C
                
                g_norm_sq = torch.sum(g_approx * g_approx)
                if g_norm_sq < 1e-12:
                    tau_B_synergy, tau_B_conflict, tau_B_ortho = torch.zeros_like(tau_B), torch.zeros_like(tau_B), tau_B
                else:
                    proj_scalar = torch.sum(g_approx * tau_B) / g_norm_sq
                    tau_B_synergy = torch.clamp_min(-proj_scalar, 0) * g_approx
                    tau_B_conflict = torch.clamp_min(proj_scalar, 0) * g_approx
                    tau_B_ortho = tau_B - tau_B_conflict - tau_B_synergy

                # --- ASAM 自适应赋权 ---
                lambda_s, lambda_c = self.args.lambda_s, self.args.lambda_c
                
                if self.args.adaptive_weighting and module_type.startswith('attn'):
                    try:
                        layers_idx = key.rfind("layers.")
                        relative_key_prefix = key[layers_idx:]
                        attn_module_name = ".".join(relative_key_prefix.split('.')[:-2])

                        Y_A_attn = activations_A[attn_module_name]['output']
                        Y_C_attn = activations_C[attn_module_name]['output']
                        delta_Y_attn_norm = torch.linalg.norm(Y_A_attn - Y_C_attn).item()

                        Q_A = activations_A[attn_module_name]['Q']
                        K_A = activations_A[attn_module_name]['K']
                        Q_C = activations_C[attn_module_name]['Q']
                        K_C = activations_C[attn_module_name]['K']
                        delta_S_norm = torch.linalg.norm((Q_A @ K_A.T) - (Q_C @ K_C.T)).item()
                        
                        if delta_Y_attn_norm > 1e-6:
                            ratio = delta_S_norm / delta_Y_attn_norm
                            lambda_s = self.args.lambda_s * max(0.5, 1 - self.args.alpha * ratio)
                            lambda_c = self.args.lambda_c * min(1.0, 0.1 + self.args.alpha * ratio)
                    except Exception:
                        # 如果计算失败，则使用默认值
                        pass

                w_star = W_A + (lambda_s * tau_B_synergy) - \
                               (lambda_c * tau_B_conflict) + \
                               (self.args.lambda_o * tau_B_ortho)
                
                merged_weights[key] = w_star.to(base_weights[key].dtype).cpu()
                
        # --- 保存模型 ---
        self._save_model(merged_weights)
        
        del activations_A, activations_C
        gc.collect()

    def _save_model(self, merged_weights):
        """辅助函数：保存模型权重和配置文件。"""
        print("\n正在保存合并后的模型...")
        try:
            index_path = os.path.join(self.args.base_model_path, "model.safetensors.index.json")
            with open(index_path, "r") as f:
                index_map = json.load(f)["weight_map"]
        except FileNotFoundError:
            print("未找到索引文件，将作为单个文件保存。")
            safetensors.torch.save_file(merged_weights, os.path.join(self.output_dir, "model.safetensors"))
            index_map = {key: "model.safetensors" for key in merged_weights.keys()}

        sharded_weights = defaultdict(dict)
        for key, value in merged_weights.items():
            if key in index_map:
                sharded_weights[index_map[key]][key] = value
        
        for filename, weights_dict in sharded_weights.items():
            safetensors.torch.save_file(weights_dict, os.path.join(self.output_dir, filename))
        
        if os.path.exists(index_path):
             shutil.copy(index_path, os.path.join(self.output_dir, os.path.basename(index_path)))
        
        create_soft_link(source_path=self.args.base_model_path, link_path=self.output_dir)
        print(f"模型成功合并并保存至: {self.output_dir}")
        
    def stage4_validate_and_finetune(self):
        """阶段四：全局验证与微调（概念实现）。"""
        print("\n--- [阶段四: 全局验证与微调 (ASAM)] ---")
        if not self.args.run_validation:
            print("跳过验证阶段。")
            return

        print("正在加载合并后的模型进行验证...")
        # 这是一个概念验证，实际评估需要一个合适的评估脚本
        # 这里我们只检查模型是否可以被成功加载
        try:
            ModelClass = AutoModelForVision2Seq if "VL" in self.args.base_model_path else AutoModelForCausalLM
            model = ModelClass.from_pretrained(self.output_dir, torch_dtype=torch.bfloat16)
            print("合并后的模型加载成功！")
            
            # 可以在这里添加计算困惑度或多模态准确率的代码
            # 例如: perplexity = calculate_perplexity(model, validation_data)
            # if perplexity > threshold:
            #     print("性能下降超过阈值，建议调整alpha并重新运行阶段三。")
            
            del model
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"加载或验证合并后的模型时出错: {e}")
            print("请检查合并过程是否完整。")

    def run_pipeline(self):
        """按顺序执行所有ASAM阶段。"""
        self.stage1_cache_activations()
        self.stage2_calculate_approx_gradients()
        self.stage3_merge_models()
        self.stage4_validate_and_finetune()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用ASAM v2.0框架进行低显存模型合并。")
    
    # 基本配置
    parser.add_argument('--base_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct", help="基础模型A的路径。")
    parser.add_argument('--donor_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si-hf", help="贡献模型B的路径。")
    parser.add_argument('--original_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-7B-Instruct", help="原始共同祖先模型C的路径。")
    parser.add_argument('--config_name', type=str, default="default", help="为本次合并配置命名，用于创建输出目录。")
    parser.add_argument('--cuda_device', type=int, default=3, help="使用的 CUDA 设备编号。")

    # 探针数据集配置
    parser.add_argument('--probe_samples', type=int, default=200, help="用于探测的样本数量。")
    parser.add_argument('--probe_batch_size', type=int, default=1, help="探测时的批处理大小。")

    # 静态合并超参数
    parser.add_argument('--lambda_s', type=float, default=1.4, help="协同分量的基础系数。")
    parser.add_argument('--lambda_c', type=float, default=0.7, help="冲突分量的基础系数。")
    parser.add_argument('--lambda_o', type=float, default=1.0, help="正交分量的系数。")
    
    # ASAM 自适应赋权配置
    parser.add_argument('--adaptive_weighting', action='store_true', help="为Attention层启用自适应赋权。")
    parser.add_argument('--alpha', type=float, default=0.5, help="自适应赋权机制的敏感度系数。")
    
    # 功能性参数
    parser.add_argument('--force_recompute', action='store_true', default=True ,help="强制重新计算缓存的激活或梯度。")
    parser.add_argument('--run_validation', action='store_true', help="在合并后执行阶段四的验证。")

    args = parser.parse_args()
    
    if torch.cuda.is_available() and args.cuda_device < torch.cuda.device_count():
        device = f"cuda:{args.cuda_device}"
    else:
        device = "cpu"

    print("--- ASAM v2.0 合并配置 ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("-------------------------")

    merger = ASAMerger(args, device)
    merger.run_pipeline()