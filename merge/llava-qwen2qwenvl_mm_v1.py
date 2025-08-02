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
from PIL import Image
import requests

# 导入指定的模型和分词器类
# MODIFIED: 引入 AutoProcessor 来处理多模态输入
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq
from torch.utils.data import DataLoader, Dataset

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

# NEW: 添加一个函数来标准化模型B (Llava) 的键名以匹配模型A和C
def normalize_donor_keys(weights_b: dict, weights_c: dict) -> dict:
    """
    将模型B (Llava) 的 LLM 部分的键名与模型C对齐。
    例如: "model.language_model.layers.0..." -> "model.layers.0..."
    """
    prefix_b = "model.language_model."
    prefix_c = "model."
    
    normalized_weights = {}
    for key_c in weights_c.keys():
        # 从模型C的键名推断出模型B中对应的键名
        # key_c = model.layers.0.mlp.down_proj.weight
        if key_c.startswith(prefix_c):
            suffix = key_c[len(prefix_c):] # layers.0.mlp.down_proj.weight
            key_b_equivalent = prefix_b + suffix # model.language_model.layers.0.mlp.down_proj.weight
            
            if key_b_equivalent in weights_b:
                # 使用模型C的键名作为标准键
                normalized_weights[key_c] = weights_b[key_b_equivalent]
                
    print(f"成功将 {len(normalized_weights)} 个 Donor LLM 权重键名标准化。")
    return normalized_weights

def need_merge(name: str) -> bool:
    """
    根据层名判断是否需要合并。我们只合并 LLM 核心层。
    """
    # MODIFIED: 简化逻辑，只关心 language_model.layers 部分
    if "language_model.layers" not in name:
        return False
    
    # 排除所有 layernorm 和 embedding
    if "layernorm" in name or "embed" in name:
        return False
        
    # 排除 rotary embedding 的频率缓存
    if name.endswith(".inv_freq"):
        return False
        
    # 其他在 layers 内部的参数都参与合并 (q,k,v,o, mlp)
    return True

# NEW: 为多模态探针数据创建一个自定义的数据集类
class VQAv2ProbeDataset(Dataset):
    def __init__(self, hf_dataset, processor, max_samples=100):
        self.samples = []
        for item in hf_dataset:
            # VQAv2 数据集结构: question, image, question_type, ...
            # 我们只需要 image 和 question
            self.samples.append({
                "image": item["image"],
                "text": item["question"]
            })
            if len(self.samples) >= max_samples:
                break
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = item["image"]
        text = f"Question: {item['text']} Answer:"

        # 如果图像是 RGBA，转换为 RGB
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # MODIFICATION: 只返回原始数据，让 collate_fn 统一处理
        return {"image": image, "text": text}

def collate_fn_factory(processor):
    """创建一个使用特定 processor 的 collate_fn"""
    def collate_fn(batch):
        """MODIFICATION: 使用 processor 和聊天模板来统一处理整个批次的数据"""
        images = [item["image"] for item in batch]
        
        # 为批次中的每个样本构建正确的聊天消息结构
        batch_messages = []
        for item in batch:
            # Qwen2-VL 的标准输入格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"}, # 图像占位符
                        {"type": "text", "text": f"Question: {item['text']} Answer:"}
                    ]
                }
            ]
            batch_messages.append(messages)

        # 使用 apply_chat_template 将消息结构转换为带 <image> token 的文本
        texts = [processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) for messages in batch_messages]

        # 现在，使用正确的文本和图像调用 processor
        # 注意：不再需要 image_sizes，因为聊天模板流程会处理好一切
        inputs = processor(
            text=texts, 
            images=images, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        return inputs # processor 的输出可以直接作为模型输入
    return collate_fn

def create_soft_link(source_path, link_path):
    # (此函数无需修改，保持原样)
    if not os.path.exists(source_path):
        print(f"Error: Source path '{source_path}' does not exist.")
        return
    if not os.path.exists(link_path):
        os.makedirs(link_path)
    for item in os.listdir(source_path):
        source_item = os.path.join(source_path, item)
        link_item = os.path.join(link_path, item)
        if item.endswith('.bin'):
            continue
        if os.path.isfile(source_item):
            try:
                if os.path.exists(link_item):
                    os.remove(link_item)
                os.symlink(source_item, link_item)
            except OSError as e:
                print(f"Error creating soft link for '{item}': {e}")
        elif os.path.isdir(source_item):
            continue

# --- 核心实现类 ---

class LowMemoryGradientMerger:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.output_dir = os.path.join("merged_models", f"mm-gradient-merge-{args.mode}")
        self.cache_dir = os.path.join(self.output_dir, "cache")
        self.grad_dir = os.path.join(self.cache_dir, "approx_grads")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.grad_dir, exist_ok=True)
        print(f"使用设备: {self.device}")
        print(f"输出将保存至: {self.output_dir}")

    def _get_target_modules(self, model):
        """获取模型中所有需要被hook的目标模块的名称。"""
        target_module_names = set()
        for name, _ in model.named_modules():
            # 使用完整的、相对于顶层模型的键名来判断
            if need_merge(name):
                target_module_names.add(name)
        return list(target_module_names)

    # MODIFIED: 完全重构此函数以适应多模态和纯文本模型的不同处理流程
    def _cache_activations_for_model(self, model_path, cache_path, is_vision_model, is_ancestor, capture_inputs=False):
        """阶段一的核心函数：为单个模型执行前向传播并缓存激活。"""
        if os.path.exists(cache_path) and not self.args.force_recompute:
            print(f"激活缓存文件已存在: {cache_path}, 跳过。")
            return

        print(f"正在为 {os.path.basename(model_path)} 缓存激活...")
        
        # --- 模型和数据处理器加载 ---
        if is_vision_model:
            model = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(self.device)
            # 使用 processor 来统一处理图像和文本
            processor = AutoProcessor.from_pretrained(model_path)
        else: # 是纯文本祖先模型
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        # --- 准备探针数据集 ---
        print("正在准备多模态探针数据集 (VQAv2)...")
        # 使用 VQAv2 数据集，它包含图像和问题
        probe_dataset_raw = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
        
        if is_vision_model:
            probe_dataset = VQAv2ProbeDataset(probe_dataset_raw, processor, max_samples=self.args.probe_samples)
            # 修复：调用工厂函数来获取实际的 collate_fn
            collate_function = collate_fn_factory(processor)
            probe_dataloader = DataLoader(probe_dataset, batch_size=self.args.probe_batch_size, collate_fn=collate_function)
        else: # 对于纯文本模型，我们只提取文本部分
            probe_texts = [f"Question: {item['question']} Answer:" for item in probe_dataset_raw.take(self.args.probe_samples)]
            probe_inputs = tokenizer(probe_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            probe_dataset = torch.utils.data.TensorDataset(probe_inputs['input_ids'], probe_inputs['attention_mask'])
            probe_dataloader = DataLoader(probe_dataset, batch_size=self.args.probe_batch_size)

        # --- 注册钩子 ---
        # 我们的目标是 'language_model.layers'
        model_to_hook = model.model # 对于 Qwen2-VL 和 Qwen2-Instruct 都是 .model
        target_module_names = self._get_target_modules(model_to_hook)
        
        if not target_module_names:
            print(f"警告: 在 {os.path.basename(model_path)} 中没有找到符合 `need_merge` 条件的模块。")
            del model, probe_dataloader
            gc.collect()
            torch.cuda.empty_cache()
            return
            
        print(f"在 {os.path.basename(model_path)} 的 LLM 部分找到 {len(target_module_names)} 个目标模块。")

        hooks = []
        captured_activations = defaultdict(lambda: {"inputs": [], "outputs": []})

        def get_hook(name):
            def hook_fn(module, input, output):
                output_tensor = output[0] if isinstance(output, tuple) else output
                if not isinstance(output_tensor, torch.Tensor): return
                captured_activations[name]["outputs"].append(output_tensor.detach().cpu())
                
                if capture_inputs:
                    # 修正：健壮地处理输入。如果输入元组为空，则直接返回。
                    # 这是因为某些模块（如自注意力）可能使用关键字参数调用，导致输入元组为空。
                    if not input:
                        return
                    input_tensor = input[0] if isinstance(input, tuple) else input
                    if not isinstance(input_tensor, torch.Tensor): return
                    captured_activations[name]["inputs"].append(input_tensor.detach().cpu())
            return hook_fn

        for name, module in model_to_hook.named_modules():
            if name in target_module_names:
                hooks.append(module.register_forward_hook(get_hook(name)))

        # --- 执行前向传播 ---
        model.eval()
        with torch.no_grad():
            for batch in tqdm(probe_dataloader, desc=f"前向传播 {os.path.basename(model_path)}"):
                if is_vision_model:
                    # MODIFICATION: 直接将 processor 输出的整个批次传给模型
                    # 将批次中的所有张量移动到目标设备
                    batch = {k: v.to(self.device, dtype=torch.bfloat16 if k == "pixel_values" else v.dtype) for k, v in batch.items()}
                    # 使用 **batch 将所有参数（包括 image_sizes）解包传入模型
                    model(**batch)
                else:
                    # 纯文本模型输入
                    input_ids, attention_mask = batch[0].to(self.device), batch[1].to(self.device)
                    model(input_ids=input_ids, attention_mask=attention_mask)

        for h in hooks: h.remove()
        
        # --- 求平均并保存 ---
        averaged_activations = {}
        for name, data in captured_activations.items():
            averaged_activations[name] = {}
            if data["outputs"]:
                try:
                    # 在拼接前确保所有张量都在CPU上且为float32
                    tensors = [t.float() for t in data["outputs"]]
                    averaged_activations[name]["output"] = torch.mean(torch.cat(tensors, dim=0), dim=0)
                except Exception as e:
                    print(f"处理 {name} 的输出激活时出错: {e}")
                    continue
            if data["inputs"]:
                try:
                    tensors = [t.float() for t in data["inputs"]]
                    averaged_activations[name]["input"] = torch.mean(torch.cat(tensors, dim=0), dim=0)
                except Exception as e:
                    print(f"处理 {name} 的输入激活时出错: {e}")
                    continue

        torch.save(averaged_activations, cache_path)
        print(f"激活已缓存至: {cache_path}")
        
        del model, captured_activations, averaged_activations, probe_dataloader
        gc.collect()
        torch.cuda.empty_cache()

    def stage1_cache_activations(self):
        """执行阶段一：缓存模型A和模型C的激活。"""
        print("\n--- [阶段一: 缓存激活] ---")
        activations_a_path = os.path.join(self.cache_dir, "activations_A.pt")
        activations_c_path = os.path.join(self.cache_dir, "activations_C.pt")

        # MODIFIED: 明确指定模型类型
        # 缓存模型A (qwen2_vl) 的输入和输出激活
        self._cache_activations_for_model(self.args.base_model_path, activations_a_path, is_vision_model=True, is_ancestor=False, capture_inputs=True)
        # 缓存模型C (original_model) 的输出激活
        self._cache_activations_for_model(self.args.original_model_path, activations_c_path, is_vision_model=False, is_ancestor=True, capture_inputs=False)


    def stage2_calculate_approx_gradients(self):
        """执行阶段二：逐层计算近似梯度。"""
        print("\n--- [阶段二: 计算近似梯度] ---")
        
        activations_a_path = os.path.join(self.cache_dir, "activations_A.pt")
        activations_c_path = os.path.join(self.cache_dir, "activations_C.pt")
        if not os.path.exists(activations_a_path) or not os.path.exists(activations_c_path):
            print("错误: 激活缓存文件不存在。请先运行阶段一。")
            sys.exit(1)
            
        print("加载缓存的激活...")
        activations_A = torch.load(activations_a_path, map_location="cpu")
        activations_C = torch.load(activations_c_path, map_location="cpu")
        
        # 以模型A的参数作为遍历蓝本
        base_weights = load_weights(self.args.base_model_path, "model.safetensors.index.json")

        for key in tqdm(base_weights.keys(), desc="计算近似梯度"):
            if not need_merge(key):
                continue
                
            grad_path = os.path.join(self.grad_dir, f"{key.replace('/', '_')}.pt")
            if os.path.exists(grad_path) and not self.args.force_recompute:
                continue

            # MODIFIED: 键名直接用于查找，因为我们已经对齐了目标模块结构
            # `key` = "model.language_model.layers.0.mlp..." for Qwen2-VL
            # 钩子注册的模块名也是这个
            module_name = ".".join(key.split('.')[:-1])

            if module_name not in activations_A or module_name not in activations_C:
                print(f"警告: 模块 {module_name} (来自键 {key}) 的激活未找到，跳过。")
                continue

            # 加载激活到设备
            if "input" not in activations_A[module_name] or "output" not in activations_A[module_name] or "output" not in activations_C[module_name]:
                print(f"警告: 模块 {module_name} 的输入/输出数据不完整，跳过。")
                continue
            
            X_A = activations_A[module_name]["input"].to(self.device)
            Y_A = activations_A[module_name]["output"].to(self.device)
            Y_C = activations_C[module_name]["output"].to(self.device)
            
            # 计算期望变化方向
            delta_Y = Y_A - Y_C
            
            g_approx = None
            if key.endswith(".weight"):
                # 确保维度匹配
                if delta_Y.shape[0] == X_A.shape[0]:
                    g_approx = delta_Y.T @ X_A
                else:
                    print(f"警告：{key} 的维度不匹配，跳过。delta_Y: {delta_Y.shape}, X_A: {X_A.shape}")
            elif key.endswith(".bias"):
                g_approx = delta_Y.sum(dim=0)
            
            if g_approx is not None:
                torch.save(g_approx.cpu(), grad_path)

        del base_weights, activations_A, activations_C
        gc.collect()
        print("所有近似梯度计算并保存完毕。")

    def stage3_merge_models(self):
        """执行阶段三：逐层加载权重，执行梯度引导的分解与合并。"""
        print("\n--- [阶段三: 分解与合并] ---")

        print("正在从磁盘加载所有模型权重...")
        base_weights_A = load_weights(self.args.base_model_path, "model.safetensors.index.json")
        donor_weights_B_raw = load_weights(self.args.donor_model_path, "model.safetensors.index.json")
        original_weights_C = load_weights(self.args.original_model_path, "model.safetensors.index.json")

        # NEW: 标准化模型B的LLM键名，以使其与模型A和C对齐
        # 注意：这里我们只关心LLM部分，非LLM部分在标准化后会被舍弃
        print("正在标准化 Donor 模型 (Llava) 的 LLM 层名...")
        donor_weights_B = normalize_donor_keys(donor_weights_B_raw, original_weights_C)
        del donor_weights_B_raw
        gc.collect()
        
        # MODIFIED: 以模型A的权重作为最终合并的起点
        final_merged_weights = base_weights_A.copy()
        print("已将基础模型A的权重作为合并起点。")

        merged_count = 0
        for key in tqdm(final_merged_weights.keys(), desc="逐层合并权重"):
            # 默认已是基础模型权重，我们只处理需要合并的层
            # MODIFIED: 对齐键名，统一用模型C的键名格式来查找
            # key in A = "model.language_model.layers.0..."
            # key in C = "model.layers.0..."
            # key in normalized B = "model.layers.0..."
            
            # 从模型A的键名构造出模型B和C中对应的键名
            prefix_a = "model.language_model."
            if key.startswith(prefix_a):
                key_in_c = "model." + key[len(prefix_a):]
            else:
                key_in_c = key
                
            # 检查是否满足合并条件
            if need_merge(key) and key_in_c in donor_weights_B and key_in_c in original_weights_C:
                # 形状检查
                if base_weights_A[key].shape != donor_weights_B[key_in_c].shape or base_weights_A[key].shape != original_weights_C[key_in_c].shape:
                    print(f"警告: 参数 {key} 形状不匹配，跳过。 A: {base_weights_A[key].shape}, B: {donor_weights_B[key_in_c].shape}, C: {original_weights_C[key_in_c].shape}")
                    continue

                grad_path = os.path.join(self.grad_dir, f"{key.replace('/', '_')}.pt")
                if not os.path.exists(grad_path):
                    print(f"警告: 参数 {key} 的近似梯度未找到，将使用基础模型权重。")
                    continue

                # 加载张量到设备
                W_A = base_weights_A[key].float().to(self.device)
                W_B = donor_weights_B[key_in_c].float().to(self.device)
                W_C = original_weights_C[key_in_c].float().to(self.device)
                g_approx = torch.load(grad_path, map_location=self.device).float()
                
                # 梯度形状必须与权重形状匹配
                if g_approx.shape != W_A.shape:
                    print(f"警告: {key} 的梯度形状 {g_approx.shape} 与权重形状 {W_A.shape} 不匹配，跳过。")
                    continue

                # 计算任务向量 τ_B
                tau_B = W_B - W_C

                # --- 梯度引导的向量分解 ---
                g_norm_sq = torch.sum(g_approx * g_approx)
                if g_norm_sq < 1e-9:
                    tau_B_synergy, tau_B_conflict = torch.zeros_like(tau_B), torch.zeros_like(tau_B)
                    tau_B_proj = torch.zeros_like(tau_B)
                else:
                    proj_scalar = torch.sum(g_approx * tau_B) / g_norm_sq
                    tau_B_proj = proj_scalar * g_approx
                    
                    if proj_scalar < 0: # 协同
                        tau_B_synergy = -tau_B_proj
                        tau_B_conflict = torch.zeros_like(tau_B)
                    else: # 冲突
                        tau_B_synergy = torch.zeros_like(tau_B)
                        tau_B_conflict = tau_B_proj
                
                # 正交分量
                tau_B_ortho = tau_B - tau_B_proj

                # --- 最终合并公式 ---
                w_star = W_A + (self.args.lambda_s * tau_B_synergy) - \
                                (self.args.lambda_c * tau_B_conflict) + \
                                (self.args.lambda_o * tau_B_ortho)
                
                final_merged_weights[key] = w_star.to(base_weights_A[key].dtype).cpu()
                merged_count += 1
        
        print(f"合并完成，共计 {merged_count} 个参数被修改。")

        # --- 保存模型 ---
        print("\n正在保存合并后的模型...")
        index_path = os.path.join(self.args.base_model_path, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index_map = json.load(f)["weight_map"]
        
        sharded_weights = defaultdict(dict)
        for key, value in final_merged_weights.items():
            if key in index_map:
                shard_file = index_map[key]
                sharded_weights[shard_file][key] = value
            else:
                print(f"警告: 权重 {key} 未在索引文件中找到，无法保存。")

        for filename, weights_dict in sharded_weights.items():
            safetensors.torch.save_file(weights_dict, os.path.join(self.output_dir, filename))
        
        # 复制/链接其他必要文件
        shutil.copy(index_path, os.path.join(self.output_dir, os.path.basename(index_path)))
        create_soft_link(source_path=self.args.base_model_path, link_path=self.output_dir)
        print(f"模型成功合并并保存至: {self.output_dir}")

    def run_pipeline(self):
        """按顺序执行所有阶段。"""
        self.stage1_cache_activations()
        self.stage2_calculate_approx_gradients()
        self.stage3_merge_models()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用多模态局部梯度近似进行低显存模型合并。")
    
    # 基本配置
    parser.add_argument('--base_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct", help="基础模型A的路径。")
    parser.add_argument('--donor_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si-hf", help="贡献模型B的路径。")
    parser.add_argument('--original_model_path', type=str, default="/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-7B-Instruct", help="原始共同祖先模型C的路径。")
    parser.add_argument('--mode', type=str, default="qwen-vl-merge", help="为本次合并配置命名。")
    parser.add_argument('--cuda_device', type=int, default=0, help="使用的 CUDA 设备编号。")

    # 探针数据集配置
    parser.add_argument('--probe_samples', type=int, default=100, help="用于探测的样本数量。")
    parser.add_argument('--probe_batch_size', type=int, default=2, help="探测时的批处理大小，如果显存不足请减小。")

    # 合并超参数
    parser.add_argument('--lambda_s', type=float, default=1.0, help="协同分量的系数。")
    parser.add_argument('--lambda_c', type=float, default=0.5, help="冲突分量的系数。")
    parser.add_argument('--lambda_o', type=float, default=0.8, help="正交分量的系数。")
    
    # 功能性参数
    parser.add_argument('--force_recompute', action='store_true', help="强制重新计算缓存的激活或梯度。")

    args = parser.parse_args()
    
    # 设置设备
    if torch.cuda.is_available() and args.cuda_device < torch.cuda.device_count():
        device = f"cuda:{args.cuda_device}"
    else:
        device = "cpu"

    print("--- 配置信息 ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("--------------------")

    merger = LowMemoryGradientMerger(args, device)
    merger.run_pipeline()