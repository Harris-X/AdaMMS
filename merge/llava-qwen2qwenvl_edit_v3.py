# llava_merging_adaptive_final.py

import os
import sys
import json
import torch
import safetensors.torch
import argparse
from tqdm import tqdm
import gc
import shutil
# 修复点：从 transformers 导入正确的模型类
from transformers import AutoTokenizer, AutoModelForVision2Seq,LlavaOnevisionForConditionalGeneration
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 关键依赖：
# 1. AlphaEdit/rome for High-Divergence Case
# 2. Hugging Face datasets library for probe data
try:
    from rome.layer_stats import layer_stats
except ImportError:
    print("="*80, file=sys.stderr)
    print("错误：无法导入 'rome.layer_stats'。", file=sys.stderr)
    print("请确保 AlphaEdit 项目中的 'rome' 文件夹在您的 Python 路径中。", file=sys.stderr)
    print("这个模块对于高冲突层的 'null_space' 策略至关重要。", file=sys.stderr)
    print("="*80, file=sys.stderr)
    layer_stats = None # 允许在没有它的情况下运行其他策略

try:
    from datasets import load_dataset
except ImportError:
    print("="*80, file=sys.stderr)
    print("错误：无法导入 'datasets' 库。请运行 `pip install datasets`。", file=sys.stderr)
    print("这个库是获取探针数据集所必需的。", file=sys.stderr)
    print("="*80, file=sys.stderr)
    sys.exit(1)

# --- 模型与路径配置 (请在使用前修改) ---
CKPT_PATH = {
    "original_model": "/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-7B-Instruct",
    "qwen2_vl": "/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct",
    "llava-onevision-qwen": "/home/user/xieqiuhao/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si-hf"
}
INDEX_FILENAME = {
    "original_model": "model.safetensors.index.json",
    "qwen2_vl": "model.safetensors.index.json",
    "llava-onevision-qwen": "model.safetensors.index.json"
}
STATS_DIR = "hparams_cache"
os.makedirs(STATS_DIR, exist_ok=True)

# --- 权重加载与辅助函数 (遵循您的模板) ---
def load_weights(base_path, index_filename):
    weights = {}
    index_path = os.path.join(base_path, index_filename)
    if not os.path.exists(index_path):
        single_file_path = os.path.join(base_path, "model.safetensors")
        if os.path.exists(single_file_path):
            print(f"Loading single weight file: {single_file_path}")
            return safetensors.torch.load_file(single_file_path)
        else:
            raise FileNotFoundError(f"Neither {index_filename} nor model.safetensors found in {base_path}")
    with open(index_path, 'r') as f:
        index = json.load(f)
    file_list = sorted(list(set(index["weight_map"].values())))
    for file in tqdm(file_list, desc=f"Loading weights from {os.path.basename(base_path)}"):
        weights.update(safetensors.torch.load_file(os.path.join(base_path, file)))
    return weights

def normalize_donor_keys(weights: dict) -> dict:
    """专门用于标准化 donor 模型（llava-onevision-qwen）的 key。"""
    prefix_to_remove = "language_model."
    normalized_weights = {}
    for key, value in weights.items():
        if key.startswith(prefix_to_remove):
            normalized_weights[key[len(prefix_to_remove):]] = value
        else:
            # 对于非语言模型部分（如 vision_tower），保留原样
            normalized_weights[key] = value
    return normalized_weights

def need_merge(name: str) -> bool:
    if any(k in name for k in ['lm_head', 'embed_tokens', 'rotary_emb.inv_freq', 'vision_tower']):
        return False
    # 我们主要合并语言模型的 Transformer 层
    return name.startswith("model.layers.") or name == 'model.norm.weight'

def create_soft_link(source_path, link_path):
    print(f"Creating symbolic links from {source_path} to {link_path} for non-weight files...")
    for item in os.listdir(source_path):
        if not item.endswith(('.safetensors', '.bin', '.py', '.md')):
            source_item, link_item = os.path.join(source_path, item), os.path.join(link_path, item)
            if not os.path.exists(link_item):
                try: os.symlink(source_item, link_item)
                except OSError as e: print(f"Error linking {item}: {e}", file=sys.stderr)

# --- 激活散度与合并逻辑 ---
def gram_linear(x):
    return x @ x.T

def cka(X, Y, kernel=gram_linear):
    X = X.cuda().float()
    Y = Y.cuda().float()
    # 中心化特征
    X -= X.mean(dim=0, keepdim=True)
    Y -= Y.mean(dim=0, keepdim=True)
    
    K_X, K_Y = kernel(X), kernel(Y)
    hsic = torch.trace(K_X @ K_Y)
    var_X = torch.sqrt(torch.trace(K_X @ K_X))
    var_Y = torch.sqrt(torch.trace(K_Y @ K_Y))
    
    return (hsic / (var_X * var_Y)).item() if (var_X * var_Y) > 0 else 0.0

@torch.no_grad()
def get_activations(model, tokenizer, layer_names, probe_dataloader, model_name, device):
    model.eval()
    activations = {name: [] for name in layer_names}
    hooks = []

    def get_hook(name):
        def hook_fn(module, input, output):
            # Qwen2VLForCausalLM的Transformer块输出是元组 (hidden_states, ...)
            # 我们需要第一个元素
            activations[name].append(output[0].detach().cpu())
        return hook_fn

    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(get_hook(name)))

    for batch in tqdm(probe_dataloader, desc=f"Probing activations for {model_name}"):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        
        # 修复点：只传入文本相关的输入，不激活视觉部分
        # 移除 dummy_pixel_values
        model(input_ids=input_ids, attention_mask=attention_mask)

    for hook in hooks: hook.remove()
    for name in layer_names: 
        if activations[name]:
            activations[name] = torch.cat(activations[name], dim=0)
    return activations

def compute_covariance_and_projector(model, tok, layer_name, hparams):
    # (此函数逻辑与上一版本相同，确保 rome/layer_stats 可用)
    if layer_stats is None:
        raise ImportError("`rome` library is required for high-divergence strategy.")
    # ... (此处省略，与上一版本实现相同) ...
    model_name_safe = hparams.base_model_path.replace("/", "_")
    cache_path = os.path.join(STATS_DIR, f"projector__{model_name_safe}__{layer_name.replace('.', '_')}__{hparams.null_space_threshold}.pt")

    if os.path.exists(cache_path) and not hparams.force_recompute:
        print(f"Loading cached projector for {layer_name}.")
        return torch.load(cache_path)

    print(f"\nComputing covariance for {layer_name}...")
    stat = layer_stats(
        model, tok, layer_name, STATS_DIR,
        hparams.mom2_dataset, to_collect=["mom2"],
        sample_size=hparams.mom2_n_samples,
        precision=hparams.mom2_dtype,
        force_recompute=hparams.force_recompute,
    )
    cov = stat.mom2.moment().float().cuda()

    print(f"Computing SVD and projector for {layer_name}...")
    U, S, _ = torch.linalg.svd(cov)
    
    projector = U[:, S < hparams.null_space_threshold]
    projector = projector @ projector.T
    
    print(f"Finished projector for {layer_name}. Null-space dim: {projector.shape[0] - torch.matrix_rank(projector).item()}")
    torch.save(projector.cpu(), cache_path)
    return projector


# --- 主转换函数 ---
def convert(args, device):
    # ... (输出路径设置) ...
    output_dir = "merged_models"
    model_name = f"adaptive-merge-{args.mode}"
    OUTPUT_PATH = os.path.join(output_dir, model_name)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # --- 模型权重加载 ---
    print("Loading all model weights from disk...")
    base_weights = load_weights(args.base_model_path, INDEX_FILENAME["qwen2_vl"])
    
    # 加载并标准化 donor 模型的权重
    donor_weights_raw = load_weights(args.donor_model_path, INDEX_FILENAME["llava-onevision-qwen"])
    print("Normalizing donor model keys...")
    donor_weights = normalize_donor_keys(donor_weights_raw)
    
    original_weights = load_weights(args.original_model_path, INDEX_FILENAME["original_model"])
    
    merged_weights = {}

    # --- 自适应合并逻辑 ---
    print("="*80); print("Applying 'Activation-Guided Adaptive Merging' strategy."); print("="*80)
    
    # 1. 准备探针数据集
    print(f"Preparing probe dataset from '{args.probe_dataset}'...")
    probe_dataset_raw = load_dataset(args.probe_dataset, "20220301.en" if "wikipedia" in args.probe_dataset else "en", split="train", streaming=True).take(args.probe_samples)
    probe_texts = [item['text'] for item in probe_dataset_raw if item['text']]

    # 2. 加载模型A用于激活探测
    print("Loading Base Model (A) to GPU for activation probing...")
    model_a = AutoModelForVision2Seq.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer_a = AutoTokenizer.from_pretrained(args.base_model_path)

    # 3. 计算模型A的激活
    # model_a 的目标层
    target_layers_a = [k for k, v in model_a.named_modules() if isinstance(v, torch.nn.Module) and k.startswith("model.language_model.layers.")]
    if not target_layers_a:
        print("错误: 在基础模型中未找到任何以 'model.language_model.layers.' 开头的层。", file=sys.stderr); sys.exit(1)
    print(f"Found {len(target_layers_a)} target layers in Base Model.")

    probe_inputs_a = tokenizer_a(probe_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    probe_dataset_a = TensorDataset(probe_inputs_a['input_ids'], probe_inputs_a['attention_mask'])
    probe_dataloader_a = DataLoader(probe_dataset_a, batch_size=args.probe_batch_size)
    activations_a = get_activations(model_a, tokenizer_a, target_layers_a, probe_dataloader_a, "Base Model", model_a.device)
    del model_a, tokenizer_a, probe_inputs_a, probe_dataset_a, probe_dataloader_a; gc.collect(); torch.cuda.empty_cache()

    # 4. 加载模型B用于激活探测
    print("\nLoading Donor Model (B) to GPU for activation probing...")
    from transformers import AutoConfig
    tokenizer_b = AutoTokenizer.from_pretrained(args.donor_model_path)
    model_b = AutoModelForVision2Seq.from_pretrained(
        args.donor_model_path,
        torch_dtype=torch.bfloat16
    ).to(device)

    # 5. 计算模型B的激活
    # 修复点：为 model_b 创建其自己的目标层列表
    target_layers_b = [k for k, v in model_b.named_modules() if isinstance(v, torch.nn.Module) and k.startswith("language_model.layers.")]
    if not target_layers_b:
        print("错误: 在增量模型中未找到任何以 'language_model.model.layers.' 开头的层。", file=sys.stderr); sys.exit(1)
    print(f"Found {len(target_layers_b)} target layers in Donor Model.")

    probe_inputs_b = tokenizer_b(probe_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    probe_dataset_b = TensorDataset(probe_inputs_b['input_ids'], probe_inputs_b['attention_mask'])
    probe_dataloader_b = DataLoader(probe_dataset_b, batch_size=args.probe_batch_size)
    # 使用 target_layers_b 获取激活
    activations_b = get_activations(model_b, tokenizer_b, target_layers_b, probe_dataloader_b, "Donor Model", model_b.device)
    del model_b, tokenizer_b, probe_inputs_b, probe_dataset_b, probe_dataloader_b; gc.collect(); torch.cuda.empty_cache()

    # 6. 计算散度并合并
    divergence_scores = {}
    # 修复点：创建一个从 model_a 层名到 model_b 层名的映射
    map_a_to_b = {a_name: b_name for a_name, b_name in zip(target_layers_a, target_layers_b)}

    for layer_name_a in tqdm(target_layers_a, desc="Calculating CKA Divergence"):
        layer_name_b = map_a_to_b[layer_name_a] # 获取对应的 model_b 层名
        
        # 检查两个模型的激活是否都已成功获取
        if layer_name_a not in activations_a or not activations_a[layer_name_a].numel():
            print(f"警告: 模型A中层 {layer_name_a} 的激活为空，跳过。")
            continue
        if layer_name_b not in activations_b or not activations_b[layer_name_b].numel():
            print(f"警告: 模型B中层 {layer_name_b} 的激活为空，跳过。")
            continue

        act_a = activations_a[layer_name_a].view(activations_a[layer_name_a].shape[0], -1)
        act_b = activations_b[layer_name_b].view(activations_b[layer_name_b].shape[0], -1)
        
        # 将散度分数存储在以 model_a 的层名为 key 的字典中，方便后续合并使用
        divergence_scores[layer_name_a] = 1 - cka(act_a, act_b)

    del activations_a, activations_b; gc.collect(); torch.cuda.empty_cache()

    if not divergence_scores:
        print("错误：未能计算任何层的散度分数，无法继续合并。请检查层名匹配逻辑。", file=sys.stderr)
        sys.exit(1)

    all_divergences = np.array(list(divergence_scores.values()))
    t_low = np.percentile(all_divergences, args.low_div_percentile)
    t_high = np.percentile(all_divergences, args.high_div_percentile)
    print(f"Automated Divergence Thresholds: Low < {t_low:.4f} | High > {t_high:.4f}")

    # 7. 逐层自适应合并
    tokenizer_for_cov = AutoTokenizer.from_pretrained(args.base_model_path) # For high-div case
    # 修复点：迭代基础模型的key，因为它是最终结构
    for key in tqdm(base_weights.keys(), desc="Applying Adaptive Merging"):
        # 使用标准化后的 donor_weights 进行检查
        if key in donor_weights and key in original_weights and need_merge(key) and base_weights[key].shape == donor_weights[key].shape:
            module_path = key.rsplit('.', 1)[0]
            divergence = divergence_scores.get(module_path, (t_low + t_high) / 2)
            
            w_c = original_weights[key].float().to(device)
            w_a = base_weights[key].float().to(device)
            w_b = donor_weights[key].float().to(device)

            if divergence > t_high and layer_stats:
                print(f"Layer {key}: High divergence ({divergence:.4f}). Using activation-space null-space grafting.")
                model_a_for_cov = AutoModelForVision2Seq.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16).to(device)
                projector = compute_covariance_and_projector(model_a_for_cov, tokenizer_for_cov, module_path, args).to(device)
                delta = w_b - w_a
                projected_delta = delta @ projector
                w_star = w_a + projected_delta
                del model_a_for_cov; gc.collect(); torch.cuda.empty_cache()
            else:
                if divergence < t_low:
                    lambda_s, lambda_c = args.lambda_s_low, args.lambda_c_low
                else:
                    lambda_s, lambda_c = args.lambda_s_mid, args.lambda_c_mid
                lambda_o = args.lambda_o

                tau_a, tau_b = w_a - w_c, w_b - w_c
                tau_a_norm_sq = torch.sum(tau_a * tau_a)
                if tau_a_norm_sq > 1e-9:
                    proj_scalar = torch.sum(tau_a * tau_b) / tau_a_norm_sq
                    tau_b_synergy = torch.clamp(proj_scalar, min=0) * tau_a
                    tau_b_conflict = torch.clamp(-proj_scalar, min=0) * tau_a
                    tau_b_ortho = tau_b - (tau_b_synergy - tau_b_conflict)
                else:
                    tau_b_synergy, tau_b_conflict, tau_b_ortho = torch.zeros_like(tau_b), torch.zeros_like(tau_b), tau_b
                w_star = w_a + (lambda_s * tau_b_synergy) - (lambda_c * tau_b_conflict) + (lambda_o * tau_b_ortho)
        
            merged_weights[key] = w_star.to(base_weights[key].dtype).cpu()
            gc.collect(); torch.cuda.empty_cache()
        else:
            # 对于不合并或在其他模型中不存在的层，直接使用基础模型的权重
            merged_weights[key] = base_weights.get(key)


    # --- 保存模型 ---
    # ... (与之前版本相同) ...
    print("\nSaving merged model...")
    index_path = os.path.join(args.base_model_path, INDEX_FILENAME["qwen2_vl"])
    with open(index_path, "r") as f: index_map = json.load(f)["weight_map"]
    
    sharded_weights = {filename: {} for filename in set(index_map.values())}
    for key, value in merged_weights.items():
        if key in index_map: sharded_weights[index_map[key]][key] = value
    
    for filename, weights_dict in sharded_weights.items():
        safetensors.torch.save_file(weights_dict, os.path.join(OUTPUT_PATH, filename))
    
    create_soft_link(source_path=args.base_model_path, link_path=OUTPUT_PATH)
    shutil.copy(index_path, os.path.join(OUTPUT_PATH, os.path.basename(index_path)))
    print(f"Merged model saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptively merge models based on activation divergence.")
    
    # 修复点：添加 cuda_device 参数
    parser.add_argument('--cuda_device', type=int, default=1, help="CUDA device to use (e.g., 0, 1, 2).")
    # ... (参数定义与上一版本相同) ...
    # Model Paths
    parser.add_argument('--base_model_path', type=str, default=CKPT_PATH["qwen2_vl"])
    parser.add_argument('--donor_model_path', type=str, default=CKPT_PATH["llava-onevision-qwen"])
    parser.add_argument('--original_model_path', type=str, default=CKPT_PATH["original_model"])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--mode', type=str, default="default", help="A name for this merging configuration.")

    # Probe Dataset Config
    parser.add_argument('--probe_dataset', type=str, default="wikipedia", help="Dataset for probing activations ('wikipedia' or 'c4').")
    parser.add_argument('--probe_samples', type=int, default=128, help="Number of samples for probing.")
    # 修复点：降低默认批处理大小
    parser.add_argument('--probe_batch_size', type=int, default=1, help="Batch size for probing. Reduce if OOM.")

    # Adaptive Strategy Config
    parser.add_argument('--low_div_percentile', type=float, default=33, help="Percentile to define low divergence threshold.")
    parser.add_argument('--high_div_percentile', type=float, default=66, help="Percentile to define high divergence threshold.")
    
    # λ coefficients for each divergence zone
    parser.add_argument('--lambda_s_low', type=float, default=1.2, help="Synergy coeff for low divergence.")
    parser.add_argument('--lambda_c_low', type=float, default=1.0, help="Conflict coeff for low divergence.")
    parser.add_argument('--lambda_s_mid', type=float, default=1.0, help="Synergy coeff for medium divergence.")
    parser.add_argument('--lambda_c_mid', type=float, default=0.5, help="Conflict coeff for medium divergence.")
    parser.add_argument('--lambda_o', type=float, default=1.0, help="Orthogonal knowledge coefficient (usually 1.0).")
    
    # Null-space projection config (for high divergence)
    parser.add_argument('--mom2_dataset', type=str, default="wikipedia")
    parser.add_argument('--mom2_n_samples', type=int, default=1000)
    parser.add_argument('--mom2_dtype', type=str, default="bfloat16")
    parser.add_argument('--null_space_threshold', type=float, default=1e-3)
    parser.add_argument('--force_recompute', action='store_true')

    args = parser.parse_args()
    
    # 修复点：设置设备
    if torch.cuda.is_available() and args.cuda_device < torch.cuda.device_count():
        device = f"cuda:{args.cuda_device}"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    CKPT_PATH.update({
        "qwen2_vl": args.base_model_path,
        "llava-onevision-qwen": args.donor_model_path,
        "original_model": args.original_model_path
    })

    print("--- Configuration ---")
    print(vars(args))
    print("--------------------")

    convert(args, device)