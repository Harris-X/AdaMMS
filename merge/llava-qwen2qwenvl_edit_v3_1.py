# llava_merging_adaptive_blocks.py

import os
import sys
import json
import torch
import safetensors.torch
import argparse
from tqdm import tqdm
import gc
import shutil
from transformers import AutoTokenizer, AutoModelForVision2Seq, LlavaOnevisionForConditionalGeneration
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# --- 关键依赖 ---
try:
    from rome.layer_stats import layer_stats
except ImportError:
    print("="*80, file=sys.stderr); print("错误：无法导入 'rome.layer_stats'。", file=sys.stderr); sys.exit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("="*80, file=sys.stderr); print("错误：无法导入 'datasets' 库。请运行 `pip install datasets`。", file=sys.stderr); sys.exit(1)

# --- 模型与路径配置 (请在使用前修改) ---
CKPT_PATH = {
    "original_model": "/path/to/your/Qwen2-7B-Instruct",
    "qwen2_vl": "/path/to/your/Qwen2-VL-7B-Instruct",
    "llava-onevision-qwen": "/path/to/your/llava-onevision-qwen2-7b-si-hf"
}
INDEX_FILENAME = { "original_model": "model.safetensors.index.json", "qwen2_vl": "model.safetensors.index.json", "llava-onevision-qwen": "model.safetensors.index.json" }
STATS_DIR = "hparams_cache"
os.makedirs(STATS_DIR, exist_ok=True)

# --- 权重加载与辅助函数 (来自您的模板) ---
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
    with open(index_path, 'r') as f: index = json.load(f)
    file_list = sorted(list(set(index["weight_map"].values())))
    for file in tqdm(file_list, desc=f"Loading weights from {os.path.basename(base_path)}"):
        weights.update(safetensors.torch.load_file(os.path.join(base_path, file)))
    return weights

def normalize_donor_keys(weights: dict) -> dict:
    prefix_to_remove = "language_model."
    return {key[len(prefix_to_remove):] if key.startswith(prefix_to_remove) else key: value for key, value in weights.items()}

def need_merge(name: str) -> bool:
    if any(k in name for k in ['lm_head', 'embed_tokens', 'rotary_emb.inv_freq', 'vision_tower']):
        return False
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
def gram_linear(x): return x @ x.T

def cka(X, Y, kernel=gram_linear):
    X, Y = X.cuda().float(), Y.cuda().float()
    X -= X.mean(dim=0, keepdim=True); Y -= Y.mean(dim=0, keepdim=True)
    K_X, K_Y = kernel(X), kernel(Y)
    hsic = torch.trace(K_X @ K_Y)
    var_X, var_Y = torch.sqrt(torch.trace(K_X @ K_X)), torch.sqrt(torch.trace(K_Y @ K_Y))
    return (hsic / (var_X * var_Y)).item() if (var_X * var_Y) > 1e-9 else 0.0

@torch.no_grad()
def get_activations_for_blocks(model, tokenizer, block_layer_names, probe_dataloader, model_name, device):
    model.eval()
    activations = {name: [] for name in block_layer_names}
    hooks = []
    
    def get_hook(name):
        def hook_fn(module, input, output):
            # Transformer块的输出是元组 (hidden_states, ...)，我们取第一个
            activations[name].append(output[0].detach().cpu())
        return hook_fn

    for name, module in model.named_modules():
        if name in block_layer_names:
            hooks.append(module.register_forward_hook(get_hook(name)))

    for batch in tqdm(probe_dataloader, desc=f"Probing activations for {model_name}"):
        inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
        model(**inputs)

    for hook in hooks: hook.remove()
    for name in block_layer_names:
        if activations[name]: activations[name] = torch.cat(activations[name], dim=0)
    return activations

def compute_covariance_and_projector(model, tok, layer_name, hparams):
    if layer_stats is None: raise ImportError("`rome` library is required for high-divergence strategy.")
    model_name_safe = hparams.base_model_path.replace("/", "_")
    cache_path = os.path.join(STATS_DIR, f"projector_{model_name_safe}_{layer_name.replace('.', '_')}_{hparams.null_space_threshold}.pt")
    if os.path.exists(cache_path) and not hparams.force_recompute:
        print(f"Loading cached projector for {layer_name}.")
        return torch.load(cache_path)
    print(f"\nComputing covariance for {layer_name}...")
    stat = layer_stats(model, tok, layer_name, STATS_DIR, hparams.mom2_dataset, to_collect=["mom2"], sample_size=hparams.mom2_n_samples, precision=hparams.mom2_dtype, force_recompute=hparams.force_recompute)
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
    # ... (输出路径设置)
    output_dir = "merged_models"
    model_name = f"adaptive-merge-{args.mode}"
    OUTPUT_PATH = os.path.join(output_dir, model_name)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # --- 权重加载 ---
    base_weights = load_weights(args.base_model_path, INDEX_FILENAME["qwen2_vl"])
    donor_weights_raw = load_weights(args.donor_model_path, INDEX_FILENAME["llava-onevision-qwen"])
    donor_weights = normalize_donor_keys(donor_weights_raw)
    original_weights = load_weights(args.original_model_path, INDEX_FILENAME["original_model"])
    
    # --- 自适应合并逻辑 ---
    print("="*80); print("Applying 'Activation-Guided Adaptive Merging' strategy."); print("="*80)

    # 1. 准备探针数据和模型
    print(f"Preparing probe dataset from '{args.probe_dataset}'...")
    probe_dataset_raw = load_dataset(args.probe_dataset, "20220301.en" if "wikipedia" in args.probe_dataset else "en", split="train", streaming=True).take(args.probe_samples)
    probe_texts = [item['text'] for item in probe_dataset_raw if item['text']]
    
    print("Loading Base Model (A) to GPU for activation probing...")
    model_a = AutoModelForVision2Seq.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer_a = AutoTokenizer.from_pretrained(args.base_model_path)

    print("\nLoading Donor Model (B) to GPU for activation probing...")
    model_b = LlavaOnevisionForConditionalGeneration.from_pretrained(args.donor_model_path, torch_dtype=torch.bfloat16).to(device)

    # 2. 定义功能块并计算散度
    num_layers = model_a.config.num_hidden_layers
    base_prefix_a = "model.language_model."
    base_prefix_b = "language_model."

    block_names_a, block_names_b = [], []
    for i in range(num_layers):
        block_names_a.extend([f"{base_prefix_a}layers.{i}.self_attn", f"{base_prefix_a}layers.{i}.mlp"])
        block_names_b.extend([f"{base_prefix_b}layers.{i}.self_attn", f"{base_prefix_b}layers.{i}.mlp"])

    probe_inputs = tokenizer_a(probe_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    probe_dataset = TensorDataset(probe_inputs['input_ids'], probe_inputs['attention_mask'])
    probe_dataloader = DataLoader(probe_dataset, batch_size=args.probe_batch_size)

    activations_a = get_activations_for_blocks(model_a, tokenizer_a, block_names_a, probe_dataloader, "Base Model", device)
    activations_b = get_activations_for_blocks(model_b, tokenizer_a, block_names_b, probe_dataloader, "Donor Model", device)

    divergence_scores = {}
    for name_a, name_b in zip(block_names_a, block_names_b):
        if activations_a[name_a] and activations_b[name_b]:
            act_a = activations_a[name_a].view(activations_a[name_a].shape[0], -1)
            act_b = activations_b[name_b].view(activations_b[name_b].shape[0], -1)
            divergence_scores[name_a] = 1 - cka(act_a, act_b)

    del model_a, model_b, activations_a, activations_b; gc.collect(); torch.cuda.empty_cache()
    
    all_divergences = np.array(list(divergence_scores.values()))
    t_low = np.percentile(all_divergences, args.low_div_percentile)
    t_high = np.percentile(all_divergences, args.high_div_percentile)
    print(f"Automated Divergence Thresholds: Low < {t_low:.4f} | High > {t_high:.4f}")

    # 3. 逐层自适应合并
    merged_weights = {}
    tokenizer_for_cov = AutoTokenizer.from_pretrained(args.base_model_path)
    
    for key in tqdm(base_weights.keys(), desc="Applying Adaptive Merging"):
        if not (key in donor_weights and key in original_weights and need_merge(key) and base_weights[key].shape == donor_weights[key].shape):
            merged_weights[key] = base_weights.get(key, original_weights.get(key))
            continue
            
        # 确定当前权重属于哪个功能块
        layer_match = re.search(r'layers\.(\d+)\.(self_attn|mlp)', key)
        if not layer_match:
            # 对于不在Transformer块内的层（如model.norm.weight），使用默认策略
            divergence = (t_low + t_high) / 2
            block_name = None
        else:
            layer_idx, block_type = layer_match.groups()
            block_name = f"{base_prefix_a}layers.{layer_idx}.{block_type}"
            divergence = divergence_scores.get(block_name, (t_low + t_high) / 2)
            
        w_c = original_weights[key].float().to(device)
        w_a = base_weights[key].float().to(device)
        w_b = donor_weights[key].float().to(device)
        
        # 高冲突区的安全策略
        if divergence > t_high and layer_stats:
            print(f"Layer {key} in block {block_name}: High divergence ({divergence:.4f}). Using activation-space null-space grafting.")
            model_a_for_cov = AutoModelForVision2Seq.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16).to(device)
            projector = compute_covariance_and_projector(model_a_for_cov, tokenizer_for_cov, key.rsplit('.', 1)[0], args).to(device)
            delta = w_b - w_a
            w_star = w_a + delta @ projector
            del model_a_for_cov; gc.collect(); torch.cuda.empty_cache()
        else: # 低/中冲突区的参数空间策略
            if divergence < t_low:
                # print(f"Layer {key} in block {block_name}: Low divergence ({divergence:.4f}). Synergistic merge.")
                lambda_s, lambda_c = args.lambda_s_low, args.lambda_c_low
            else:
                # print(f"Layer {key} in block {block_name}: Medium divergence ({divergence:.4f}). Balanced merge.")
                lambda_s, lambda_c = args.lambda_s_mid, args.lambda_c_mid
            
            tau_a, tau_b = w_a - w_c, w_b - w_c
            tau_a_norm_sq = torch.sum(tau_a * tau_a)
            if tau_a_norm_sq > 1e-9:
                proj_scalar = torch.sum(tau_a * tau_b) / tau_a_norm_sq
                tau_b_synergy = torch.clamp(proj_scalar, min=0) * tau_a
                tau_b_conflict = torch.clamp(-proj_scalar, min=0) * tau_a
                tau_b_ortho = tau_b - (tau_b_synergy - tau_b_conflict)
            else:
                tau_b_synergy, tau_b_conflict, tau_b_ortho = torch.zeros_like(tau_b), torch.zeros_like(tau_b), tau_b

            w_star = w_a + (lambda_s * tau_b_synergy) - (lambda_c * tau_b_conflict) + (args.lambda_o * tau_b_ortho)
        
        merged_weights[key] = w_star.to(base_weights[key].dtype).cpu()
        gc.collect(); torch.cuda.empty_cache()

    # --- 保存模型 ---
    # ... (与之前版本相同)
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
    parser = argparse.ArgumentParser(description="Adaptively merge models based on block-level activation divergence.")
    
    # 设备参数
    parser.add_argument('--cuda_device', type=int, default=0, help="CUDA device to use.")
    
    # ... (其他参数与上一版本相同)
    # Model Paths
    parser.add_argument('--base_model_path', type=str, default=CKPT_PATH["qwen2_vl"])
    parser.add_argument('--donor_model_path', type=str, default=CKPT_PATH["llava-onevision-qwen"])
    parser.add_argument('--original_model_path', type=str, default=CKPT_PATH["original_model"])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--mode', type=str, default="default", help="A name for this merging configuration.")

    # Probe Dataset Config
    parser.add_argument('--probe_dataset', type=str, default="wikipedia", help="Dataset for probing activations ('wikipedia' or 'c4').")
    parser.add_argument('--probe_samples', type=int, default=128, help="Number of samples for probing.")
    parser.add_argument('--probe_batch_size', type=int, default=4, help="Batch size for probing. Reduce if OOM.")

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
    
    device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"

    CKPT_PATH.update({
        "qwen2_vl": args.base_model_path,
        "llava-onevision-qwen": args.donor_model_path,
        "original_model": args.original_model_path
    })

    print("--- Configuration ---"); print(vars(args)); print("--------------------")

    convert(args, device)