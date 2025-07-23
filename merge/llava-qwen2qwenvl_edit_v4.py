import os
import sys
import json
import torch
import safetensors.torch
import argparse
from tqdm import tqdm
import gc
from transformers import AutoTokenizer, AutoModelForVision2Seq
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

# --- Dependencies ---
try:
    from datasets import load_dataset
except ImportError:
    print("="*80, file=sys.stderr)
    print("ERROR: Could not import the 'datasets' library. Please run `pip install datasets`.", file=sys.stderr)
    print("This library is essential for fetching the probe dataset.", file=sys.stderr)
    print("="*80, file=sys.stderr)
    sys.exit(1)

# --- CONFIGURATION (MODIFY THESE PATHS) ---
CKPT_PATH = {
    "original_model": "/root/autodl-tmp/AdaMMS/downloaded_models/Qwen2-7B-Instruct",
    "base_model": "/root/autodl-tmp/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct",
    "donor_model": "/root/autodl-tmp/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si-hf"
}
INDEX_FILENAME = "model.safetensors.index.json"

# --- HELPER FUNCTIONS ---

def load_weights(base_path, index_filename):
    """Loads model weights from sharded safetensors or a single file."""
    weights = {}
    index_path = os.path.join(base_path, index_filename)
    if not os.path.exists(index_path):
        single_file_path = os.path.join(base_path, "model.safetensors")
        if os.path.exists(single_file_path):
            return safetensors.torch.load_file(single_file_path)
        else:
            raise FileNotFoundError(f"Neither {index_filename} nor model.safetensors found in {base_path}")
    with open(index_path, 'r') as f:
        index = json.load(f)
    file_list = sorted(list(set(index["weight_map"].values())))
    for file in tqdm(file_list, desc=f"Loading index from {os.path.basename(base_path)}"):
        weights.update(safetensors.torch.load_file(os.path.join(base_path, file)))
    return weights

def normalize_donor_keys(weights: dict) -> dict:
    """Standardizes keys for the LLaVA donor model."""
    prefix_to_remove = "language_model."
    normalized_weights = {}
    for key, value in weights.items():
        if key.startswith(prefix_to_remove):
            normalized_weights[key[len(prefix_to_remove):]] = value
        else:
            normalized_weights[key] = value
    return normalized_weights

def get_mergeable_modules(model):
    """Identifies the modules (e.g., nn.Linear) whose parameters should be merged."""
    mergeable_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "model.layers" in name:
            mergeable_modules.add(name)
    return list(mergeable_modules)

def create_soft_links(source_path, link_path):
    """Creates symbolic links for non-weight files."""
    if not os.path.exists(source_path): return
    os.makedirs(link_path, exist_ok=True)
    for item in os.listdir(source_path):
        source_item = os.path.join(source_path, item)
        link_item = os.path.join(link_path, item)
        if not item.endswith(('.safetensors', '.bin')) and not os.path.exists(link_item):
            os.symlink(os.path.abspath(source_item), link_item)

# --- CORE LOGIC FUNCTIONS ---

@torch.no_grad()
def get_averaged_activations(model_path, probe_texts, mergeable_modules, args, device, cache_inputs=False, cache_outputs=True):
    """
    Loads a model, performs a forward pass, and returns its averaged activations.
    This function carefully manages VRAM by loading and deleting the model within its scope.
    """
    print(f"\nProcessing model from {model_path}...")
    model = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    probe_inputs_tokenized = tokenizer(probe_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    probe_dataset = TensorDataset(probe_inputs_tokenized['input_ids'], probe_inputs_tokenized['attention_mask'])
    probe_dataloader = DataLoader(probe_dataset, batch_size=args.probe_batch_size)

    captured_activations = defaultdict(lambda: defaultdict(list))
    hooks = []

    def get_hook(name, cache_type):
        def hook_fn(module, ins, outs):
            tensor_to_cache = ins[0] if cache_type == 'input' else outs
            if isinstance(tensor_to_cache, tuple): tensor_to_cache = tensor_to_cache[0]
            captured_activations[name][cache_type].append(tensor_to_cache.detach().cpu())
        return hook_fn

    for name, module in model.named_modules():
        if name in mergeable_modules:
            if cache_inputs: hooks.append(module.register_forward_hook(get_hook(name, 'input')))
            if cache_outputs: hooks.append(module.register_forward_hook(get_hook(name, 'output')))
    
    print(f"Registered {len(hooks)} hooks. Running forward pass...")
    for batch in tqdm(probe_dataloader, desc=f"Probing {os.path.basename(model_path)}"):
        input_ids, attention_mask = [b.to(device) for b in batch]
        model(input_ids=input_ids, attention_mask=attention_mask)
    
    for hook in hooks: hook.remove()

    averaged_activations = {}
    print("Averaging captured activations...")
    for module_name, data in captured_activations.items():
        averaged_activations[module_name] = {}
        if 'input' in data: averaged_activations[module_name]['input'] = torch.cat(data['input'], dim=0).mean(dim=0)
        if 'output' in data: averaged_activations[module_name]['output'] = torch.cat(data['output'], dim=0).mean(dim=0)
    
    # Critical VRAM cleanup
    del model, tokenizer, hooks, captured_activations
    gc.collect()
    torch.cuda.empty_cache()
    
    return averaged_activations

def calculate_approx_gradients(activations_a, activations_c, base_weights, device):
    """
    Uses the in-memory activations to calculate a 'local approximate gradient' for each layer.
    This function has very low VRAM usage.
    """
    approx_grads = {}
    mergeable_keys = [k for k in base_weights.keys() if "model.layers" in k and k.endswith(".weight") and "norm" not in k]
    
    for key in tqdm(mergeable_keys, desc="Step 2/3: Calculating Approx Gradients"):
        module_name = key.rsplit('.', 1)[0]
        
        if module_name not in activations_a or 'input' not in activations_a[module_name] or 'output' not in activations_a[module_name] or 'output' not in activations_c.get(module_name, {}):
            continue

        input_a = activations_a[module_name]['input'].to(device).float()
        output_a = activations_a[module_name]['output'].to(device).float()
        output_c = activations_c[module_name]['output'].to(device).float()

        delta_y = output_a - output_c
        
        if input_a.dim() > 2: input_a = input_a.view(-1, input_a.shape[-1])
        if delta_y.dim() > 2: delta_y = delta_y.view(-1, delta_y.shape[-1])

        approx_grad = torch.matmul(delta_y.T, input_a)
        
        weight_shape = base_weights[key].shape
        if approx_grad.shape != weight_shape:
            if approx_grad.T.shape == weight_shape: approx_grad = approx_grad.T
            else: continue

        approx_grads[key] = approx_grad.cpu()
        
        del input_a, output_a, output_c, delta_y, approx_grad
        gc.collect()
        torch.cuda.empty_cache()
        
    return approx_grads

def merge_weights_with_grads(base_weights, donor_weights, original_weights, approx_grads, args, device):
    """
    Loads weights and approximate gradients layer by layer to perform the final merge.
    This function also has very low VRAM usage.
    """
    merged_weights = {}
    
    for key in tqdm(base_weights.keys(), desc="Step 3/3: Merging Tensors"):
        if key in approx_grads and key in donor_weights and key in original_weights:
            g_a = approx_grads[key].float().to(device)
            w_c = original_weights[key].float().to(device)
            w_a = base_weights[key].float().to(device)
            w_b = donor_weights[key].float().to(device)
            
            tau_b = w_b - w_c
            g_a_norm_sq = torch.sum(g_a * g_a)

            if g_a_norm_sq > 1e-9:
                dot_product = torch.sum(g_a * tau_b)
                proj_scalar_synergy = torch.clamp(-dot_product / g_a_norm_sq, min=0)
                tau_b_synergy = proj_scalar_synergy * (-g_a)
                
                proj_scalar_conflict = torch.clamp(dot_product / g_a_norm_sq, min=0)
                tau_b_conflict = proj_scalar_conflict * g_a
                
                tau_b_ortho = tau_b - (tau_b_synergy - tau_b_conflict)
            else:
                tau_b_synergy, tau_b_conflict, tau_b_ortho = torch.zeros_like(tau_b), torch.zeros_like(tau_b), tau_b
            
            w_star = w_a + (args.lambda_s * tau_b_synergy) - (args.lambda_c * tau_b_conflict) + (args.lambda_o * tau_b_ortho)
            merged_weights[key] = w_star.to(base_weights[key].dtype).cpu()
            
            del g_a, w_c, w_a, w_b, tau_b, w_star
            gc.collect()
            torch.cuda.empty_cache()
        else:
            merged_weights[key] = base_weights[key].clone()
            
    return merged_weights


def convert(args, device):
    """
    Orchestrates the entire unified, low-memory merging process.
    """
    # --- Step 0: Initial Setup ---
    output_path = os.path.join(args.output_dir, f"unified-low-mem-merge-{args.mode}")
    os.makedirs(output_path, exist_ok=True)
    
    print("Loading probe dataset...")
    # 修改数据集加载方式，使用更可靠的wikitext数据集
    try:
        probe_dataset_raw = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True).take(args.probe_samples)
        probe_texts = [item['text'] for item in probe_dataset_raw if item['text'] and len(item['text']) > 50]
        print(f"Successfully loaded wikitext dataset with {args.probe_samples} samples")
    except Exception as e:
        print(f"Error loading wikitext: {e}")
        # 备用方案：创建一些简单的探测文本
        print("Using fallback text samples...")
        probe_texts = [
            "The quick brown fox jumps over the lazy dog. This sentence contains all letters in the English alphabet.",
            "Machine learning models are trained on large datasets to recognize patterns and make predictions.",
            "Neural networks consist of layers of nodes that process information in a hierarchical manner."
        ] * (args.probe_samples // 3 + 1)
        probe_texts = probe_texts[:args.probe_samples]
    
    # Temporarily load a model to identify mergeable modules
    temp_model = AutoModelForVision2Seq.from_pretrained(args.base_model_path)
    mergeable_modules = get_mergeable_modules(temp_model)
    del temp_model
    gc.collect()
    print(f"Identified {len(mergeable_modules)} mergeable modules (e.g., Linear layers).")
    
    # --- Step 1: Cache Activations (In-Memory) ---
    print("\n--- Starting Step 1/3: Caching Activations ---")
    activations_a = get_averaged_activations(args.base_model_path, probe_texts, mergeable_modules, args, device, cache_inputs=True, cache_outputs=True)
    activations_c = get_averaged_activations(args.original_model_path, probe_texts, mergeable_modules, args, device, cache_inputs=False, cache_outputs=True)
    print("\n✅ Activation Caching Complete.")

    # --- Step 2: Calculate Approximate Gradients (In-Memory) ---
    print("\n--- Starting Step 2/3: Calculating Approximate Gradients ---")
    # Load base weights just for shape info
    base_weights = load_weights(args.base_model_path, INDEX_FILENAME)
    approx_grads = calculate_approx_gradients(activations_a, activations_c, base_weights, device)
    del activations_a, activations_c # Free up CPU memory
    gc.collect()
    print("\n✅ Approximate Gradient Calculation Complete.")
    
    # --- Step 3: Merge Weights ---
    print("\n--- Starting Step 3/3: Merging All Weights ---")
    print("Loading all model weights for the final merge...")
    # Base weights already loaded, load others
    original_weights = load_weights(args.original_model_path, INDEX_FILENAME)
    donor_weights_raw = load_weights(args.donor_model_path, INDEX_FILENAME)
    donor_weights = normalize_donor_keys(donor_weights_raw)
    
    merged_weights = merge_weights_with_grads(base_weights, donor_weights, original_weights, approx_grads, args, device)
    print("\n✅ Weight Merging Complete.")
    
    # --- Step 4: Save Final Model ---
    print("\nSaving merged model...")
    try:
        index_path = os.path.join(args.base_model_path, INDEX_FILENAME)
        with open(index_path, "r") as f: index_map = json.load(f)["weight_map"]
        sharded_weights = {filename: {} for filename in set(index_map.values())}
        for key, value in merged_weights.items():
            if key in index_map: sharded_weights[index_map[key]][key] = value
        for filename, weights_dict in sharded_weights.items():
            safetensors.torch.save_file(weights_dict, os.path.join(output_path, filename))
    except FileNotFoundError:
        safetensors.torch.save_file(merged_weights, os.path.join(output_path, "model.safetensors"))

    create_soft_links(source_path=args.base_model_path, link_path=output_path)
    print(f"\n🎉🎉🎉 Process complete! Merged model saved to: {output_path}")

# --- Main Execution Block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified, low-memory model merging using local gradient approximation.")
    
    # Paths
    parser.add_argument('--base_model_path', type=str, default=CKPT_PATH["base_model"])
    parser.add_argument('--donor_model_path', type=str, default=CKPT_PATH["donor_model"])
    parser.add_argument('--original_model_path', type=str, default=CKPT_PATH["original_model"])
    parser.add_argument('--output_dir', type=str, default="merged_models")
    parser.add_argument('--mode', type=str, default="default", help="A name for this merging configuration.")

    # Probe Dataset Config
    parser.add_argument('--probe_dataset', type=str, default="wikipedia")
    parser.add_argument('--probe_samples', type=int, default=256)
    parser.add_argument('--probe_batch_size', type=int, default=1, help="Reduce if OOM during activation caching.")

    # Merging Coefficients
    parser.add_argument('--lambda_s', type=float, default=1.0, help="Synergy component coefficient.")
    parser.add_argument('--lambda_c', type=float, default=0.0, help="Conflict component coefficient (0.0 means remove conflict).")
    parser.add_argument('--lambda_o', type=float, default=1.0, help="Orthogonal component coefficient.")
    
    # System
    parser.add_argument('--cuda_device', type=int, default=0)

    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = f"cuda:{args.cuda_device}"
    else:
        device = "cpu"

    print("\n--- Configuration ---")
    print(json.dumps(vars(args), indent=2))
    print(f"Using device: {device}")
    print("---------------------\n")
    
    convert(args, device)