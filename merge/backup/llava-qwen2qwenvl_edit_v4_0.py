# llava_merging_gradient_guided.py

import os
import sys
import json
import torch
import safetensors.torch
import argparse
from tqdm import tqdm
import gc
import shutil
from transformers import AutoTokenizer, AutoModelForVision2Seq
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# --- Dependencies ---
# Ensure you have the 'datasets' library installed: pip install datasets
try:
    from datasets import load_dataset
except ImportError:
    print("="*80, file=sys.stderr)
    print("ERROR: Could not import the 'datasets' library. Please run `pip install datasets`.", file=sys.stderr)
    print("This library is essential for fetching the probe dataset.", file=sys.stderr)
    print("="*80, file=sys.stderr)
    sys.exit(1)

# --- Model & Path Configuration (Modify Before Use) ---
CKPT_PATH = {
    "original_model": "/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-7B-Instruct",
    "base_model": "/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct",
    "donor_model": "/home/user/xieqiuhao/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si-hf"
}
INDEX_FILENAME = "model.safetensors.index.json"

# --- Weight Loading & Helper Functions ---

def load_weights(base_path, index_filename):
    """Loads model weights from sharded safetensors or a single file."""
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
    """Standardizes keys for the LLaVA donor model by removing the 'language_model.' prefix."""
    prefix_to_remove = "language_model."
    normalized_weights = {}
    for key, value in weights.items():
        if key.startswith(prefix_to_remove):
            normalized_weights[key[len(prefix_to_remove):]] = value
        else:
            # Keep non-language model parts (e.g., vision tower) as is.
            normalized_weights[key] = value
    return normalized_weights

def get_mergeable_keys(weights_dict: dict) -> set:
    """Identifies the keys for layers that should be merged."""
    excluded_patterns = [
        'model.norm.weight', 
        'lm_head.weight', 
        'model.embed_tokens.weight',
        '.rotary_emb.inv_freq'
    ]
    mergeable_keys = set()
    for key in weights_dict.keys():
        if key.startswith("model.layers.") and not any(pattern in key for pattern in excluded_patterns):
            mergeable_keys.add(key)
    return mergeable_keys

def create_soft_links(source_path, link_path):
    """Creates symbolic links for non-weight files from the base model directory to the output directory."""
    if not os.path.exists(source_path):
        print(f"ERROR: Source path for linking '{source_path}' does not exist.", file=sys.stderr)
        return
    os.makedirs(link_path, exist_ok=True)
    
    for item in os.listdir(source_path):
        source_item = os.path.join(source_path, item)
        link_item = os.path.join(link_path, item)
        # Exclude weight files, link everything else (configs, tokenizer files, etc.)
        if not item.endswith(('.safetensors', '.bin')) and not os.path.exists(link_item):
            try:
                os.symlink(os.path.abspath(source_item), link_item)
                print(f"Linked '{link_item}' -> '{source_item}'")
            except OSError as e:
                print(f"Error creating soft link for '{item}': {e}", file=sys.stderr)

# --- Core Gradient-Guided Merging Logic ---

@torch.enable_grad()
def get_gradients(model, tokenizer, target_keys, probe_dataloader, model_name, device):
    """
    Computes gradients of the model on a probe dataset.
    This is the core of the "Gradient-Guided" method, capturing the model's
    "intent" on general-purpose data.
    """
    model.train()
    model.zero_grad()
    
    # Store accumulated gradients on CPU to save VRAM
    accumulated_grads = {name: torch.zeros_like(p, device='cpu') for name, p in model.named_parameters() if name in target_keys}
    
    total_loss = 0.0
    num_samples = 0

    print(f"Calculating 'generalization-preserving' gradients for {model_name}...")
    for batch in tqdm(probe_dataloader, desc=f"Probing Gradients for {model_name}"):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        
        # Causal LM loss: labels are the input_ids, with padding ignored
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        # Forward pass to get loss
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels,
            # We don't provide visual inputs for the generalization gradient
            pixel_values=None 
        )
        loss = outputs.loss
        
        # Backward pass to compute gradients for this batch
        loss.backward()
        
        total_loss += loss.item() * len(input_ids)
        num_samples += len(input_ids)

        # Accumulate gradients on CPU
        for name, param in model.named_parameters():
            if name in target_keys and param.grad is not None:
                accumulated_grads[name] += param.grad.detach().cpu()
        
        # Reset gradients for the next batch
        model.zero_grad()
        
    # Average the gradients over all samples
    for name in accumulated_grads:
        if num_samples > 0:
            accumulated_grads[name] /= num_samples
            
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    print(f"Average probe loss for {model_name}: {avg_loss:.4f}")

    return accumulated_grads

def convert(args, device):
    """
    Main function to perform gradient-guided model merging.
    """
    # --- Output Path Setup ---
    output_dir = args.output if args.output else "merged_models"
    model_name = f"gradient-guided-merge-{args.mode}"
    OUTPUT_PATH = os.path.join(output_dir, model_name)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # --- Weight Loading ---
    print("Loading all model weights from disk...")
    base_weights = load_weights(args.base_model_path, INDEX_FILENAME)
    original_weights = load_weights(args.original_model_path, INDEX_FILENAME)
    donor_weights_raw = load_weights(args.donor_model_path, INDEX_FILENAME)
    print("Normalizing donor model keys...")
    donor_weights = normalize_donor_keys(donor_weights_raw)
    
    merged_weights = {}

    # --- Gradient-Guided Merging ---
    print("\n" + "="*80)
    print("Applying 'Gradient-Guided Task Vector Decomposition & Merging'")
    print("="*80 + "\n")
    
    # 1. Prepare Probe Dataset
    print(f"Preparing probe dataset from '{args.probe_dataset}'...")
    # Use streaming to handle large datasets and take a small sample
    probe_dataset_raw = load_dataset(args.probe_dataset, "20220301.en" if "wikipedia" in args.probe_dataset else "en", split="train", streaming=True).take(args.probe_samples)
    probe_texts = [item['text'] for item in probe_dataset_raw if item['text'] and len(item['text']) > 50]
    print(f"Using {len(probe_texts)} samples for probing.")

    # 2. Identify Keys to Merge
    mergeable_keys = get_mergeable_keys(base_weights)
    print(f"Identified {len(mergeable_keys)} mergeable parameter tensors.")

    # 3. Calculate "Generalization-Preserving" Gradient (g_A) for the Base Model
    print("\nLoading Base Model (M_A) to GPU for gradient probing...")
    model_a = AutoModelForVision2Seq.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer_a = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer_a.pad_token is None:
        tokenizer_a.pad_token = tokenizer_a.eos_token

    probe_inputs = tokenizer_a(probe_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    probe_dataset = TensorDataset(probe_inputs['input_ids'], probe_inputs['attention_mask'])
    probe_dataloader = DataLoader(probe_dataset, batch_size=args.probe_batch_size)
    
    # Get the gradients for all mergeable keys
    gradients_a = get_gradients(model_a, tokenizer_a, mergeable_keys, probe_dataloader, "Base Model (M_A)", device)
    
    del model_a, probe_inputs, probe_dataset, probe_dataloader; gc.collect(); torch.cuda.empty_cache()

    # 4. Perform Gradient-Guided Decomposition and Merging
    print("\nDecomposing task vectors and merging layers...")
    for key in tqdm(base_weights.keys(), desc="Merging Tensors"):
        # Check if the key is mergeable and exists in all models
        if key in mergeable_keys and key in donor_weights and key in original_weights:
            g_a = gradients_a.get(key)
            if g_a is None or torch.all(g_a == 0):
                print(f"Warning: Zero gradient for {key}. Copying base weight.")
                merged_weights[key] = base_weights[key].clone()
                continue

            # Move tensors to device for computation
            w_c = original_weights[key].float().to(device)
            w_a = base_weights[key].float().to(device)
            w_b = donor_weights[key].float().to(device)
            g_a = g_a.float().to(device)
            
            # Calculate task vector for the donor model
            tau_b = w_b - w_c
            
            # Decompose tau_b based on the gradient g_a
            g_a_norm_sq = torch.sum(g_a * g_a)
            
            if g_a_norm_sq > 1e-9:
                # Project tau_b onto the gradient direction
                dot_product = torch.sum(g_a * tau_b)
                
                # Synergy component: Aligns with the *negative* gradient (beneficial direction)
                proj_scalar_synergy = torch.clamp(-dot_product / g_a_norm_sq, min=0)
                tau_b_synergy = proj_scalar_synergy * (-g_a)
                
                # Conflict component: Aligns with the *positive* gradient (detrimental direction)
                proj_scalar_conflict = torch.clamp(dot_product / g_a_norm_sq, min=0)
                tau_b_conflict = proj_scalar_conflict * g_a
                
                # Orthogonal component: The remainder
                tau_b_ortho = tau_b - (tau_b_synergy - tau_b_conflict)
            else:
                # If gradient is zero, there's no defined direction. Treat all of tau_b as orthogonal.
                tau_b_synergy = torch.zeros_like(tau_b)
                tau_b_conflict = torch.zeros_like(tau_b)
                tau_b_ortho = tau_b
            
            # Final Merge Formula from the document
            w_star = w_a + (args.lambda_s * tau_b_synergy) - (args.lambda_c * tau_b_conflict) + (args.lambda_o * tau_b_ortho)
            
            merged_weights[key] = w_star.to(base_weights[key].dtype).cpu()
            
            # Clean up GPU memory per layer
            del w_c, w_a, w_b, g_a, tau_b, tau_b_synergy, tau_b_conflict, tau_b_ortho, w_star
            gc.collect(); torch.cuda.empty_cache()
            
        else:
            # For non-merged or missing keys, copy from the base model
            merged_weights[key] = base_weights[key].clone()

    # --- Save Merged Model ---
    print("\nSaving merged model...")
    # Find the original index file to map weights back to sharded files
    index_path = os.path.join(args.base_model_path, INDEX_FILENAME)
    if not os.path.exists(index_path):
        # Handle single-file model case
        print(f"Saving merged model to a single file: {os.path.join(OUTPUT_PATH, 'model.safetensors')}")
        safetensors.torch.save_file(merged_weights, os.path.join(OUTPUT_PATH, "model.safetensors"))
    else:
        # Handle sharded model case
        with open(index_path, "r") as f:
            index_map = json.load(f)["weight_map"]
        
        sharded_weights = {filename: {} for filename in set(index_map.values())}
        for key, value in merged_weights.items():
            if key in index_map:
                sharded_weights[index_map[key]][key] = value
        
        for filename, weights_dict in sharded_weights.items():
            safetensors.torch.save_file(weights_dict, os.path.join(OUTPUT_PATH, filename))
    
    # Copy/link all other necessary files (config.json, tokenizer.json, etc.)
    create_soft_links(source_path=args.base_model_path, link_path=OUTPUT_PATH)
    print(f"\n✅ Merged model saved successfully to: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge models using Gradient-Guided Task Vector Decomposition.")
    
    # --- Paths ---
    parser.add_argument('--base_model_path', type=str, default=CKPT_PATH["base_model"], help="Path to the base model (M_A).")
    parser.add_argument('--donor_model_path', type=str, default=CKPT_PATH["donor_model"], help="Path to the donor model (M_B).")
    parser.add_argument('--original_model_path', type=str, default=CKPT_PATH["original_model"], help="Path to the common ancestor model (M_C).")
    parser.add_argument('--output', type=str, default=None, help="Custom output directory for the merged model.")
    parser.add_argument('--mode', type=str, default="default", help="A name for this merging configuration, used in the output folder name.")

    # --- Probe Dataset Config ---
    parser.add_argument('--probe_dataset', type=str, default="wikipedia", help="Dataset for probing gradients ('wikipedia' or 'c4').")
    parser.add_argument('--probe_samples', type=int, default=128, help="Number of samples for probing gradients.")
    parser.add_argument('--probe_batch_size', type=int, default=1, help="Batch size for gradient probing. Reduce if you encounter OOM errors.")

    # --- Gradient-Guided Merging Coefficients (λ) ---
    parser.add_argument('--lambda_s', type=float, default=1.0, help="Scaling coefficient for the synergy component (τ_B_synergy). >1.0 amplifies, <1.0 dampens.")
    parser.add_argument('--lambda_c', type=float, default=0.0, help="Scaling coefficient for the conflict component (-τ_B_conflict). 0.0 removes conflict, 1.0 cancels it out.")
    parser.add_argument('--lambda_o', type=float, default=1.0, help="Scaling coefficient for the orthogonal component (τ_B_ortho). Usually 1.0 to retain all new, non-interfering knowledge.")
    
    # --- System ---
    parser.add_argument('--cuda_device', type=int, default=2, help="CUDA device to use (e.g., 0, 1, 2).")

    args = parser.parse_args()
    
    if torch.cuda.is_available() and args.cuda_device < torch.cuda.device_count():
        device = f"cuda:{args.cuda_device}"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Update global paths from arguments for clarity
    CKPT_PATH.update({
        "base_model": args.base_model_path,
        "donor_model": args.donor_model_path,
        "original_model": args.original_model_path
    })

    print("\n--- Configuration ---")
    print(json.dumps(vars(args), indent=2))
    print("---------------------\n")

    convert(args, device)