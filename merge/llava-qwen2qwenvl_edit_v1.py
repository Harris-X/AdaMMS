# llava_merging_with_task_vectors.py

import os
import sys
import json
import torch
import safetensors.torch
import argparse
from tqdm import tqdm
import gc

# --- Model & Path Configuration (Please update with your paths) ---
CKPT_PATH = {
    "original_model": "/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-7B-Instruct",  # The original pretrained model M_C
    "qwen2_vl": "/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct",      # Base model M_A
    "llava-onevision-qwen": "/home/user/xieqiuhao/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si" # Donor model M_B
}

INDEX_FILENAME = {
    "original_model": "model.safetensors.index.json",
    "qwen2_vl": "model.safetensors.index.json",
    "llava-onevision-qwen": "model.safetensors.index.json"
}

# --- Weight Loading Functions (from your template) ---
def load_weights(base_path, index_filename):
    weights = {}
    index_path = os.path.join(base_path, index_filename)
    if not os.path.exists(index_path):
        single_file_path = os.path.join(base_path, "model.safetensors")
        if os.path.exists(single_file_path):
            print(f"Loading single weight file from {single_file_path}")
            return safetensors.torch.load_file(single_file_path)
        else:
            raise FileNotFoundError(f"Neither {index_filename} nor model.safetensors found in {base_path}")
            
    with open(index_path, 'r') as f:
        index = json.load(f)
    file_list = sorted(list(set(index["weight_map"].values())))
    
    for file in tqdm(file_list, desc=f"Loading weights from {os.path.basename(base_path)}"):
        path = os.path.join(base_path, file)
        weights.update(safetensors.torch.load_file(path))
    return weights

# --- Helper Functions (from your template) ---
def need_merge(name: str) -> bool:
    # We merge all parameters except the embeddings and final head for stability
    if name in ['lm_head.weight', 'model.embed_tokens.weight']:
        return False
    if "rotary_emb.inv_freq" in name:
        return False
    return True

def create_soft_link(source_path, link_path):
    print(f"Creating symbolic links from {source_path} to {link_path} for non-weight files...")
    for item in os.listdir(source_path):
        source_item = os.path.join(source_path, item)
        link_item = os.path.join(link_path, item)
        if item.endswith(('.safetensors', '.bin', '.py', '.md')):
             continue
        if os.path.exists(link_item):
            continue
        try:
            os.symlink(source_item, link_item)
        except OSError as e:
            print(f"Error creating soft link for '{item}': {e}", file=sys.stderr)

# --- Main Conversion and Merging Logic ---
def convert(args):
    # --- Output Path Setup ---
    output_dir = "merged_models"
    if args.output is not None:
        model_name = os.path.basename(args.output)
        output_dir = os.path.dirname(args.output)
    else:
        strategy_name = args.strategy
        if strategy_name == "task_vector_grafting":
            model_name = f"grafted-s{args.lambda_s}-c{args.lambda_c}"
        else: # Default to interpolation
            model_name = f"interpolated-a{args.alpha}"
    
    OUTPUT_PATH = os.path.join(output_dir, model_name)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"Merging output path: {OUTPUT_PATH}")

    # --- Model Loading ---
    print("Loading Base Model (M_A)...")
    base_weights = load_weights(args.base_model_path, INDEX_FILENAME["qwen2_vl"])
    
    print("Loading Donor Model (M_B)...")
    donor_weights = load_weights(args.donor_model_path, INDEX_FILENAME["llava-onevision-qwen"])
    
    merged_weights = {}

    # --- Strategy Selection ---
    if args.strategy == "task_vector_grafting":
        print("="*80)
        print("Applying 'Task Vector Grafting' strategy.")
        print(f"Synergy coefficient (lambda_s): {args.lambda_s}, Conflict coefficient (lambda_c): {args.lambda_c}")
        print("="*80)

        print("Loading Original Pretrained Model (M_C)...")
        original_weights = load_weights(args.original_model_path, INDEX_FILENAME["original_model"])

        for key in tqdm(original_weights.keys(), desc="Applying Task Vector Grafting"):
            if key not in base_weights or key not in donor_weights:
                merged_weights[key] = original_weights[key]
                continue

            if need_merge(key) and base_weights[key].shape == donor_weights[key].shape:
                w_c = original_weights[key].float().cuda()
                w_a = base_weights[key].float().cuda()
                w_b = donor_weights[key].float().cuda()

                # 1. Calculate Task Vectors
                tau_a = w_a - w_c
                tau_b = w_b - w_c

                # 2. Decompose tau_b into synergistic, conflicting, and orthogonal components
                # Frobenius inner product and norm squared
                tau_a_norm_sq = torch.sum(tau_a * tau_a)
                inner_product = torch.sum(tau_a * tau_b)

                if tau_a_norm_sq > 1e-9: # Avoid division by zero
                    # Calculate the scalar projection coefficient
                    proj_scalar = inner_product / tau_a_norm_sq
                    
                    # Decompose the parallel component
                    tau_b_synergy = torch.clamp(proj_scalar, min=0) * tau_a
                    tau_b_conflict = torch.clamp(-proj_scalar, min=0) * tau_a
                    
                    # Calculate the orthogonal component
                    tau_b_ortho = tau_b - (tau_b_synergy - tau_b_conflict)
                else: # If tau_a is a zero vector, all of tau_b is orthogonal
                    tau_b_synergy = torch.zeros_like(tau_b)
                    tau_b_conflict = torch.zeros_like(tau_b)
                    tau_b_ortho = tau_b

                # 3. Apply the controlled merging formula
                # W* = W_A + (λ_s-1)*τ_B_syn + (1-λ_c)*(-τ_B_conf) + τ_B_ortho
                # Note: W_A already contains τ_A. We start with W_A and add modifications.
                
                # We start with the base model's weights
                w_star = w_a.clone()
                # Add scaled synergistic component
                w_star += (args.lambda_s - 1.0) * tau_b_synergy
                # Add scaled (and direction-corrected) conflicting component
                w_star += (1.0 - args.lambda_c) * (-tau_b_conflict)
                # Add the orthogonal component
                w_star += tau_b_ortho
                
                merged_weights[key] = w_star.to(original_weights[key].dtype).cpu()
            else:
                # For layers not being merged, we keep the base model's weights
                merged_weights[key] = base_weights[key]
            
            # Clean up GPU memory
            gc.collect()
            torch.cuda.empty_cache()

    else: # Default to linear interpolation
        print("="*80)
        print(f"Applying linear interpolation with alpha = {args.alpha}")
        print("="*80)
        for key in tqdm(base_weights.keys(), desc="Applying Linear Interpolation"):
            if key in donor_weights and need_merge(key) and base_weights[key].shape == donor_weights[key].shape:
                 merged_weights[key] = (1 - args.alpha) * base_weights[key] + args.alpha * donor_weights[key]
            else:
                 merged_weights[key] = base_weights[key]

    # --- Saving the Merged Model ---
    print("\nSaving merged model...")
    index_path = os.path.join(args.base_model_path, INDEX_FILENAME["qwen2_vl"])
    with open(index_path, "r") as f:
        index_map = json.load(f)["weight_map"]
    
    sharded_weights = {}
    for filename in set(index_map.values()):
        sharded_weights[filename] = {}
    
    for key, value in merged_weights.items():
        if key in index_map:
            sharded_weights[index_map[key]][key] = value
        else:
            print(f"Warning: key '{key}' not in weight map, will not be saved.", file=sys.stderr)
            
    for filename, weights_dict in sharded_weights.items():
        save_path = os.path.join(OUTPUT_PATH, filename)
        safetensors.torch.save_file(weights_dict, save_path)
    
    create_soft_link(source_path=args.base_model_path, link_path=OUTPUT_PATH)
    with open(index_path, "r") as f_in, \
         open(os.path.join(OUTPUT_PATH, os.path.basename(index_path)), "w") as f_out:
        f_out.write(f_in.read())

    print("Convert Done.")
    print(f"Merged model and associated files saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two models using advanced task vector projection.")
    
    parser.add_argument('--strategy', type=str, default="task_vector_grafting", 
                        choices=['interpolation', 'task_vector_grafting'], 
                        help="Merging strategy to use.")
    
    # Model paths
    parser.add_argument('--base_model_path', type=str, default=CKPT_PATH["qwen2_vl"], 
                        help="Path to the base model (M_A).")
    parser.add_argument('--donor_model_path', type=str, default=CKPT_PATH["llava-onevision-qwen"], 
                        help="Path to the donor model (M_B).")
    parser.add_argument('--original_model_path', type=str, default=CKPT_PATH["original_model"], 
                        help="Path to the original pre-trained model (M_C). Required for 'task_vector_grafting'.")

    # Strategy-specific parameters
    parser.add_argument('--alpha', type=float, default=0.5, help="Coefficient for 'interpolation' strategy.")
    parser.add_argument('--lambda_s', type=float, default=1.0, help="Synergy coefficient for 'task_vector_grafting'.")
    parser.add_argument('--lambda_c', type=float, default=1.0, help="Conflict mitigation coefficient for 'task_vector_grafting'.")

    parser.add_argument('--output', type=str, default=None, help="Output directory and name for the merged model.")
    
    args = parser.parse_args()
    
    # Update paths from args
    CKPT_PATH['qwen2_vl'] = args.base_model_path
    CKPT_PATH['llava-onevision-qwen'] = args.donor_model_path
    CKPT_PATH['original_model'] = args.original_model_path

    print("--- Configuration ---")
    print(f"Strategy: {args.strategy}")
    print(f"Base Model (M_A): {args.base_model_path}")
    print(f"Donor Model (M_B): {args.donor_model_path}")
    if args.strategy == 'task_vector_grafting':
        print(f"Original Model (M_C): {args.original_model_path}")
        print(f"Synergy Coefficient (λ_s): {args.lambda_s}")
        print(f"Conflict Coefficient (λ_c): {args.lambda_c}")
    else:
        print(f"Interpolation Alpha: {args.alpha}")
    print("--------------------")

    convert(args)