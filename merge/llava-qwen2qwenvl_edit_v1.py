# llava_merging_with_task_vectors.py

import os
import sys
import json
import torch
import safetensors.torch
import argparse
from tqdm import tqdm
import gc
import shutil

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

# --- Weight Loading Functions (from llava-qwen2qwenvl.py) ---
def load_safetensors_weights(base_path, file_list):
    weights = {}
    for file in tqdm(file_list, desc=f"Loading weights from {os.path.basename(base_path)}"):
        path = os.path.join(base_path, file)
        weights.update(safetensors.torch.load_file(path))
    return weights

def load_pytorch_weights(base_path, file_list):
    weights = {}
    for file in file_list:
        path = os.path.join(base_path, file)
        x = torch.load(path)
        weights.update(x)
    return weights

# # Define hardcoded file lists
# qwenvl_file_list = [f'model-0000{i}-of-00005.safetensors' for i in range(1, 6)]
# llava_onevision_qwen_file_list = [f'model-0000{i}-of-00004.safetensors' for i in range(1, 5)]




vicuna_file_list = ['pytorch_model-00001-of-00002.bin', 'pytorch_model-00002-of-00002.bin']
llama_file_list = ['pytorch_model-00001-of-00003.bin', 'pytorch_model-00002-of-00003.bin', 'pytorch_model-00003-of-00003.bin']
def load_llama_weights(base_path, file_list=llama_file_list):
    return load_pytorch_weights(base_path, file_list)

llava_file_list = ['pytorch_model-00001-of-00002.bin', 'pytorch_model-00002-of-00002.bin']
def load_llava_weights(base_path, file_list=llava_file_list):
    return load_pytorch_weights(base_path, file_list)

mplug_owl_file_list_template = "pytorch_model-{}-of-33.bin"
mplug_owl_file_list = [mplug_owl_file_list_template.format(str(i+1)) for i in range(33)]
def load_mplug_owl_weights(base_path, file_list=mplug_owl_file_list):
    return load_pytorch_weights(base_path, file_list)

cogvlm_file_list = ['model-00001-of-00008.safetensors', 'model-00002-of-00008.safetensors', 'model-00003-of-00008.safetensors', 'model-00004-of-00008.safetensors', 'model-00005-of-00008.safetensors', 'model-00006-of-00008.safetensors', 'model-00007-of-00008.safetensors', 'model-00008-of-00008.safetensors']
def load_cogvlm_weights(base_path, file_list=cogvlm_file_list):
    return load_safetensors_weights(base_path, file_list)

qwenvl_file_list = ['model-00001-of-00005.safetensors', 'model-00002-of-00005.safetensors', 'model-00003-of-00005.safetensors', 'model-00004-of-00005.safetensors', 'model-00005-of-00005.safetensors']
def load_qwenvl_weights(base_path, file_list=qwenvl_file_list):
    return load_safetensors_weights(base_path, file_list)

llava_onevision_qwen_file_list = ['model-00001-of-00004.safetensors', 'model-00002-of-00004.safetensors', 'model-00003-of-00004.safetensors', 'model-00004-of-00004.safetensors']
def load_llava_onevision_weights(base_path, file_list=llava_onevision_qwen_file_list):
    return load_safetensors_weights(base_path, file_list)

qwen2_7b_file_list = [f'model-0000{i}-of-00004.safetensors' for i in range(1, 4)]
def load_original_weights(base_path, file_list=qwen2_7b_file_list):
    return load_safetensors_weights(base_path, file_list)

# --- Helper Functions ---
def need_merge(name:str) -> bool:
    if name in ['model.norm.weight']:
        return True
    if name in ['lm_head.weight', 'model.embed_tokens.weight']:
        return False
    if name.startswith("model.layers."):
        if name.endswith(".self_attn.rotary_emb.inv_freq"):
            return False
        return True
    return False

def create_soft_link(source_path, link_path):
    # Check if source path exists
    if not os.path.exists(source_path):
        print(f"Error: Source path '{source_path}' does not exist.")
        return

    # Check if link path exists, if not create it
    if not os.path.exists(link_path):
        os.makedirs(link_path)
        print(f"Created directory '{link_path}'")

    # Iterate through all files and directories in the source path
    for item in os.listdir(source_path):
        source_item = os.path.join(source_path, item)
        link_item = os.path.join(link_path, item)

        # Skip files that end with '.bin'
        if item.endswith('.bin'):
            print(f"Skipping '{item}' as it ends with '.bin'")
            continue

        # If it's a file, create a symbolic link
        if os.path.isfile(source_item):
            try:
                os.symlink(source_item, link_item)
                print(f"Created soft link '{link_item}' -> '{source_item}'")
            except OSError as e:
                print(f"Error creating soft link for '{item}': {e}")

        # If it's a directory, ignore it
        elif os.path.isdir(source_item):
            continue


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
    base_weights = load_qwenvl_weights(args.base_model_path)
    
    print("Loading Donor Model (M_B)...")
    donor_weights = load_llava_onevision_weights(args.donor_model_path)
    
    merged_weights = {}

    # --- Strategy Selection ---
    if args.strategy == "task_vector_grafting":
        print("="*80)
        print("Applying 'Task Vector Grafting' strategy.")
        print(f"Synergy coefficient (lambda_s): {args.lambda_s}, Conflict coefficient (lambda_c): {args.lambda_c}")
        print("="*80)

        print("Loading Original Pretrained Model (M_C)...")
        original_weights = load_original_weights(args.original_model_path)

        # Use base_weights keys as the reference for iteration
        for key in tqdm(base_weights.keys(), desc="Applying Task Vector Grafting"):
            if key in original_weights and key in donor_weights and need_merge(key) and base_weights[key].shape == donor_weights[key].shape:
                w_c = original_weights[key].float().cuda()
                w_a = base_weights[key].float().cuda()
                w_b = donor_weights[key].float().cuda()

                tau_a = w_a - w_c
                tau_b = w_b - w_c

                tau_a_norm_sq = torch.sum(tau_a * tau_a)
                inner_product = torch.sum(tau_a * tau_b)

                if tau_a_norm_sq > 1e-9:
                    proj_scalar = inner_product / tau_a_norm_sq
                    tau_b_synergy = torch.clamp(proj_scalar, min=0) * tau_a
                    tau_b_conflict = torch.clamp(-proj_scalar, min=0) * tau_a
                    tau_b_ortho = tau_b - (tau_b_synergy - tau_b_conflict)
                else:
                    tau_b_synergy = torch.zeros_like(tau_b)
                    tau_b_conflict = torch.zeros_like(tau_b)
                    tau_b_ortho = tau_b

                w_star = w_a.clone()
                w_star += (args.lambda_s - 1.0) * tau_b_synergy
                w_star += (1.0 - args.lambda_c) * (-tau_b_conflict)
                w_star += tau_b_ortho
                
                merged_weights[key] = w_star.to(base_weights[key].dtype).cpu()
            else:
                # For layers not being merged or not present in all models, keep the base model's weights
                merged_weights[key] = base_weights[key]
            
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

    # --- Saving the Merged Model (as per llava-qwen2qwenvl.py) ---
    print("\nSaving merged model...")

    metadata = {'format': 'pt'}
    llava_index_path = os.path.join(CKPT_PATH["qwen2_vl"], INDEX_FILENAME["qwen2_vl"])
    with open(llava_index_path, "r") as f:
        llava_index = json.load(f)
        llava_index = llava_index["weight_map"]
    
    split_llava = {}
    for file in qwenvl_file_list:
        split_llava[file] = {}
    for key in llava_index:
        split_llava[llava_index[key]][key] = base_weights[key]
    for file in qwenvl_file_list:
        if not os.path.isdir(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)
        save_path = os.path.join(OUTPUT_PATH, file)
        safetensors.torch.save_file(split_llava[file], save_path, metadata)
        
    create_soft_link(source_path=CKPT_PATH["qwen2_vl"], link_path=OUTPUT_PATH)

    print("Convert Done.")
    print(save_path)


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