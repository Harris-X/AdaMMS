# 7B version - General purpose model merger for compatible architectures

import os
import sys
import json
import torch
import safetensors.torch
import argparse
from tqdm import tqdm
from collections import defaultdict

# --- Model & File Configs ---

CKPT_PATH = {
    "cogvlm_chat": "/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/THUDM_cogvlm-base-490-hf",
    "cogvlm_grounding": "/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/THUDM_cogvlm-grounding-generalist-hf",
    "llava": "/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/liuhaotian_llava-v1.5-7b",
    "sharegpt": "/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/Lin-Chen_ShareGPT4V-7B",
    "vicuna-v1.5": "/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/lmsys_vicuna-7b-v1.5",
    "qwen2-7B": "/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/Qwen_Qwen2-7B-Instruct",
    "qwen2_vl" : "/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/Qwen_Qwen2-VL-7B-Instruct",
    "llava-onevision-qwen" : "/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/lmms-lab_llava-onevision-qwen2-7b-si",
}

INDEX_FILENAME = {
    "cogvlm_chat": "model.safetensors.index.json",
    "cogvlm_grounding": "model.safetensors.index.json",
    "llava": "pytorch_model.bin.index.json",
    "sharegpt": "pytorch_model.bin.index.json",
    "vicuna-v1.5": "pytorch_model.bin.index.json",
    "llava-onevision-qwen": "model.safetensors.index.json",
    "qwen2_vl" : "model.safetensors.index.json",
    "qwen2-7B": "model.safetensors.index.json"
}

# --- Weight Loading Functions ---

def load_pytorch_weights(base_path, file_list):
    weights = {}
    for file in file_list:
        path = os.path.join(base_path, file)
        weights.update(torch.load(path, map_location="cpu"))
    return weights

def load_safetensors_weights(base_path, file_list):
    weights = {}
    for file in file_list:
        path = os.path.join(base_path, file)
        weights.update(safetensors.torch.load_file(path, device="cpu"))
    return weights

# Define file lists for each model type
vicuna_file_list = ['pytorch_model-00001-of-00002.bin', 'pytorch_model-00002-of-00002.bin']
sharegpt_file_list = ['pytorch_model-00001-of-00002.bin', 'pytorch_model-00002-of-00002.bin']
llava_file_list = ['pytorch_model-00001-of-00002.bin', 'pytorch_model-00002-of-00002.bin']
qwen_file_list = ['model-00001-of-00004.safetensors', 'model-00002-of-00004.safetensors', 'model-00003-of-00004.safetensors', 'model-00004-of-00004.safetensors']
qwenvl_file_list = ['model-00001-of-00005.safetensors', 'model-00002-of-00005.safetensors', 'model-00003-of-00005.safetensors', 'model-00004-of-00005.safetensors', 'model-00005-of-00005.safetensors']
llava_onevision_qwen_file_list = ['model-00001-of-00004.safetensors', 'model-00002-of-00004.safetensors', 'model-00003-of-00004.safetensors', 'model-00004-of-00004.safetensors']
cogvlm_file_list = ['model-00001-of-00008.safetensors', 'model-00002-of-00008.safetensors', 'model-00003-of-00008.safetensors', 'model-00004-of-00008.safetensors', 'model-00005-of-00008.safetensors', 'model-00006-of-00008.safetensors', 'model-00007-of-00008.safetensors', 'model-00008-of-00008.safetensors']

# Create a dictionary of loader functions for dynamic loading
model_loaders = {
    "cogvlm_chat": lambda path: load_safetensors_weights(path, cogvlm_file_list),
    "cogvlm_grounding": lambda path: load_safetensors_weights(path, cogvlm_file_list),
    "llava": lambda path: load_pytorch_weights(path, llava_file_list),
    "sharegpt": lambda path: load_pytorch_weights(path, sharegpt_file_list),
    "vicuna-v1.5": lambda path: load_pytorch_weights(path, vicuna_file_list),
    "qwen2-7B": lambda path: load_safetensors_weights(path, qwen_file_list),
    "qwen2_vl": lambda path: load_safetensors_weights(path, qwenvl_file_list),
    "llava-onevision-qwen": lambda path: load_safetensors_weights(path, llava_onevision_qwen_file_list),
}

# --- Helper Functions ---

def need_merge(name: str) -> bool:
    """A filter for parameters that should be merged."""
    if name.endswith(".inv_freq"):
        return False
    return True

def create_soft_link(source_path, link_path):
    """Creates symbolic links for non-model files (e.g., config.json)."""
    if not os.path.exists(link_path):
        os.makedirs(link_path)
    for item in os.listdir(source_path):
        source_item = os.path.join(source_path, item)
        link_item = os.path.join(link_path, item)
        if item.endswith(('.safetensors', '.bin')):
            continue
        if os.path.lexists(link_item):
            os.remove(link_item)
        os.symlink(os.path.abspath(source_item), link_item)

# --- Main Merging Logic ---

def convert(args):
    # Validate arguments
    if len(args.models) != len(args.weights):
        raise ValueError("The number of models and weights must be the same.")
    if not all(model in CKPT_PATH for model in args.models):
        raise ValueError("One or more specified models are not defined in CKPT_PATH.")
    
    print(f"Merging models: {args.models} with weights: {args.weights}")
    print(f"Output path: {args.output}")
    os.makedirs(args.output, exist_ok=True)

    # Load all specified models
    loaded_models = []
    for model_name in args.models:
        print(f"Loading {model_name}...")
        loader = model_loaders[model_name]
        model_path = CKPT_PATH[model_name]
        loaded_models.append(loader(model_path))

    # --- Compatibility Check ---
    print("Checking for model architecture compatibility...")
    reference_keys = set(loaded_models[0].keys())
    for i, model_sd in enumerate(loaded_models[1:], 1):
        current_keys = set(model_sd.keys())
        if current_keys != reference_keys:
            print(f"\n!!! ERROR: Model '{args.models[i]}' is not compatible with '{args.models[0]}'.")
            print("Model merging aborted due to architecture mismatch.")
            missing_keys = reference_keys - current_keys
            extra_keys = current_keys - reference_keys
            if missing_keys:
                print(f"  - Keys missing in '{args.models[i]}': {len(missing_keys)} (e.g., {list(missing_keys)[:3]})")
            if extra_keys:
                print(f"  - Extra keys in '{args.models[i]}': {len(extra_keys)} (e.g., {list(extra_keys)[:3]})")
            return

    print("All models are compatible. Starting merge...")
    
    # --- Weighted Averaging ---
    final_model = {}
    for key in tqdm(reference_keys, desc="Averaging weights"):
        if need_merge(key):
            summed_tensor = torch.zeros_like(loaded_models[0][key], dtype=torch.float32)
            for i, model_sd in enumerate(loaded_models):
                summed_tensor += args.weights[i] * model_sd[key].to(torch.float32)
            final_model[key] = summed_tensor
        else:
            # For non-merged weights, take them from the first model in the list
            final_model[key] = loaded_models[0][key]

    # --- Saving Logic ---
    print("Saving merged model...")
    # Use qwen2_vl as the template for saving, as it's the target architecture
    saving_template_model = "qwen2_vl"
    template_index_path = os.path.join(CKPT_PATH[saving_template_model], INDEX_FILENAME[saving_template_model])
    with open(template_index_path, "r") as f:
        template_index = json.load(f)["weight_map"]

    split_model = defaultdict(dict)
    for key, filename in template_index.items():
        if key in final_model:
            split_model[filename][key] = final_model[key]

    for filename, state_dict in split_model.items():
        save_path = os.path.join(args.output, filename)
        safetensors.torch.save_file(state_dict, save_path)
    
    create_soft_link(source_path=CKPT_PATH[saving_template_model], link_path=args.output)
    print(f"\nMerge complete. Model saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple models with compatible architectures via weighted averaging.")
    parser.add_argument('--models', nargs='+', required=True, help=f"Space-separated list of model names to merge. Choices: {list(CKPT_PATH.keys())}")
    parser.add_argument('--weights', nargs='+', type=float, required=True, help="Space-separated list of weights for each model. Should sum to 1 for linear interpolation.")
    parser.add_argument('--output', type=str, required=True, help="Output directory for the merged model.")
    
    args = parser.parse_args()
    convert(args)