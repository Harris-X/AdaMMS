# 7B version with SVD Merging

import os
import sys
import json
import torch
import safetensors.torch
import argparse
from collections import defaultdict, OrderedDict
from copy import deepcopy
from tqdm import tqdm

# --- Model Paths and Configs (from blueprint) ---
CKPT_PATH = {
    "cogvlm_chat": "/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/cogvlm-base-490-hf",
    "cogvlm_grounding": "/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/cogvlm-grounding-generalist-hf",
    "llava": "/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/llava-v1.5-7b",
    "sharegpt": "/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/ShareGPT4V-7B",
    "vicuna-v1.5": "/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/vicuna-7b-v1.5",
    "qwen2-7B": "/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/Qwen2-7B-Instruct",
    "qwen2_vl" : "/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct",
    "llava-onevision-qwen" : "/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si",
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

N = 28 # layers count

# --- Loading Functions (from blueprint) ---
def load_safetensors_weights(base_path, file_list):
    weights = {}
    for file in file_list:
        path = os.path.join(base_path, file)
        x = safetensors.torch.load_file(path)
        weights.update(x)
    return weights

qwen_file_list = ['model-00001-of-00004.safetensors', 'model-00002-of-00004.safetensors', 'model-00003-of-00004.safetensors', 'model-00004-of-00004.safetensors']
def load_qwen_weights(base_path, file_list=qwen_file_list):
    return load_safetensors_weights(base_path, file_list)

qwenvl_file_list = ['model-00001-of-00005.safetensors', 'model-00002-of-00005.safetensors', 'model-00003-of-00005.safetensors', 'model-00004-of-00005.safetensors', 'model-00005-of-00005.safetensors']
def load_qwenvl_weights(base_path, file_list=qwenvl_file_list):
    return load_safetensors_weights(base_path, file_list)

llava_onevision_qwen_file_list = ['model-00001-of-00004.safetensors', 'model-00002-of-00004.safetensors', 'model-00003-of-00004.safetensors', 'model-00004-of-00004.safetensors']
def load_minicpm_weights(base_path, file_list=llava_onevision_qwen_file_list):
    return load_safetensors_weights(base_path, file_list)

# --- SVD Helper Functions (inspired by task_merger.py) ---

def get_layer_names(state_dict):
    """Parses state_dict keys into layer groups for matrix conversion."""
    layer_names = defaultdict(lambda: dict())
    for key in state_dict:
        if '.weight' in key:
            strip_key = key.replace('.weight', '')
            layer_names[strip_key]['weight'] = key
        elif '.bias' in key:
            strip_key = key.replace('.bias', '')
            layer_names[strip_key]['bias'] = key
        else:
            layer_names[key]['other'] = key
    return layer_names

def directions_to_matrices(directions, layer_names):
    """Converts a state_dict of weight differences into a dictionary of matrices."""
    matrices = {}
    for layer_name, parameter_names in layer_names.items():
        if 'other' in parameter_names:
            other_parameter = directions[parameter_names['other']].to(torch.float32)
            if len(other_parameter.shape) == 1: other_parameter = other_parameter[None, :]
            elif len(other_parameter.shape) > 2: other_parameter = other_parameter.flatten(1)
            matrices[layer_name + ':other'] = other_parameter
        elif 'weight' in parameter_names:
            weight = directions[parameter_names['weight']]
            if 'norm' in layer_name or 'ln' in layer_name:
                weight = torch.diag(weight)
            matrices[layer_name] = weight.flatten(1)
            if 'bias' in parameter_names:
                bias = directions[parameter_names['bias']]
                matrices[layer_name] = torch.concat((matrices[layer_name], bias.reshape(-1, 1)), dim=1)
    return matrices

def matrix_to_state_dict(matrix, reference_dict, layer_names):
    """Converts a dictionary of matrices back to a state_dict."""
    merged_state_dict = {}
    for layer_name, value in matrix.items():
        parameter_types = layer_names[layer_name.replace(':other', '')]
        if 'other' in parameter_types:
            name = parameter_types['other']
            merged_state_dict[name] = value.reshape(reference_dict[name].shape)
        else:
            if 'bias' in parameter_types:
                bias_index = value.shape[1] - 1
                value, bias = value[:, :bias_index], value[:, -1].flatten()
                merged_state_dict[parameter_types['bias']] = bias
            if 'norm' in layer_name or 'ln' in layer_name:
                value = torch.diagonal(value)
            name = parameter_types['weight']
            merged_state_dict[name] = value.reshape(*(reference_dict[name].shape))
    return merged_state_dict

def need_merge_llava(name:str) -> bool:
    """Filter for weights to be merged."""
    if name in ['lm_head.weight', 'model.embed_tokens.weight' ]:
        return True
    if name in [ 'model.norm.weight']:
        return True   
    if name.startswith("model.layers."):
        if name.endswith(".self_attn.rotary_emb.inv_freq"):
            return False
        return True
    return False

def create_soft_link(source_path, link_path):
    if not os.path.exists(source_path):
        print(f"Error: Source path '{source_path}' does not exist.")
        return
    if not os.path.exists(link_path):
        os.makedirs(link_path)
    for item in os.listdir(source_path):
        source_item = os.path.join(source_path, item)
        link_item = os.path.join(link_path, item)
        if item.endswith(('.safetensors', '.bin')):
            continue
        if os.path.isfile(source_item):
            try:
                if os.path.lexists(link_item):
                    os.remove(link_item)
                os.symlink(source_item, link_item)
            except OSError as e:
                print(f"Error creating soft link for '{item}': {e}")

# --- Main Conversion Logic ---

def convert(args):
    """Main function to perform SVD merging."""
    print(f"Merging output path: {args.output}")
    os.makedirs(args.output, exist_ok=True)

    print("Loading base model (Qwen2-7B)...")
    base_model = load_qwen_weights(CKPT_PATH["qwen2-7B"])
    
    print("Loading target model (Qwen2-VL)...")
    model1 = load_qwenvl_weights(CKPT_PATH["qwen2_vl"])
    
    print("Loading source model (LLaVA-OneVision)...")
    model2 = load_minicpm_weights(CKPT_PATH['llava-onevision-qwen'])

    print("Calculating task vectors (deltas)...")
    delta1, delta2 = {}, {}
    for key in tqdm(model1.keys(), desc="Calculating Deltas"):
        if need_merge_llava(key) and key in base_model and key in model2:
            delta1[key] = model1[key].to(torch.float32) - base_model[key].to(torch.float32)
            delta2[key] = model2[key].to(torch.float32) - base_model[key].to(torch.float32)

    layer_names = get_layer_names(delta1)

    print("Converting task vectors to matrices...")
    matrices1 = directions_to_matrices(delta1, layer_names)
    matrices2 = directions_to_matrices(delta2, layer_names)

    print("Performing SVD merging...")
    merged_matrices = {}
    for key in tqdm(matrices1.keys(), desc="SVD Merging Progress"):
        mat1, mat2 = matrices1[key], matrices2[key]
        
        combined_mat = torch.cat([mat1, mat2], dim=1)
        U, S, Vh = torch.linalg.svd(combined_mat, full_matrices=False)

        rank = args.rank
        if rank > 0 and rank < len(S):
            U_trunc, S_trunc, Vh_trunc = U[:, :rank], S[:rank], Vh[:rank, :]
        else:
            U_trunc, S_trunc, Vh_trunc = U, S, Vh

        V1h, V2h = Vh_trunc[:, :mat1.shape[1]], Vh_trunc[:, mat1.shape[1]:]
        
        merged_SVh = (torch.diag(S_trunc) @ V1h + torch.diag(S_trunc) @ V2h) / 2.0
        merged_matrices[key] = U_trunc @ merged_SVh

    print("Converting merged matrices back to state dict...")
    merged_delta = matrix_to_state_dict(merged_matrices, delta1, layer_names)

    print("Adding merged delta to base model...")
    final_model = base_model
    for key, value in merged_delta.items():
        final_model[key] = final_model[key].to(torch.float32) + value
    
    for key in model1:
        if key not in merged_delta:
            final_model[key] = model1[key]

    print("Saving...")
    llava_index_path = os.path.join(CKPT_PATH["qwen2_vl"], INDEX_FILENAME["qwen2_vl"])
    with open(llava_index_path, "r") as f:
        llava_index = json.load(f)["weight_map"]
    
    split_llava = defaultdict(dict)
    for key, file in llava_index.items():
        if key in final_model:
            split_llava[file][key] = final_model[key]

    for file, state_dict in split_llava.items():
        save_path = os.path.join(args.output, file)
        safetensors.torch.save_file(state_dict, save_path)
        
    create_soft_link(source_path=CKPT_PATH["qwen2_vl"], link_path=args.output)
    print(f"Convert Done. Model saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True, help="Output checkpoint path")
    parser.add_argument('--rank', type=int, default=64, help="Rank for SVD truncation. 0 for no truncation.")
    args = parser.parse_args()
    convert(args)


