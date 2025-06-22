# Transfer grounding ability from cogvlm to llava *Using cogvlm arch*

import os
import sys
import json
import torch
import safetensors.torch
import argparse

from typing import Optional

OUTPUT_PATH = "converted-cogvlm-base-noNorm"

CKPT_PATH = {
    # --base option will override cogvlm_chat
    "cogvlm_chat": "/yeesuanAI05/thumt/dyy/model/cogvlm-base-490-hf", #"/yeesuanAI05/thumt/dyy/model/cogvlm-chat-hf",
    "cogvlm_grounding": "/yeesuanAI05/thumt/dyy/model/cogvlm-grounding-generalist-hf",
    "llava": "/yeesuanAI05/thumt/dyy/model/llava-v1.5-7b",
    "mplug_owl": "/yeesuanAI05/thumt/dyy/model/mplug-owl2-llama2-7b"
}

INDEX_FILENAME = {
    "cogvlm_chat": "model.safetensors.index.json",
    "cogvlm_grounding": "model.safetensors.index.json",
    "llava": "pytorch_model.bin.index.json",
    "mplug_owl" : "pytorch_model.bin.index.json",
}

N = 32 # layers count

def compare_index():
    indices = {}
    for model in CKPT_PATH:
        if model == 'cogvlm_grounding':
            continue
        index_path = os.path.join(CKPT_PATH[model], INDEX_FILENAME[model])
        with open(index_path, 'r') as f:
            index = json.load(f)
        indices[model] = index['weight_map']
    
    llava_key = indices['llava'].keys()
    cogvlm_key = indices['cogvlm_chat'].keys()

    common_key = list(set(llava_key) & set(cogvlm_key))
    # print(llava_key)
    # print(common_key)
    print(list(set(llava_key) - set(common_key)))
    # print(common_key)

def load_pytorch_weights(base_path, file_list):
    weights = {}
    for file in file_list:
        path = os.path.join(base_path, file)
        x = torch.load(path)
        weights.update(x)
    return weights
def load_safetensors_weights(base_path, file_list):
    weights = {}
    for file in file_list:
        path = os.path.join(base_path, file)
        x = safetensors.torch.load_file(path)
        weights.update(x)
    return weights

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

def need_merge(name:str) -> bool:
    if name in ['lm_head.weight', 'model.embed_tokens.weight', 'model.norm.weight']:
        return True
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
        if item.endswith('.safetensors'):
            print(f"Skipping '{item}' as it ends with '.safetensors'")
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
    

def convert(args):
    global OUTPUT_PATH
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    interpolation = args.interpolation

    if args.output is not None:
        OUTPUT_PATH = args.output
    else: # when --output is not provided
        # Default path name
        if alpha != 1.0:
            assert alpha != 0
            OUTPUT_PATH = OUTPUT_PATH + f"-alpha-{alpha}"
        if interpolation:
            OUTPUT_PATH = OUTPUT_PATH + "-interpolation"
    print(f"Merging output path: {OUTPUT_PATH}")

    print("Loading...")
    cogvlm_chat = load_cogvlm_weights(CKPT_PATH["cogvlm_chat"])
    cogvlm_grounding = load_cogvlm_weights(CKPT_PATH["cogvlm_grounding"])
    cogvlm_diff = {}
    # if interpolation: scale by alpha if this key is to be merged
    # if not interpolation: calculate diff, then scale if needed
    for key in cogvlm_grounding:
        if interpolation:
            if not args.with_grounding: # interpolate with base(chat)
                cogvlm_diff[key] = (cogvlm_chat[key] * alpha) if need_merge(key) else cogvlm_chat[key]
            else: # interpolate with grounding
                cogvlm_diff[key] = (cogvlm_grounding[key] * alpha) if need_merge(key) else cogvlm_grounding[key]
        elif args.double_interpolation:
            if not args.with_grounding: # interpolate with base(chat)
                cogvlm_diff[key] = (cogvlm_chat[key] * (alpha-beta) + cogvlm_grounding[key] * beta) if need_merge(key) else cogvlm_chat[key]
            else: # interpolate with(to) grounding
                cogvlm_diff[key] = (cogvlm_chat[key] * alpha*(1-beta) + cogvlm_grounding[key] * beta) if need_merge(key) else cogvlm_chat[key]
        elif args.triple_interpolation:
            if not args.with_grounding:
                cogvlm_diff[key] = (cogvlm_chat[key] * (alpha*gamma + beta*(1-gamma)) + cogvlm_grounding[key] * ((1-beta)*(1-gamma))) if need_merge(key) else (cogvlm_chat[key] * beta + cogvlm_grounding[key] * (1-beta))
            else:
                cogvlm_diff[key] = (cogvlm_chat[key] * beta*(1-gamma) + cogvlm_grounding[key] * (alpha*gamma + (1-beta)*(1-gamma))) if need_merge(key) else (cogvlm_chat[key] * beta + cogvlm_grounding[key] * (1-beta))
        elif need_merge(key):
            cogvlm_diff[key] = (cogvlm_grounding[key] - cogvlm_chat[key]) * alpha
        else:
            cogvlm_diff[key] = cogvlm_grounding[key]
    
    llava = load_mplug_owl_weights(CKPT_PATH["mplug_owl"])

    # re-scale llava by (1-alpha) as interpolation if needed
    if interpolation:
        for key in llava:
            llava[key] = llava[key] * (1 - alpha)
    elif args.double_interpolation:
        for key in llava:
            llava[key] = llava[key] * (1 - alpha - beta + alpha*beta)
    elif args.triple_interpolation:
        for key in llava:
            llava[key] = llava[key] * (1 - alpha) * gamma

    print("Merging...")
    # merge lm_head
    cogvlm_diff['lm_head.weight'] += llava['lm_head.weight']

    # merge model.embed_tokens.weight
    cogvlm_diff['model.embed_tokens.weight'] += llava['model.embed_tokens.weight']

    # merge transformer layers
    for i in range(N):
        # LN
        if not args.noLN:
            cogvlm_diff[f'model.layers.{i}.input_layernorm.weight'] += llava[f'model.layers.{i}.input_layernorm.multiway.0.weight'] # .multiway.1 not used
            cogvlm_diff[f'model.layers.{i}.post_attention_layernorm.weight'] += llava[f'model.layers.{i}.post_attention_layernorm.multiway.0.weight']
        else:
            cogvlm_diff[f'model.layers.{i}.input_layernorm.weight'] += cogvlm_chat[f'model.layers.{i}.input_layernorm.weight']
            cogvlm_diff[f'model.layers.{i}.post_attention_layernorm.weight'] += cogvlm_chat[f'model.layers.{i}.post_attention_layernorm.weight']

        # MLP
        cogvlm_diff[f'model.layers.{i}.mlp.language_mlp.down_proj.weight'] += llava[f'model.layers.{i}.mlp.down_proj.weight']
        cogvlm_diff[f'model.layers.{i}.mlp.language_mlp.gate_proj.weight'] += llava[f'model.layers.{i}.mlp.gate_proj.weight']
        cogvlm_diff[f'model.layers.{i}.mlp.language_mlp.up_proj.weight'] += llava[f'model.layers.{i}.mlp.up_proj.weight']

        cogvlm_diff[f'model.layers.{i}.mlp.vision_mlp.down_proj.weight'] += llava[f'model.layers.{i}.mlp.down_proj.weight']
        cogvlm_diff[f'model.layers.{i}.mlp.vision_mlp.gate_proj.weight'] += llava[f'model.layers.{i}.mlp.gate_proj.weight']
        cogvlm_diff[f'model.layers.{i}.mlp.vision_mlp.up_proj.weight'] += llava[f'model.layers.{i}.mlp.up_proj.weight']

        # ATTENTION
        cat_tensor_0 = torch.cat((llava[f'model.layers.{i}.self_attn.q_proj.weight'], llava[f'model.layers.{i}.self_attn.k_proj.multiway.0.weight'], llava[f'model.layers.{i}.self_attn.v_proj.multiway.0.weight']), dim=0)
        cat_tensor_1 = torch.cat((llava[f'model.layers.{i}.self_attn.q_proj.weight'], llava[f'model.layers.{i}.self_attn.k_proj.multiway.1.weight'], llava[f'model.layers.{i}.self_attn.v_proj.multiway.1.weight']), dim=0)

        assert cat_tensor_0.shape == cogvlm_diff[f'model.layers.{i}.self_attn.language_expert_query_key_value.weight'].shape
        cogvlm_diff[f'model.layers.{i}.self_attn.language_expert_query_key_value.weight'] += cat_tensor_0
        cogvlm_diff[f'model.layers.{i}.self_attn.vision_expert_query_key_value.weight'] += cat_tensor_1

        cogvlm_diff[f'model.layers.{i}.self_attn.language_expert_dense.weight'] += llava[f'model.layers.{i}.self_attn.o_proj.weight']
        cogvlm_diff[f'model.layers.{i}.self_attn.vision_expert_dense.weight'] += llava[f'model.layers.{i}.self_attn.o_proj.weight']
        
        # no need to merge rotary_emb.inv_freq
    if not args.noLN:
        cogvlm_diff['model.norm.weight'] += llava['model.norm.weight']
    else:
        cogvlm_diff['model.norm.weight'] += cogvlm_chat['model.norm.weight']
    
    # save
    print("Saving...")
    cogvlm_index_path = os.path.join(CKPT_PATH["cogvlm_grounding"], INDEX_FILENAME["cogvlm_grounding"])
    with open(cogvlm_index_path, "r") as f:
        cogvlm_index = json.load(f)
        cogvlm_index = cogvlm_index["weight_map"]
    
    split_cogvlm = {}
    metadata = {'format': 'pt'}
    for file in cogvlm_file_list:
        split_cogvlm[file] = {}
    for key in cogvlm_index:
        split_cogvlm[cogvlm_index[key]][key] = cogvlm_diff[key]
    for file in cogvlm_file_list:
        if not os.path.isdir(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)
        save_path = os.path.join(OUTPUT_PATH, file)
        safetensors.torch.save_file(split_cogvlm[file], save_path, metadata)
    
    create_soft_link(source_path=CKPT_PATH["cogvlm_chat"], link_path=OUTPUT_PATH)

    print("Convert Done.")
    # import ipdb; ipdb.set_trace();

if __name__ == "__main__":
    # compare_index()
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=None, help="Output checkpoint path")
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--interpolation', default=True,action='store_true')
    parser.add_argument('--with-grounding', action='store_true')
    parser.add_argument('--noLN', action='store_true')
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--double-interpolation', action='store_true')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--triple-interpolation', action='store_true')    
    parser.add_argument('--base', type=str, default=None)
    
    args = parser.parse_args()
    print(args)

    assert not (args.interpolation and args.double_interpolation)
    if args.base is not None:
        CKPT_PATH['cogvlm_chat'] = args.base

    convert(args)