# Transfer grounding ability from cogvlm to llava *Using mplugowl arch*

# 7B version

import os
import sys
import json
import torch
import safetensors.torch
import argparse

try:
    from ties_merging import do_merging, do_merging_strategy
except:
    print("ties_merging.py not found, couldn't perform ties-merging.")

from typing import Optional
from copy import deepcopy

OUTPUT_PATH = "converted-cogvlm-base-noNorm"

CKPT_PATH = {
    "cogvlm_chat": "/yeesuanAI05/thumt/dyy/model/cogvlm-chat-hf", #"/yeesuanAI05/thumt/dyy/model/cogvlm-chat-hf",
    "cogvlm_grounding": "/yeesuanAI05/thumt/dyy/model/cogvlm-grounding-generalist-hf",
    "llava": "/yeesuanAI05/thumt/dyy/model/llava-v1.5-7b",
    "mplug_owl": "/yeesuanAI05/thumt/dyy/model/mplug-owl2-llama2-7b",
    "llama2": "/yeesuanAI05/thumt/dyy/model/Llama-2-7b-hf",
}

INDEX_FILENAME = {
    "cogvlm_chat": "model.safetensors.index.json",
    "cogvlm_grounding": "model.safetensors.index.json",
    "llava": "pytorch_model.bin.index.json",
    "mplug_owl" : "pytorch_model.bin.index.json",
    "llama2": "pytorch_model.bin.index.json",
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
    interpolation = False # args.interpolation

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
    llava = load_llava_weights(CKPT_PATH["llava"])
    
    mplug_owl = load_mplug_owl_weights(CKPT_PATH["mplug_owl"])

    llama = load_llama_weights(CKPT_PATH["llama2"])

    print("Merging...")
    def mplug2llava(key):
        # this only applies to keys that 'need merge'
        key = key.replace("multiway.0.", "")\
                 .replace("multiway.1.", "")
        return key
    
    mplug_owl_merge = {}
    llava_merge = {}
    for key in mplug_owl:
        if need_merge(key):
            llava_key = mplug2llava(key)
            mplug_owl_merge[key] = mplug_owl[key] - llama[llava_key]
            llava_merge[key] = llava[llava_key] - llama[llava_key]
    if args.strategy == 'ties':
        merged = do_merging([llava_merge, mplug_owl_merge], merge_func="dis-sum", K=args.K)
    else:
        merged = do_merging_strategy([llava_merge, mplug_owl_merge], args.strategy, K=args.K)
    for key in mplug_owl:
        if need_merge(key):
            llava_key = mplug2llava(key)
            mplug_owl[key] = merged[key] + llama[llava_key]

    # save
    print("Saving...")
    mplug_owl_index_path = os.path.join(CKPT_PATH["mplug_owl"], INDEX_FILENAME["mplug_owl"])
    with open(mplug_owl_index_path, "r") as f:
        mplug_owl_index = json.load(f)
        mplug_owl_index = mplug_owl_index["weight_map"]
    
    split_mplug_owl = {}
    # metadata = {'format': 'pt'}
    for file in mplug_owl_file_list:
        split_mplug_owl[file] = {}
    for key in mplug_owl_index:
        split_mplug_owl[mplug_owl_index[key]][key] = mplug_owl[key]
    for file in mplug_owl_file_list:
        if not os.path.isdir(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)
        save_path = os.path.join(OUTPUT_PATH, file)
        torch.save(split_mplug_owl[file], save_path)
        # safetensors.torch.save_file(split_mplug_owl[file], save_path, metadata)
    
    create_soft_link(source_path=CKPT_PATH["mplug_owl"], link_path=OUTPUT_PATH)
    
    print("Convert Done.")
    # import ipdb; ipdb.set_trace();

if __name__ == "__main__":
    # compare_index()
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=None, help="Output checkpoint path")
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--with-grounding', action='store_true')
    parser.add_argument('--noLN', action='store_true')

    parser.add_argument('--beta', type=float, default=1.0)

    parser.add_argument('--gamma', type=float, default=1.0)
    
    parser.add_argument('--base', type=str, default=None)
    parser.add_argument('--base-mplug', type=str, default=None)
    parser.add_argument('--base-llava', type=str, default=None)

    parser.add_argument('--strategy', type=str, default=None) 
    parser.add_argument('-K', type=float, default=1.0)
    
    
    args = parser.parse_args()
    print(args)

    if args.base is not None:
        CKPT_PATH['cogvlm_chat'] = args.base
    if args.base_mplug is not None:
        CKPT_PATH['mplug_owl'] = args.base_mplug
    if args.base_llava is not None:
        CKPT_PATH['llava'] = args.base_llava
    

    convert(args)