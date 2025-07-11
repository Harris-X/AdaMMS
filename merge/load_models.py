import os
import sys
import json
import torch
import safetensors.torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor,AutoModelForVision2Seq


qwen_file_list = ['model-00001-of-00004.safetensors', 'model-00002-of-00004.safetensors', 'model-00003-of-00004.safetensors', 'model-00004-of-00004.safetensors']
vicuna_file_list = ['pytorch_model-00001-of-00002.bin', 'pytorch_model-00002-of-00002.bin']
llama_file_list = ['pytorch_model-00001-of-00003.bin', 'pytorch_model-00002-of-00003.bin', 'pytorch_model-00003-of-00003.bin']

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

def load_llama_weights(base_path, file_list=llama_file_list):
    return load_pytorch_weights(base_path, file_list)

def load_qwen_weights(base_path, file_list=qwen_file_list):
    return load_safetensors_weights(base_path, file_list)


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
def load_minicpm_weights(base_path, file_list=llava_onevision_qwen_file_list):
    return load_safetensors_weights(base_path, file_list)



################## Load Models completely ##################
def load_complete_qwen2(model_id): 
    print(f"正在加载模型: {model_id}...")
    # 加载模型，使用 bfloat16 以提高效率并减少显存占用 (如果GPU支持)
    # torch_dtype="auto" 会自动选择最佳的数据类型
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto" # 自动将模型分片加载到可用设备上
    )
    print("模型加载完成。")

    print(f"正在加载分词器: {model_id}...")
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer, model

def load_complete_llama2(model_id):
    print(f"正在加载模型: {model_id}...")
    # 加载模型，使用 bfloat16 以提高效率并减少显存占用 (如果GPU支持)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto" # 自动将模型分片加载到可用设备上
    )
    print("模型加载完成。")

    print(f"正在加载分词器: {model_id}...")
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer, model