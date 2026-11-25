"""RAPT V0.3 共享权重加载工具 (与 rapt_V0_1/utils.py 保持一致, 版本目录自包含)。

只做两件事:
  - need_merge: 判定某个参数是否属于语言侧可合并 affine 模块;
  - load_weights: 安全地加载 safetensors / pytorch 分片权重。
"""
from __future__ import annotations

import json
import os

import safetensors.torch
import torch
from tqdm import tqdm


def need_merge(name: str) -> bool:
    """仅处理 transformer layers 内部的线性权重与 bias, 显式排除 norm / rotary_emb。"""
    is_in_layers = (
        name.startswith("model.layers.")
        or name.startswith("language_model.layers.")
        or name.startswith("language_model.model.layers.")
    )
    if not is_in_layers:
        return False
    if "layernorm" in name or "norm" in name or "rotary_emb" in name:
        return False
    if name.endswith(".weight") or name.endswith(".bias"):
        return True
    return False


def load_weights(base_path, index_filename="model.safetensors.index.json"):
    """加载模型权重, 支持 safetensors 索引/单文件与 pytorch bin 索引/单文件。"""
    weights = {}

    st_index = os.path.join(base_path, index_filename)
    st_single = os.path.join(base_path, "model.safetensors")
    if os.path.exists(st_index):
        with open(st_index, "r") as f:
            index = json.load(f)
        file_list = sorted(list(set(index["weight_map"].values())))
        for file in tqdm(file_list, desc=f"从 {os.path.basename(base_path)} 加载 safetensors 分片"):
            weights.update(safetensors.torch.load_file(os.path.join(base_path, file)))
        return weights
    if os.path.exists(st_single):
        print(f"正在加载单个权重文件: {st_single}")
        return safetensors.torch.load_file(st_single)

    pt_index = os.path.join(base_path, "pytorch_model.bin.index.json")
    pt_single = os.path.join(base_path, "pytorch_model.bin")
    if os.path.exists(pt_index):
        with open(pt_index, "r") as f:
            index = json.load(f)
        file_list = sorted(list(set(index["weight_map"].values())))
        for file in tqdm(file_list, desc=f"从 {os.path.basename(base_path)} 加载 PyTorch .bin 分片"):
            shard_path = os.path.join(base_path, file)
            state = torch.load(shard_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict):
                weights.update(state)
            else:
                raise ValueError(f"未识别的分片格式: {shard_path}")
        return weights
    if os.path.exists(pt_single):
        print(f"正在加载单个权重文件: {pt_single}")
        state = torch.load(pt_single, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            return state
        raise ValueError(f"未识别的权重文件格式: {pt_single}")

    raise FileNotFoundError(
        f"在 {base_path} 中未找到以下任一权重格式: "
        f"model.safetensors.index.json / model.safetensors / "
        f"pytorch_model.bin.index.json / pytorch_model.bin"
    )
