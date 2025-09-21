
import json
import os
import safetensors
import safetensors.torch
import torch
from tqdm import tqdm


def need_merge(name: str) -> bool:
    """
    SAFE的复杂合并目标：
    - 仅处理 transformer layers 内部的线性权重与 bias
    - 显式排除所有 norm 与 rotary_emb
    """
    is_in_layers = name.startswith("model.layers.") or name.startswith("language_model.layers.") or name.startswith("language_model.model.layers.")
    if not is_in_layers:
        return False

    # 显式排除 norm 和 rotary
    if 'layernorm' in name or 'norm' in name or 'rotary_emb' in name:
        return False

    # 线性层的 .weight/.bias 进入复杂合并
    if name.endswith('.weight') or name.endswith('.bias'):
        return True

    return False


def load_weights(base_path, index_filename="model.safetensors.index.json"):
    """加载模型权重，支持以下几种情况：
    1) model.safetensors.index.json + sharded safetensors
    2) 单个 model.safetensors
    3) pytorch_model.bin.index.json + sharded .bin（PyTorch）
    4) 单个 pytorch_model.bin
    """
    weights = {}

    # 优先 safetensors（索引 + 单文件）
    st_index = os.path.join(base_path, index_filename)
    st_single = os.path.join(base_path, "model.safetensors")
    if os.path.exists(st_index):
        with open(st_index, 'r') as f:
            index = json.load(f)
        file_list = sorted(list(set(index["weight_map"].values())))
        for file in tqdm(file_list, desc=f"从 {os.path.basename(base_path)} 加载 safetensors 分片"):
            weights.update(safetensors.torch.load_file(os.path.join(base_path, file)))
        return weights
    if os.path.exists(st_single):
        print(f"正在加载单个权重文件: {st_single}")
        return safetensors.torch.load_file(st_single)

    # 再尝试 PyTorch .bin（索引 + 单文件）
    pt_index = os.path.join(base_path, "pytorch_model.bin.index.json")
    pt_single = os.path.join(base_path, "pytorch_model.bin")
    if os.path.exists(pt_index):
        with open(pt_index, 'r') as f:
            index = json.load(f)
        file_list = sorted(list(set(index["weight_map"].values())))
        for file in tqdm(file_list, desc=f"从 {os.path.basename(base_path)} 加载 PyTorch .bin 分片"):
            shard_path = os.path.join(base_path, file)
            state = torch.load(shard_path, map_location='cpu')
            # 有些分片是包含 key->tensor 的 dict（常见），直接合并
            if isinstance(state, dict):
                weights.update(state)
            else:
                raise ValueError(f"未识别的分片格式: {shard_path}")
        return weights
    if os.path.exists(pt_single):
        print(f"正在加载单个权重文件: {pt_single}")
        state = torch.load(pt_single, map_location='cpu')
        if isinstance(state, dict):
            return state
        raise ValueError(f"未识别的权重文件格式: {pt_single}")

    raise FileNotFoundError(
        f"在 {base_path} 中未找到以下任一权重格式: "
        f"model.safetensors.index.json / model.safetensors / pytorch_model.bin.index.json / pytorch_model.bin"
    )

def normalize_llm_keys(weights_to_norm: dict, reference_keys: list) -> dict:
    """将待对齐模型的键名映射到参考模型(A)的键名。

    策略：
    1) 构建“规范化键名”，统一不同前缀（language_model.model./language_model. → model.），
       并从 layers 起对齐为 model.layers.* 路径，保留尾部 .weight/.bias。
    2) 用参考键构建 canon→ref_key 的映射；
    3) 将待对齐权重按规范名查表，若匹配则重命名为 ref_key；否则回退到简单前缀替换方案。
    """

    def canonical_param_key(k: str) -> str:
        k2 = k.replace("language_model.model.", "model.").replace("language_model.", "model.")
        if "layers" in k2:
            pos = k2.find("layers")
            k2 = "model." + k2[pos:]
        return k2

    # 1) 构建参考映射：canon -> 参考原始键
    ref_canon_to_key = {}
    for rk in reference_keys:
        if "layers" not in rk:
            # 非 layers 参数（如嵌入、lm_head），原样保留映射（也构建 canon 以便匹配）
            ref_canon_to_key[canonical_param_key(rk)] = rk
        else:
            ref_canon_to_key[canonical_param_key(rk)] = rk

    # 2) 先尝试按规范名一一映射
    normalized_weights: dict = {}
    unmatched: dict = {}
    for key, value in weights_to_norm.items():
        ckey = canonical_param_key(key)
        if ckey in ref_canon_to_key:
            normalized_weights[ref_canon_to_key[ckey]] = value
        else:
            unmatched[key] = value

    if not unmatched:
        return normalized_weights

    # 3) 回退到简单前缀替换以覆盖少量非 layers 的键或异常键
    ref_prefix = ""
    for key in reference_keys:
        if "layers" in key:
            ref_prefix = key.split("layers")[0]
            break
    norm_prefix = ""
    for key in weights_to_norm.keys():
        if "layers" in key:
            norm_prefix = key.split("layers")[0]
            break

    for key, value in unmatched.items():
        if ref_prefix and norm_prefix and key.startswith(norm_prefix):
            new_key = ref_prefix + key[len(norm_prefix):]
            normalized_weights[new_key] = value
        else:
            # 最终兜底：保持原名
            normalized_weights[key] = value

    return normalized_weights

