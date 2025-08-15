#!/usr/bin/env python3
import datasets
import os

# 设置缓存目录
cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")

# 预下载常用的评测数据集
datasets_to_download = [
    "textvqa",
    "MME-RealWorld", 
    "ok_vqa",
    "gqa",
    "vizwiz_vqa",
    "seedbench"
]

for dataset_name in datasets_to_download:
    try:
        print(f"Downloading {dataset_name}...")
        dataset = datasets.load_dataset(dataset_name, cache_dir=cache_dir)
        print(f"✓ {dataset_name} downloaded successfully")
    except Exception as e:
        print(f"✗ Failed to download {dataset_name}: {e}")

print("Dataset download completed!")
