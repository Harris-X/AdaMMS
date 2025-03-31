

~~~markdown
<h1 align="center">🚀 AdaMMS: Adaptive Model Merging for Heterogeneous Multimodal LLMs</h1>
<p align="center">
  <img src="https://img.shields.io/badge/CVPR-2025-blue.svg" alt="CVPR 2025"/>
  <img src="https://img.shields.io/github/stars/your-org-name/AdaMMS?style=social" alt="GitHub stars"/>
  <img src="https://img.shields.io/github/issues/your-org-name/AdaMMS" alt="Issues"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
</p>

<p align="center">🔥 Accepted to CVPR 2025!</p>

---

## 📘 项目简介

随着大模型融合方法在多个任务中展现出强大性能，其在多模态大语言模型（MLLMs）上的应用也日益受到关注。然而，现有方法主要集中于**同构模型**的融合，难以适应**异构模型**之间在架构与参数空间上的差异。

我们提出 **AdaMMS**：**Ada**ptive **M**apping, **M**erging, and **S**earching，一种面向异构多模态大模型的无监督融合框架。其核心包含三大步骤：

1. 🧠 **Mapping**  
   构建模型间的映射函数，解决架构不一致问题。
2. ⚖️ **Merging**  
   在参数空间中进行加权插值，适应异构模型的不对称性。
3. 🔍 **Searching**  
   提出无监督的超参搜索方法，自动寻找最佳融合系数。

📊 实验表明，AdaMMS 在多个视觉语言任务中均优于现有融合方法。

---

## 🛠️ 环境配置

> ⚠️ 由于涉及多个模型架构，建议**分别配置模型环境**后再安装 `lmms-eval`。

### ✅ CogVLM 配置示例

```bash
conda create -n lmms-cogvlm python=3.10
conda activate lmms-cogvlm

wget https://github.com/THUDM/CogVLM/blob/main/requirements.txt --no-check-certificate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval && pip install -e .

conda install openjdk=8
#################################

### ✅ mPLUG-Owl 配置示例
conda create -n lmms-mplug python=3.10
conda activate lmms-mplug

git clone https://github.com/X-PLUG/mPLUG-Owl.git
cd mPLUG-Owl/mPLUG-Owl2
pip install --upgrade pip && pip install -e .

git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval && pip install -e .

conda install openjdk=8
pip install deepspeed  # 可选：加速推理

~~~



------

## 🔄 融合脚本说明

> 脚本命名规则：`xxx2yyy.py` 表示将模型 `xxx` 的参数融合进 `yyy` 架构。

### 📈 线性插值脚本

| 源模型               | 目标模型 | 脚本文件               |
| -------------------- | -------- | ---------------------- |
| LLaVA                | CogVLM   | `llava2cogvlm.py`      |
| mPLUG-Owl            | CogVLM   | `mplugowl2cogvlm.py`   |
| LLaVA-OneVision-Qwen | QwenVL2  | `llava-qwen2qwenvl.py` |

### 🧬 非线性融合（Baseline）

| 源模型               | 目标模型 | 脚本文件                            |
| -------------------- | -------- | ----------------------------------- |
| LLaVA                | CogVLM   | `llava2cogvlm_ties_merging.py`      |
| mPLUG-Owl            | CogVLM   | `mplugowl2cogvlm_ties_merging.py`   |
| LLaVA-OneVision-Qwen | QwenVL2  | `llava-qwen2qwenvl_ties_merging.py` |

------

## ⚙️ 模型融合与推理

> 📝 可参考 `runs/` 中的历史脚本，记录推理结果便于后续评估 alpha。

### 🧪 执行融合脚本

```bash
conda activate lmms-cogvlm
python $MERGE_SCRIPT --output $ckpt_path --alpha $alpha \
       --base $BASE_MODEL_PATH --base_llava $LLAVA_PATH \
       --interpolation
```

### 🚀 批量测试多个 alpha（0.4~1.0）

```bash
#!/bin/bash

for alpha in 1.0 0.9 0.8 0.7 0.6 0.5 0.4; do
    echo "===> 当前 alpha: $alpha"
    # 融合
    python3 $MERGE_SCRIPT --output $ckpt_path --alpha $alpha --interpolation \
        --base COGVLM_PATH --llava_base LLAVA_PATH

    # 评测
    for task in "mme" "mmmu_val" "nocaps_val" "vizwiz_vqa_val" ...; do
        CUDA_VISIBLE_DEVICES=$GPU accelerate launch \
            --num_processes=1 \
            -m lmms_eval \
            --model cogvlm \
            --model_args pretrained=$ckpt_path,... \
            --tasks $task \
            --log_samples \
            --output_path $output_path
    done

    # 清除临时模型
    rm -rf $ckpt_path
done
```

------

## 🔍 搜索最优 alpha

完成多轮评测后，使用以下脚本自动选择最优 alpha：

```bash
python search/view_log_delta_perdata_search_limit.py
```

输出内容包括最优 alpha 及其对应的评测结果。

------

## 🧩 融合逻辑详解（以 `llava2cogvlm.py` 为例）

### 1️⃣ 加载参数

- 判断是否参与融合：`need_merge(key)`

- 缩放融合模型参数：

  ```python
  cogvlm_diff[key] = (cogvlm_chat[key] * alpha)
  ```

### 2️⃣ 参数融合

- 线性插值：

  ```python
  cogvlm_diff['lm_head.weight'] += llava['lm_head.weight']
  ```

- 非线性融合：调用 `ties_merging.py` 中的 `do_merging` 或 `do_merging_strategy`。

### 3️⃣ 保存参数

- 兼容 `torch` 与 `safetensors` 格式。
- `safetensors` 保存需添加 `metadata`。

------

## 🤝 贡献与反馈

欢迎大家提交 PR 或 Issue，我们非常期待社区的反馈与改进建议！🌟
 本项目旨在为多模态大模型的研究和部署提供更高效的融合方法，希望对您的工作有所帮助！

------

## 📄 引用

如果本项目对你有帮助，请引用我们的论文：

```bibtex
@misc{adamms2025,
  title={AdaMMS: Model Merging for Heterogeneous Multimodal Large Language Models with Unsupervised Coefficient Optimization},
  author={Yiyang Du, Xioachen Wang},
  booktitle={CVPR},
  year={2025}
}

```