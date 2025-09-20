#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cache neuron activations per layer for a VLMEvalKit-supported model on a chosen dataset.

This script blends two ideas:
- Load models via VLMEvalKit (supported_VLM) in the same safe manner as inference.py
- Load a meta probing dataset from Hugging Face (similar to my_llava-* script), and cache activations via forward hooks

Key features:
- Pick a VLMEvalKit-supported local model (API models are not supported since we need torch hooks)
- Choose "--hf-dataset meta" to build a mixed probing dataset (MMBench, VCR, DocVQA, VQAv2, ScienceQA, ST-VQA)
- Or fallback to VLMEvalKit datasets via --data
- Register forward hooks on selected target modules (regex + optional class filters)
- Run generation to trigger forwards; aggregate input/output activations (sum over token dimension) and average
- Save a dictionary {module_name: {input: 1D tensor, output: 1D tensor}}

Example:
  python cache_activation_new.py \
    --model mPLUG-Owl2 \
    --hf-dataset meta \
    --n-mmbench 50 --n-vqa 50 \
    --req-act input output \
    --module-regex "mlp\\.|self_attn\\.|down_proj|up_proj|gate_proj|q_proj|k_proj|v_proj|o_proj|dense|fc|ffn" \
    --save activations/mmplug-meta.pt

Notes:
- Only local torch models are supported (API models cannot be hooked).
- Hooks try kwargs['hidden_states'] first or args[0] as input tensor.
- Output is assumed to be a Tensor or first element of a tuple, pooled by flattening to [tokens, hidden] and summing.
- When using --hf-dataset meta, images are saved to a tmp folder as files and referenced by path in messages.

Tip: run scripts/pre_download_hf_meta.py beforehand to cache HF datasets locally and avoid long waits at first run.

New (added):
- Support caching activations for pure-text LLMs loaded via Hugging Face transformers (e.g., meta-llama/Llama-2-7b-hf).
- Use only the textual part of datasets (questions + choices/hints) and ignore images.
- Example for LLM:
        python cache_activation_new.py \
            --hf-llm-id meta-llama/Llama-2-7b-hf \
            --hf-dataset meta \
            --n-mmbench 50 \
            --req-act input output \
            --module-regex "mlp\\.|self_attn\\.|down_proj|up_proj|gate_proj|q_proj|k_proj|v_proj|o_proj|dense|fc|ffn" \
            --probe-batch-size 4 --llm-max-length 1024
"""

from __future__ import annotations

import argparse
import functools
import gc
import os
import os.path as osp
import random
import re
import uuid
import warnings
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

# VLMEvalKit imports
from vlmeval.config import supported_VLM
try:
    from vlmeval.dataset import build_dataset as vlmeval_build_dataset
except Exception:
    vlmeval_build_dataset = None

# HF datasets
try:
    from datasets import load_dataset, DownloadConfig
except Exception as e:
    load_dataset = None
    _DATASETS_IMPORT_ERR = e
else:
    _DATASETS_IMPORT_ERR = None

try:
    from PIL import Image
except Exception:
    Image = None

# New: optional transformers import for pure LLM path
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None

# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache layer activations for a VLM on a dataset (meta/HF or VLMEval)")
    # model/dataset
    # Note: either --model (VLMEvalKit) or --hf-llm-id (transformers) should be provided
    parser.add_argument("--model", required=False, type=str, help="Model name key in supported_VLM (vlmeval/config.py)")
    parser.add_argument("--hf-llm-id", type=str, default=None,
                        help="HuggingFace model id or local path for transformers AutoModelForCausalLM; e.g., meta-llama/Llama-2-7b-hf")
    parser.add_argument("--data", required=False, type=str, default=None,
                        help="Dataset name supported by VLMEvalKit (ignored if --hf-dataset is set)")
    parser.add_argument("--hf-dataset", type=str, default=None,
                        help="Special HF loader key; currently supports: 'meta'. If set, overrides --data")

    # meta dataset knobs
    parser.add_argument("--n-mmbench", type=int, default=0, help="Samples to draw from MMBench (en/test)")
    parser.add_argument("--n-vcr", type=int, default=0, help="Samples to draw from VCR (validation, Q->A)")
    parser.add_argument("--n-docvqa", type=int, default=0, help="Samples to draw from DocVQA (validation)")
    parser.add_argument("--n-vqa", type=int, default=0, help="Samples to draw from VQAv2 (validation)")
    parser.add_argument("--n-scienceqa", type=int, default=0, help="Samples to draw from ScienceQA (validation, has image)")
    parser.add_argument("--n-stvqa", type=int, default=0, help="Samples to draw from ST-VQA task1 (test)")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap total samples (after composition)")
    # 新增：HF 下载与镜像控制
    parser.add_argument("--hf-no-streaming", action="store_true",
                        help="Disable datasets streaming mode; use regular download (more stable, larger download).")
    parser.add_argument("--hf-endpoint", type=str, default=None,
                        help="Override HF_ENDPOINT (e.g., https://huggingface.co or a mirror). Use 'disable' to unset.")
    parser.add_argument("--hf-offline", action="store_true",
                        help="Force offline mode for HF datasets (use local cache only); implies --hf-no-streaming.")

    # hook + selection
    parser.add_argument("--req-act", nargs="+", default=["output"], choices=["input", "output"],
                        help="Which activations to record: input/output (one or both)")
    parser.add_argument("--module-regex", type=str,
                        default=r"mlp\. |self_attn\.|attention\.|down_proj|up_proj|gate_proj|q_proj|k_proj|v_proj|o_proj|dense|fc|ffn".replace(" ", ""),
                        help="Regex to select modules by name. Applied to named_modules() full path.")
    parser.add_argument("--include-types", nargs="*", default=["Linear"],
                        help="Optional nn.Module class name filters, e.g. Linear Conv2d LayerNorm; empty=all")
    parser.add_argument("--exclude-regex", type=str, default=r"lm_head|embed|embedding",
                        help="Regex to exclude modules by name")

    # misc
    parser.add_argument("--work-dir", type=str, default=".", help="Work dir for tmp files")
    parser.add_argument("--save", type=str, default=None, help="Output .pt file path; default under activations/")
    parser.add_argument("--verbose", action="store_true", help="Print progress and matched modules")
    parser.add_argument("--use-vllm", action="store_true",
                        help="Pass use_vllm to certain models (e.g., Llama-4, Qwen2-VL series)")

    # New: transformers LLM runtime knobs
    parser.add_argument("--llm-device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for transformers LLM (cuda/cpu)")
    parser.add_argument("--llm-dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="torch dtype for transformers LLM load; auto=HF default")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Pass trust_remote_code=True to transformers.from_pretrained")
    parser.add_argument("--probe-batch-size", type=int, default=4,
                        help="Batch size for forward pass when using transformers LLM (text-only)")
    parser.add_argument("--llm-max-length", type=int, default=1024,
                        help="Max token length for tokenizer(truncation)")

    return parser.parse_args()


# -------------------------
# Meta dataset builder (HF)
# -------------------------

def _ensure_hf_import():
    if load_dataset is None:
        raise RuntimeError(
            f"datasets is not available for --hf-dataset; please install `datasets`. Import error: {_DATASETS_IMPORT_ERR}"
        )


def _dump_image_to_file(img: Any, root: str) -> str:
    os.makedirs(root, exist_ok=True)
    # Convert to RGB if it's a PIL Image with alpha
    if Image is not None and isinstance(img, Image.Image):
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            try:
                img = img.convert('RGB')
            except Exception:
                pass
    # save
    fname = f"{uuid.uuid4().hex}.jpg"
    path = osp.join(root, fname)
    try:
        if Image is not None and isinstance(img, Image.Image):
            img.save(path, format='JPEG', quality=95)
        else:
            # datasets Image feature returns PIL Image; if not, try array-like
            from PIL import Image as _PILImage
            _PILImage.fromarray(img).save(path, format='JPEG', quality=95)
    except Exception as e:
        raise RuntimeError(f"Failed to save image to {path}: {e}")
    return path


def build_meta_probe_dataset(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Compose a meta probing dataset by sampling from several HF datasets.

    Returns list of dict with keys: {image (PIL), question (str), optional answer/answers}.
    """
    _ensure_hf_import()
    # 若指定端点，优先设置（可覆盖全局环境的镜像设置）
    if args.hf_endpoint is not None:
        if args.hf_endpoint.strip().lower() == "disable":
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = args.hf_endpoint.strip()

    # Offline/streaming config
    offline = bool(getattr(args, "hf_offline", False))
    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    streaming = False if offline else (not getattr(args, "hf_no_streaming", False))
    dl_cfg = DownloadConfig(
        max_retries=0 if offline else 10,
        resume_download=not offline,
        use_etag=not offline,
        local_files_only=offline,
    )

    def _set_ep(ep: Optional[str]):
        if ep is None:
            return
        if ep.strip().lower() == "disable":
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = ep.strip()

    def _load_ds(path, name=None, split="validation"):
        # 在当前端点失败时，自动回退到官方端点，并在需要时回退到非流式
        current_ep = os.environ.get("HF_ENDPOINT")
        trial_eps = [current_ep, "https://huggingface.co"]
        seen = set(); ordered_eps = []
        for ep in trial_eps:
            key = ep or "__none__"
            if key in seen: continue
            seen.add(key); ordered_eps.append(ep)

        last_exc = None
        for ep in ordered_eps:
            _set_ep(ep)
            ep_str = ep or os.environ.get("HF_ENDPOINT") or "<default>"
            for sflag in [streaming, False] if streaming else [False]:
                try:
                    print(f"[Meta] try load {path} ({name or ''}) split={split} streaming={sflag} ep={ep_str}")
                    if name is None:
                        return load_dataset(path, split=split, streaming=sflag, download_config=None if sflag else dl_cfg)
                    else:
                        return load_dataset(path, name, split=split, streaming=sflag, download_config=None if sflag else dl_cfg)
                except Exception as e:
                    last_exc = e
                    print(f"[Meta][warn] load failed at ep={ep_str} streaming={sflag}: {type(e).__name__}: {e}")
        raise RuntimeError(f"[Meta] Failed to load {path} ({name or ''}) split={split} after retries: {last_exc}")

    def _yield_first_n(path: str, name: Optional[str], split: str, n: int):
        """Robustly yield first n items with retries across endpoint and streaming modes.

        This handles failures that occur during iteration (common in streaming mode) by
        switching to non-streaming and/or the official endpoint.
        """
        if n <= 0:
            return
        current_ep = os.environ.get("HF_ENDPOINT")
        trial_eps = [current_ep, "https://huggingface.co"]
        seen = set(); ordered_eps = []
        for ep in trial_eps:
            key = ep or "__none__"
            if key in seen: continue
            seen.add(key); ordered_eps.append(ep)

        # try combinations: requested streaming first, then non-streaming; across endpoints
        attempts = []
        if streaming:
            attempts.append(True)
        attempts.append(False)

        for ep in ordered_eps:
            _set_ep(ep)
            ep_str = ep or os.environ.get("HF_ENDPOINT") or "<default>"
            for sflag in attempts:
                try:
                    print(f"[Meta] try iterate {path} ({name or ''}) split={split} streaming={sflag} ep={ep_str}")
                    # load
                    if name is None:
                        ds = load_dataset(path, split=split, streaming=sflag, download_config=None if sflag else dl_cfg)
                    else:
                        ds = load_dataset(path, name, split=split, streaming=sflag, download_config=None if sflag else dl_cfg)
                    # iterate
                    cnt = 0
                    if sflag:
                        for item in ds.shuffle(seed=42).take(n):
                            yield item
                            cnt += 1
                        if cnt >= n:
                            return
                    else:
                        dss = ds.shuffle(seed=42)
                        n_eff = min(n, len(dss))
                        if n_eff <= 0:
                            return
                        for item in dss.select(range(n_eff)):
                            yield item
                        return
                except Exception as e:
                    print(f"[Meta][warn] iterate failed at ep={ep_str} streaming={sflag}: {type(e).__name__}: {e}")
                    continue
        raise RuntimeError(f"[Meta] Failed to iterate dataset {path} ({name or ''}) split={split} for first {n} samples after retries")

    meta_probe_samples: List[Dict[str, Any]] = []

    # 1) MMBench EN (test)
    if getattr(args, 'n_mmbench', 0) > 0:
        print(f"[Meta] Loading {args.n_mmbench} from MMBench (en/test)...")
        for item in _yield_first_n("lmms-lab/MMBench", "en", "test", args.n_mmbench):
            q = item['question']
            options = []
            for key in ['A', 'B', 'C', 'D', 'E', 'F']:
                if key in item and item[key] is not None:
                    options.append(f"{key}. {item[key]}")
            options_str = "\n".join(options)
            if item.get('hint'):
                full_q = f"{item['hint']}\n{q}\n{options_str}" if options_str else f"{item['hint']}\n{q}"
            else:
                full_q = f"{q}\n{options_str}" if options_str else q
            meta_probe_samples.append({
                "image": item["image"],
                "question": full_q,
                "answer": item.get("answer", None)
            })

    # 2) VCR (validation, Q->A)
    if getattr(args, 'n_vcr', 0) > 0:
        print(f"[Meta] Loading {args.n_vcr} from VCR (validation, Q->A)...")
        for item in _yield_first_n("pingzhili/vcr-qa", None, "validation", args.n_vcr):
            q = item['question']
            choices = item.get('answer_choices', [])
            choices_str = "\n".join([f"- {c}" for c in choices])
            full_q = f"{q}\n\nChoices:\n{choices_str}" if choices_str else q
            label = item.get('answer_label', None)
            correct_text = choices[label] if (isinstance(label, int) and 0 <= label < len(choices)) else None
            meta_probe_samples.append({
                "image": item["image"],
                "question": full_q,
                "answer": correct_text
            })

    # 3) DocVQA (validation)
    if getattr(args, 'n_docvqa', 0) > 0:
        print(f"[Meta] Loading {args.n_docvqa} from DocVQA (validation)...")
        for item in _yield_first_n("lmms-lab/DocVQA", "DocVQA", "validation", args.n_docvqa):
            meta_probe_samples.append({
                "image": item["image"],
                "question": item["question"],
                "answers": item.get("answers", None)
            })

    # 4) VQAv2 (validation)
    if getattr(args, 'n_vqa', 0) > 0:
        print(f"[Meta] Loading {args.n_vqa} from VQAv2 (validation)...")
        for item in _yield_first_n("lmms-lab/VQAv2", None, "validation", args.n_vqa):
            meta_probe_samples.append({
                "image": item["image"],
                "question": item["question"],
            })

    # 5) ScienceQA (validation, has image) 仅非流式
    if getattr(args, 'n_scienceqa', 0) > 0:
        print(f"[Meta] Loading {args.n_scienceqa} from ScienceQA (validation, has image)...")
        ds = _load_ds("derek-thomas/ScienceQA", None, "validation")
        if streaming:
            # streaming 模式下 filter 不便，直接顺序筛选
            cnt = 0
            for item in ds:
                if item.get('image') is None:
                    continue
                hint = item.get('hint', None)
                q = item.get('question', '')
                full_q = f"{hint} {q}" if hint else q
                meta_probe_samples.append({"image": item["image"], "question": full_q})
                cnt += 1
                if cnt >= args.n_scienceqa:
                    break
        else:
            ds_img = ds.filter(lambda x: x.get('image') is not None)
            dss = ds_img.shuffle(seed=42)
            n_eff = min(args.n_scienceqa, len(dss))
            for item in dss.select(range(n_eff)):
                hint = item.get('hint', None)
                q = item.get('question', '')
                full_q = f"{hint} {q}" if hint else q
                meta_probe_samples.append({"image": item["image"], "question": full_q})
        del ds

    # 6) ST-VQA task1 (test)
    if getattr(args, 'n_stvqa', 0) > 0:
        print(f"[Meta] Loading {args.n_stvqa} from ST-VQA task1 (test)...")
        for item in _yield_first_n("danjacobellis/stvqa_task1", None, "test", args.n_stvqa):
            meta_probe_samples.append({"image": item["image"], "question": item["question"]})

    random.shuffle(meta_probe_samples)
    if args.max_samples is not None:
        meta_probe_samples = meta_probe_samples[: args.max_samples]
    print(f"[Meta] Built meta probing dataset, total samples: {len(meta_probe_samples)}")
    return meta_probe_samples


# -------------------------
# Utilities for text-only prompts (LLM path)
# -------------------------

def _get_texts_from_meta_samples(samples: List[Dict[str, Any]]) -> List[str]:
    """Extract text prompts from meta samples (ignore images).

    Each sample has 'question' already including choices/hints if applicable.
    """
    texts = []
    for s in samples:
        q = s.get("question", "")
        if not isinstance(q, str):
            q = str(q)
        texts.append(q)
    return texts

def _extract_texts_from_vlmeval_dataset(dataset, max_n: Optional[int]) -> List[str]:
    """Build text prompts from a VLMEval dataset by concatenating text segments in messages.

    This is a best-effort extraction: we collect all dicts with type='text' and join them.
    """
    data = dataset.data
    max_n = len(data) if max_n is None else min(len(data), max_n)
    texts: List[str] = []
    for i in range(max_n):
        struct = dataset.build_prompt(data.iloc[i])  # list of {type: 'image'|'text', value:...}
        parts = []
        for seg in struct:
            if isinstance(seg, dict) and seg.get("type") == "text":
                parts.append(str(seg.get("value", "")))
        texts.append("\n".join([p for p in parts if p]))
    return texts


# -------------------------
# Torch model inspection & selection
# -------------------------

def get_underlying_torch_model(vlm_obj) -> Optional[nn.Module]:
    """Try to retrieve the underlying torch.nn.Module from a VLMEvalKit model wrapper.

    Many wrappers use attribute `model` to hold the HF/torch model. If not present but the
    wrapper itself is an nn.Module, return the wrapper. Otherwise return None.
    """
    if hasattr(vlm_obj, "model") and isinstance(getattr(vlm_obj, "model"), nn.Module):
        return getattr(vlm_obj, "model")
    if isinstance(vlm_obj, nn.Module):
        return vlm_obj
    return None


def _class_name(m: nn.Module) -> str:
    return m.__class__.__name__


def get_target_module_map(
    model: nn.Module,
    module_regex: str,
    include_types: List[str],
    exclude_regex: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, nn.Module]:
    pat = re.compile(module_regex)
    ex_pat = re.compile(exclude_regex) if exclude_regex else None
    allow_set = set(include_types or [])

    matched: Dict[str, nn.Module] = {}
    for name, mod in model.named_modules():
        if name == "":
            continue
        if not pat.search(name):
            continue
        if ex_pat and ex_pat.search(name):
            continue
        if allow_set and _class_name(mod) not in allow_set:
            continue
        matched[name] = mod

    if verbose:
        print(f"[Hook] Matched {len(matched)} modules:")
        for n, m in matched.items():
            print(f" - {n} ({_class_name(m)})")
    return matched


# -------------------------
# Hook function (input/output sum & average)
# -------------------------

def get_hook_with_kwargs(name: str, req_act: Iterable[str], activation_stats: dict):
    def hook_fn(module, args, kwargs, output):
        # Output
        if "output" in req_act:
            out_tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(out_tensor, torch.Tensor):
                t_float = out_tensor.detach().cpu().float()
                try:
                    t_reshaped = t_float.reshape(-1, t_float.shape[-1])
                except Exception:
                    return
                current_sum = torch.sum(t_reshaped, dim=0)
                if activation_stats[name]["output_sum"] is None:
                    activation_stats[name]["output_sum"] = current_sum
                else:
                    activation_stats[name]["output_sum"] += current_sum
                activation_stats[name]["output_tokens"] += t_reshaped.shape[0]
        # Input
        if "input" in req_act:
            in_tensor = kwargs.get("hidden_states", args[0] if (args and isinstance(args[0], torch.Tensor)) else None)
            if isinstance(in_tensor, torch.Tensor):
                t_float = in_tensor.detach().cpu().float()
                try:
                    t_reshaped = t_float.reshape(-1, t_float.shape[-1])
                except Exception:
                    return
                current_sum = torch.sum(t_reshaped, dim=0)
                if activation_stats[name]["input_sum"] is None:
                    activation_stats[name]["input_sum"] = current_sum
                else:
                    activation_stats[name]["input_sum"] += current_sum
                activation_stats[name]["input_tokens"] += t_reshaped.shape[0]
    return hook_fn


# -------------------------
# Main caching routine
# -------------------------

@torch.no_grad()
def main():
    args = parse_args()

    work_dir = args.work_dir
    os.makedirs(work_dir, exist_ok=True)
    tmp_img_dir = osp.join(work_dir, 'tmp_images')
    os.makedirs(tmp_img_dir, exist_ok=True)

    # -------- Determine mode: VLMEval VLM vs transformers LLM --------
    use_hf_llm = args.hf_llm_id is not None and len(args.hf_llm_id) > 0
    if not use_hf_llm and not args.model:
        raise ValueError("Please provide either --model (VLMEvalKit) or --hf-llm-id (transformers).")

    # -------- Build dataset payload --------
    dataset_id: str
    messages: List[List[Dict[str, Any]]] = []  # for VLM path (multi-modal messages)
    texts_for_llm: List[str] = []              # for LLM path (plain text)

    if args.hf_dataset is not None:
        key = args.hf_dataset.strip().lower()
        if key != 'meta':
            raise ValueError("--hf-dataset currently only supports 'meta'")
        if load_dataset is None:
            raise RuntimeError(
                f"datasets is not available; please install `datasets`. Import error: {_DATASETS_IMPORT_ERR}"
            )
        samples = build_meta_probe_dataset(args)
        if use_hf_llm:
            # New: text-only prompts for transformers LLM
            texts_for_llm = _get_texts_from_meta_samples(samples)
        else:
            for s in samples:
                img = s["image"]
                # convert + save
                if Image is not None and isinstance(img, Image.Image) and img.mode == 'RGBA':
                    img = img.convert('RGB')
                img_path = _dump_image_to_file(img, tmp_img_dir)
                text = s.get("question", "")
                messages.append([
                    dict(type='image', value=img_path),
                    dict(type='text', value=text)
                ])
        dataset_id = 'HF:meta'
    elif args.data is not None:
        if vlmeval_build_dataset is None:
            raise RuntimeError("vlmeval.dataset.build_dataset is unavailable; cannot load --data dataset.")
        dataset = vlmeval_build_dataset(name=args.data, work_dir=work_dir)
        dataset_id = dataset.dataset_name
        if use_hf_llm:
            texts_for_llm = _extract_texts_from_vlmeval_dataset(dataset, args.max_samples)
        else:
            data = dataset.data
            max_n = len(data) if args.max_samples is None else min(len(data), args.max_samples)
            for i in range(max_n):
                struct = dataset.build_prompt(data.iloc[i])
                messages.append(struct)
    else:
        raise ValueError("Please specify either --hf-dataset meta or --data <DatasetName>.")

    # -------- Instantiate model (either VLMEvalKit VLM or transformers LLM) --------
    torch_model: Optional[nn.Module] = None
    vlm = None
    if use_hf_llm:
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise RuntimeError("transformers is not available; please install transformers to use --hf-llm-id")

        # parse dtype
        dt = args.llm_dtype
        torch_dtype = None
        if dt == "float16":
            torch_dtype = torch.float16
        elif dt == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dt == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = None  # auto

        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_llm_id, use_fast=True, trust_remote_code=args.trust_remote_code
        )
        # Ensure pad token exists for batching (common for LLaMA)
        if tokenizer.pad_token_id is None and hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_llm_id, torch_dtype=torch_dtype, trust_remote_code=args.trust_remote_code
        )
        device = torch.device(args.llm_device)
        model.to(device)
        model.eval()
        torch_model = model
    else:
        # VLMEvalKit (safe WORLD_SIZE handling)
        ws_bak = os.environ.pop('WORLD_SIZE', None)
        model_kwargs = {}
        model_name = args.model
        if model_name is not None and (
            'Llama-4' in model_name or 'Qwen2-VL' in model_name or 'Qwen2.5-VL' in model_name
        ):
            model_kwargs = {'use_vllm': args.use_vllm}

        constructor = supported_VLM[model_name]
        if isinstance(constructor, functools.partial):
            kw = dict(constructor.keywords or {})
            if 'model_path' in kw and isinstance(kw['model_path'], str):
                kw['model_path'] = kw['model_path'].strip()
            constructor = functools.partial(constructor.func, *(constructor.args or ()), **kw)
        vlm = constructor(**model_kwargs) if 'constructor' in locals() else supported_VLM[model_name](**model_kwargs)
        if ws_bak:
            os.environ['WORLD_SIZE'] = ws_bak

        if getattr(vlm, 'is_api', False):
            raise RuntimeError("API models are not supported for activation caching (no torch hooks).")

        # If we used a VLMEval dataset (not HF meta), let model know how to dump images if needed
        if args.hf_dataset is None and args.data is not None and hasattr(vlm, 'set_dump_image') and hasattr(dataset, 'dump_image'):
            vlm.set_dump_image(dataset.dump_image)

        torch_model = get_underlying_torch_model(vlm)
        if torch_model is None:
            raise RuntimeError("Cannot find underlying torch model to hook (no .model and wrapper is not nn.Module)")
        torch_model.eval()

    target_modules = get_target_module_map(
        torch_model,
        module_regex=args.module_regex,
        include_types=args.include_types or [],
        exclude_regex=args.exclude_regex,
        verbose=args.verbose,
    )
    if not target_modules:
        raise RuntimeError("No modules matched given --module-regex/--include-types/--exclude-regex filters.")

    activation_stats = defaultdict(lambda: {
        "input_sum": None, "input_tokens": 0,
        "output_sum": None, "output_tokens": 0
    })

    hooks = [
        module.register_forward_hook(
            get_hook_with_kwargs(name, args.req_act, activation_stats), with_kwargs=True
        )
        for name, module in target_modules.items()
    ]

    # -------- Run forwards to collect activations --------
    processed = 0
    err_count = 0
    if use_hf_llm:
        # New: batch text-only forwards with transformers
        assert torch_model is not None
        model = torch_model  # type: ignore
        device = torch.device(args.llm_device)
        # tokenizer defined above in LLM branch
        # To avoid mypy confusion, re-create here (safe, loads from cache)
        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_llm_id, use_fast=True, trust_remote_code=args.trust_remote_code
        )
        if tokenizer.pad_token_id is None and hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        bsz = max(1, int(args.probe_batch_size))
        for i in tqdm(range(0, len(texts_for_llm), bsz), desc=f"Forward LLM on {dataset_id}"):
            batch_texts = texts_for_llm[i:i+bsz]
            enc = tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=args.llm_max_length 
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            try:
                _ = model(**enc, use_cache=False)
            except RuntimeError as err:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if os.environ.get('SKIP_ERR', '0') == '1':
                    warnings.warn(f"forward failed: {type(err).__name__}: {err}")
                    err_count += 1
                else:
                    raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            processed += len(batch_texts)
    else:
        # Original: drive forwards via vlm.generate on multi-modal messages
        for struct in tqdm(messages, desc=f"Forward {args.model} on {dataset_id}"):
            if os.environ.get('SKIP_ERR', '0') == '1':
                try:
                    _ = vlm.generate(message=struct, dataset=dataset_id)
                except RuntimeError as err:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    warnings.warn(f"generation failed: {type(err).__name__}: {err}")
                    err_count += 1
            else:
                _ = vlm.generate(message=struct, dataset=dataset_id)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            processed += 1

    for h in hooks:
        h.remove()

    if args.verbose:
        print(f"Processed samples: {processed}, errors: {err_count}")

    # -------- Aggregate & save --------
    averaged_activations: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, stats in activation_stats.items():
        averaged_activations[name] = {}
        if stats["input_sum"] is not None and stats["input_tokens"] > 0:
            averaged_activations[name]["input"] = stats["input_sum"] / stats["input_tokens"]
        if stats["output_sum"] is not None and stats["output_tokens"] > 0:
            averaged_activations[name]["output"] = stats["output_sum"] / stats["output_tokens"]

    save_path = args.save
    if not save_path:
        os.makedirs('activations', exist_ok=True)
        suffix = (args.hf_dataset or args.data or 'unknown').replace('/', '_')
        save_path = osp.join('activations', f"{args.model}_{suffix}.pt")
    torch.save(averaged_activations, save_path)

    # Cleanup
    del hooks, activation_stats, averaged_activations
    if not use_hf_llm:
        del vlm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Done] Saved averaged activations to: {save_path}")


if __name__ == "__main__":
    main()
