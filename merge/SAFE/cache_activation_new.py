"""
Cache neuron activations per layer for a VLMEvalKit-supported local model on a mixed HF meta-probe dataset.

What this script does
- Build a small "meta probe" dataset in the my_llava-qwen2qwenvl_mm_v3.n.1.1.py style (currently: MMBench EN test split, streaming)
- Instantiate a VLM from vlmeval.config.supported_VLM (transformers/local models only; API models are not supported)
- Locate underlying torch.nn.Module and register forward hooks on selected target modules
- For each sample, build a VLMEvalKit-style message [image, text] and call vlm.generate(...) to trigger forwards
- Aggregate input/output activations by summing across tokens and averaging at the end
- Save a dict {module_name: {input: 1D tensor, output: 1D tensor}} to a .pt file

Notes
- Hooks: input looks for kwargs['hidden_states'] or first tensor arg; output uses output or output[0]
- Pooling: flatten features to [tokens, hidden] and sum across token dim; track token counts to average
- Only torch local models are supported. API models (e.g., OpenAI, Gemini) cannot be hooked
- We follow VLMEvalKit inference behaviors for WORLD_SIZE unsetting and SKIP_ERR handling

Example
python cache_activation_new.py \
  --model mPLUG-Owl2 \
  --n-mmbench 64 \
  --max-samples 128 \
  --req-act input output \
  --module-regex "mlp\\.|self_attn\\.|down_proj|up_proj|gate_proj|q_proj|k_proj|v_proj|o_proj|dense|fc|ffn" \
  --include-types Linear LayerNorm \
  --save activations/mmplug_meta.pt \
  --verbose
"""

from __future__ import annotations

import argparse
import functools
import gc
import os
import os.path as osp
import re
import uuid
import warnings
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from datasets import load_dataset  # huggingface datasets
except Exception as e:  # pragma: no cover
    load_dataset = None

# VLMEvalKit imports
from vlmeval.config import supported_VLM


# ---------------------------
# Dataset construction (HF)
# ---------------------------
def _create_meta_probe_dataset(n_mmbench: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Build a meta-probe dataset in the style of my_llava-qwen2qwenvl_mm_v3.n.1.1.py.

    Currently includes:
    - MMBench EN test (streaming): questions are MCQ; we format options into the question text

    Returns a list of samples, each: {"image": PIL.Image.Image, "question": str, "answer": Optional[str]}
    """
    if load_dataset is None:
        raise RuntimeError("HuggingFace datasets is not installed. Please `pip install datasets`.")

    meta_probe_samples: List[Dict[str, Any]] = []
    if n_mmbench and n_mmbench > 0:
        print(f"[MetaProbe] Loading {n_mmbench} samples from lmms-lab/MMBench (en, test, streaming)...")
        ds = load_dataset("lmms-lab/MMBench", "en", split="test", streaming=True)
        ds = ds.shuffle(seed=seed).take(n_mmbench)
        for item in ds:
            question = item.get("question", "")
            # format options
            options_parts = []
            for opt_key in ["A", "B", "C", "D", "E", "F"]:
                if opt_key in item and item[opt_key] is not None and item[opt_key] != "":
                    options_parts.append(f"{opt_key}. {item[opt_key]}")
            options = "\n".join(options_parts)

            if item.get("hint"):
                full_q = f"{item['hint']}\n{question}\n{options}" if options else f"{item['hint']}\n{question}"
            else:
                full_q = f"{question}\n{options}" if options else question

            meta_probe_samples.append({
                "image": item["image"],  # PIL image from HF datasets
                "question": full_q,
                "answer": item.get("answer")
            })
        # ensure iterator is freed
        del ds

    if len(meta_probe_samples) == 0:
        raise RuntimeError("No samples constructed. Please increase --n-mmbench or extend the loader.")

    return meta_probe_samples


# ---------------------------
# Utilities for model & hooks
# ---------------------------
def get_underlying_torch_model(vlm_obj) -> Optional[nn.Module]:
    """Try to retrieve the underlying torch.nn.Module from a VLMEvalKit model wrapper."""
    if hasattr(vlm_obj, "model") and isinstance(getattr(vlm_obj, "model"), nn.Module):
        return getattr(vlm_obj, "model")
    if isinstance(vlm_obj, nn.Module):
        return vlm_obj
    return None


def _type_filter(include_types: List[str]) -> Optional[Tuple[type, ...]]:
    if not include_types:
        return None
    # map provided names to classes when available; otherwise compare by __name__ later
    classes: List[type] = []
    for name in include_types:
        if not name:
            continue
        if hasattr(nn, name):
            cls = getattr(nn, name)
            if isinstance(cls, type):
                classes.append(cls)
    return tuple(classes) if classes else None


def get_target_module_map(
    model: nn.Module,
    module_regex: str,
    include_types: List[str],
    exclude_regex: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, nn.Module]:
    """Select named modules from the model using regex and type filters."""
    inc_pat = re.compile(module_regex) if module_regex else None
    exc_pat = re.compile(exclude_regex) if exclude_regex else None
    include_type_tuple = _type_filter(include_types)
    include_names = set(t for t in include_types) if include_types else set()

    selected: Dict[str, nn.Module] = {}
    for name, mod in model.named_modules():
        # leaf modules often suffice
        # still allow non-leaf if it matches and is a known op type
        if inc_pat and not inc_pat.search(name):
            continue
        if exc_pat and exc_pat.search(name):
            continue

        ok_type = True
        if include_type_tuple is not None:
            ok_type = isinstance(mod, include_type_tuple)
        elif include_names:
            ok_type = (mod.__class__.__name__ in include_names)

        if not ok_type:
            continue

        selected[name] = mod

    if verbose:
        print(f"[HookSelect] Matched {len(selected)} modules.")
        for k in sorted(selected.keys())[:50]:
            print("  ", k, "|", selected[k].__class__.__name__)
        if len(selected) > 50:
            print("  ...")

    if not selected:
        raise RuntimeError("No modules matched. Please relax --module-regex or --include-types.")
    return selected


def make_activation_hook(name: str, req_act: Iterable[str], activation_stats: dict):
    """Forward hook function factory collecting input/output sums and token counts."""
    def hook_fn(module, args, kwargs, output):
        # output activations
        if "output" in req_act:
            out_tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(out_tensor, torch.Tensor):
                t = out_tensor.detach().cpu().float()
                t2 = t.reshape(-1, t.shape[-1])
                s = torch.sum(t2, dim=0)
                if activation_stats[name]["output_sum"] is None:
                    activation_stats[name]["output_sum"] = s
                else:
                    activation_stats[name]["output_sum"] += s
                activation_stats[name]["output_tokens"] += t2.shape[0]

        # input activations
        if "input" in req_act:
            in_tensor = kwargs.get("hidden_states", args[0] if args and isinstance(args[0], torch.Tensor) else None)
            if isinstance(in_tensor, torch.Tensor):
                t = in_tensor.detach().cpu().float()
                t2 = t.reshape(-1, t.shape[-1])
                s = torch.sum(t2, dim=0)
                if activation_stats[name]["input_sum"] is None:
                    activation_stats[name]["input_sum"] = s
                else:
                    activation_stats[name]["input_sum"] += s
                activation_stats[name]["input_tokens"] += t2.shape[0]

    return hook_fn


# ---------------------------
# Image dump helper
# ---------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def dump_pil_to_file(img, work_dir: str) -> str:
    img_dir = osp.join(work_dir, "tmp_images")
    ensure_dir(img_dir)
    fname = f"{uuid.uuid4().hex}.png"
    fpath = osp.join(img_dir, fname)
    # normalize mode
    try:
        from PIL import Image
        if isinstance(img, Image.Image):
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img.save(fpath)
        else:
            # datasets may return PIL already; otherwise try .save
            img.save(fpath)
    except Exception:
        # fallback via torchvision if available
        try:
            import torchvision.transforms.functional as TF
            TF.to_pil_image(img).save(fpath)
        except Exception as e:
            raise RuntimeError(f"Failed to dump image: {e}")
    return fpath


# ---------------------------
# Main logic
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache layer activations (meta-probe HF dataset) for a VLMEvalKit model")
    p.add_argument("--model", type=str, required=True, help="Model name in vlmeval.config.supported_VLM")
    p.add_argument("--n-mmbench", type=int, default=64, help="Number of samples to take from MMBench (streaming)")
    p.add_argument("--max-samples", type=int, default=128, help="Global limit on total samples processed")
    p.add_argument("--module-regex", type=str, default=r"mlp\.|self_attn\.|attention\.|down_proj|up_proj|gate_proj|q_proj|k_proj|v_proj|o_proj|dense|fc|ffn",
                   help="Regex to select module names (from model.named_modules())")
    p.add_argument("--include-types", nargs="*", default=["Linear"],
                   help="Optional nn.Module class names to include (e.g., Linear LayerNorm Conv2d); empty=all")
    p.add_argument("--exclude-regex", type=str, default=r"lm_head|embed|embedding",
                   help="Regex to exclude module names")
    p.add_argument("--req-act", nargs="+", choices=["input", "output"], default=["output"],
                   help="Which activations to record")
    p.add_argument("--save", type=str, default=None, help="Path to save .pt (default activations/{model}_meta.pt)")
    p.add_argument("--work-dir", type=str, default=".", help="Work directory (for temp images, etc.)")
    p.add_argument("--use-vllm", action="store_true", help="Pass use_vllm to Llama-4/Qwen2-VL/Qwen2.5-VL series")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    work_dir = args.work_dir
    ensure_dir(work_dir)
    ensure_dir(osp.join(work_dir, "tmp_images"))

    # Build meta-probe dataset (HF style)
    samples = _create_meta_probe_dataset(n_mmbench=args.n_mmbench, seed=args.seed)
    if args.max_samples and len(samples) > args.max_samples:
        samples = samples[: args.max_samples]

    # Instantiate model (VLMEvalKit style, compatible with inference.py semantics)
    if args.model not in supported_VLM:
        raise KeyError(f"Unknown model '{args.model}'. Check vlmeval/config.py supported_VLM.")

    ws_bak = os.environ.pop("WORLD_SIZE", None)
    model_kwargs: Dict[str, Any] = {}
    if args.model is not None and (
        "Llama-4" in args.model or "Qwen2-VL" in args.model or "Qwen2.5-VL" in args.model
    ):
        model_kwargs = {"use_vllm": args.use_vllm}

    constructor = supported_VLM[args.model]
    if isinstance(constructor, functools.partial):
        kw = dict(constructor.keywords or {})
        if "model_path" in kw and isinstance(kw["model_path"], str):
            kw["model_path"] = kw["model_path"].strip()
        constructor = functools.partial(constructor.func, *(constructor.args or ()), **kw)
    vlm = constructor(**model_kwargs) if "constructor" in locals() else supported_VLM[args.model](**model_kwargs)
    if ws_bak:
        os.environ["WORLD_SIZE"] = ws_bak

    if getattr(vlm, "is_api", False):
        raise RuntimeError("API models are not supported for activation caching (no torch hooks).")

    # Try to set dump_image if wrapper uses it (not required here, we pre-dump to paths)
    if hasattr(vlm, "set_dump_image"):
        def _dump_image_adapter(line: Any) -> str:
            # Accepts dict with 'image' holding a PIL Image
            img = line["image"] if isinstance(line, dict) and "image" in line else line
            return dump_pil_to_file(img, work_dir)
        try:
            vlm.set_dump_image(_dump_image_adapter)
        except Exception:
            pass

    torch_model = get_underlying_torch_model(vlm)
    if torch_model is None:
        raise RuntimeError("Cannot find underlying torch model to hook (no .model and wrapper is not nn.Module)")
    torch_model.eval()

    # Select target modules and register hooks
    target_modules = get_target_module_map(
        torch_model,
        module_regex=args.module_regex,
        include_types=args.include_types,
        exclude_regex=args.exclude_regex,
        verbose=args.verbose,
    )

    activation_stats = defaultdict(lambda: {
        "input_sum": None, "input_tokens": 0,
        "output_sum": None, "output_tokens": 0,
    })

    hooks = []
    for name, module in target_modules.items():
        h = module.register_forward_hook(make_activation_hook(name, args.req_act, activation_stats), with_kwargs=True)
        hooks.append(h)

    # Iterate samples; build VLMEvalKit message and call vlm.generate to trigger forwards
    dataset_id = "HF:meta_probe"
    processed = 0
    for i, item in enumerate(tqdm(samples, desc=f"Forwarding {args.model} on {dataset_id}")):
        img = item.get("image")
        if img is None:
            continue
        img_path = dump_pil_to_file(img, work_dir)
        struct = [
            {"type": "image", "value": img_path},
            {"type": "text", "value": item.get("question", "") or ""},
        ]

        # With SKIP_ERR handling consistent with vlmeval.inference
        if os.environ.get("SKIP_ERR", "0") == "1":
            try:
                _ = vlm.generate(message=struct, dataset=dataset_id)
            except RuntimeError as err:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                warnings.warn(f"generation failed at sample {i}: {type(err).__name__}: {err}")
        else:
            _ = vlm.generate(message=struct, dataset=dataset_id)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        processed += 1

    # Remove hooks
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    # Average activations
    averaged_activations: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, stats in activation_stats.items():
        averaged_activations[name] = {}
        if stats["input_sum"] is not None and stats["input_tokens"] > 0:
            averaged_activations[name]["input"] = stats["input_sum"] / stats["input_tokens"]
        if stats["output_sum"] is not None and stats["output_tokens"] > 0:
            averaged_activations[name]["output"] = stats["output_sum"] / stats["output_tokens"]

    # Save
    save_path = args.save or osp.join("activations", f"{args.model}_meta.pt")
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    torch.save(averaged_activations, save_path)

    # Cleanup
    del vlm, torch_model, hooks, activation_stats, averaged_activations
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if args.verbose:
        print(f"[Done] Processed {processed} samples. Saved activations to: {save_path}")


if __name__ == "__main__":
    main()
