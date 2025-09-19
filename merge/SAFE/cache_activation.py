"""
Cache neuron activations per layer for a VLMEvalKit-supported model on a chosen dataset.

This script:
- Instantiates a VLM from vlmeval.config.supported_VLM
- Builds a dataset via vlmeval.dataset.build_dataset
- Registers forward hooks on selected target modules
- Runs generation on up to N samples to trigger forwards
- Aggregates input/output activations (sum over tokens) and averages them
- Saves a dictionary {module_name: {input: 1D tensor, output: 1D tensor}}

Limitations:
- Only local torch models are supported (API models cannot be hooked).
- The hook looks for kwargs['hidden_states'] or first tensor arg as input.
- Output is assumed to be a Tensor or first element of a tuple, and we pool
  across non-feature dimensions by flattening to 2D [tokens, hidden] then summing.

Usage example:
  python cache_activation.py \
	--model mPLUG-Owl2 \
	--data MMBench_DEV_EN \
	--max-samples 50 \
	--req-act input output \
	--module-regex "mlp\\.|self_attn\\.|down_proj|up_proj|gate_proj|q_proj|k_proj|v_proj|o_proj|dense|fc" \
	--save activations/mmplug-owl2_MMBench_DEV_EN.pt
"""

from __future__ import annotations

import argparse
import gc
import os
import os.path as osp
from random import random
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple
import functools
from datasets import load_dataset
import torch
import torch.nn as nn
from tqdm import tqdm
# VLMEvalKit imports
from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Cache layer activations for a VLM on a dataset")
	parser.add_argument("--model", required=True, type=str, help="Model name key in supported_VLM (vlmeval/config.py)")
	parser.add_argument("--data", required=True, type=str, help="Dataset name supported by VLMEvalKit")
	parser.add_argument("--max-samples", type=int, default=100, help="Max number of samples from dataset to run")
	parser.add_argument("--req-act", nargs="+", default=["output"], choices=["input", "output"],
						help="Which activations to record: input/output (one or both)")
	parser.add_argument("--module-regex", type=str,
						default=r"mlp\.|self_attn\.|attention\.|down_proj|up_proj|gate_proj|q_proj|k_proj|v_proj|o_proj|dense|fc|ffn",
						help="Regex to select modules by name. Applied to named_modules() full path.")
	parser.add_argument("--include-types", nargs="*", default=["Linear"],
						help="Optional nn.Module class name filters, e.g. Linear Conv2d LayerNorm; empty=all")
	parser.add_argument("--exclude-regex", type=str, default=r"lm_head|embed|embedding",
						help="Regex to exclude modules by name")
	parser.add_argument("--work-dir", type=str, default=".", help="Work dir for intermediate files")
	parser.add_argument("--save", type=str, default=None, help="Output .pt file path; default under activations/")
	parser.add_argument("--verbose", action="store_true", help="Print progress and matched modules")
	parser.add_argument("--use-vllm", action="store_true", help="Pass use_vllm to certain models (e.g., Llama-4, Qwen2-VL series)")
	parser.add_argument("--only-llm", action="store_true", help="用于缓存LLM模型")
	return parser.parse_args()


def get_underlying_torch_model(vlm_obj) -> Optional[nn.Module]:
	"""Try to retrieve the underlying torch.nn.Module from a VLMEvalKit model wrapper.

	Many wrappers use attribute `model` to hold the HF/torch model. If not present but the
	wrapper itself is an nn.Module, return the wrapper. Otherwise return None.
	"""
	if hasattr(vlm_obj, "model") and isinstance(getattr(vlm_obj, "model"), nn.Module):
		return vlm_obj.model
	if isinstance(vlm_obj, nn.Module):
		return vlm_obj
	return None


def _class_name(m: nn.Module) -> str:
	return m.__class__.__name__


def _type_filter(include_types: List[str]) -> Optional[Tuple[type, ...]]:
	if not include_types:
		return None
	name_to_type = {cls.__name__: cls for cls in [
		nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
		nn.MultiheadAttention, nn.Embedding
	]}
	types: List[type] = []
	for n in include_types:
		if n in name_to_type:
			types.append(name_to_type[n])
	return tuple(types) if types else None


def get_target_module_map(model: nn.Module,
						  module_regex: str,
						  include_types: List[str],
						  exclude_regex: Optional[str] = None,
						  verbose: bool = False) -> Dict[str, nn.Module]:
	"""Enumerate named modules and select those matching filters.

	- module_regex: select by name regex
	- include_types: select by class names (subset of known nn layers); empty means all
	- exclude_regex: skip names matching this regex
	Returns name->module dict with unique modules (leaf modules preferred).
	"""
	name_pat = re.compile(module_regex) if module_regex else None
	excl_pat = re.compile(exclude_regex) if exclude_regex else None
	type_tuple = _type_filter(include_types)

	tgt: Dict[str, nn.Module] = {}
	for name, m in model.named_modules():
		if name == "":
			continue
		if name_pat and not name_pat.search(name):
			continue
		if excl_pat and excl_pat.search(name):
			continue
		if type_tuple is not None and not isinstance(m, type_tuple):
			continue
		# Prefer leaf modules to avoid double-counting parent containers
		is_leaf = len(list(m.children())) == 0
		if is_leaf:
			tgt[name] = m
	if verbose:
		print(f"[hook] Selected {len(tgt)} modules")
		for k in sorted(tgt.keys()):
			print(f"  - {k} ({_class_name(tgt[k])})")
	# Fallback: if nothing matched, take all Linear layers as a safe default
	if not tgt:
		for name, m in model.named_modules():
			if isinstance(m, nn.Linear):
				tgt[name] = m
		if verbose:
			print(f"[hook] Fallback to Linear layers: {len(tgt)} modules")
	return tgt


def get_hook_with_kwargs(name: str, req_act: Iterable[str], activation_stats: dict):
	"""Creates a forward hook that accumulates token-wise sums for input/output.

	The hook tries to read input hidden states from kwargs['hidden_states'] or the first tensor arg.
	The output is taken as the tensor itself or output[0] if tuple.
	"""

	def hook_fn(module: nn.Module, args, kwargs, output):
		try:
			if "output" in req_act:
				out_tensor = None
				if isinstance(output, tuple):
					out_tensor = output[0]
				elif isinstance(output, list):
					out_tensor = output[0] if len(output) and isinstance(output[0], torch.Tensor) else None
				elif isinstance(output, dict):
					if "hidden_states" in output and isinstance(output["hidden_states"], torch.Tensor):
						out_tensor = output["hidden_states"]
					else:
						# try find first tensor value
						for v in output.values():
							if isinstance(v, torch.Tensor):
								out_tensor = v
								break
				else:
					out_tensor = output
				if isinstance(out_tensor, torch.Tensor):
					t = out_tensor.detach()
					t2d = t.reshape(-1, t.shape[-1]) if t.ndim >= 2 else t.view(1, -1)
					current_sum = torch.sum(t2d, dim=0).to(dtype=torch.float32).cpu()

					if activation_stats[name]["output_sum"] is None:
						activation_stats[name]["output_sum"] = current_sum
					else:
						activation_stats[name]["output_sum"] += current_sum
					activation_stats[name]["output_tokens"] += int(t2d.shape[0])

			if "input" in req_act:
				in_tensor = None
				if isinstance(kwargs, dict) and "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
					in_tensor = kwargs["hidden_states"]
				elif args and isinstance(args[0], torch.Tensor):
					in_tensor = args[0]

				if isinstance(in_tensor, torch.Tensor):
					t = in_tensor.detach()
					t2d = t.reshape(-1, t.shape[-1]) if t.ndim >= 2 else t.view(1, -1)
					current_sum = torch.sum(t2d, dim=0).to(dtype=torch.float32).cpu()

					if activation_stats[name]["input_sum"] is None:
						activation_stats[name]["input_sum"] = current_sum
					else:
						activation_stats[name]["input_sum"] += current_sum
					activation_stats[name]["input_tokens"] += int(t2d.shape[0])
		except Exception as e:
			# Be robust: don't break generation because of a hook error
			print(f"[hook:{name}] skipped due to: {type(e).__name__}: {e}")

	return hook_fn


def load_opt_dataset(args):
	"""
	构建并返回一个由多个数据源组成的元探测数据集。
	此版本包含了更具挑战性的数据集。
	"""
	print("--- [元探测数据集构建] ---")
	meta_probe_samples = []
	
	# ======================================================================
	# 类别 1: 综合评估 (Comprehensive-Evaluation)
	# ======================================================================

	# 推荐数据集: MMBench (替代 MME, SEED-Bench)
	if hasattr(args, 'n_mmbench') and args.n_mmbench > 0:
		print(f"从 MMBench 加载 {args.n_mmbench} 个样本...")
		# MMBench 的问题是选择题，需要格式化
		mmbench_dataset = load_dataset("lmms-lab/MMBench", 'en',split="test", streaming=True).shuffle(seed=42).take(args.n_mmbench)
		for item in mmbench_dataset:
			# 组合问题和选项
			question = item['question']
			options = f"A. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}"
			# 如果有提示，可以加在问题前面
			if item.get('hint'):
				full_question = f"{item['hint']}\n{question}\n{options}"
			else:
				full_question = f"{question}\n{options}"
			
			meta_probe_samples.append({"image": item["image"], "question": full_question, "answer": item["answer"]})
		del mmbench_dataset  # 释放内存
	# ======================================================================
	# 类别 2: 认知与推理 (Cognition and Reasoning)
	# ======================================================================

	# 推荐数据集: VCR (Visual Commonsense Reasoning) (替代 GQA, MMMU)
	if hasattr(args, 'n_vcr') and args.n_vcr > 0:
		print(f"从 VCR 加载 {args.n_vcr} 个样本...")
		# VCR 包含问题->答案(Q->A)和答案->理由(A->R)两个子任务。这里我们使用 Q->A。
		# 注意: VCR 的 'image' 字段是字符串路径，需要特殊处理，但 huggingface 会自动加载。
		vcr_dataset = load_dataset("pingzhili/vcr-qa", split="validation", streaming=True).shuffle(seed=42).take(args.n_vcr)
		for item in vcr_dataset:
			# 格式化问题和答案选项
			question = item['question']
			choices = "\n".join([f"- {c}" for c in item['answer_choices']])
			full_question = f"{question}\n\nChoices:\n{choices}"
			
			# 记录正确答案的文本以供参考
			correct_answer_text = item['answer_choices'][item['answer_label']]
			meta_probe_samples.append({"image": item["image"], "question": full_question, "answer": correct_answer_text})
		del vcr_dataset  # 释放内存
	# ======================================================================
	# 类别 3: 富文本视觉问答 (Text-rich VQA)
	# ======================================================================

	# 推荐数据集: DocVQA (替代 TextVQA, OCRBench)
	if hasattr(args, 'n_docvqa') and args.n_docvqa > 0:
		print(f"从 DocVQA 加载 {args.n_docvqa} 个样本...")
		# DocVQA 的问题字段名为 'question'
		docvqa_dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation", streaming=True).shuffle(seed=42).take(args.n_docvqa)
		for item in docvqa_dataset:
			# DocVQA 的 'answers' 是一个列表，这里我们只用问题
			meta_probe_samples.append({"image": item["image"], "question": item["question"], "answers": item["answers"]})
		del docvqa_dataset  # 释放内存

	# (此处可以保留或添加您原来的数据集加载逻辑)
	# 1. 加载并处理 VQA v2 (综合能力)
	if args.n_vqa > 0:
		print(f"从 VQA v2 加载 {args.n_vqa} 个样本...")
		vqa_dataset = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True).shuffle(seed=42).take(args.n_vqa)
		for item in vqa_dataset:
			meta_probe_samples.append({"image": item["image"], "question": item["question"]})
	
		del vqa_dataset  # 释放内存

	# 2. 加载并处理 ScienceQA (认知与推理)
	if args.n_scienceqa > 0:
		print(f"从 ScienceQA 加载 {args.n_scienceqa} 个样本...")
		# ScienceQA non-streaming for easier filtering
		scienceqa_dataset = load_dataset("derek-thomas/ScienceQA", split="validation").shuffle(seed=42)
		# 筛选出包含图像的样本
		scienceqa_with_images = scienceqa_dataset.filter(lambda x: x['image'] is not None)
		count = 0
		for item in scienceqa_with_images:
			if count >= args.n_scienceqa: break
			question = f"{item['hint']} {item['question']}" if item['hint'] else item['question']
			meta_probe_samples.append({"image": item["image"], "question": question})
			count += 1
		del scienceqa_dataset  # 释放内存

	# 3. 加载并处理 ST-VQA (富文本VQA)
	if args.n_stvqa > 0:
		print(f"从 ST-VQA 加载 {args.n_stvqa} 个样本...")
		# ST-VQA 字段名为 'question', 'image'
		stvqa_dataset = load_dataset("danjacobellis/stvqa_task1", split="test", streaming=True).shuffle(seed=42).take(args.n_stvqa)
		for item in stvqa_dataset:
				meta_probe_samples.append({"image": item["image"], "question": item["question"]})
		del stvqa_dataset  # 释放内存
	# 打乱最终的数据集
	random.shuffle(meta_probe_samples)
	print(f"元探测数据集构建完成，总样本数: {len(meta_probe_samples)}")
	print("--------------------------")
	return meta_probe_samples


def llm_run_and_cache(
		model_name,
		dataset,
		req_act: Iterable[str],
		module_regex: str,
		include_types: List[str],
		exclude_regex: Optional[str],
		save_path: Optional[str] = None,
):
	pass
	
	


@torch.no_grad()
def run_and_cache(model_name: str,
				  dataset_name: str,
				  max_samples: int,
				  req_act: Iterable[str],
				  module_regex: str,
				  include_types: List[str],
				  exclude_regex: Optional[str],
				  work_dir: str,
				  save_path: Optional[str] = None,
				  use_vllm: bool = False,
				  verbose: bool = False) -> str:
	# Build dataset
	dataset = build_dataset(dataset_name)
	assert dataset is not None, f"Dataset not found or unsupported: {dataset_name}"

	# Instantiate model
	# (25.06.05) 与框架保持一致：在构建模型前移除 WORLD_SIZE，避免 transformers 新版在 torchrun 下自动采用 TP
	ws_bak = os.environ.pop('WORLD_SIZE', None)
	model_kwargs = {}
	if model_name is not None and (
		'Llama-4' in model_name or 'Qwen2-VL' in model_name or 'Qwen2.5-VL' in model_name
	):
		model_kwargs = {'use_vllm': use_vllm}
	# Sanitize potential trailing spaces in local model_path provided via functools.partial in config
	constructor = supported_VLM[model_name]
	if isinstance(constructor, functools.partial):
		kw = dict(constructor.keywords or {})
		if 'model_path' in kw and isinstance(kw['model_path'], str):
			kw['model_path'] = kw['model_path'].strip()
		constructor = functools.partial(constructor.func, *(constructor.args or ()), **kw)
	vlm = constructor(**model_kwargs) if 'constructor' in locals() else supported_VLM[model_name](**model_kwargs)
	if ws_bak:
		os.environ['WORLD_SIZE'] = ws_bak
	if getattr(vlm, "is_api", False):
		raise RuntimeError("API models are not supported for activation caching (no torch hooks).")

	# For non-API models, let dataset provide image dumping if needed
	if hasattr(vlm, "set_dump_image"):
		vlm.set_dump_image(dataset.dump_image)

	# Try to set eval mode on the underlying torch model
	torch_model = get_underlying_torch_model(vlm)
	if torch_model is not None:
		torch_model.eval()

	# Determine targets and register hooks
	if torch_model is None:
		raise RuntimeError("Cannot find underlying torch model to hook (no .model and wrapper is not nn.Module)")

	target_modules = get_target_module_map(
		torch_model, module_regex=module_regex, include_types=include_types,
		exclude_regex=exclude_regex, verbose=verbose
	)

	activation_stats = defaultdict(lambda: {
		"input_sum": None, "input_tokens": 0,
		"output_sum": None, "output_tokens": 0
	})

	hooks: List[torch.utils.hooks.RemovableHandle] = []
	for name, module in target_modules.items():
		# Prefer new API with kwargs; fallback to legacy hook signature on older PyTorch
		try:
			hook = module.register_forward_hook(
				get_hook_with_kwargs(name, req_act, activation_stats), with_kwargs=True
			)
		except TypeError:
			def legacy_hook(mod, args, output):
				return get_hook_with_kwargs(name, req_act, activation_stats)(mod, args, {}, output)
			hook = module.register_forward_hook(legacy_hook)
		hooks.append(hook)

	# Run up to max_samples items to trigger forwards
	ds = dataset.data
	total = min(len(ds), max_samples) if max_samples > 0 else len(ds)
	if verbose:
		print(f"[run] dataset={dataset_name}, samples={total}, model={model_name}")

	processed = 0
	for i in tqdm(range(total), desc=f"Infer {model_name}/{dataset_name}"):
		item = ds.iloc[i]
		# Build the input struct (message) using model-specific prompt if requested
		if hasattr(vlm, 'use_custom_prompt') and vlm.use_custom_prompt(dataset_name):
			struct = vlm.build_prompt(item, dataset=dataset_name)
		else:
			struct = dataset.build_prompt(item)

		# 与 inference.py 保持一致：支持 SKIP_ERR 环境变量
		if os.environ.get('SKIP_ERR', '0') == '1':
			try:
				_ = vlm.generate(message=struct, dataset=dataset_name)
			except RuntimeError as err:
				if torch.cuda.is_available():
					torch.cuda.synchronize()
				print(f"[warn] generation failed at index {item['index']}: {type(err).__name__}: {err}")
		else:
			_ = vlm.generate(message=struct, dataset=dataset_name)
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
	averaged_activations = {}
	for name, stats in activation_stats.items():
		averaged_activations[name] = {}
		if stats["input_sum"] is not None and stats["input_tokens"] > 0:
			averaged_activations[name]["input"] = stats["input_sum"] / stats["input_tokens"]
		if stats["output_sum"] is not None and stats["output_tokens"] > 0:
			averaged_activations[name]["output"] = stats["output_sum"] / stats["output_tokens"]

	# Prepare save path
	if save_path is None:
		os.makedirs("activations", exist_ok=True)
		base = f"{model_name}_{dataset_name}.pt"
		save_path = osp.join("activations", base)

	# Ensure parent directory exists for custom --save path
	if save_path is not None:
		parent = osp.dirname(save_path)
		if parent:
			os.makedirs(parent, exist_ok=True)

	# Save torch tensors
	# meta = {
	# 	"model": model_name,
	# 	"dataset": dataset_name,
	# 	"samples": processed,
	# 	"req_act": list(req_act),
	# 	"module_regex": module_regex,
	# 	"include_types": include_types,
	# 	"exclude_regex": exclude_regex,
	# }
	# pkg = {"activations": averaged_activations, "meta": meta}
	# torch.save(pkg, save_path)
	torch.save(averaged_activations, save_path)

	# Cleanup to free GPU/CPU memory
	del vlm, torch_model, hooks, activation_stats, averaged_activations
	gc.collect();
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	if verbose:
		print(f"[done] saved to {save_path}")
	return save_path


def main():
	args = parse_args()
	save_path = run_and_cache(
		model_name=args.model,
		dataset_name=args.data,
		max_samples=args.max_samples,
		req_act=args.req_act,
		module_regex=args.module_regex,
		include_types=args.include_types,
		exclude_regex=args.exclude_regex,
		work_dir=args.work_dir,
		save_path=args.save,
		use_vllm=args.use_vllm,
		verbose=args.verbose,
	)
	print(save_path)


if __name__ == "__main__":
	main()

