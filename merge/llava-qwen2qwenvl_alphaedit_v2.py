# llava-qwen2qwenvl_with_alphaedit.py

import os
import sys
import json
import torch
import safetensors.torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoModelForCausalLM
import gc

# 假设 rome/layer_stats.py 存在于 Python 路径中
# 这是从 AlphaEdit 项目中借鉴的关键部分，用于计算协方差
try:
    from rome.layer_stats import layer_stats
except ImportError:
    print("="*80)
    print("错误：无法导入 'rome.layer_stats'。")
    print("请确保您已经将 AlphaEdit 项目中的 'rome' 文件夹放置在您的工作目录或 Python 路径中。")
    print("这个模块对于计算 'null_space' 策略所需的协方差矩阵至关重要。")
    print("您可以从 AlphaEdit 的 GitHub 仓库获取相关文件: https://github.com/jianghoucheng/AlphaEdit")
    print("="*80)
    sys.exit(1)

# --- 模型路径配置 (保持不变) ---
# CKPT_PATH = {
#     'qwen2_vl': "/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct",
#     'llava-onevision-qwen': "/home/user/xieqiuhao/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si"
# }
# # 请将上面的 "/home/user/xieqiuhao/AdaMMS/..." 替换为您的实际模型路径

CKPT_PATH = {
    "cogvlm_chat": "/home/user/xieqiuhao/AdaMMS/downloaded_models/cogvlm-base-490-hf", #"/home/user/xieqiuhao/AdaMMS/downloaded_models/cogvlm-chat-hf",
    "cogvlm_grounding": "/home/user/xieqiuhao/AdaMMS/downloaded_models/cogvlm-grounding-generalist-hf",
    "llava": "/home/user/xieqiuhao/AdaMMS/downloaded_models/llava-v1.5-7b",
    "sharegpt": "/home/user/xieqiuhao/AdaMMS/downloaded_models/ShareGPT4V-7B-llava",
    "vicuna-v1.5": "/yeesuanAI05/thumt/cc/checkpoints/vicuna-7b-v1.5",
    "qwen2_vl" : "/home/user/xieqiuhao/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct",
    "llava-onevision-qwen" : "/home/user/xieqiuhao/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si"
}

INDEX_FILENAME = {
    "cogvlm_chat": "model.safetensors.index.json",
    "cogvlm_grounding": "model.safetensors.index.json",
    "llava": "pytorch_model.bin.index.json",
    "sharegpt": "pytorch_model.bin.index.json",
    "vicuna-v1.5": "pytorch_model.bin.index.json",
    "llava-onevision-qwen": "model.safetensors.index.json",
    "qwen2_vl" : "model.safetensors.index.json"
}


COMPUTE_DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"计算设备: {COMPUTE_DEVICE}")



# 用于缓存协方差统计数据和投影矩阵的目录
STATS_DIR = "hparams_cache"
os.makedirs(STATS_DIR, exist_ok=True)

# --- AlphaEdit 核心逻辑实现 ---
PROJECTOR_CACHE = {}

def compute_covariance_and_projector(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    hparams: argparse.Namespace,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    结合 AlphaEdit 的思想，计算给定层的协方差矩阵，然后进行SVD分解，
    最终返回用于零空间投影的投影矩阵 P。
    这个函数是整个“零空间嫁接”方法的核心。

    Args:
        model: 用于计算协方差的基础模型 (e.g., Qwen2-VL)。
        tok: 对应的分词器。
        layer_name: 需要计算投影矩阵的层的名称。
        hparams: 包含数据集、样本数等超参数的命名空间。
        force_recompute: 是否强制重新计算而不是使用缓存。

    Returns:
        torch.Tensor: 投影矩阵 P。
    """
    model_name_safe = hparams.base_model_path.replace("/", "_")
    # 增加阈值到缓存键中，以防不同阈值的结果混淆
    key = (model_name_safe, layer_name, hparams.null_space_threshold)

    cache_path = os.path.join(STATS_DIR, f"projector__{key[0]}__{key[1]}__{key[2]}.pt")

    if os.path.exists(cache_path) and not force_recompute:
        print(f"Loading cached projector for {model_name_safe} @ {layer_name}.")
        return torch.load(cache_path)

    print(f"Computing covariance for {model_name_safe} @ {layer_name}.")
    # 使用 layer_stats (类似AlphaEdit中的 get_cov) 来获取二阶矩 (协方差)
    # mom2_dataset 指定了用于统计的数据集，例如'c4'或'wikipedia'

    # 修复: 确保模型的权重是float32类型以避免BFloat16不兼容问题
    original_dtype = None
    for param in model.parameters():
        if hasattr(param, 'dtype'):
            original_dtype = param.dtype
            break
            
    # 临时将模型转换为float32进行计算
    if original_dtype == torch.bfloat16:
        print(f"临时将模型从BFloat16转换为Float32以计算协方差...")
        model = model.float()

    stat = layer_stats(
        model,
        tok,
        layer_name,
        STATS_DIR,
        hparams.mom2_dataset,
        to_collect=["mom2"],
        sample_size=hparams.mom2_n_samples,
        precision="float32", #"float32", hparams.mom2_dtype
        force_recompute=force_recompute,
        device=COMPUTE_DEVICE,  # 使用统一的计算设备
    )
    
    cov = stat.mom2.moment().float().to(COMPUTE_DEVICE)

    print(f"Computing SVD and projector for {layer_name}.")
    # 对协方差矩阵进行SVD分解
    U, S, _ = torch.linalg.svd(cov)
    
    # 根据阈值确定零空间
    # AlphaEdit 论文中使用的阈值是 1e-2
    threshold = hparams.null_space_threshold
    null_space_vectors = U[:, S < threshold]
    
    # 计算投影矩阵 P = Û * Ûᵀ
    projector = null_space_vectors @ null_space_vectors.T
    
    print(f"Finished computing projector for {layer_name}. "
          f"Original dim: {cov.shape[0]}, Null-space dim: {null_space_vectors.shape[1]}")
    
    # 缓存结果到文件
    torch.save(projector.cpu(), cache_path)
    # 将模型恢复到原始数据类型
    if original_dtype == torch.bfloat16:
        model = model.to(original_dtype)
    
    return projector

# # --- 原始脚本中的模型加载和辅助函数 ---
# def load_qwenvl_weights(ckpt_path):
#     # (此函数内容保持不变)
#     return safetensors.torch.load_file(os.path.join(ckpt_path, "model.safetensors"))

# def load_minicpm_weights(ckpt_path):
#     # (此函数内容保持不变)
#     return safetensors.torch.load_file(os.path.join(ckpt_path, "model.safetensors"))
def compute_all_projectors(
    model: AutoModelForVision2Seq,
    tokenizer: AutoTokenizer,
    layer_names: list,
    hparams: argparse.Namespace,
    force_recompute: bool = False,
) -> dict:
    """
    一次性计算所有层的投影矩阵，避免重复模型前向传播
    
    Args:
        model: 用于计算协方差的基础模型
        tokenizer: 分词器
        layer_names: 需要计算的层名称列表
        hparams: 超参数
        force_recompute: 是否强制重新计算
        
    Returns:
        dict: 层名称到投影矩阵的映射
    """
    from util.nethook import TraceDict
    from rome.layer_stats import layer_stats_for_multiple_layers
    
    # 检查缓存
    model_name_safe = hparams.base_model_path.replace("/", "_").replace("-", "_")
    projectors = {}
    uncached_layers = []
    
    # 首先检查哪些层需要计算
    for layer_name in layer_names:
        cache_path = os.path.join(
            STATS_DIR, 
            f"projector__{model_name_safe}__{layer_name.replace('.', '_')}__{hparams.null_space_threshold}.pt"
        )
        if os.path.exists(cache_path) and not force_recompute:
            print(f"Loading cached projector for {layer_name}")
            projectors[layer_name] = torch.load(cache_path).to(COMPUTE_DEVICE)
        else:
            uncached_layers.append(layer_name)
    
    # 如果所有层都已缓存，直接返回
    if not uncached_layers:
        return projectors
    
    # 修复: 确保模型的权重是float32类型以避免BFloat16不兼容问题
    original_dtype = None
    for param in model.parameters():
        if hasattr(param, 'dtype'):
            original_dtype = param.dtype
            break
            
    # 临时将模型转换为float32进行计算
    if original_dtype == torch.bfloat16:
        print(f"临时将模型从BFloat16转换为Float32以计算协方差...")
        model = model.float()
    
    print(f"Computing statistics for {len(uncached_layers)} layers in one forward pass")
    
    # 一次性计算所有未缓存层的统计数据
    layer_stats_dict = layer_stats_for_multiple_layers(
        model=model,
        tokenizer=tokenizer,
        layer_names=uncached_layers,
        stats_dir=STATS_DIR,
        ds_name=hparams.mom2_dataset,
        to_collect=["mom2"],
        sample_size=hparams.mom2_n_samples,
        precision="float32",
        force_recompute=force_recompute,
        device=COMPUTE_DEVICE
    )
    
    # 计算每一层的投影矩阵
    for layer_name in uncached_layers:
        if layer_name not in layer_stats_dict:
            print(f"Warning: No statistics found for layer {layer_name}")
            continue
            
        stat = layer_stats_dict[layer_name]
        cov = stat.mom2.moment().float().to(COMPUTE_DEVICE)
        
        print(f"Computing SVD and projector for {layer_name}")
        # 对协方差矩阵进行SVD分解
        U, S, _ = torch.linalg.svd(cov)
        
        # 根据阈值确定零空间
        threshold = hparams.null_space_threshold
        null_space_vectors = U[:, S < threshold]
        
        # 计算投影矩阵 P = Û * Ûᵀ
        projector = null_space_vectors @ null_space_vectors.T
        
        print(f"Finished computing projector for {layer_name}. "
              f"Original dim: {cov.shape[0]}, Null-space dim: {null_space_vectors.shape[1]}")
        
        # 缓存结果到文件
        cache_path = os.path.join(
            STATS_DIR, 
            f"projector__{model_name_safe}__{layer_name.replace('.', '_')}__{hparams.null_space_threshold}.pt"
        )
        torch.save(projector.cpu(), cache_path)
        
        projectors[layer_name] = projector
    
    # 将模型恢复到原始数据类型
    if original_dtype == torch.bfloat16:
        model = model.to(original_dtype)
    
    return projectors


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

vicuna_file_list = ['pytorch_model-00001-of-00002.bin', 'pytorch_model-00002-of-00002.bin']
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

qwenvl_file_list = ['model-00001-of-00005.safetensors', 'model-00002-of-00005.safetensors', 'model-00003-of-00005.safetensors', 'model-00004-of-00005.safetensors', 'model-00005-of-00005.safetensors']
def load_qwenvl_weights(base_path, file_list=qwenvl_file_list):
    return load_safetensors_weights(base_path, file_list)

llava_onevision_qwen_file_list = ['model-00001-of-00004.safetensors', 'model-00002-of-00004.safetensors', 'model-00003-of-00004.safetensors', 'model-00004-of-00004.safetensors']
def load_minicpm_weights(base_path, file_list=llava_onevision_qwen_file_list):
    return load_safetensors_weights(base_path, file_list)


def need_merge(name:str) -> bool:
    # 这个函数保持不变，用于识别需要合并的层
    if name in ['model.norm.weight']:
        return True
    if name in ['lm_head.weight', 'model.embed_tokens.weight']:
        return False
    if name.startswith("model.layers."):
        if name.endswith(".self_attn.rotary_emb.inv_freq"):
            return False
        return True
    return False


def fix_layer_name_for_qwen2vl(name):
    """修正 Qwen2-VL 模型的层名称路径"""
    # 对于语言模型层
    if name.startswith("model.layers."):
        return name.replace("model.layers.", "model.language_model.layers.")
    
    # 对于模型最终的norm层
    if name == "model.norm.weight":
        return "model.language_model.norm.weight"
    
    return name


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




# --- 主合并逻辑 ---
def convert(args):
    """
    主转换函数，集成了新的 'null_space' 合并策略。
    """
    # 输出路径设置
    output_dir = "merged_models"
    if args.output is not None:
        model_name = os.path.basename(args.output)
        output_dir = os.path.dirname(args.output)
    else:
        strategy_name = args.strategy if args.strategy else "interpolation"
        if strategy_name == "null_space":
            model_name = f"qwen-grafted-t{args.null_space_threshold}"
        else:
            model_name = f"qwen-merged-{strategy_name}-a{args.alpha}"
    
    OUTPUT_PATH = os.path.join(output_dir, model_name)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"Merging output path: {OUTPUT_PATH}")

    # 加载模型权重
    print("Loading base model (Qwen2-VL) and donor model (LLaVA-OneVision-Qwen)...")
    base_weights = load_qwenvl_weights(args.base_model_path)
    donor_weights = load_minicpm_weights(args.donor_model_path)
    
    # 选择合并策略
    if args.strategy == "null_space":
        print("="*80)
        print("Applying 'null_space' grafting strategy.")
        print("="*80)
        
        # 加载用于计算协方差的基础模型和分词器
        print(f"Loading base model '{args.base_model_path}' for covariance computation...")
        # torch.bfloat16 以bfloat16加载以节省内存，计算时会转为float32 # AutoModelForCausalLM, AutoModelForVision2Seq
        base_model_for_cov = AutoModelForVision2Seq.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16).to(COMPUTE_DEVICE)
        print(base_model_for_cov)
        base_tok = AutoTokenizer.from_pretrained(args.base_model_path)


        # 获取需要计算的所有层
        layers_to_process = []
        for key in tqdm(base_weights.keys(), desc="Applying Null-Space Grafting"):
            if need_merge(key) and key in donor_weights:
                print(f"\nProcessing layer: {key}")
                # 1. 计算投影矩阵 P
                module_path = key.rsplit('.', 1)[0]
                module_path = fix_layer_name_for_qwen2vl(module_path)

                # print(f"Computing projector for layer '{module_path}'...")
                # projector = compute_covariance_and_projector(
                #     base_model_for_cov,
                #     base_tok,
                #     module_path,
                #     args,
                # ).to(COMPUTE_DEVICE)
                layers_to_process.append(module_path)
        

        # 去重
        layers_to_process = list(set(layers_to_process))
        print(f"需要计算投影矩阵的层数: {len(layers_to_process)}")
        # 一次性计算所有投影矩阵
        projectors = compute_all_projectors(
            base_model_for_cov,
            base_tok,
            layers_to_process,
            args
        )

        # 应用计算好的投影矩阵进行合并
        for key in tqdm(base_weights.keys(), desc="Applying Null-Space Grafting"):
            if need_merge(key) and key in donor_weights:
                module_path = key.rsplit('.', 1)[0]
                module_path = fix_layer_name_for_qwen2vl(module_path)
                
                if module_path not in projectors:
                    print(f"Warning: No projector found for {module_path}, using linear interpolation")
                    base_weights[key] = (1 - args.alpha) * base_weights[key] + args.alpha * donor_weights[key]
                    continue
                
                # 1. 使用预先计算好的投影矩阵
                projector = projectors[module_path]
                # 2. 计算权重差异 Δ
                w_a = base_weights[key].float().to(COMPUTE_DEVICE)
                w_b = donor_weights[key].float().to(COMPUTE_DEVICE)
                delta = w_b - w_a
                
                # 3. 投影扰动 Δ' = Δ @ P
                # 权重矩阵 W 的形状通常是 (d_out, d_in)。投影作用于输入特征空间，因此是右乘。
                projected_delta = delta @ projector
                
                # 4. 应用嫁接 W* = W_A + Δ'
                base_weights[key] = (w_a + projected_delta).to(base_weights[key].dtype).cpu()
                

        del base_model_for_cov, base_tok, projectors
        gc.collect()
        torch.cuda.empty_cache()

    # --- 其他合并策略的逻辑 (保持不变) ---
    elif args.strategy in ['ties', 'dare_ties', 'dare_linear']:
        # ... (此处为原始的 TIES/DARE 等合并逻辑) ...
        print(f"Applying '{args.strategy}' strategy... (Placeholder)")
        pass
    else: 
        # 默认的线性插值
        print("="*80)
        print("Applying default linear interpolation strategy.")
        print("="*80)
        for key in tqdm(base_weights.keys(), desc="Applying Linear Interpolation"):
            if key in donor_weights and need_merge(key):
                 base_weights[key] = (1 - args.alpha) * base_weights[key] + args.alpha * donor_weights[key]

    # --- 保存合并后的模型 ---
    print("\nSaving merged model...")

    # save
    print("Saving...")
    metadata = {'format': 'pt'}
    llava_index_path = os.path.join(CKPT_PATH["qwen2_vl"], INDEX_FILENAME["qwen2_vl"])
    with open(llava_index_path, "r") as f:
        print(f"Loading LLaVA index from {llava_index_path}...")
        llava_index = json.load(f)
        llava_index = llava_index["weight_map"]
    
    split_llava = {}
    for file in qwenvl_file_list:
        split_llava[file] = {}
    for key in llava_index:
        split_llava[llava_index[key]][key] = base_weights[key]
    for file in qwenvl_file_list:
        if not os.path.isdir(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)
        save_path = os.path.join(OUTPUT_PATH, file)
        safetensors.torch.save_file(split_llava[file], save_path, metadata)
        
    create_soft_link(source_path=CKPT_PATH["qwen2_vl"], link_path=OUTPUT_PATH)

    print("Convert Done.")
    print(save_path)



    # # 找到基础模型的配置文件并复制
    # config_path = os.path.join(args.base_model_path, "config.json")
    # if os.path.exists(config_path):
    #     with open(config_path, 'r') as f_in, open(os.path.join(OUTPUT_PATH, "config.json"), 'w') as f_out:
    #         f_out.write(f_in.read())
    
    # # 保存权重
    # safetensors.torch.save_file(base_weights, os.path.join(OUTPUT_PATH, "model.safetensors"))
    
    print("Convert Done.")
    print(f"Merged model saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 通用参数
    parser.add_argument('--output', type=str, default=None, help="Output checkpoint path (including model name)")
    parser.add_argument('--strategy', type=str, default="null_space", 
                        help="Merging strategy: 'ties', 'dare_ties', or the new 'null_space'. Default is linear interpolation.") 
    
    # 线性插值和TIES/DARE的参数
    parser.add_argument('--alpha', type=float, default=0.5, help="Interpolation coefficient for linear merge")
    parser.add_argument('-K', type=float, default=0.5, help="Parameter for TIES/DARE merging")

    # 'null_space' 策略的新增参数
    parser.add_argument('--base_model_path', type=str, default=CKPT_PATH["qwen2_vl"], 
                        help="Path to the base model for covariance computation (e.g., Qwen2-VL).")
    parser.add_argument('--donor_model_path', type=str, default=CKPT_PATH["llava-onevision-qwen"], 
                        help="Path to the donor model (e.g., LLaVA-OneVision).")
    parser.add_argument('--mom2_dataset', type=str, default="wikipedia", 
                        help="Dataset to use for statistics, e.g., 'wikipedia' or 'c4'.")
    parser.add_argument('--mom2_n_samples', type=int, default=10000, 
                        help="Number of samples for statistics. Reduce if VRAM is limited.")
    parser.add_argument('--mom2_dtype', type=str, default="bfloat16", 
                        help="Precision for statistics computation (e.g., bfloat16, float16).")
    parser.add_argument('--null_space_threshold', type=float, default=1e-2, 
                        help="Threshold for SVD singular values to define the null-space.")

    args = parser.parse_args()
    
    # 更新模型路径
    CKPT_PATH['qwen2_vl'] = args.base_model_path
    CKPT_PATH['llava-onevision-qwen'] = args.donor_model_path

    print("--- Configuration ---")
    print(args)
    print("--------------------")

    convert(args)