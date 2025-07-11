# 7B version with Functional Alignment (CKA) and Direct Merging
import os
import sys
import json
import torch
import safetensors.torch
import argparse
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
import numpy as np

# --- NEW IMPORTS for Functional Alignment ---
from transformers import AutoModelForCausalLM, AutoTokenizer, LlavaNextForConditionalGeneration, LlavaNextProcessor
from datasets import load_dataset

# --- Model Paths and Configs ---
# Updated to focus on the two chosen heterogeneous models
CKPT_PATH = {
    "llava": "/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/liuhaotian_llava-v1.5-7b",
    "qwen2": "/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/Qwen_Qwen2-7B-Instruct",
}

# =======================================================================================
# == STEP 1: CKA IMPLEMENTATION and REPRESENTATION EXTRACTION HELPERS
# =======================================================================================

def gram_linear(x):
    """Computes the linear Gram matrix."""
    # Ensure tensor is 2D
    if x.ndim > 2:
        x = x.view(x.shape[0], -1)
    return x @ x.T

def center_gram(gram, unbiased=False):
    """Centers the Gram matrix."""
    if not torch.allclose(gram, gram.T):
        gram = (gram + gram.T) / 2.0 # Symmetrize
    
    if unbiased:
        n = gram.shape[0]
        gram.fill_diagonal_(0)
        means = torch.sum(gram, dim=0) / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        gram.fill_diagonal_(0)
    else:
        means = torch.mean(gram, dim=0)
        means -= torch.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]
    return gram

def cka(x, y, unbiased=False):
    """Computes the CKA score between two representation matrices X and Y."""
    gram_x = center_gram(gram_linear(x), unbiased=unbiased)
    gram_y = center_gram(gram_linear(y), unbiased=unbiased)
    scaled_hsic = (gram_x.T.ravel() @ gram_y.ravel())
    norm_x = torch.linalg.norm(gram_x)
    norm_y = torch.linalg.norm(gram_y)
    return (scaled_hsic / (norm_x * norm_y)).item() if (norm_x * norm_y) != 0 else 0.0

def get_layer_representations(model, tokenizer, texts, device, max_samples=32, max_length=128):
    """
    Extracts layer representations using forward hooks for a given set of texts.
    """
    model.eval()
    layer_reps = defaultdict(list)
    
    # Identify the transformer layers based on model type
    if "Qwen2" in model.config.architectures[0]:
        layers = model.model.layers
    elif "Llava" in model.config.architectures[0] or "Llama" in model.config.architectures[0]:
        # For LLaVA, we operate on its language_model component
        layers = model.language_model.model.layers if hasattr(model, 'language_model') else model.model.layers
    else:
        raise NotImplementedError(f"Model architecture {model.config.architectures[0]} not supported for layer extraction.")

    hooks = []
    for i, layer in enumerate(layers):
        def create_hook_fn(layer_idx):
            def hook_fn(module, input, output):
                # output[0] is the hidden state. Move to CPU to save VRAM.
                layer_reps[layer_idx].append(output[0].detach().cpu())
            return hook_fn
        hooks.append(layer.register_forward_hook(create_hook_fn(i)))
    
    # Process texts in a batch-like manner
    with torch.no_grad():
        for i in tqdm(range(0, min(len(texts), max_samples)), desc="Extracting Representations"):
            text = texts[i]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            model(**inputs)

    for hook in hooks:
        hook.remove()
        
    # Aggregate representations: concatenate across the sample dimension
    # Result: {layer_idx: tensor of shape (num_samples, seq_len, hidden_dim)}
    aggregated_reps = {i: torch.cat(reps, dim=0) for i, reps in layer_reps.items()}
    return aggregated_reps

# =======================================================================================
# == STEP 2: CORE LOGIC FOR FUNCTIONAL MERGING
# =======================================================================================

def find_functional_layer_mapping(model_a, tokenizer_a, model_b, tokenizer_b, device):
    """
    Finds the optimal layer mapping from model_a to model_b using CKA.
    """
    print("\n--- Starting Functional Layer Mapping ---")
    
    # 1. Prepare probe dataset
    print("1. Loading probe dataset...")
    # Use a small but diverse subset for efficiency
    probe_dataset = load_dataset("wikipedia", "20220301.en", split='train[:50]', trust_remote_code=True)
    probe_texts = [text for text in probe_dataset['text'] if len(text) > 200]
    
    # 2. Extract representations
    print("2. Extracting representations from Model A (LLaVA)...")
    reps_a = get_layer_representations(model_a, tokenizer_a, probe_texts, device, max_samples=50)
    
    print("3. Extracting representations from Model B (Qwen2)...")
    reps_b = get_layer_representations(model_b, tokenizer_b, probe_texts, device, max_samples=50)

    num_layers_a = len(reps_a)
    num_layers_b = len(reps_b)
    print(f"Model A layers: {num_layers_a}, Model B layers: {num_layers_b}")

    # 4. Compute CKA similarity matrix
    print("4. Computing CKA similarity matrix...")
    similarity_matrix = np.zeros((num_layers_a, num_layers_b))
    for i in tqdm(range(num_layers_a), desc="Calculating CKA"):
        for j in range(num_layers_b):
            # Ensure sequence lengths match for CKA. Simple truncation.
            len_a = reps_a[i].shape[1]
            len_b = reps_b[j].shape[1]
            min_len = min(len_a, len_b)
            similarity_matrix[i, j] = cka(reps_a[i][:, :min_len, :], reps_b[j][:, :min_len, :])

    # 5. Find optimal mapping (greedy approach)
    print("5. Finding optimal layer mapping...")
    layer_mapping = {}
    for i in range(num_layers_a):
        best_match_for_a = np.argmax(similarity_matrix[i, :])
        layer_mapping[i] = int(best_match_for_a)
        
    print("\n--- CKA Similarity Matrix (Rounded) ---")
    print(np.round(similarity_matrix, 2))
    
    print("\n--- Functional Layer Mapping (Model A -> Model B) ---")
    for layer_a, layer_b in layer_mapping.items():
        print(f"LLaVA Layer {layer_a:2d}  ==>  Qwen2 Layer {layer_b:2d}  (CKA Score: {similarity_matrix[layer_a, layer_b]:.4f})")
        
    return layer_mapping

def merge_models_with_mapping(model_a, model_b, layer_mapping, alpha=0.5):
    """
    Merges model_a into model_b using the provided functional mapping.
    This function modifies model_b in place.
    """
    print("\n--- Performing Direct Parameter Fusion based on CKA Mapping ---")
    
    # Identify layer structures
    layers_a = model_a.language_model.model.layers
    layers_b = model_b.model.layers
    
    with torch.no_grad():
        # Merge Transformer Layers
        for i_a, i_b in tqdm(layer_mapping.items(), desc="Merging Layers"):
            layer_a = layers_a[i_a]
            layer_b = layers_b[i_b]

            # Merge parameters within the mapped layers
            for name_a, param_a in layer_a.named_parameters():
                # Try to find a corresponding parameter in layer_b
                # This requires heuristic name matching, as names can differ
                # e.g., 'self_attn.q_proj' in LLaMA vs Qwen2
                if name_a in dict(layer_b.named_parameters()):
                    param_b = dict(layer_b.named_parameters())[name_a]
                    if param_a.shape == param_b.shape:
                        # Direct linear interpolation
                        param_b.data = (1 - alpha) * param_b.data + alpha * param_a.data.to(param_b.device)
        
        # Merge Embedding and Final Layer Norm / LM Head
        # This part remains heuristic and requires careful alignment
        print("Merging Embeddings...")
        model_b.model.embed_tokens.weight.data = (1 - alpha) * model_b.model.embed_tokens.weight.data + alpha * model_a.language_model.model.embed_tokens.weight.data.to(model_b.device)
        
        print("Merging Final LayerNorm...")
        model_b.model.norm.weight.data = (1 - alpha) * model_b.model.norm.weight.data + alpha * model_a.language_model.model.norm.weight.data.to(model_b.device)
        
        print("Merging LM Head...")
        # Check for shape mismatch, which is common
        if model_b.lm_head.weight.shape == model_a.lm_head.weight.shape:
             model_b.lm_head.weight.data = (1 - alpha) * model_b.lm_head.weight.data + alpha * model_a.lm_head.weight.data.to(model_b.device)
        else:
            print(f"LM Head shape mismatch: Qwen2={model_b.lm_head.weight.shape} vs LLaVA={model_a.lm_head.weight.shape}. Skipping LM Head merge.")

    return model_b

# =======================================================================================
# == MAIN EXECUTION
# =======================================================================================

def main(args):
    """Main function to perform functional merging."""
    print(f"Output path for merged model: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Load Full Models using Transformers ---
    print("Loading Model A: LLaVA (Llama-based)...")
    # Llava is our source model (model_a)
    model_a = LlavaNextForConditionalGeneration.from_pretrained(CKPT_PATH["llava"])
    processor_a = LlavaNextProcessor.from_pretrained(CKPT_PATH["llava"])
    tokenizer_a = processor_a.tokenizer
    model_a.to(device)

    print("Loading Model B: Qwen2 (Base for merging)...")
    # Qwen2 is our target model (model_b), which will be modified
    model_b = AutoModelForCausalLM.from_pretrained(CKPT_PATH["qwen2"])
    tokenizer_b = AutoTokenizer.from_pretrained(CKPT_PATH["qwen2"])
    model_b.to(device)

    # --- 2. Find Functional Layer Mapping ---
    layer_mapping = find_functional_layer_mapping(model_a, tokenizer_a, model_b, tokenizer_b, device)
    
    # --- 3. Perform Merging ---
    # We create a copy to keep the original model_b intact in memory
    merged_model = merge_models_with_mapping(model_a, deepcopy(model_b), layer_mapping, alpha=args.alpha)
    
    # --- 4. Save the Merged Model ---
    print("\nSaving merged model...")
    # We save the full model in the standard Hugging Face format
    merged_model.save_pretrained(args.output)
    tokenizer_b.save_pretrained(args.output)
    print(f"Merge Done. Model saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True, help="Output directory path for the merged model.")
    parser.add_argument('--alpha', type=float, default=0.5, help="Linear interpolation factor. 0.0 means all Qwen2, 1.0 means all LLaVA.")
    args = parser.parse_args()
    main(args)