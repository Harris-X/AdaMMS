import os
import gc
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from copy import deepcopy

# --- Import from the provided repository files ---
# Ensure these files are in the correct directories as specified in the setup
from torch_cka.cka import CKA
from graphs.transformer_enc_graph import TransformerEncoderGraph
from model_merger import ModelMerge
from matching_functions import match_tensors_zipit
from metric_calculators import CovarianceMetric

# --- 1. Core Configuration ---
CKPT_PATH = {
    "llama2": "./downloaded_models/Llama-2-7b-hf",
    "qwen2": "./downloaded_models/Qwen2-7B-Instruct",
}
# Automatically select CUDA if available
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 2. Helper Functions (Model Loading, Dataset, Graph Creation) ---

def load_model_and_tokenizer(model_id):
    """Loads a model and tokenizer, using bfloat16 for memory efficiency."""
    print(f"Loading model: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"{model_id.split('/')[-1]} model loaded.")
    return tokenizer, model

def prepare_dataloader(dataset_name="wikitext", split="test", max_samples=32, batch_size=4, max_length=128):
    """Loads and prepares a dataset for feature extraction."""
    print(f"Loading dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)
    
    # Use a subset for efficiency
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    
    # Filter out empty or whitespace-only texts
    dataset = dataset.filter(lambda example: example['text'] and example['text'].strip())
    
    # We only need the raw text for this dataloader
    return DataLoader(dataset, batch_size=batch_size)
    
def create_transformer_graph(model, model_config):
    """Creates a graph representation for a given transformer model."""
    print(f"Creating graph for {model_config['name']}...")

    # Adapt the 'bert' graph function from the repository for our models
    # We define the module names that the graph will hook into
    module_map = {
        'emb': 'model.embed_tokens',
        'emb_ln': 'model.norm',
        'q': 'self_attn.q_proj',
        'k': 'self_attn.k_proj',
        'v': 'self_attn.v_proj',
        'lin_attn': 'self_attn.o_proj',
        'attn_ln': 'post_attention_layernorm',
        'fc1': 'mlp.gate_proj', # Using gate_proj as one of the FF layers
        'fc2': 'mlp.down_proj',
        'final_ln': 'input_layernorm',
        'head_pref': None,
        'pooler': None,
        'classifier': 'lm_head',
    }

    return TransformerEncoderGraph(
        model,
        modules=module_map,
        layer_name='model.layers',
        enc_prefix='model',
        merge_type='all',
        num_layers=model.config.num_hidden_layers,
        num_heads=model.config.num_attention_heads,
        qk=True, # Assumes Q and K are processed together for similarity
        name=model_config['name'],
        classifier=True
    ).graphify()


# --- 3. Main Execution ---

def main():
    # --- Load Models and Tokenizers ---
    tokenizer_llama, model_llama = load_model_and_tokenizer(CKPT_PATH["llama2"])
    tokenizer_qwen, model_qwen = load_model_and_tokenizer(CKPT_PATH["qwen2"])

    # --- Prepare DataLoader ---
    # The dataloader will yield batches of raw text. We'll tokenize on the fly.
    dataloader_raw = prepare_dataloader(max_samples=64, batch_size=4, max_length=256)

    # --- Model Merging Process ---
    print("\n--- Starting Heterogeneous Model Merging ---")

    # 1. Create model graphs
    # Use deepcopy as the merger will modify the models in place
    model_a, model_b = deepcopy(model_llama), deepcopy(model_qwen)
    graph_a = create_transformer_graph(model_a, {"name": "llama2"})
    graph_b = create_transformer_graph(model_b, {"name": "qwen2"})
    
    # 2. Initialize the ModelMerge class
    merger = ModelMerge(graph_a, graph_b, device=device)

    # 3. Create a custom dataloader for the merger
    # This loader tokenizes each batch for both models
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        inputs_a = tokenizer_llama(texts, padding=True, truncation=True, return_tensors="pt", max_length=256)
        inputs_b = tokenizer_qwen(texts, padding=True, truncation=True, return_tensors="pt", max_length=256)
        # The merger expects a tuple of (inputs, labels or None)
        return (inputs_a.to(device), inputs_b.to(device)), None

    merger_dataloader = DataLoader(dataloader_raw.dataset, batch_size=dataloader_raw.batch_size, collate_fn=collate_fn)
    
    # Override the default compute_intermediates to handle two different tokenizations
    def custom_compute_intermediates(self, x):
        x_a, x_b = x
        self.graphs[0].model.eval()
        self.graphs[1].model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            self.graphs[0].intermediates = {}
            self.graphs[0].model(**x_a)
            
            self.graphs[1].intermediates = {}
            self.graphs[1].model(**x_b)

            return [self.graphs[0].intermediates, self.graphs[1].intermediates]
            
    merger.compute_intermediates = custom_compute_intermediates.__get__(merger, ModelMerge)

    # 4. Compute transformations
    print("Computing transformations (this may take a while)...")
    merger.transform(
        model=deepcopy(model_a),  # Base architecture for the merged model
        dataloader=merger_dataloader,
        transform_fn=match_tensors_zipit,
        metric_classes=(CovarianceMetric,),
        # fix_rate controls the final dimension after zipping
        fix_rate=0.5  # Retain 50% of combined neuron dimension
    )

    # 5. Get and load the merged state dictionary
    print("Averaging weights and creating merged model...")
    merged_state_dict = merger.get_merged_state_dict()
    
    merged_model = deepcopy(model_llama)  # Use LLaMA2 as the base architecture
    merged_model.load_state_dict(merged_state_dict, strict=False)
    
    print("\n--- Model Merging Complete ---")

    # --- 6. Test the Merged Model ---
    prompt = "The future of artificial intelligence is"
    print(f"\nTesting merged model with prompt: '{prompt}'")
    
    inputs = tokenizer_llama(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = merged_model.generate(**inputs, max_new_tokens=50, do_sample=True, top_k=10, top_p=0.95)
    
    generated_text = tokenizer_llama.decode(outputs[0], skip_special_tokens=True)
    print("\n--- Merged Model Generation ---")
    print(generated_text)

    # --- Cleanup ---
    del model_llama, model_qwen, model_a, model_b, merged_model, merger, dataloader_raw, merger_dataloader
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()