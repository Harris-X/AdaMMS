import torch
pkg = torch.load("VLMEvalKit/activations/mplug-owl2_MMBench_DEV_EN.pt", map_location="cpu")
# acts = pkg["activations"]
# meta = pkg["meta"]
print(pkg["model.vision_model.encoder.layers.3.mlp.fc2"])
# print(pkg.keys())
# first_key = sorted(acts.keys())[0]
# print(first_key, acts[first_key].keys(), acts[first_key].get("output").shape)