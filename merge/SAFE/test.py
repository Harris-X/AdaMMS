import torch
pkg = torch.load("merged_models/idream-sams-dream-0.1-0.8/cache/activations_A.pt", map_location="cpu")
# acts = pkg["activations"]
# meta = pkg["meta"]
print(pkg["layers.2"]["input"].shape)
# print(pkg.keys())
# first_key = sorted(acts.keys())[0]
# print(first_key, acts[first_key].keys(), acts[first_key].get("output").shape)