# Reference: https://github.com/prateeky2806/ties-merging/blob/main/src/ties_minimal.ipynb
# @inproceedings{yadav2023ties-merging,
#       title={Resolving Interference When Merging Models}, 
#       author={Prateek Yadav and Derek Tam and Leshem Choshen and Colin Raffel and Mohit Bansal},
#     booktitle = "NeurIPS",
#     year = "2023",
#     address = "New Orleans, USA",
#     publisher = "Proceedings of Machine Learning Research",
# }

import sys
import os, copy
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import OrderedDict
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

### Start of Model conversion utils ###
def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict


def add_ptm_to_tv(tv_dict, ptm_dict):
    assert set(tv_dict.keys()) == set(
        ptm_dict.keys()
    ), "Differing parameter names in models."
    final_dict = copy.deepcopy(tv_dict)
    for k, v in ptm_dict.items():
        final_dict[k] = tv_dict[k] + v
    return final_dict


def check_parameterNamesMatch(checkpoints):
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )

def check_state_dicts_equal(state_dict1, state_dict2):
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True
### End of Model conversion utils ###

### Start of Merge Utils ###
def topk_values_mask(M, K=0.7, return_mask=False):
    if K >= 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values_list = []
    print("get kth value")
    for i in range(M.shape[0]):
        kth_values, _ = M[i:i+1, ...].abs().kthvalue(k, dim=1, keepdim=True)
        print(i)
        kth_values_list.append(kth_values)
    # import ipdb; ipdb.set_trace();
    kth_values = torch.cat(kth_values_list, dim=0)
    # kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    print("calculate mask")
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


def resolve_sign(Tensor):
    sign_to_mult = torch.sign(Tensor.sum(dim=0))
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult


def disjoint_merge(Tensor, merge_func, sign_to_mult):

    merge_func = merge_func.split("-")[-1]

    print(1)
    # import ipdb; ipdb.set_trace();
    # for i in range(Tensor.shape[0]):
    #     for j in tqdm(range(Tensor.shape[1])):
    #         Tensor[i][j] = Tensor[i][j] if ((sign_to_mult[j] > 0) == (Tensor[i][j] > 0)) else 0
    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        # rows_to_keep = torch.where(
        #     sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        # )
        rows_to_keep = (sign_to_mult > 0) == (Tensor > 0)
        print(1.5)
        # import ipdb; ipdb.set_trace();
        # chunk_count = 10
        # chunk_size = (Tensor.shape[1] - 1 // chunk_count) + 1
        # chunk_start = 0
        # for i in range(chunk_count):
        #     Tensor[chunk_start:chunk_start+chunk_size] *= rows_to_keep[chunk_start:chunk_start+chunk_size]

        Tensor *= rows_to_keep
        selected_entries = Tensor
        # selected_entries = Tensor * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep

    print(2)
    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")
    print(3)
    return disjoint_aggs


def ties_merging(
    flat_task_checks,
    reset_thresh=None,
    merge_func="",
    alpha=0.5
):
    all_checks = flat_task_checks.clone()
    updated_checks, *_ = topk_values_mask(
        all_checks, K=reset_thresh, return_mask=False
    )
    print(f"RESOLVING SIGN")
    final_signs = resolve_sign(updated_checks)
    assert final_signs is not None
    # print(updated_checks.shape)
    # updated_checks[0]=2*alpha*updated_checks[0]
    # updated_checks[1]=2*(1-alpha)*updated_checks[1]
    print(f"Disjoint AGGREGATION: {merge_func}")
    merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)
    
    print("merged")
    return merged_tv
### End of Merge Utils ###

### Start of Ties Merging ###
def do_merging(ft_checks, K=20, merge_func="dis-mean", lamda=1,alpha=0.5):
    """
        [INPUT]

        ft_checks is a list of dicts, each dict is a state_dict

        K = 20 (bigger K conserve more params)

        merge_func = "dis-mean" # "dis-sum" "dis-max"

        lamda = 1

        [OUTPUT]

        merged_state_dict
    """

    remove_keys = []

    print("torch.vstack ...")
    tv_flat_checks = torch.vstack(
        [state_dict_to_vector(check, remove_keys) for check in ft_checks]
    )

    
    print("ties merging...")
    # return merged flat task vector
    merged_tv = ties_merging(
        tv_flat_checks,
        reset_thresh=K,
        merge_func=merge_func,
        alpha=alpha
    )

    # we *don't* add back the PTM to the flat merged task vector here
    # merged_check = flat_ptm + lamda * merged_tv
    merged_check = lamda * merged_tv

    print("vector_to_state_dict...")
    # convert the flat merged checkpoint to a state dict
    ptm_check = ft_checks[0] # we use ft_checks[0] to pretend ptm_check, since we only need the state_dict keys in them
    # import ipdb; ipdb.set_trace();
    merged_state_dict = vector_to_state_dict(
        merged_check, ptm_check, remove_keys=remove_keys
    )
    return merged_state_dict
### End of Ties Merging ###

def do_merging_strategy(ft_checks, strategy, K=20, alpha=0.5):
    remove_keys = []

    if K > 1:
        K /= 100
    assert K > 0

    print("torch.vstack ...")
    tv_flat_checks = torch.vstack(
        [state_dict_to_vector(check, remove_keys) for check in ft_checks]
    )

    if strategy == 'task_arithmetic':
        merged_check = tv_flat_checks.mean(dim=0)
    elif strategy == 'metagpt':
        print("Model Exclusive Task Arithmetic...")
        dist = (tv_flat_checks**2).sum(dim=1)
        dist_sum = dist.sum()
        scale_factor = dist / dist_sum
        print(scale_factor)
        scaled_check = tv_flat_checks * scale_factor.unsqueeze(1)
        merged_check = scaled_check.sum(dim=0)
    elif strategy == 'dare_ties':
        merge_func = "dis-sum"
        drop_mask = torch.bernoulli(
            torch.full_like(input=tv_flat_checks, fill_value=K)
        )
        drop_tv = tv_flat_checks * drop_mask
        updated_checks = drop_tv / K
        print(f"RESOLVING SIGN")
        # updated_checks[0]=2*alpha*updated_checks[0]
        # updated_checks[1]=2*(1-alpha)*updated_checks[1]
        final_signs = resolve_sign(updated_checks)
        assert final_signs is not None
        
        print(f"Disjoint AGGREGATION: {merge_func}")
        merged_check = disjoint_merge(updated_checks, merge_func, final_signs)
        
        print("merged")
    elif strategy == 'dare_linear':
        drop_mask = torch.bernoulli(
            torch.full_like(input=tv_flat_checks, fill_value=K)
        )
        drop_tv = tv_flat_checks * drop_mask
        rescale_tv = drop_tv / K
        merged_check = rescale_tv.mean(dim=0)
    elif strategy == 'drop_linear':
        drop_mask = torch.bernoulli(
            torch.full_like(input=tv_flat_checks, fill_value=K)
        )
        drop_tv = tv_flat_checks * drop_mask
        rescale_tv = drop_tv / K
        merged_check = rescale_tv.mean(dim=0)
    elif strategy == 'average':
        merged_check = tv_flat_checks.mean(dim=0)
    elif strategy == 'ties':
        merge_func="dis-sum"
        print("ties merging...")
    # return merged flat task vector
        merged_tv = ties_merging(
            tv_flat_checks,
            reset_thresh=K,
            merge_func=merge_func,
        )
        merged_check = merged_tv # lambda = 1
    else:
        raise NotImplementedError(f'strategy [{strategy}] is not implemented')

    print("vector_to_state_dict...")
    # convert the flat merged checkpoint to a state dict
    ptm_check = ft_checks[0] # we use ft_checks[0] to pretend ptm_check, since we only need the state_dict keys in them
    merged_state_dict = vector_to_state_dict(
        merged_check, ptm_check, remove_keys=remove_keys
    )
    return merged_state_dict
    

### Start of Model Exclusive Task Arithmetic ###
def do_merging_metagpt(ft_checks):
    """
        [INPUT]

        ft_checks is a list of dicts, each dict is a state_dict

        K = 20 (bigger K conserve more params)

        merge_func = "dis-mean" # "dis-sum" "dis-max"

        lamda = 1

        [OUTPUT]

        merged_state_dict
    """

    remove_keys = []

    print("torch.vstack ...")
    tv_flat_checks = torch.vstack(
        [state_dict_to_vector(check, remove_keys) for check in ft_checks]
    )

    # Since we already have the delta weights from LoRA, we don't minus the flat_ptm
    # tv_flat_checks = flat_ft - flat_ptm
    
    print("Model Exclusive Task Arithmetic...")
    dist = (tv_flat_checks**2).sum(dim=1)
    dist_sum = dist.sum()
    scale_factor = dist / dist_sum
    print(scale_factor)
    scaled_check = tv_flat_checks * scale_factor.unsqueeze(1)
    merged_check = scaled_check.sum(dim=0)

    # import ipdb; ipdb.set_trace();
    # return merged flat task vector
    # merged_tv = ties_merging(
    #     tv_flat_checks,
    #     reset_thresh=K,
    #     merge_func=merge_func,
    # )

    # # we *don't* add back the PTM to the flat merged task vector here
    # # merged_check = flat_ptm + lamda * merged_tv
    # merged_check = lamda * merged_tv

    print("vector_to_state_dict...")
    # convert the flat merged checkpoint to a state dict
    ptm_check = ft_checks[0] # we use ft_checks[0] to pretend ptm_check, since we only need the state_dict keys in them
    merged_state_dict = vector_to_state_dict(
        merged_check, ptm_check, remove_keys=remove_keys
    )
    return merged_state_dict
### End of Model Exclusive Task Arithmetic ###

def convert_delta_to_ft(delta_weights):
    """
        input: delta_weights in our script.

        output: (ft_checks, unique) .
            ft_checks: a list of state_dicts that can be the input of func `do_merging()`.
            uniques: keys that are unique (that only appear once)
    """

    # first, we check the keys are the same, and get the length.
    N = -1
    for key in delta_weights.keys():
        N = max(N, len(delta_weights[key]))
    assert N > 0

    
    ft_checks = [{} for _ in range(N)]
    uniques = {}
    for key in delta_weights.keys():
        if len(delta_weights[key]) == N:
            for i in range(N):
                ft_checks[i][key] = delta_weights[key][i]
        else:
            assert len(delta_weights[key]) == 1
            uniques[key] = delta_weights[key][0]
    
    return (ft_checks, uniques)


def demo():
    ft_a = {'x': torch.Tensor([1,2,3]), 'y': torch.Tensor([4,5,6])}
    ft_b = {'x': torch.Tensor([-1,2,3]), 'y': torch.Tensor([0,0,0])}
    print(do_merging([ft_a, ft_b], K=0.9)) # bigger K conserve more params
    # import ipdb; ipdb.set_trace();
    
def merge2():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model1', type=str, required=True)
    parser.add_argument('--model2', type=str, required=True)
    parser.add_argument('--strategy', type=str, required=True)
    parser.add_argument('-K', type=float, default=1.0)

    args = parser.parse_args()
    
    model1 = torch.load(args.model1)
    model2 = torch.load(args.model2)

    working_dtype = torch.bfloat16
    for key in model1:
        model1[key] = model1[key].to(working_dtype)
    for key in model2:
        model2[key] = model2[key].to(working_dtype)
    

    # import ipdb; ipdb.set_trace();

    merged = do_merging_strategy([model1, model2], args.strategy, K=args.K)
    torch.save(merged, args.output)
if __name__ == '__main__':
    # demo()
    merge2()