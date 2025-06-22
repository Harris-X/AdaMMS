import os
import re
import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np

from tqdm import tqdm
import random

def load_file(path):
    with open(path, 'r') as f:
        data = json.load(f)
    # if 'results' not in path:
    #     print(path)
    return data
# 
# alpha_pattern = re.compile(".*_([01]\.\d+)_.*")
alpha_pattern = re.compile(".*([01]\.\d+).*")

def load_folder_list(args):
    folder_path = args.path
    log_file_path = f'{args.task}.json'
    result_file_path = 'results.json'
    ls = os.listdir(folder_path)
    logs = {}
    results = {}
    for run_folder in ls:
        if args.dirname in run_folder:
            alpha = alpha_pattern.match(run_folder)
            if alpha and len(alpha[1]) <= 3:
                alpha = alpha[1]
                logs[alpha] = load_file(os.path.join(folder_path, run_folder, log_file_path))
                results[alpha] = load_file(os.path.join(folder_path, run_folder, result_file_path))
        
    if args.reverse:
        reversed_logs = {}
        reversed_results = {}
        for alpha in logs:
            reversed_alpha = str(round(1 - float(alpha), 1))
            reversed_logs[reversed_alpha] = logs[alpha]
            reversed_results[reversed_alpha] = results[alpha]
        return reversed_logs, reversed_results
    
    return logs, results


def load_new_folder_list(args):
    folder_path = args.path
    log_file_path = f'{args.task}.json'
    result_file_path = 'results.json'
    ls = os.listdir(folder_path)
    logs = {}
    results = {}
    for run_folder in ls:
        if args.dirname in run_folder:
            alpha = alpha_pattern.match(run_folder)
            if alpha and len(alpha[1]) <= 3:
                alpha = alpha[1]
                logs[alpha] = load_file(os.path.join(folder_path, run_folder, log_file_path))
                results[alpha] = load_file(os.path.join(folder_path, run_folder, result_file_path))
        
    if args.reverse:
        reversed_logs = {}
        reversed_results = {}
        for alpha in logs:
            reversed_alpha = str(round(1 - float(alpha), 1))
            reversed_logs[reversed_alpha] = logs[alpha]
            reversed_results[reversed_alpha] = results[alpha]
        return reversed_logs, reversed_results
    
    return logs, results

def check_items_not_in_path(items, path):
    """
    
    """
    for item in items:
        if item in path:
            return False
    return True

def search_files(args):
    
    logs_paths=["/yeesuanAI05/thumt/dyy/model/Merge/eval/paper/sample100",'/yeesuanAI05/thumt/dyy/model/Merge/eval/newdifflog',"/yeesuanAI05/thumt/dyy/model/Merge/eval/logs-new"]
   
    log_file_path = f'{args.task}.json'
    result_file_path = 'results.json'
    other_merge=['triple','ties','metagpt','task_arithmetic','dare']
    logs = {}
    results = {}
    count=0
    for root_dir in logs_paths:
        for path in os.listdir(root_dir):
            
            if args.merge in path :
                alpha = alpha_pattern.match(path)
                if alpha and len(alpha[1]) <= 3:
                    alpha = alpha[1]
                    
                    for p in os.listdir(os.path.join(root_dir,path)):
                        
                        if args.task in p: 
                            if 'ok_vqa' in args.task:
                                log_file_path='ok_vqa_val2014.json'                    
                            if alpha in logs.keys():
                                continue
                                print(path)    
                            
                            logs[alpha] = load_file(os.path.join(root_dir,path,p, log_file_path))
                            results[alpha] = load_file(os.path.join(root_dir,path,p, result_file_path))
                            count+=1
                            print(os.path.join(root_dir,path,p, log_file_path))

   
    return logs, results
def load_folder(args):
    folder_path = args.path
    log_file_path = f'{args.task}.json'
    result_file_path = 'results.json'
    ls = os.listdir(folder_path)
    logs = {}
    results = {}
    for run_folder in ls:
        if args.task not in run_folder:
            continue
        alpha = alpha_pattern.match(run_folder)
        if alpha and len(alpha[1]) <= 3:
            alpha = alpha[1]
            logs[alpha] = load_file(os.path.join(folder_path, run_folder, log_file_path))
            results[alpha] = load_file(os.path.join(folder_path, run_folder, result_file_path))
    
    if args.reverse:
        reversed_logs = {}
        reversed_results = {}
        for alpha in logs:
            reversed_alpha = str(round(1 - float(alpha), 1))
            reversed_logs[reversed_alpha] = logs[alpha]
            reversed_results[reversed_alpha] = results[alpha]
        return reversed_logs, reversed_results
    
    return logs, results

ppl_alpha_pattern = re.compile(".*_([01]\.\d+).*")

def smooth_list(inp, K=1):
    return [np.mean(inp[max(0,i-K):min(i+K+1,len(inp))]) for i in range(len(inp))]
    
def draw_delta(args):
    logs, results = search_files(args)
    
    parsed_results = {}

    parsed_resps = {}
    parsed_resps_eval = {}
    
   
    if 'mme' in args.task:
        for alpha in logs:
            logs[alpha]["logs"] = [doc for doc in logs[alpha]["logs"] if 'mme_percetion_score' in doc]
    
    if args.sample and "0.3" in logs and  len(logs['0.5']["logs"])>args.sample :
        #random sample
        
        alpha = list(logs.keys())[0]
        sample_idx = random.sample(range(len(logs[alpha]["logs"])), int(args.sample))
        for alpha in logs:
            logs[alpha]["logs"] = [logs[alpha]["logs"][i] for i in sample_idx]

    # get output of each doc from logs
    for alpha in results:
        if 'mmmu' in args.task:
            parsed_resps[alpha] = [doc["filtered_resps"][0] for doc in logs[alpha]["logs"]]
        else:
            parsed_resps[alpha] = [doc["resps"][0][0] for doc in logs[alpha]["logs"]]

    #get the score of every sample
    for alpha in results:
        if 'ok_vqa' in args.task:
            parsed_resps_eval[alpha] = [float(doc["exact_match"]) for doc in logs[alpha]["logs"]]
        elif 'mme' in args.task:
            parsed_resps_eval[alpha] = [float(doc["mme_percetion_score"]["score"]) for doc in logs[alpha]["logs"]]
        elif 'ocrbench' in args.task:
            parsed_resps_eval[alpha] = [float(doc["ocrbench_accuracy"]["score"]) for doc in logs[alpha]["logs"]]
        elif 'mmmu' in args.task:
            parsed_resps_eval[alpha] = [float(doc["mmmu_acc"]["answer"] == doc["mmmu_acc"]["parsed_pred"]) for doc in logs[alpha]["logs"]]
        elif 'textvqa_val' in args.task:
            parsed_resps_eval[alpha] = [float(doc["exact_match"]) for doc in logs[alpha]["logs"]]
        elif 'refcoco' in args.task:
            parsed_resps_eval[alpha] = [float(1.0) for doc in logs[alpha]["logs"]]
        else:
            raise NotImplementedError()
            parsed_resps_eval[alpha] = [doc["resps"][0][0] for doc in logs[alpha]["logs"]]


    # Sort the alphas
    alpha_sorted = sorted(parsed_resps.keys())
    
    # Calculate differences between adjacent parsed_resps lists
    differences = []
  
    if args.diff_strategy == 'embedding':
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print(alpha_sorted)
    for i in tqdm(range(1, len(alpha_sorted)-1)):
       
        current_alpha = alpha_sorted[i]
        previous_alpha = alpha_sorted[i - 1]
        next_alpha = alpha_sorted[i + 1]
    
        current_resps = parsed_resps[str(current_alpha)]
        previous_resps = parsed_resps[str(previous_alpha)]
        next_resps = parsed_resps[str(next_alpha)]

        diff_strategy = args.diff_strategy
        # Calculate the difference in the number of positions with different strings
        if diff_strategy == 'embedding':
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            embedder = SentenceTransfwormer("sentence-transformers/all-MiniLM-L6-v2")
            # import ipdb; ipdb.set_trace();
            current_resps_embed = embedder.encode(current_resps)
            previous_resps_embed = embedder.encode(previous_resps)
            next_resps_embed = embedder.encode(next_resps)
            corresponding_similarity = [cosine_similarity([current_resps_embed[j]], [previous_resps_embed[j]]) for j in range(len(current_resps))]
            diff = 1 - np.average(corresponding_similarity)
            corresponding_similarity2 = [cosine_similarity([current_resps_embed[j]], [next_resps_embed[j]]) for j in range(len(current_resps))]
            diff += 1 - np.average(corresponding_similarity2)
        elif diff_strategy == 'exact_match':
            diff = sum(1 for j in range(len(current_resps)) if not( (str(current_resps[j]).lower() == str(previous_resps[j]).lower()) and (str(current_resps[j]).lower() == str(next_resps[j]).lower())) )
            # print(current_alpha)
            # diff += sum(1 for j in range(len(current_resps)) if str(current_resps[j]).lower() != str(next_resps[j]).lower())
        else:
            raise NotImplementedError()
        differences.append(diff)

      
    # smooth
    differences = smooth_list(differences, K=args.smooth)
    # delta_result = smooth_list(delta_result, K=args.smooth)

    print("response differences", (differences))
    try:
        min_value=min(differences)
        all_min_indices = [i for i, x in enumerate(differences) if x == min_value]
        for i in all_min_indices:
            print(alpha_sorted[i+1])
        
    except:
        print("fail...")
    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots()

    fig.set_size_inches(12.8,4.8)

    # Plot the first list on the left y-axis
    ax1.plot(alpha_sorted[1:-1], differences, label='Differences on response', color='tab:blue')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Differences', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    try:
        ax1.set_ylim(min(differences), max(differences))
    except:
        print("difference is None")
        ax1.set_ylim(0, 100)
    # Add a legend
    fig.legend(loc='upper left')

    # Add a title
    plt.title('Comparison of Differences and Delta Result')

    # Show the plot
    # plt.savefig(f"{os.path.basename(args.path)}_{args.diff_strategy}_smooth{args.smooth}{f'_sample{args.sample}' if args.sample else ''}{'_ppl' if args.ppl_path else ''}.png")
    if differences is not  None:
        plt.savefig(f"figure/{args.merge}_{args.task}_{args.diff_strategy}_smooth{args.smooth}_{f'_sample{args.sample}'}.png")
 
    # Now differences contains the number of positions with different strings for each pair of adjacent alphas
    return differences

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default="/yeesuanAI05/thumt/dyy/model/Merge/eval/wxc-new/0919--llava2cogvlm--interpolation")
    parser.add_argument('--dirname', type=str, default='0911-llava2sharegpt_0.') 
    parser.add_argument('--ppl-path', type=str, default=None)
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--smooth', type=int, default=0)
    parser.add_argument('--diff_strategy', type=str, default='exact_match')
    parser.add_argument('--sample', type=int, default=100)
    parser.add_argument('--merge', type=str, default='llava2cogvlm') #mplugowl2cogvlm ','cogvlm2mplugowl','sharegpt2llava
    parser.add_argument('--merge2', type=str, default='mplugowl_llava')  #cogvlm_mplugowl2','mplugowl2_cogvlm_','llava_sharegpt
    parser.add_argument('--reverse', action="store_true")
    args = parser.parse_args() #'gqa',
    # task_list_exact = ['gqa','ocrbench', 'seedbench',  'mmmu_val', 'textvqa_val', 'ok_vqa', "mme", "vizwiz_vqa_val"  ]
    task_list_exact = ["mme"  ]
    # task_list_embedding = ['refcoco_bbox_testA', 'refcocog_bbox_test' , 'refcoco+_bbox_testA']
    omit_list=['ok_vqa',  "vizwiz_vqa_val"]
    reflist=[ 'refcocog_bbox_test' ,'refcoco_bbox_testA']
    merge_list=['llava2mplugowl','mplugowl2cogvlm' ,'cogvlm2mplugowl','sharegpt2llava','llava2sharegpt', 'llava2cogvlm', 'sharegpt2cogvlm'] 
    random.seed(2)
    # merge_list=['mplugowl2cogvlm']
    for x in merge_list: 
                   
        args.merge = x
        # args.merge2 = y      
        
        for task in task_list_exact:        
            args.task = task
            # args.diff_strategy = 'embedding'                           
            draw_delta(args)
            print(args.task,"-----------------------over!")
    
    