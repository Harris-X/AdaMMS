import os
import re
import json
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import pandas as pd
from tqdm import tqdm
import random

from view_ppl_file import load_file as load_ppl_file
from view_ppl_file import calc_first_k_ppl_resp

def xlsx_to_json(file_path, output_path):
    # 使用 pandas 读取 xlsx 文件
    df = pd.read_excel(file_path, engine='openpyxl')
    
    # 将 DataFrame 转换为 JSON 格式
    json_str = df.to_json(orient='records', force_ascii=False)
    
    # 将 JSON 字符串写入到文件
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json_str)


def load_file(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    return data
# 
# alpha_pattern = re.compile(".*_([01]\.\d+)_.*")
alpha_pattern = re.compile(".*([01]\.\d+).*")


def check_items_not_in_path(items, path):
    """
    
    """
    for item in items:
        if item in path:
            return False
    return True

def search_files(args):
    
    path_base="/yeesuanAI05/thumt/dyy/model/Merge/eval/logs-kit"
    
    other_merge=['triple','score']
    logs = {}
    results = {}
    count=0
    task=args.task
    print(task,"--------------")
    for alpha in range(0,7):
        alpha/=10
        alpha=str(alpha)
        path1="llava2qwen2-"+alpha
        
        path2=path1+"_"+task+".json"
        path3=path1+"_"+task+".xlsx"
        
        path=os.path.join(path_base,path1,path2)
        path4=os.path.join(path_base,path1,path3)
        if not os.path.exists(path) :
            if not os.path.exists(path4):
                logs[alpha]=[]
            else:
                xlsx_to_json(path4,path)
                logs[alpha] = load_file(path)
        else:
            logs[alpha] = load_file(path)
        
     
    return logs, results

ppl_alpha_pattern = re.compile(".*_([01]\.\d+).*")

def smooth_list(inp, K=1):
    return [np.mean(inp[max(0,i-K):min(i+K+1,len(inp))]) for i in range(len(inp))]
    
def draw_delta(args):
    logs, results = search_files(args)
    
    parsed_results = {}

    parsed_resps = {}
    parsed_resps_eval = {}
        
    if args.sample:
        #random sample
        
        alpha = list(logs.keys())[0]
        if args.sample<1:
            samplenum=args.sample * len(logs[alpha])
        else:
            samplenum=args.sample
        sample_idx = random.sample(range(len(logs[alpha])), int(samplenum))
        for alpha in logs:
            logs[alpha] = [logs[alpha][i] for i in sample_idx]

    # get output of each doc from logs
    for alpha in logs:        
        parsed_resps[alpha] = [str(doc["prediction"]) for doc in logs[alpha]]


    # Sort the alphas
    alpha_sorted = sorted(parsed_resps.keys())
    
    # Calculate differences between adjacent parsed_resps lists
    differences = []
    original = []
    delta_result = []
    positive_delta_result = []
    negative_delta_result = []
    # for alph in alpha_sorted:


    for i in tqdm(range(1, len(alpha_sorted))):
        # if 10*(float(alpha_sorted[-1]) - float(alpha_sorted[0])) + 1 != len(alpha_sorted):
        #     print(alpha_sorted)
        #     break
        current_alpha = alpha_sorted[i]
        previous_alpha = alpha_sorted[i - 1]
        # next_alpha = alpha_sorted[i + 1]
                
        current_resps = parsed_resps[str(current_alpha)]
        previous_resps = parsed_resps[str(previous_alpha)]
        # next_resps = parsed_resps[str(next_alpha)]

        diff_strategy = args.diff_strategy
        # Calculate the difference in the number of positions with different strings
        if diff_strategy == 'embedding':
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            # import ipdb; ipdb.set_trace();
            current_resps_embed = embedder.encode(current_resps)
            previous_resps_embed = embedder.encode(previous_resps)
            # next_resps_embed = embedder.encode(next_resps)
            corresponding_similarity = [cosine_similarity([current_resps_embed[j]], [previous_resps_embed[j]]) for j in range(len(current_resps))]
            diff = 1 - np.average(corresponding_similarity)
            # corresponding_similarity2 = [cosine_similarity([current_resps_embed[j]], [next_resps_embed[j]]) for j in range(len(current_resps))]
            # diff += 1 - np.average(corresponding_similarity2)
        elif diff_strategy == 'exact_match':
            diff = sum(1 for j in range(len(current_resps)) if not( (str(current_resps[j]).lower() == str(previous_resps[j]).lower()) ) )
            # for j in range(len(current_resps)) :
            #     if  (str(current_resps[j]).lower() == str(previous_resps[j]).lower()) and (str(current_resps[j]).lower() == str(next_resps[j]).lower()):
            #         print(current_resps[j])
            #     else:
            #         print(str(previous_resps[j]).lower())
            #         print(str(current_resps[j]).lower())
            #         print(str(next_resps[j]).lower())
                    

            
            # print(current_alpha)
            # diff += sum(1 for j in range(len(current_resps)) if str(current_resps[j]).lower() != str(next_resps[j]).lower())
        else:
            raise NotImplementedError()
        differences.append(diff)


    # smooth
    # differences = smooth_list(differences, K=args.smooth)
    # delta_result = smooth_list(delta_result, K=args.smooth)

    print("response differences", (differences))
    try:
        min_value=min(differences)
        all_min_indices = [i for i, x in enumerate(differences) if x == min_value]
        for i in all_min_indices:
            print(alpha_sorted[i+1])
        
    except:
        print("失败。。。")
    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots()

    fig.set_size_inches(12.8,4.8)

    # Plot the first list on the left y-axis
    ax1.plot(alpha_sorted[1:], differences, label='Differences on response', color='tab:blue')
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
        plt.savefig(f"/yeesuanAI05/thumt/dyy/model/Merge/eval/new_diff3_pic_qwen_inter_alpha/{args.merge}_{args.task}_{args.diff_strategy}_smooth{args.smooth}_{f'_sample{args.sample}'}.png")
 
    # Now differences contains the number of positions with different strings for each pair of adjacent alphas
    return differences

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default="/yeesuanAI05/thumt/dyy/model/Merge/eval/wxc-new/0919--llava2cogvlm--interpolation")
    parser.add_argument('--dirname', type=str, default='0911-llava2sharegpt_0.') 
    parser.add_argument('--ppl-path', type=str, default=None)
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--smooth', type=int, default=0)
    parser.add_argument('--diff_strategy', type=str, default='embedding')
    parser.add_argument('--sample', type=int, default=None)
    # /yeesuanAI05/thumt/dyy/model/Merge/eval/logs/interpolation_mplugowl2_cogvlm_chat_vqa_mme/0606_0252_interpolation_mplugowl2_cogvlm_chat_vqa_0.4_mme_mplugowl2_model_args_73cb15/results.json
    parser.add_argument('--merge', type=str, default='qwen') #mplugowl2cogvlm ','cogvlm2mplugowl','sharegpt2llava
    parser.add_argument('--merge2', type=str, default='mplugowl_llava')  #cogvlm_mplugowl2','mplugowl2_cogvlm_','llava_sharegpt
    parser.add_argument('--reverse', action="store_true")
    args = parser.parse_args()
    
    reflist=[ 'refcocog_bbox_test' ,'refcoco_bbox_testA']
    # task_list_exact=[ "OCRBench","ScienceQA_VAL","TextVQA_VAL","MMMU_DEV_VAL","MME","SEEDBench_IMG","OK-VQA"]
    task_list_exact=["MMMU_DEV_VAL", "OCRBench","ScienceQA_VAL","TextVQA_VAL", "MME","SEEDBench_IMG"]
    # task_list_exact=["SEEDBench_IMG","OK-VQA"]

    random.seed(2)
    
        
    for task in task_list_exact:        
        args.task = task
        # args.diff_strategy = 'exact_match'  #exact_match  embedding
                        
        draw_delta(args)
        print(args.task,"-----------------------over!")
    
    