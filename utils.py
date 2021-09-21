
import glob
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns

from node_utils import node_divergence

DOMAIN_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),'domains/')
AVAILABLE_DOMAINS = [ domain for domain in os.listdir(DOMAIN_ROOT) if domain not in ['.ipynb_checkpoints']]

try:
    with open('assets/lengths.json','r') as f:
        DOMAIN_LENGTHS  = json.load(f)
except FileNotFoundError:
    print("No `lengths.json` was found, Using zero values for lengths, Please note that you would not be able to get reproduce results without it.")
    DOMAIN_LENGTHS = {temp_name: {"inflated":0,"deflated":0} for temp_name in AVAILABLE_DOMAINS}

def get_domain_example(domain_key: str,domain_root:str = "domains/") -> Dict:
    domain_path = os.path.join(domain_root,domain_key)
    domain_files = glob.glob(domain_path+"/*.json")
    if not domain_files:
        return {}
    else:
        return json.load(open(random.choice(domain_files)))

def calculate_dd(domain_1: Union[os.PathLike,str],domain_2: Union[os.PathLike,str],verbose=False) -> float:
    if domain_1 in AVAILABLE_DOMAINS and domain_2 in AVAILABLE_DOMAINS:
        domain_1 = os.path.join(DOMAIN_ROOT,domain_1)
        domain_2 = os.path.join(DOMAIN_ROOT,domain_2)

    if os.path.isdir(domain_1) and os.path.isdir(domain_2):
        if verbose:
            print("Calculating Distance b/w two domains")
            print(f"Selected domains are {domain_1.split('/')[-1]} & {domain_2.split('/')[-1]}")
        files_domain_1 = glob.glob(domain_1+"/*.json")
        files_domain_2 = glob.glob(domain_2+"/*.json")

        ovr_values = []
        for f1 in files_domain_1:
            json_file_1 = json.load(open(f1))
            scores = []
            for f2 in files_domain_2:
                json_file_2 = json.load(open(f2))

                score = node_divergence(json_file_1["flow"], json_file_2["flow"])
                scores.append(score)
                if score == 0:
                    break
            ovr_values.append(np.min(scores))  
        return np.mean(ovr_values)

    elif os.path.isfile(domain_1) and os.path.isfile(domain_2):
        if verbose:
            print("Calculating Domain Distance b/w two JSON files")
            print(f"Selected JSONs are {domain_1.split('/')[-1]} & {domain_2.split('/')[-1]}")
        json_file_1 = json.load(open(domain_1))
        json_file_2 = json.load(open(domain_2))
        return node_divergence( json_file_1['flow'],json_file_2['flow'])

def cache_dd():
    dd_cache = defaultdict()
    for ta in AVAILABLE_DOMAINS:
        for tb in AVAILABLE_DOMAINS:
            if (tb,ta) in dd_cache.keys():
                dd_cache[(ta,tb)] = dd_cache[(tb,ta)]
            else:
                ta_root = os.path.join(DOMAIN_ROOT,ta)
                tb_root = os.path.join(DOMAIN_ROOT,tb)
                dd_cache[(ta,tb)] = calculate_dd(ta_root,tb_root,verbose=False)
    return dd_cache

def drop_duplicate_tuples(l):
    """
    Drop the tuples (b,a) if (a,b) exists already
    """
    uniq = []
    for i in l:
        if not (i in uniq or tuple([i[1], i[0]]) in uniq):
            uniq.append(i)
    return uniq

def generate_dd_matrix(domains_list=None):
    if domains_list is None:
        domains_list = AVAILABLE_DOMAINS
    df = pd.DataFrame(columns=domains_list,index=domains_list,dtype='float').fillna(-1)
    
    for d1 in domains_list:
        d1_p = os.path.join(DOMAIN_ROOT,d1)
        for d2 in domains_list:
            d2_p = os.path.join(DOMAIN_ROOT,d2)
            score = calculate_dd(d1_p,d2_p)
            df[d1][d2] = score
    
    sns.heatmap(df,vmin=0.0,vmax=1.0,annot=True)

def scale_dots(gi_min,gi_max,current):
    return 120 * (1 +((current - gi_min) / (gi_min )))

def get_exp_components(exp_file: Union[str,os.PathLike],truncate_domains: bool = True,truncation_count: int = 2):
    try:
        json_dict = json.load(open(exp_file))

        IS = json_dict["model"]["train_params"]["model_name"]
        IS = IS.replace("/"," ") if "/" in IS else IS
        CurriculaDomains = json_dict["train_set"]["data"]
        TaskDomains = json_dict["test_set"]["data"]
        E = np.log2(json_dict["train_cost"]["compute"])
        GD = np.exp( json_dict["domain_distance"]*10 )
    
        if len(CurriculaDomains) > truncation_count and truncate_domains:
            print(f"Truncated Domains Data to {truncation_count} values")
            CurriculaDomains = CurriculaDomains[:4]
        if len(TaskDomains) > 4 and truncate_domains:
            TaskDomains = TaskDomains[:4]

        print(f"IS: {IS}")
        print(f"CurriculaDomains: {CurriculaDomains}")
        print(f"TaskDomains: {TaskDomains}")
        print(f"E: {E}")
        print(f"GD: {GD}")

        
    except FileNotFoundError as fe:
        print("File doesn't exist")


def get_template_lengths(temp_name: str) -> float:





