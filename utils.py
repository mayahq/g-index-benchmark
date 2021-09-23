
from dataclasses import dataclass
import glob
import json
import os
import random
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import seaborn as sns

from node_utils import node_divergence

DOMAIN_ROOT = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'domains/')
AVAILABLE_DOMAINS = [domain for domain in os.listdir(
    DOMAIN_ROOT) if domain not in ['.ipynb_checkpoints']]


def nsplit(n_samples_per_domain: int, num_domains: int) -> np.array:
    n_samples_total = n_samples_per_domain * num_domains
    result = []
    lim = n_samples_total - num_domains + 1
    for i in range(num_domains - 1):
        result.append(np.random.randint(1, lim, 1).item())
        lim = lim - result[-1] + 1
    result.append(lim)
    return np.array(result)


def resplit(n_samples_per_domain: int, num_domains: int) -> np.array:
    while True:
        sample = nsplit(n_samples_per_domain, num_domains)
        _, counts = np.unique(sample, return_counts=True)
        # No Curricula Distribution distribution has same counts for more than 3 templates
        if np.all(counts < int(num_domains/5)):
            return sample


def get_domain_example(domain_key: str, domain_root: str = "domains/") -> Dict:
    domain_path = os.path.join(domain_root, domain_key)
    domain_files = glob.glob(domain_path+"/*.json")
    if not domain_files:
        return {}
    else:
        return json.load(open(random.choice(domain_files)))


def calculate_dd(domain_1: Union[os.PathLike, str], domain_2: Union[os.PathLike, str], verbose=False) -> float:
    if domain_1 in AVAILABLE_DOMAINS and domain_2 in AVAILABLE_DOMAINS:
        domain_1 = os.path.join(DOMAIN_ROOT, domain_1)
        domain_2 = os.path.join(DOMAIN_ROOT, domain_2)

    if os.path.isdir(domain_1) and os.path.isdir(domain_2):
        if verbose:
            print("Calculating Distance b/w two domains")
            print(
                f"Selected domains are {domain_1.split('/')[-1]} & {domain_2.split('/')[-1]}")
        files_domain_1 = glob.glob(domain_1+"/*.json")
        files_domain_2 = glob.glob(domain_2+"/*.json")

        ovr_values = []
        for f1 in files_domain_1:
            json_file_1 = json.load(open(f1))
            scores = []
            for f2 in files_domain_2:
                json_file_2 = json.load(open(f2))

                score = node_divergence(
                    json_file_1["flow"], json_file_2["flow"])
                scores.append(score)
                if score == 0:
                    break
            ovr_values.append(np.min(scores))
        return np.mean(ovr_values)

    elif os.path.isfile(domain_1) and os.path.isfile(domain_2):
        if verbose:
            print("Calculating Domain Distance b/w two JSON files")
            print(
                f"Selected JSONs are {domain_1.split('/')[-1]} & {domain_2.split('/')[-1]}")
        json_file_1 = json.load(open(domain_1))
        json_file_2 = json.load(open(domain_2))
        return node_divergence(json_file_1['flow'], json_file_2['flow'])


def cache_dd() -> Dict:
    dd_cache = defaultdict()
    for da in AVAILABLE_DOMAINS:
        for db in AVAILABLE_DOMAINS:
            if (db, da) in dd_cache.keys():
                dd_cache[(da, db)] = dd_cache[(db, da)]
            else:
                da_root = os.path.join(DOMAIN_ROOT, da)
                db_root = os.path.join(DOMAIN_ROOT, db)
                dd_cache[(da, db)] = calculate_dd(
                    da_root, db_root, verbose=False)
    return dd_cache


def generate_dd_matrix(domains: list = None):
    if domains is None:
        domains = AVAILABLE_DOMAINS
    df = pd.DataFrame(columns=domains,
                      index=domains, dtype='float').fillna(-1)

    for d1 in domains:
        d1_p = os.path.join(DOMAIN_ROOT, d1)
        for d2 in domains:
            d2_p = os.path.join(DOMAIN_ROOT, d2)
            score = calculate_dd(d1_p, d2_p)
            df[d1][d2] = score

    sns.heatmap(df, vmin=0.0, vmax=1.0, annot=True)


def scale_dots(gi_min: float, current: float) -> float:
    return 120 * (1 + ((current - gi_min) / (gi_min)))


def get_domain_lengths(domain_list=AVAILABLE_DOMAINS) -> Dict:
    DOMAIN_LENGTHS = defaultdict()
    for domain in domain_list:
        domain_files = glob.glob(os.path.join(DOMAIN_ROOT, domain)+"/*.json")
        lengths = []
        for domain_file in domain_files:
            json_file = json.load(open(domain_file))
            lengths.append(len(json.dumps(json_file['flow'])))

        DOMAIN_LENGTHS[domain] = mean(lengths)
    return dict(DOMAIN_LENGTHS)


def print_dataclass(dc: dataclass):
    for field in dc.__dataclass_fields__:
        value = getattr(dc, field)
        print(field, ": ", value)


def random_row_mean(multilevel_dict: Dict[str,Dict[str,float]]) -> float:
    random_row = multilevel_dict.get(
        np.random.choice(list(multilevel_dict.keys()))
    ).values()
    return np.mean(list(random_row))
