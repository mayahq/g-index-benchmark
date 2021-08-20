from g_index import Benchmark
import glob
import os
import json
from collections import defaultdict

EXPERIMENT_DIR = os.path.join(os.getcwd(),'experiments/')
exp_files = glob.glob(os.path.join(EXPERIMENT_DIR,"*.json"))

records = defaultdict(dict)
for fpath in exp_files:
    exp = Benchmark(json.load(open(fpath)))
    records[fpath.split("/")[-1]] = exp.GetExperimentIndices()

print(records)