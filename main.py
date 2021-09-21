import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime

from g_index import Benchmark, cache_dd


def main(args):
    if not os.path.isdir(args.exp_dir):
        raise ValueError(f"Experiment Directory {args.exp_dir} doesn't exist")
    if not os.path.isdir(args.temp_dir):
        raise ValueError(f"Template Directory {args.temp_dir} doesn't exist")
    
    exp_files = glob.glob(os.path.join(args.exp_dir,"*.json"))
    records = defaultdict(dict)
    if not len(exp_files):
        print("No Experiment Files found in this directory, Please select the right directory")
        return
    dd_cache = cache_dd()
    for fpath in exp_files:
        exp = Benchmark(json.load(open(fpath)),dd_cache=dd_cache)
        records[fpath.split("/")[-1]] = exp.GetExperimentIndices(return_dict=True)
        if args.print_metrics:
            print(records[fpath.split("/")[-1]])
    if args.save_metrics:
        with open(f'results/results_{datetime.now().strftime("%Y_%m_%d %H_%M_%S.json")}','w') as f:
            json.dump(records,f,indent=2)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--exp_dir',
                        default='experiments',
                        help='Set the directory where the experiment files are stored')
    parser.add_argument('-t','--temp_dir',
                        default='templates',
                        help='Set the directory where the template files are stored')
    parser.add_argument('-p','--print_metrics',
                        default=False,
                        help='Set whether to print the metrics on the command line.')
    parser.add_argument('-s','--save_metrics',
                        default=True,
                        help='Set whether to dump metrics to a JSON file.')

    args = parser.parse_args()
    main(args)
