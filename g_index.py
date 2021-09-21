import itertools
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from statistics import mean
from time import time
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import seaborn as sns
from node_utils import node_divergence
from utils import (AVAILABLE_TEMPLATES, TEMPLATE_DETAILS, Dataset,
                   DatasetDetails, domain_distance,cache_dd)

def nsplit(n_samples_total, num_domains):
    ans = []
    lim = n_samples_total - num_domains + 1
    for i in range(num_domains - 1):
        ans.append(np.random.randint(1, lim, 1).item())
        lim = lim - ans[-1] + 1
    ans.append(lim)
    return np.array(ans)

def resplit(n_samples_total,num_domains):
    runs_to_sample=0
    while True:
        runs_to_sample +=1
        sample = nsplit(n_samples_total,num_domains)
        unique, counts = np.unique(sample, return_counts=True)
        if np.all( counts < int(num_domains/5) ):
            return sample

@dataclass(frozen=True)
class ExperimentIndices:
    GD: Dict[str, Dict[str,float]] 
    PC: Dict[str, float]
    PTheta: Dict[str, float]
    TaskDomains: List[str] = field(default_factory=list)
    CurriculaDomains: List[str] = field(default_factory=list)
    P: float = 0.0
    E: float = 0.0
    GIndex: float = 0.0
    AvgPerf: float = 0.0

class Benchmark:

    def __init__(self,experiment=None,taskDomains=None,CurriculaDomains=None,GD=None,PC=None,P=None,E=None,PTheta=None,dd=None,use_dd_cache=True,dd_cache=None):
        if experiment is not None and isinstance(experiment,dict):
            self.experiment = experiment

        elif experiment is not None and isinstance(experiment,Union[str,os.PathLike].__args__):
            self.experiment = json.load(open(experiment))
        else:
            self.experiment = None

        self.taskDomains = taskDomains
        self.CurriculaDomains = CurriculaDomains
        self.GD = GD
        self.PC = PC
        self.P = P
        self.E = E
        self.PTheta = PTheta
        self.dd = dd
        self.use_dd_cache = use_dd_cache
        if self.use_dd_cache and dd_cache is not None:
            self.dd_cache = dd_cache
        elif self.use_dd_cache :
            self.dd_cache = cache_dd()
            print(dd_cache)
        else:
            self.dd_cache = None

    def get_curricula_domains(self) -> Dict:
        """
        This takes in a list of templates in the training array and returns
        a set of curriculum domains. In initial versions this is just a list of all templates, with 
        number of samples per template.
        """
        return self.experiment["train_set"]["data"]

    def probability_of_curricula(self, curriculum, curricula_domains):
        """
        What is the probability of the given curriculum domain occurring in 
        the complete curricula. Initially, this is just (number_of_samples_of_templateN/total_no_of_samples)
        """
        probability = 1/(1+ np.log2(curriculum["num_samples"]))
        return probability

    def get_task_domains(self) -> Dict:
        """
        Get a list of all tasks, i.e. scope of tasks
        """
        return self.experiment["test_set"]["data"]

    def get_priors(self)-> float:
        """
        `priors (P)` : data which is built into the system _before_ training (are you fine-tuning something already fine-tuned?). This is negligible (~0) for most of current models.
        """
        return 0.0001
    
    def get_task_compute(self) ->float:
        """
        compute exposure to a curricula
        """
        return self.experiment["train_cost"]["compute"] + self.experiment["test_cost"]["compute"]

    def get_experience(self):
        if self.E is not None:
            return np.log2(self.E)
        elif self.experiment is not None:
            self.E = self.experiment['train_cost']['compute']
            return np.log2(self.E)

    def get_domain_details(self, domain: Dict ) -> Dict:
        """
        In this case, this returns the template details for a given template.
        Output : 
        {
                "dag_length" : {
                     "inflated": 937.0,
                     "deflated": 429.2,
                }
        """
        if domain["name"] in AVAILABLE_TEMPLATES:
            extracted_details = {
                "dag_length" : TEMPLATE_DETAILS.get(domain["name"]),
            }

            return extracted_details
        else :
            return {
                "dag_length" : {
                     "inflated": 0,
                     "deflated": 0,
                }
            }

    def get_performance_theta(self,task,raw_perf=False):
        if self.experiment:
            performance_per_template = self.experiment["performance"]["templates"]
            selected_task = next(
            (x for x in performance_per_template if x["name"] == task["name"]), None)
            domain_details = self.get_domain_details(selected_task)
            performance = 1 - selected_task["divergence"]
            return np.exp(performance*12)
        elif self.PTheta:
            performance = self.Ptheta.get(task)
            return performance


    def get_generalization_difficulty(self, task_domain, curriculum_domain):
        """
        returns the exponential generalization difficulty
        (domain_distance_score * 10)^e
        """
        score = self.domain_distance_score(task_domain, curriculum_domain)
        return np.exp(score *10)

    def domain_distance_score(self, task_domain, curriculum_domain):
        if self.dd is not None:
            if isinstance(self.dd,Union[int,float].__args__):
                return self.dd
            elif isinstance(self.dd,dict):
                return self.dd.get(task_domain['name']).get(curriculum_domain['name'],1.0)
        
        if self.use_dd_cache and self.dd_cache is not None:
            return self.dd_cache[(task_domain['name'],curriculum_domain['name'])]
        
        total_samples = 2
        task_dataset_details = DatasetDetails(total=total_samples, data=[{"name": task_domain["name"], "num_samples":total_samples}])
        curriculum_dataset_details = DatasetDetails(total=total_samples, data=[{"name": curriculum_domain["name"], "num_samples":total_samples}])
        domain_distance_score = domain_distance(Dataset.from_details(
            task_dataset_details), Dataset.from_details(curriculum_dataset_details))

        return domain_distance_score

    def _drop_duplicate_tuples(self,l):
        """
        Drop the tuples (b,a) if (a,b) exists already
        """
        uniq = []
        for i in l:
            if not (i in uniq or tuple([i[1], i[0]]) in uniq):
                uniq.append(i)
        return uniq

    def GetExperimentIndices(self,return_dict: bool =False) -> Union[ExperimentIndices,Dict]:
        TaskDomains = self.get_task_domains() if self.taskDomains is None else self.taskDomains
        CurriculaDomains = self.get_curricula_domains() if self.CurriculaDomains is None else self.CurriculaDomains
        GD = { task.get("name"): {curricula.get("name"): self.get_generalization_difficulty(task,curricula) for curricula in CurriculaDomains} for task in TaskDomains } if self.GD is None else self.GD
        PC = { curricula.get("name"):self.probability_of_curricula(curricula,CurriculaDomains) for curricula in CurriculaDomains } if self.PC is None else self.PC
        P = self.get_priors() if self.P is None else self.P
        E = self.get_experience()
        Ptheta = { task.get("name"):self.get_performance_theta(task,raw_perf=True) for task in TaskDomains } if self.PTheta is None else self.PTheta
        AvgPerf = round(mean(list(Ptheta.values())),3)
        
        TaskContributions = defaultdict()
        for task in TaskDomains:
            TaskContributions[task.get("name")] = np.sqrt(Ptheta.get(task.get("name")) * sum( [ GD.get(task.get("name")).get(curricula.get("name")) * (PC.get(curricula.get("name"))/ (P+E)) for curricula in CurriculaDomains]))
        TaskContributions = dict(TaskContributions)
        GIndex = round(mean(list(TaskContributions.values())),3)
        exp_indices = ExperimentIndices(GD=GD,
                                        PC=PC,
                                        PTheta=Ptheta,
                                        TaskDomains=TaskDomains,
                                        CurriculaDomains=CurriculaDomains,
                                        P=P,E=E,
                                        GIndex=GIndex,
                                        AvgPerf=AvgPerf)
        if return_dict:
            return asdict(exp_indices)
        return exp_indices

    def generate_dd_matrix(self):
        template_pair_half_grid = self._drop_duplicate_tuples(list(itertools.product(AVAILABLE_TEMPLATES,AVAILABLE_TEMPLATES)))
        tp_dict = {f: json.load(open(os.path.join(os.getcwd(),'templates',f)))['flow'] for f in AVAILABLE_TEMPLATES }
        df = pd.DataFrame(columns=AVAILABLE_TEMPLATES,index=AVAILABLE_TEMPLATES)
        
        for ent in template_pair_half_grid:
            tp0,tp1 = ent[0], ent[1]
            if df[tp1][tp0] != -1:
                df[tp0][tp1] = df[tp1][tp0]
            else:
                df[tp0][tp1] = node_divergence(tp_dict.get(tp0,[]),tp_dict.get(tp1,[]))

            if df[tp1][tp0] == -1.0 and df[tp0][tp1] != -1.0:
                df[tp1][tp0] = df[tp0][tp1]

        sns.heatmap(df,vmin=0.0,vmax=1.0,annot=True)
