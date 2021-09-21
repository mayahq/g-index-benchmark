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

from utils import (AVAILABLE_DOMAINS, DOMAIN_LENGTHS, DOMAIN_ROOT, cache_dd,
                   calculate_dd)


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
        else:
            self.dd_cache = None

    def get_curricula_domains(self,return_total: bool = False)-> Union[Dict,int]:
        """
        This takes in a list of templates in the training array and returns
        a set of curriculum domains. In initial versions this is just a list of all templates, with 
        number of samples per Domain.
        """
        if self.experiment:
            if return_total:
                return self.experiment["train_set"]["total"]
            return self.experiment["train_set"]["data"]
            
    def probability_of_curricula(self, curriculum: Dict) -> float:
        """
        What is the probability of the given curriculum domain occurring in 
        the complete curricula. Initially, this is just (number_of_samples_of_templateN/total_no_of_samples)
        """
        probability = 1/(1+ np.log2(curriculum["num_samples"]))
        return probability

    def get_task_domains(self,return_total: bool = False) -> Union[Dict,int]:
        """
        Get a list of all tasks, i.e. scope of tasks
        """
        if self.experiment:
            if return_total:
                return self.experiment["test_set"]["total"]
            return self.experiment["test_set"]["data"]

    def get_priors(self)-> float:
        """
        `priors (P)` : data which is built into the system _before_ training (are you fine-tuning something already fine-tuned?). This is negligible (~0) for most of current models.
        """
        return 1e-4
    
    def get_task_compute(self) -> float:
        """
        compute exposure to a curricula
        """
        return self.experiment["train_cost"]["compute"] + self.experiment["test_cost"]["compute"]

    def get_is_name(self) -> str:
        if self.experiment:
            return self.experiment["model"]["train_params"]["model_name"]

    def get_experience(self) -> float:
        if self.E is not None:
            return np.log2(self.E)
        elif self.experiment is not None:
            self.E = self.experiment['train_cost']['compute']
            return np.log2(self.E)

    def get_domain_details(self, domain: Dict ) -> Dict:
        """
        In this case, this returns the details for a given Domain.
        Output : 
        {
                "dag_length" : {
                     "inflated": 937.0,
                     "deflated": 429.2,
                }
        """
        if domain["name"] in AVAILABLE_DOMAINS:
            extracted_details = {
                "dag_length" : DOMAIN_LENGTHS.get(domain["name"]),
            }

            return extracted_details
        else :
            return {
                "dag_length" : {
                     "inflated": 0,
                     "deflated": 0,
                }
            }

    def get_performance_theta(self,task: Dict) -> float:
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


    def get_generalization_difficulty(self, task_domain: Dict , curriculum_domain: Dict) -> float:
        """
        returns the exponential generalization difficulty
        (domain_distance_score * 10)^e
        """
        score = self.domain_distance_score(task_domain, curriculum_domain)
        return np.exp(score *10)

    def domain_distance_score(self, task_domain: Dict, curriculum_domain: Dict) ->float:
        if self.dd is not None:
            if isinstance(self.dd,Union[int,float].__args__):
                return self.dd
            elif isinstance(self.dd,dict):
                return self.dd.get(task_domain['name']).get(curriculum_domain['name'],1.0)
        
        if self.use_dd_cache and self.dd_cache is not None:
            return self.dd_cache[(task_domain['name'],curriculum_domain['name'])]
        
        task_domain_details = os.path.join(DOMAIN_ROOT,curriculum_domain["name"])
        curriculum_domain_details = os.path.join(DOMAIN_ROOT,curriculum_domain["name"])
        return calculate_dd(task_domain_details,curriculum_domain_details)

    def GetExperimentIndices(self,return_dict: bool =False) -> Union[ExperimentIndices,Dict]:
        TaskDomains = self.get_task_domains() if self.taskDomains is None else self.taskDomains
        CurriculaDomains = self.get_curricula_domains() if self.CurriculaDomains is None else self.CurriculaDomains
        GD = { task.get("name"): {curricula.get("name"): self.get_generalization_difficulty(task,curricula) for curricula in CurriculaDomains} for task in TaskDomains } if self.GD is None else self.GD
        PC = { curricula.get("name"):self.probability_of_curricula(curricula) for curricula in CurriculaDomains } if self.PC is None else self.PC
        P = self.get_priors() if self.P is None else self.P
        E = self.get_experience()
        Ptheta = { task.get("name"):self.get_performance_theta(task) for task in TaskDomains } if self.PTheta is None else self.PTheta
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

    def calculate_g_index(self) -> float:
        return self.GetExperimentIndices().GIndex

    def get_avg_perf(self) ->float:
        if self.experiment:
            return (1 - self.experiment["performance"]["divergence"])
    