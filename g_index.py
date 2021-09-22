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

from utils import (AVAILABLE_DOMAINS, DOMAIN_ROOT, cache_dd, calculate_dd,
                   get_domain_lengths, print_dataclass)

DOMAIN_LENGTHS = get_domain_lengths(AVAILABLE_DOMAINS)


@dataclass(frozen=True)
class ExperimentIndices:
    GD: Dict[str, Dict[str, float]]
    PC: Dict[str, float]
    PTheta: Dict[str, float]
    TaskDomains: List[str] = field(default_factory=list)
    CurriculaDomains: List[str] = field(default_factory=list)
    P: float = 0.0
    E: float = 0.0
    GIndex: float = 0.0
    AvgPTheta: float = 0.0


@dataclass(frozen=True)
class ExperimentComponents:
    IS: str
    CurriculaDomains: List[Dict]
    TaskDomains: List[Dict]
    E: float = 0.0
    GD: float = 0.0


class Experiment:

    def __init__(self, experiment=None, taskDomains=None, CurriculaDomains=None, GD=None, PC=None, P=None, E=None, PTheta=None, dd=None, use_dd_cache=True, dd_cache=None):
        if experiment is not None and isinstance(experiment, dict):
            self.experiment = experiment
            self.AVAILABLE_DOMAINS = set([domain['name'] for domain in self.experiment['train_set']['data']]).union(
                set([domain['name'] for domain in self.experiment['test_set']['data']]))

        elif experiment is not None and isinstance(experiment, Union[str, os.PathLike].__args__):
            self.experiment = json.load(open(experiment))
            self.AVAILABLE_DOMAINS = set([domain['name'] for domain in self.experiment['train_set']['data']]).union(
                set([domain['name'] for domain in self.experiment['test_set']['data']]))

        else:
            self.experiment = None
            self.AVAILABLE_DOMAINS = AVAILABLE_DOMAINS

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
        elif self.use_dd_cache:
            self.dd_cache = cache_dd()
        else:
            self.dd_cache = None

    def probability_of_curricula(self, curriculum: Dict) -> float:
        """
        What is the probability of the given curriculum domain occurring in
        the complete curricula. Initially, this is just (number_of_samples_of_domain(N) /total_no_of_samples)
        """
        probability = 1/(1 + np.log2(curriculum["num_samples"]))
        return probability

    def get_priors(self) -> float:
        """
        `priors (P)` : data which is built into the system _before_ training (are you fine-tuning something already fine-tuned?). This is negligible (~0) for most of current models.
        """
        return 1e-4

    def get_experience(self, sim_E=None) -> float:
        if self.experiment is not None:
            self.E = self.experiment['train_cost']['compute']
            return np.log2(self.E)

        elif sim_E is not None:
            return np.log2(sim_E)

    def get_performance_theta(self, task: Dict = None, sim_PTheta: float = None) -> float:
        if self.experiment and task is not None:
            performance_per_domain = self.experiment["performance"]["domains"]
            selected_task = next(
                (x for x in performance_per_domain if x["name"] == task["name"]), None)
            performance = 1 - selected_task["divergence"]
            return np.exp(performance*12)
        elif sim_PTheta:
            return np.exp(sim_PTheta * 12)

    def get_generalization_difficulty(self, task_domain: Dict = None, curriculum_domain: Dict = None, sim_GD: float = None) -> float:
        """
        returns the exponential generalization difficulty
        (domain_distance_score * 10)^e
        """
        if task_domain is None and curriculum_domain is None and sim_GD is not None:
            return np.exp(sim_GD * 10)

        score = self.domain_distance_score(task_domain, curriculum_domain)
        return np.exp(score * 10)

    def domain_distance_score(self, task_domain: Dict, curriculum_domain: Dict) -> float:
        if self.dd is not None:
            if isinstance(self.dd, Union[int, float].__args__):
                return self.dd
            elif isinstance(self.dd, dict):
                return self.dd.get(task_domain['name']).get(curriculum_domain['name'], 1.0)

        if self.use_dd_cache and self.dd_cache is not None:
            return self.dd_cache[(task_domain['name'], curriculum_domain['name'])]

        task_domain_details = os.path.join(
            DOMAIN_ROOT, curriculum_domain["name"])
        curriculum_domain_details = os.path.join(
            DOMAIN_ROOT, curriculum_domain["name"])
        return calculate_dd(task_domain_details, curriculum_domain_details)

    def get_exp_components(self, truncate_domains: bool = False, truncation_count: int = 2, return_dict: bool = False, print_members: bool = False):
        if self.experiment:
            IS = self.experiment["model"]["train_params"]["model_name"]
            IS = IS.replace("/", " ") if "/" in IS else IS
            CurriculaDomains = self.experiment["train_set"]["data"]
            TaskDomains = self.experiment["test_set"]["data"]
            E = np.log2(self.experiment["train_cost"]["compute"])
            GD = np.exp(self.experiment["domain_distance"]*10)

            if len(CurriculaDomains) > truncation_count and truncate_domains:
                print(f"Truncated Domains Data to {truncation_count} values")
                CurriculaDomains = CurriculaDomains[:truncation_count]
            if len(TaskDomains) > truncation_count and truncate_domains:
                TaskDomains = TaskDomains[:truncation_count]

            exp_components = ExperimentComponents(IS=IS, CurriculaDomains=CurriculaDomains,
                                                  TaskDomains=TaskDomains, E=E, GD=GD)
            if print_members:
                print_dataclass(exp_components)
                return
            if return_dict:
                return asdict(exp_components)
            return exp_components
        else:
            print("No experiment file was provided, returning default values")
            IS = "GPT2-345M"
            CurriculaDomains = [{"name": domain_name, "num_samples": 40}
                                for domain_name in self.AVAILABLE_DOMAINS]
            TaskDomains = [{"name": domain_name, "num_samples": 5}
                           for domain_name in self.AVAILABLE_DOMAINS]
            E = np.log2(1000)
            GD = np.exp(0.5 * 10)
            exp_components = ExperimentComponents(IS=IS, CurriculaDomains=CurriculaDomains,
                                                  TaskDomains=TaskDomains, E=E, GD=GD)
            if print_members:
                print_dataclass(exp_components)
                return
            if return_dict:
                return asdict(exp_components)
            return exp_components

    def _GetExperimentIndices(self, return_dict: bool = False, print_members: bool = False) -> Union[ExperimentIndices, Dict]:
        exp_components = self.get_exp_components()
        TaskDomains = exp_components.TaskDomains if self.taskDomains is None else self.taskDomains
        CurriculaDomains = exp_components.CurriculaDomains if self.CurriculaDomains is None else self.CurriculaDomains

        GD = {task.get("name"): {curricula.get("name"): self.get_generalization_difficulty(task, curricula)
                                 for curricula in CurriculaDomains} for task in TaskDomains} if self.GD is None else self.GD
        PC = {curricula.get("name"): self.probability_of_curricula(
            curricula) for curricula in CurriculaDomains} if self.PC is None else self.PC
        P = self.get_priors() if self.P is None else self.P
        E = self.get_experience()
        Ptheta = {task.get("name"): self.get_performance_theta(
            task) for task in TaskDomains} if self.PTheta is None else self.PTheta
        AvgPTheta = round(mean(list(Ptheta.values())), 3)

        TaskContributions = defaultdict()
        for task in TaskDomains:
            TaskContributions[task.get("name")] = np.sqrt(Ptheta.get(task.get("name")) * sum([GD.get(task.get("name")).get(
                curricula.get("name")) * (PC.get(curricula.get("name")) / (P+E)) for curricula in CurriculaDomains]))
        TaskContributions = dict(TaskContributions)
        GIndex = round(mean(list(TaskContributions.values())), 3)
        exp_indices = ExperimentIndices(GD=GD,
                                        PC=PC,
                                        PTheta=Ptheta,
                                        TaskDomains=TaskDomains,
                                        CurriculaDomains=CurriculaDomains,
                                        P=P, E=E,
                                        GIndex=GIndex,
                                        AvgPTheta=AvgPTheta)
        if print_members:
            print_dataclass(exp_components)
            return
        if return_dict:
            return asdict(exp_indices)
        return exp_indices

    def calculate_g_index(self) -> float:
        return self._GetExperimentIndices().GIndex

    def get_avg_perf(self) -> float:
        if self.experiment:
            return (1 - self.experiment["performance"]["divergence"])

    def _sim_gi(self, n_tasks_domain=5, n_curricula_domain=40, sim_P=1e-4, sim_E=1000, sim_PTheta=0.8,sim_GD=None):
        
        TaskDomains = [{"name": domain_name, "num_samples": n_tasks_domain}
                       for domain_name in self.AVAILABLE_DOMAINS]
        CurriculaDomains = [{"name": domain_name, "num_samples": n_curricula_domain}
                            for domain_name in self.AVAILABLE_DOMAINS]
        if sim_GD is None:
            GD = {task.get("name"): {curricula.get("name"):self.get_generalization_difficulty(task_domain=task,curriculum_domain=curricula) for curricula in CurriculaDomains} for task in TaskDomains}
        elif sim_GD is not None:
            GD = {task.get("name"): {curricula.get("name"):self.get_generalization_difficulty(sim_GD=sim_GD) for curricula in CurriculaDomains} for task in TaskDomains}

        E = self.get_experience(sim_E)
        Ptheta = {task.get("name"): self.get_performance_theta(sim_PTheta=sim_PTheta) for task in TaskDomains}  
        PC = {curricula.get("name"): self.probability_of_curricula(curricula) for curricula in CurriculaDomains}

        TaskContributions = defaultdict()
        for task in TaskDomains:
            TaskContributions[task.get("name")] = np.sqrt(Ptheta.get(task.get("name")) * sum([GD.get(task.get("name")).get(
                curricula.get("name")) * (PC.get(curricula.get("name")) / (sim_P+sim_E)) for curricula in CurriculaDomains]))
        TaskContributions = dict(TaskContributions)
        GIndex = round(mean(list(TaskContributions.values())), 3)

        sim_indices = ExperimentIndices(GD=GD,
                                        PC=PC,
                                        PTheta=Ptheta,
                                        TaskDomains=TaskDomains,
                                        CurriculaDomains=CurriculaDomains,
                                        P=sim_P, E=E,
                                        GIndex=GIndex,
                                        AvgPTheta=0.0)


        return sim_indices


    def simulate_g_index(self, n_tasks_domain=5, n_curricula_domain=40, sim_GD=0.2, sim_P=1e-4, sim_E=1000, sim_PTheta=0.8,print_members=False,return_dict=False):
        sim_indices = self._sim_gi(n_tasks_domain=n_tasks_domain, n_curricula_domain=n_curricula_domain, sim_GD=sim_GD, sim_P=sim_P, sim_E=sim_E, sim_PTheta=sim_PTheta)
        
        if print_members:
            print_dataclass(sim_indices)
        
        if return_dict:
            return asdict(sim_indices)
        
        return sim_indices.GIndex
