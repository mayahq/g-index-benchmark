import dataclasses
import glob
import json
import os
from datetime import datetime
from typing import Dict, List, Union

import numpy as np
from pydantic import BaseModel, Field, validator
from pydantic.dataclasses import dataclass
from node_utils import node_divergence

files = glob.glob(os.path.join("templates","*.json"))
AVAILABLE_TEMPLATES = [os.path.basename(fname).split(".")[0] for fname in files]

try:
    with open('assets/lengths.json','r') as f:
        TEMPLATE_DETAILS  = json.load(f)
except FileNotFoundError:
    print("No `lengths.json` was found, Using zero values for lengths, Please note that you would not be able to get reproduce results without it.")
    TEMPLATE_DETAILS = {temp_name: {"inflated":0,"deflated":0} for temp_name in AVAILABLE_TEMPLATES}

def get_template(template_key: str,template_dir:str = "templates") -> Dict:
    json_file = os.path.join(template_dir,template_key + ".json")
    try:
        return json.load(open(json_file))
    except FileNotFoundError as fe:
        print(f"No template of the type {template_key} found")

def cache_dd():
    dd_cache = defaultdict()
    for ta in AVAILABLE_TEMPLATES:
        for tb in AVAILABLE_TEMPLATES:
            if (tb,ta) in dd_cache.keys():
                dd_cache[(ta,tb)] = dd_cache[(tb,ta)]
            else:
                ta_details = DatasetDetails(total=2, data=[{"name": ta, "num_samples":2}])
                tb_details = DatasetDetails(total=2, data=[{"name":tb, "num_samples":2}])
                dd_cache[(ta,tb)] = domain_distance(Dataset.from_details(
                ta_details), Dataset.from_details(tb_details))
    return dd_cache

        
@dataclass
class TemplateInfo:
    name: str
    num_samples: int = dataclasses.field(default=0)

    @validator("num_samples")
    def _check_samples(cls, v):
        if v < 0:
            raise ValueError(f"num_samples {v} < 0 !!")
        return v

@dataclass
class DatasetDetails:
    data: List[TemplateInfo] = dataclasses.field(default_factory=lambda: [])
    total: int = dataclasses.field(default=0)


class Dataset(BaseModel):
    data: Dict[str, List[Dict]]
    total: int = Field(ge=0)

    @classmethod
    def from_dir(self, dirpath: str):
        assert os.path.isdir(dirpath), f"{dirpath} is not a valid directory"
        files = glob.glob(os.path.join(dirpath, "*.json"))
        data = {}
        for fname in files:
            tmpname = "-".join(os.path.basename(fname).split("-")[:-1])
            with open(fname, "r") as f:
                loaded = json.load(f)
                sample = loaded.get("flow", [])

            tmpname = loaded.get("name") or tmpname
            if data.get(tmpname) is not None:
                data[tmpname].append(sample)
            else:
                data[tmpname] = [sample]

        total = sum([len(v) for v in data.values()])
        return Dataset(data=data, total=total)

    @classmethod
    def from_details(self, details: DatasetDetails):
        data = {}
        for info in details.data:
            tmp = get_template(info.name)
            data[info.name] = data.get(info.name, [])
            data[info.name].append(tmp)

        total = sum([len(v) for v in data.values()])
        return Dataset.construct(data=data, total=total)

    def to_details(self) -> DatasetDetails:
        answer = {
            "data": [{"name": k, "num_samples": len(v)} for k, v in self.data.items()],
            "total": self.total,
        }
        return DatasetDetails(**answer)

    def __len__(self):
        return self.total

    def __iter__(self) -> Dict:
        for k, v in self.data.items():
            for tmp in v:
                yield tmp

def domain_distance(train_set: Dataset, test_set: Dataset) -> float:
    ovr_values = []
    for samp1 in test_set:
        scores = []
        for samp2 in train_set:
            score = node_divergence(samp1["flow"], samp2["flow"])
            scores.append(score)
            if score == 0:
                break
        ovr_values.append(np.mean(scores))

    return np.mean(ovr_values)
