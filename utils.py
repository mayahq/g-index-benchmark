import os
import json
import dataclasses
import numpy as np
from datetime import datetime
from typing import Union, List, Dict
from pydantic import BaseModel, Field, validator
from pydantic.dataclasses import dataclass

from node_utils import node_divergence
import glob

files = glob.glob(os.path.join(os.getcwd(), "templates", "*.json"))
AVAILABLE_TEMPLATES = [os.path.basename(fname) for fname in files]


def get_template(template_key: str, template_dir: str = "templates") -> Dict:
    return json.load(open(os.path.join(template_dir, template_key)))


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
        ovr_values.append(np.min(scores))

    return np.mean(ovr_values)
