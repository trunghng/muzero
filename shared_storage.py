from typing import Dict, List, Any
from copy import deepcopy

import ray
import torch


@ray.remote
class SharedStorage:

    def __init__(self, checkpoint: Dict[str, Any]) -> None:
        self.checkpoint = checkpoint


    def get_checkpoint(self) -> Dict[str, Any]:
        return deepcopy(self.checkpoint)


    def save_checkpoint(self, model_path: str=None) -> None:
        torch.save(self.checkpoint, model_path)


    def get_info(self, keys: 'List[str] | str') -> 'Dict[str, Any] | Any':
        try:
            if isinstance(keys, list):
                return {k: self.checkpoint[k] for k in keys}
            elif isinstance(keys, str):
                return self.checkpoint[keys]
        except TypeError as err:
            print(err)


    def set_info(self, data_dict: Dict[str, Any]) -> None:
        self.checkpoint.update(data_dict)
