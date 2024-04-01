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


    def set_checkpoint(self, model_path: str=None) -> None:
        torch.save(self.checkpoint, model_path)


    def get_info(self, keys: List[str] | str) -> Dict[str, Any] | Any:
        try:
            if isinstance(keys, list):
                return {k: self.checkpoint[k] for k in keys}
            else isinstance(keys, str):
                return self.checkpoint[key]
        except TypeError as err:
            print(err)


    def set_info(self,
                keys: List[str] | str,
                values: List[str] | str) -> None:
        try:
            if isinstance(keys, list) and isinstance(values, list):
                self.checkpoint.update({k: v} for k, v in zip(keys, values))
            else:
                self.checkpoint[keys] = values
        except TypeError as err:
            print(err)
