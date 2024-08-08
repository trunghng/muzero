from typing import Dict, List, Any
from copy import deepcopy

import ray


@ray.remote
class SharedStorage:

    def __init__(self, checkpoint: Dict[str, Any]) -> None:
        self.checkpoint = checkpoint

    def get_checkpoint(self) -> Dict[str, Any]:
        return deepcopy(self.checkpoint)

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
