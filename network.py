from abc import ABC, abstractmethod
from typing import Tuple

import torch

from game import ActType


class Network(ABC):
    
    @abstractmethod
    def initial_inference(self, observation: torch.Tensor)\
                            -> Tuple[float, float, torch.Tensor, torch.Tensor]:
        """representation + prediction function"""

    @abstractmethod
    def recurrent_inference(self, hidden_state: torch.Tensor, action: ActType)\
                            -> Tuple[float, float, torch.Tensor, torch.Tensor]:
        """dynamics + prediction function"""


    def support_to_scalar(self, support):
        pass


    def scalar_to_support(self, scalar: float):
        pass


