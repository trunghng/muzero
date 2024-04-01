from typing import Dict, Any, Tuple

import ray
import torch
from torch.optim import Adam

from network import MuZeroNetwork


@ray.remote
class Trainer:
    

    def __init__(self, initial_checkpoint: Dict[str, Any], config) -> None:
        self.config = config
        self.network = MuZeroNetwork(self.config)
        self.network.set_weights(initial_checkpoint['model_state_dict'])
        self.network.train()
        self.optimizer = Adam(self.model.parameters(), lr=self.config.lr)
        self.optimizer.load_state_dict(initial_checkpoint['optimizer_state_dict'])
        self.training_step = initial_checkpoint['training_step']


    def train(self) -> None:
        pass


    def loss(self,
            value: torch.Tensor,
            reward: torch.Tensor,
            policy: torch.Tensor,
            value_target: torch.Tensor,
            reward_target: torch.Tensor,
            policy_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass