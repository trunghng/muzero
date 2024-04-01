from collections import deque
from typing import Tuple
import random

import ray
import torch

from game import GameHistory


@ray.remote
class ReplayBuffer:

    def __init__(self, size: int) -> None:
        self._memory = deque(maxlen=size)


    def add(self, game_history: GameHistory) -> None:
        self._memory.append(game_history)


    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of game positions"""
        game_histories = random.sample(self._memory, k=batch_size)

        # value, reward, policy, action

        for game_history in game_histories:
            t = random.randint(0, len(game_history))



