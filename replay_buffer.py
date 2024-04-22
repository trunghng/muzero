# from collections import deque
from typing import Tuple, Dict, Any
import random

import ray
import torch
import numpy as np

from game import GameHistory
from shared_storage import SharedStorage


@ray.remote
class ReplayBuffer:

    def __init__(self, initial_checkpoint: Dict[str, Any], config) -> None:
        self.config = config
        # self.memory = deque(maxlen=self.config.buffer_size)
        self.memory = []
        self.played_games = initial_checkpoint['played_games']
        self.played_steps = initial_checkpoint['played_steps']


    def len(self) -> int:
        return len(self.memory)


    def add(self, game_history: GameHistory, shared_storage: SharedStorage) -> None:
        """Store history of a new game into the buffer"""
        self.memory.append(game_history)
        self.played_games += 1
        self.played_steps += len(game_history)

        shared_storage.set_info.remote({
            'played_games': self.played_games,
            'played_steps': self.played_steps
        })


    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :return observation_batch:  (B x (stack_obs * channels) x h x w)
        :return action_batch:       (B x (unroll_steps + 1))
        :return value_target_batch: (B x (unroll_steps + 1))
        :return reward_target_batch:(B x (unroll_steps + 1))
        :return policy_target_batch:(B x (unroll_steps + 1) x A)
        """
        # game_histories = random.sample(self.memory, k=self.config.batch_size)
        selected_indices = np.random.choice(range(len(self.memory)), size=self.config.batch_size)
        game_histories = np.asarray(self.memory)[selected_indices]

        batch = [[], [], [], [], []]

        for game_history in game_histories:
            t = random.randint(0, len(game_history))

            observations = game_history.stack_observations(
                t, self.config.stacked_observations, len(self.config.action_space), self.config.stack_action
            )

            encoded_actions = game_history.encoded_actions[t:t + self.config.unroll_steps + 1]
            if len(encoded_actions) < self.config.unroll_steps + 1:
                absorbed_indices = np.random.choice(range(len(game_history.encoded_actions)),
                                            size=self.config.unroll_steps + 1 - len(encoded_actions))
                encoded_actions.extend([game_history.encoded_actions[i] for i in absorbed_indices])

            value_targets, reward_targets, policy_targets = game_history.make_target(
                t, self.config.td_steps, self.config.gamma, self.config.unroll_steps, len(self.config.action_space)
            )
            batch[0].append(observations)
            batch[1].append(encoded_actions)
            batch[2].append(value_targets)
            batch[3].append(reward_targets)
            batch[4].append(policy_targets)

        for i in range(len(batch)):
            batch[i] = torch.as_tensor(np.asarray(batch[i]), dtype=torch.float32)
        return tuple(batch)
