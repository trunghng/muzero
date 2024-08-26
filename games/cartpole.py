from typing import List, Tuple

import gymnasium as gym
import numpy as np

from games.game import Game, ActType, ObsType


class CartPole(Game):

    def __init__(self, render: bool=False) -> None:
        super().__init__(1, [1, 1, 4], 2)
        if render:
            self.env = gym.make('CartPole-v1', render_mode='human')
        else:
            self.env = gym.make('CartPole-v1')

    def reset(self) -> ObsType:
        return np.array([[self.env.reset()[0]]])

    def legal_actions(self) -> List[ActType]:
        return list(range(2))

    def step(self, action: ActType) -> Tuple[ObsType, float, bool]:
        observation, reward, terminated, _, _ = self.env.step(action)
        return np.array([[observation]]), reward, terminated

    def action_encoder(self, action: ActType) -> ActType:
        one_hot_action = np.zeros(self.action_space_size)
        one_hot_action[action] = 1
        return one_hot_action

    def visit_softmax_temperature_func(self,
                                       training_steps: int,
                                       training_step: int) -> float:
        """
        :param training_steps: number of training steps
        :param training_step: current training step
        """
        if training_step < 0.5 * training_steps:
            return 1.0
        elif training_step < 0.75 * training_steps:
            return 0.5
        else:
            return 0.25

    def render(self) -> None:
        pass
