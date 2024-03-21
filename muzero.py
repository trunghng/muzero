import random

import numpy as np
import torch

from game import Game, GameHistory
from mcts import MCTS
from utils import set_seed
import network


class MuZero:

    def __init__(self, game: Game, config) -> None:
        set_seed(config.seed)
        self.config = config
        self.game = game
        self.mcts = MCTS(self.config)
        self.model = None


    def run(self) -> None:
        pass


    def play(self) -> GameHistory:
        """Run a self-play game"""
        observation = self.game.reset()
        game_history = GameHistory(self.game, self.config.gamma)

        while True:
            stacked_observations = game_history.get_stacked_observation(-1, 
                    self.config.n_stacked_observations, len(self.config.action_space))
            root = self.mcts.search(self.network, stacked_observations, self.game.legal_actions(),
                                        game_history.action_history)
            action = self.mcts.select_action(root, None) # TODO

            observation, reward, terminated = self.game.step(action)
            game_history.store_experience(action, reward, self.game.to_play(), observation)
            game_history.store_search_statistics(root, len(self.config.action_space))

            if terminated or len(game_history) > self.config.max_moves:
                break
        return game_history
