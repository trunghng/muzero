from datetime import datetime as dt
from typing import Any, Dict

import numpy as np
import ray
import torch

from games.game import Game, GameHistory
from mcts.mcts import MCTS
from player import HumanPlayer, RandomPlayer
from network import MuZeroNetwork
from replay_buffer import ReplayBuffer
from shared_storage import SharedStorage
from utils.utils import set_seed


class SelfPlay:

    def __init__(self,
                 game: Game,
                 initial_checkpoint: Dict[str, Any],
                 config,
                 seed: int) -> None:
        set_seed(seed)
        self.config = config
        self.game = game
        self.mcts = MCTS(self.config)
        self.network = MuZeroNetwork(config)
        self.network.set_weights(initial_checkpoint['model_state_dict'])
        self.network.eval()

    def play_continuously(self,
                          shared_storage: SharedStorage,
                          replay_buffer: ReplayBuffer,
                          test: bool=False) -> None:
        while ray.get(shared_storage.get_info.remote('training_step')) < self.config.training_steps:
            self.network.set_weights(ray.get(shared_storage.get_info.remote('model_state_dict')))
            if test:
                game_history = self.play(
                    0,  # select action with max #visits
                    'self' if self.config.players == 1 else self.config.opponent,
                    self.config.muzero_player
                )
                shared_storage.set_info.remote({
                    'episode_length': len(game_history),
                    'episode_return': game_history.compute_return(
                        self.config.gamma, self.config.muzero_player, self.config.players
                    ),
                    'mean_value': np.mean([v for v in game_history.root_values if v])
                })
            else:
                game_history = self.play(
                    self.config.visit_softmax_temperature_func(
                        self.config.training_steps,
                        ray.get(shared_storage.get_info.remote('training_step'))
                    ), 'self', -1
                )
                replay_buffer.add.remote(game_history, shared_storage)

    def play(self,
             temperature: float,
             opponent: str,
             muzero_player: int,
             render: bool = False) -> GameHistory:
        """Play a game with MuZero player

        :param temperature:
        :param opponent: Opponent of MuZero agent
            1. 'self'   play with itself
            2. 'human'  play with a human
            3. 'random' play with a random player
        :param muzero_player: MuZero's turn order
            1. 0        play first or MuZero is the only player (self-play or 1p games)
            2. 1        play second
        :param render: Whether to render the game
        """
        observation = self.game.reset()
        game_history = GameHistory(self.game.action_encoder, observation)
        if render:
            self.game.render()

        with torch.no_grad():
            while True:
                to_play = self.game.to_play()
                if opponent == 'self' or muzero_player == to_play:
                    stacked_observations = game_history.stack_n_observations(
                        -1,
                        self.config.n_stacked_observations,
                        self.config.n_actions,
                        self.config.stack_action
                    )
                    action, root_value, action_probs = self.mcts.search(
                        self.network,
                        stacked_observations,
                        self.game.legal_actions(),
                        self.game.action_encoder,
                        to_play,
                        temperature
                    )
                elif opponent == 'human':
                    action = HumanPlayer().play(self.game)
                    root_value, action_probs = None, None
                else:
                    action = RandomPlayer().play(self.game)
                    root_value, action_probs = None, None

                next_observation, reward, terminated = self.game.step(action)
                game_history.save(observation, action, reward, to_play, action_probs, root_value)
                observation = next_observation
                if render:
                    self.game.render()

                if terminated or len(game_history) > self.config.max_moves:
                    break
        return game_history
