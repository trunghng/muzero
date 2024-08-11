from typing import Any, Dict

import numpy as np
import ray
import torch

from game import Game, GameHistory
from mcts import MCTS
from player import HumanPlayer, RandomPlayer
from network import MuZeroNetwork
from replay_buffer import ReplayBuffer
from shared_storage import SharedStorage
from utils import set_seed


@ray.remote
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
        self.network = MuZeroNetwork(config.observation_dim,
                                     config.action_space_size,
                                     config.stacked_observations,
                                     config.blocks,
                                     config.channels,
                                     config.reduced_channels_reward,
                                     config.reduced_channels_policy,
                                     config.reduced_channels_value,
                                     config.fc_reward_layers,
                                     config.fc_policy_layers,
                                     config.fc_value_layers,
                                     config.downsample,
                                     config.support_limit)
        self.network.set_weights(initial_checkpoint['model_state_dict'])
        self.network.eval()

    def play_continuously(self,
                          shared_storage: SharedStorage,
                          replay_buffer: ReplayBuffer,
                          test: bool=False) -> None:
        while ray.get(shared_storage.get_info.remote('training_step')) < self.config.training_steps \
                and not ray.get(shared_storage.get_info.remote('terminated')):
            self.network.set_weights(ray.get(shared_storage.get_info.remote('model_state_dict')))
            if test:
                game_history = self.play(
                    0,  # select action with max #visits
                    'self' if self.config.players == 1 else self.config.opponent,
                    self.config.muzero_player)
                shared_storage.set_info.remote({
                    'episode_length': len(game_history),
                    'episode_return': game_history.compute_return(self.config.gamma),
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
             render: bool=False) -> GameHistory:
        """Play a game with MuZero player

        :param temperature:
        :param opponent: Opponent of MuZero agent
            1. 'self'   play with itself
            2. 'human'  play with a human
            3. 'random' play with a random player
        :param muzero_player: MuZero's turn order
            1. -1       play first or MuZero is the only player (self-play or 1p games)
            2. 1        play second
        :param render: Whether to render the game
        """
        observation = self.game.reset()
        game_history = GameHistory(self.game)
        if render:
            self.game.render()

        with torch.no_grad():
            while True:
                if opponent == 'self' or muzero_player == self.game.to_play:
                    stacked_observations = game_history.stack_observations(
                        -1, self.config.stacked_observations, self.config.action_space_size, self.config.stack_action
                    )
                    root = self.mcts.search(self.network, stacked_observations, self.game.legal_actions(),
                                            game_history.actions, self.game.action_encoder, self.game.to_play)
                    action = self.mcts.select_action(root, temperature)
                    action_probs = self.mcts.action_probabilities(root)
                elif opponent == 'human':
                    action = HumanPlayer().play(self.game)
                else:
                    action = RandomPlayer().play(self.game)

                next_observation, reward, terminated = self.game.step(action)
                game_history.save(observation, action, reward, self.game.to_play, action_probs, root.value())
                observation = next_observation
                if render:
                    self.game.render()

                if terminated or len(game_history) > self.config.max_moves:
                    break
        return game_history
