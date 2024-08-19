import time
from typing import Any, Dict

import ray

from games.game import Game
from mcts import MCTS
from network import MuZeroNetwork
from replay_buffer import ReplayBuffer
from shared_storage import SharedStorage
from utils.utils import ftensor, set_seed


@ray.remote
class Reanalyser:

    def __init__(self,
                 game: Game,
                 initial_checkpoint: Dict[str, Any],
                 config,
                 seed: int) -> None:
        set_seed(seed)
        self.game = game
        self.config = config
        self.target_network = MuZeroNetwork(config.observation_dim,
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
        self.target_network.set_weights(initial_checkpoint['model_state_dict'])
        self.target_network.eval()
        self.mcts = MCTS(self.config)

    def reanalyse(self,
                  shared_storage: SharedStorage,
                  replay_buffer: ReplayBuffer) -> None:
        while ray.get(shared_storage.get_info.remote('played_games')) < 1:
            time.sleep(0.1)

        while ray.get(shared_storage.get_info.remote('training_step'))\
                < self.config.training_steps:
            if ray.get(shared_storage.get_info.remote('training_step'))\
                    % self.config.target_network_update_freq == 0:
                self.target_network.set_weights(
                    ray.get(shared_storage.get_info.remote('model_state_dict'))
                )

            game_idx, game_history = ray.get(replay_buffer.sample_n_games.remote(1))

            # Policy is updated via re-executing MCTS
            # Value is updated either via MCTS re-run or from a target network
            observations = [
                game_history.stack_n_observations(
                    t,
                    self.config.stacked_observations,
                    self.config.action_space_size,
                    self.config.stack_action
                ) for t in range(len(game_history))
            ]

            if not self.config.mcts_target_value:
                values = [
                    self.target_network.initial_inference(
                        ftensor(observation).unsqueeze(0)
                    )[2] for observation in observations
                ]

            action_probabilities, root_values = [], []
            for t, observation in enumerate(observations):
                root = self.mcts.search(
                    self.target_network,
                    observation,
                    self.game.legal_actions(),
                    game_history.actions,
                    self.game.action_encoder,
                    game_history.to_plays[t]
                )
                action_probabilities.append(self.mcts.action_probabilities(root))
                root_values.append(root.value())

            game_history.save_reanalysed_stats(
                action_probabilities,
                root_values if self.config.mcts_target_value else values
            )
            replay_buffer.update_game.remote(game_idx, game_history, shared_storage)
