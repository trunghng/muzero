import time
from typing import Any, Dict

import ray

from network import MuZeroNetwork
from network_utils import ftensor
from replay_buffer import ReplayBuffer
from shared_storage import SharedStorage
from utils import set_seed


@ray.remote
class Reanalyser:

    def __init__(self, initial_checkpoint: Dict[str, Any], config) -> None:
        set_seed(config.seed)
        self.config = config
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
        self.reanalysed_games = initial_checkpoint['reanalysed_games']

    def reanalyse(self,
                  shared_storage: SharedStorage,
                  replay_buffer: ReplayBuffer) -> None:
        while ray.get(shared_storage.get_info.remote('played_games')) < 1:
            time.sleep(0.1)

        while ray.get(shared_storage.get_info.remote('training_step')) < self.config.training_steps:
            self.network.set_weights(ray.get(shared_storage.get_info.remote('model_state_dict')))
            game_idx, game_history = ray.get(replay_buffer.sample_n_games.remote(1))

            observations = [
                game_history.stack_n_observations(
                    t,
                    self.config.stacked_observations,
                    self.config.action_space_size,
                    self.config.stack_action
                ) for t in range(len(game_history))
            ]
            values = [
                self.network.initial_inference(
                    ftensor(observation).unsqueeze(0)
                )[2] for observation in observations
            ]
            game_history.reanalysed_root_values = values
            replay_buffer.update_game.remote(game_idx, game_history)
            self.reanalysed_games += 1
            shared_storage.set_info.remote({'reanalysed_games': self.reanalysed_games})
