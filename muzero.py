from copy import deepcopy
import time
import os

import ray
import torch

from game import Game
from self_play import SelfPlay
from trainer import Trainer
from replay_buffer import ReplayBuffer
from shared_storage import SharedStorage
from logger import Logger
from utils import set_seed


class MuZero:
    
    def __init__(self, game: Game, config) -> None:
        self.game = game
        self.config = config
        set_seed(self.config.seed)
        ray.init()

        self.checkpoint = {
            'model_state_dict': None,       # Model state dict
            'optimizer_state_dict': None,   # Optimizer state dict
            'episode_length': 0,            # Episode length
            'episode_return': 0,            # Episode return
            'mean_value': 0,                # Mean across non-zero value functions produced by MCTS
            'lr': 0,                        # Current learning rate
            'loss': 0,                      # Total loss
            'value_loss': 0,                # Value loss
            'reward_loss': 0,               # Reward loss
            'policy_loss': 0,               # Policy loss
            'training_step': 0,             # Current training step
            'terminated': False,            # Whether the current game is over
            'played_games': 0,              # Number of games played
            'played_steps': 0               # Number of steps played
        }

        self.self_play_workers = None
        self.training_worker = None
        self.test_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

        if config.exp_name:
            exp_name = config.exp_name
            log_dir = os.path.join(os.getcwd(), 'data', exp_name, f'{exp_name}_s{config.seed}')
        else:
            log_dir = None
        self.logger = Logger(log_dir, self.shared_storage_worker)
        self.logger.save_config(vars(deepcopy(config)))


    def train(self) -> None:
        n_gpus = 0 if self.config.device == 'cpu' else torch.cuda.device_count()
        n_cpus = 0 if n_gpus > 0 else 1

        self.self_play_workers = [SelfPlay.remote(deepcopy(self.game), self.checkpoint, self.config)
                                    for _ in range(self.config.workers)]
        self.training_worker = Trainer.options(num_cpus=n_cpus, num_gpus=n_gpus).remote(self.checkpoint, self.config)
        self.replay_buffer_worker = ReplayBuffer.remote(self.checkpoint, self.config)
        self.shared_storage_worker = SharedStorage.remote(self.checkpoint)
        self.shared_storage_worker.set_info.remote({'terminated': False})

        for self_play_worker in self.self_play_workers:
            self_play_worker.play_continuously.remote(self.shared_storage_worker, self.replay_buffer_worker)

        self.training_worker.update_weights_continuously.remote(self.shared_storage_worker, self.replay_buffer_worker)
        self.log()


    def log(self) -> None:
        self.test_worker = SelfPlay.remote(deepcopy(self.game), self.checkpoint, self.config)
        self.test_worker.play_continuously.remote(self.shared_storage_worker, None, test=True)
        keys = [
            'episode_length', 'episode_return', 'mean_value', 'training_step', 'played_games', 'loss'
        ]
        info = ray.get(self.shared_storage_worker.get_info.remote(keys))

        try:
            while info['training_step'] < self.config.training_steps:
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
                print(f'\rEpisode return: {info["episode_return"]:.2f}. Training step: {info["training_step"]}/{self.config.training_steps}. Played games: {info["played_games"]}. Loss: {info["loss"]:.2f}', end="")
        except KeyboardInterrupt:
            pass

        if self.config.save_model:
            pass
