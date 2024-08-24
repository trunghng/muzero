from copy import deepcopy
import math
import os.path as osp
import pickle
import time

import numpy as np
import ray
import torch

from games.game import Game
from logger import Logger
from reanalyse import Reanalyser
from replay_buffer import ReplayBuffer
from self_play import SelfPlay
from shared_storage import SharedStorage
from trainer import Trainer
from utils.utils import set_seed


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
            'mean_value': 0,                # Mean across non-zero value funcs produced by MCTS
            'lr': 0,                        # Current learning rate
            'loss': 0,                      # Total loss
            'value_loss': 0,                # Value loss
            'reward_loss': 0,               # Reward loss
            'policy_loss': 0,               # Policy loss
            'training_step': 0,             # Current training step
            'played_games': 0,              # Number of games played
            'played_steps': 0,              # Number of steps played
            'reanalysed_games': 0           # Number of reanalysed games played
        }
        self.replay_buffer = []
        self.logger = Logger(self.config.exp_name)
        self.logger.save_config(vars(deepcopy(self.config)))

        if config.logdir:
            self.load_model()

    def train(self) -> None:
        n_gpus = 0 if self.config.device == 'cpu' else torch.cuda.device_count()
        n_cpus = 0 if n_gpus > 0 else 1
        training_worker = Trainer.options(
            num_cpus=n_cpus, num_gpus=n_gpus
        ).remote(self.checkpoint, self.config)
        self_play_workers = [
            SelfPlay.remote(
                deepcopy(self.game), self.checkpoint, self.config,
                self.config.seed + 10 * i
            ) for i in range(self.config.workers)
        ]
        replay_buffer_worker = ReplayBuffer.remote(
            self.checkpoint, self.replay_buffer, self.config
        )
        shared_storage_worker = SharedStorage.remote(self.checkpoint)
        reanalyse_workers = [
            Reanalyser.remote(
                deepcopy(self.game), self.checkpoint, self.config,
                self.config.seed + 10 * i
            ) for i in range(self.config.reanalyse_workers)
        ]
        test_worker = SelfPlay.remote(
            deepcopy(self.game), self.checkpoint, self.config,
            self.config.seed + 10 * self.config.workers
        )

        print('\nTraining...')
        for self_play_worker in self_play_workers:
            self_play_worker.play_continuously.remote(
                shared_storage_worker, replay_buffer_worker
            )
        training_worker.update_weights_continuously.remote(
            shared_storage_worker, replay_buffer_worker
        )
        for reanalyse_worker in reanalyse_workers:
            reanalyse_worker.reanalyse.remote(
                shared_storage_worker, replay_buffer_worker
            )
        self.logger.log_continuously(
            self.config, test_worker, shared_storage_worker, replay_buffer_worker
        )

    def test(self) -> None:
        self_play_workers = [
            SelfPlay.remote(
                deepcopy(self.game), self.checkpoint, self.config, self.config.seed + 10 * i
            ) for i in range(self.config.workers)
        ]

        histories = []

        print('\nTesting...')
        for _ in range(math.ceil(self.config.tests / self.config.workers)):
            histories += [
                worker.play.remote(
                    0,  # select actions with max #visits
                    self.config.opponent,
                    self.config.muzero_player,
                    self.config.render
                ) for worker in self_play_workers
            ]
        histories = ray.get(histories)
        self.logger.log_result(self.config, histories)

    def load_model(self):
        checkpoint_path = osp.join(self.config.logdir, 'model.checkpoint')
        self.checkpoint = torch.load(checkpoint_path)
        print(f'\nLoading model checkpoint from {checkpoint_path}')

        if self.config.mode == 'train':
            replay_buffer_path = osp.join(self.config.logdir, 'replay_buffer.pkl')
            with open(replay_buffer_path, 'rb') as f:
                replay_buffer = pickle.load(f)
            self.replay_buffer = replay_buffer['buffer']
            self.checkpoint['played_steps'] = replay_buffer['played_steps']
            self.checkpoint['played_games'] = replay_buffer['played_games']
            self.checkpoint['reanalysed_games'] = replay_buffer['reanalysed_games']
            print(f'Loading replay buffer from {replay_buffer_path}')
