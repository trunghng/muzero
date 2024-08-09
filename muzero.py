from copy import deepcopy
import os
import time

import numpy as np
import ray
import torch
from tqdm import tqdm

from game import Game
from logger import Logger
from replay_buffer import ReplayBuffer
from self_play import SelfPlay
from shared_storage import SharedStorage
from trainer import Trainer
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

        self.logger = Logger(config.exp_name, config.mode)
        self.logger.save_config(vars(deepcopy(config)))

    def train(self) -> None:
        n_gpus = 0 if self.config.device == 'cpu' else torch.cuda.device_count()
        n_cpus = 0 if n_gpus > 0 else 1

        self.self_play_workers = [
            SelfPlay.remote(deepcopy(self.game), self.checkpoint, self.config)
            for _ in range(self.config.workers)
        ]
        self.training_worker = Trainer.options(num_cpus=n_cpus, num_gpus=n_gpus).remote(self.checkpoint, self.config)
        self.replay_buffer_worker = ReplayBuffer.remote(self.checkpoint, self.config)
        self.shared_storage_worker = SharedStorage.remote(self.checkpoint)
        self.shared_storage_worker.set_info.remote({'terminated': False})

        for self_play_worker in self.self_play_workers:
            self_play_worker.play_continuously.remote(self.shared_storage_worker, self.replay_buffer_worker)

        self.training_worker.update_weights_continuously.remote(self.shared_storage_worker, self.replay_buffer_worker)
        self.log_continuously()

    def log_continuously(self) -> None:
        self.test_worker = SelfPlay.remote(deepcopy(self.game), self.checkpoint, self.config)
        self.test_worker.play_continuously.remote(self.shared_storage_worker, None, test=True)
        keys = [
            'episode_length', 'episode_return', 'mean_value', 'training_step', 'played_games', 'loss'
        ]
        info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        last_step = 0

        try:
            while info['training_step'] < self.config.training_steps:
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
                if info['training_step'] > last_step:
                    print(f'\rEpisode return: {info["episode_return"]:.2f}. '
                          + f'Training step: {info["training_step"]}/{self.config.training_steps}. '
                          + f'Played games: {info["played_games"]}. '
                          + f'Loss: {info["loss"]:.2f}', end="")

                    if info['training_step'] % self.config.checkpoint_interval == 0:
                        checkpoint = ray.get(self.shared_storage_worker.get_checkpoint.remote())
                        self.logger.save_checkpoint(checkpoint)
                    self.logger.log_loss(info['loss'])
                    last_step += 1
        except KeyboardInterrupt:
            pass

    def test(self) -> None:
        checkpoint = torch.load(os.path.join(self.config.log_dir, 'model.checkpoint'))
        self_play_worker = SelfPlay.remote(self.game, checkpoint, self.config)
        histories = []
        for _ in tqdm(range(self.config.tests), desc=f'Testing'):
            history = ray.get(self_play_worker.play.remote(
                0,  # select actions with max #visits
                self.config.opponent,
                self.config.muzero_player,
                self.config.render)
            )
            self.logger.log_reward(history.rewards)
            histories.append(history)

        if self.config.players == 1:
            result = np.mean([sum(history.rewards) for history in histories])
            print(result)
        else:
            p1_wr = np.mean([
                sum(reward for i, reward in enumerate(history.rewards)
                if history.to_plays[i - 1] == -1) for history in histories
            ])
            p2_wr = np.mean([
                sum(reward for i, reward in enumerate(history.rewards)
                if history.to_plays[i - 1] == 1) for history in histories
            ])
            time.sleep(1)
            print(f'P1 win rate: {p1_wr * 100:.2f}%\nP2 win rate: {p2_wr * 100:.2f}%\
                \nDraw: {(1 - p1_wr - p2_wr) * 100:.2f}%')
