from copy import deepcopy
import time

import ray

from game import Game
from self_play import SelfPlay
from trainer import Trainer
from replay_buffer import ReplayBuffer
from shared_storage import SharedStorage
from utils import set_seed


class MuZero:
    
    def __init__(self, game: Game, config) -> None:
        self.game = game
        self.config = config
        set_seed(self.config.seed)

        ray.init()
        self.checkpoint = {
            'model_state_dict': None,
            'optimizer_state_dict': None,
            'episode_return': 0,
            'episode_length': 0,
            'lr': 0,
            'total_loss': 0,
            'value_loss': 0,
            'policy_loss': 0,
            'reward_loss': 0,
            'training_step': 0,
            'terminated': False,
            'played_games': 0,
            'played_steps': 0
        }

        self.self_play_workers = None
        self.training_worker = None
        self.replay_buffer = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None


    def train(self) -> None:
        self.self_play_workers = [SelfPlay.remote(deepcopy(self.game), self.checkpoint, self.config)
                                    for _ in range(self.config.workers)]

        self.training_worker = Trainer.remote(self.checkpoint, self.config)
        self.replay_buffer_worker = ReplayBuffer.remote(self.checkpoint, self.config)
        self.shared_storage_worker = SharedStorage.remote(self.checkpoint)

        for self_play_worker in self.self_play_workers:
            self_play_worker.play_continuously.remote(self.shared_storage_worker, self.replay_buffer_worker)

        self.training_worker.update_weights_continuously.remote(self.shared_storage_worker, self.replay_buffer_worker)
        time.sleep(10)