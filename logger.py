from collections import defaultdict
from datetime import datetime as dt
import json
import os
import os.path as osp
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import ray
import torch

from self_play import SelfPlay
from shared_storage import SharedStorage


class Logger:

    def __init__(self,
                 exp_name: str,
                 mode: str):
        self.log_dir = osp.join(os.getcwd(), 'data', exp_name)\
            if exp_name else f'/tmp/experiments/{str(dt.now())}'
        if not osp.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if mode == 'train':
            self.log_file = open(osp.join(self.log_dir, 'loss.txt'), 'w')
            self.losses = defaultdict(list)
        else:
            self.log_file = open(osp.join(self.log_dir, 'rewards.txt'), 'w')

    def save_config(self, config: Dict) -> None:
        del config['visit_softmax_temperature_func']
        output = json.dumps(config, separators=(',', ':\t'), indent=4)
        print('Experiment config:\n', output)
        with open(osp.join(self.log_dir, 'config.json'), 'w') as out:
            out.write(output)
        self.config = config

    def save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        torch.save(checkpoint, osp.join(self.log_dir, 'model.checkpoint'))

    def log_continuously(self,
                         config,
                         test_worker: SelfPlay,
                         shared_storage_worker: SharedStorage) -> None:
        test_worker.play_continuously.remote(shared_storage_worker, None, test=True)
        keys = [
            'episode_length', 'episode_return', 'mean_value', 'training_step',
            'played_games', 'loss', 'value_loss', 'reward_loss', 'policy_loss'
        ]
        info = ray.get(shared_storage_worker.get_info.remote(keys))
        last_step = 0

        try:
            while info['training_step'] < config.training_steps:
                info = ray.get(shared_storage_worker.get_info.remote(keys))
                if info['training_step'] > last_step:
                    print(f'\rEpisode return: {info["episode_return"]:.2f}. '
                          + f'Training step: {info["training_step"]}/{config.training_steps}. '
                          + f'Played games: {info["played_games"]}. '
                          + f'Loss: {info["loss"]:.2f}', end="")

                    if info['training_step'] % config.checkpoint_interval == 0:
                        checkpoint = ray.get(shared_storage_worker.get_checkpoint.remote())
                        self.save_checkpoint(checkpoint)
                    self.log_loss({
                        'value_loss': info['value_loss'],
                        'reward_loss': info['reward_loss'],
                        'policy_loss': info['policy_loss'],
                        'total': info['loss']
                    })
                    last_step += 1
            self.close()
        except KeyboardInterrupt:
            pass

    def log_loss(self, losses: Dict[str, float]) -> None:
        if not self.losses:
            self.log_file.write(','.join(list(losses.keys())) + '\n')

        for k, v in losses.items():
            self.losses[k].append(v)
        self.log_file.write(','.join(['{:.4f}'.format(l) for l in list(losses.values())]) + '\n')
        self.log_file.flush()

        for k, v in self.losses.items():
            plt.plot(range(1, len(v) + 1), v, label=k)
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.legend(loc='upper right')
        plt.savefig(osp.join(self.log_dir, 'loss.png'))
        plt.close()

    def log_reward(self, rewards: List[float]) -> None:
        self.log_file.write(','.join(map(str, rewards)) + '\n')
        self.log_file.flush()

    def close(self):
        self.log_file.close()
