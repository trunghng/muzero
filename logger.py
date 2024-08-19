from datetime import datetime as dt
import json
import os
import os.path as osp
import pickle
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from self_play import SelfPlay
from shared_storage import SharedStorage


class Logger:

    def __init__(self, exp_name: str):
        self.logdir = osp.join(os.getcwd(), 'data', exp_name)\
            if exp_name else f'/tmp/experiments/{str(dt.now())}'
        if not osp.exists(self.logdir):
            os.makedirs(self.logdir)

    def save_config(self, config: Dict) -> None:
        if 'visit_softmax_temperature_func' in config:
            del config['visit_softmax_temperature_func']
        output = json.dumps(config, separators=(',', ':\t'), indent=4)
        print('Experiment config:\n', output)
        with open(osp.join(self.logdir, 'config.json'), 'w') as f:
            f.write(output)

    def save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        torch.save(checkpoint, osp.join(self.logdir, 'model.checkpoint'))

    def save_replay_buffer(self,
                           replay_buffer: ReplayBuffer,
                           checkpoint: Dict[str, Any]) -> None:
        replay_buffer_path = osp.join(self.logdir, 'replay_buffer.pkl')
        print(f'\n\nSaving replay buffer at {replay_buffer_path}')
        pickle.dump({
            'buffer': replay_buffer,
            'played_games': checkpoint['played_games'],
            'played_steps': checkpoint['played_steps'],
            'reanalysed_games': checkpoint['reanalysed_games']
        }, open(replay_buffer_path, 'wb')
    )

    def log_continuously(self,
                         config,
                         test_worker: SelfPlay,
                         shared_storage_worker: SharedStorage,
                         replay_buffer_worker: ReplayBuffer) -> None:
        test_worker.play_continuously.remote(shared_storage_worker, None, test=True)
        writer = SummaryWriter(self.logdir)
        keys = [
            'episode_length',
            'episode_return',
            'mean_value',
            'lr',
            'loss',
            'value_loss',
            'reward_loss',
            'policy_loss',
            'training_step',
            'played_games',
            'played_steps',
            'reanalysed_games'
        ]
        info = ray.get(shared_storage_worker.get_info.remote(keys))
        last_step, counter = 0, 0

        try:
            while info['training_step'] < config.training_steps:
                info = ray.get(shared_storage_worker.get_info.remote(keys))

                writer.add_scalar(
                    '1.Total_reward/1.Episode_return', info['episode_return'], counter
                )
                writer.add_scalar(
                    '1.Total_reward/2.Mean_value', info['mean_value'], counter
                )
                writer.add_scalar(
                    '1.Total_reward/3.Episode_length', info['episode_length'], counter
                )
                writer.add_scalar(
                    '2.Workers/1.Self_played_games', info['played_games'], counter
                )
                writer.add_scalar(
                    '2.Workers/2.Training_steps', info['training_step'], counter
                )
                writer.add_scalar(
                    '2.Workers/3.Self_played_steps', info['played_steps'], counter
                )
                writer.add_scalar(
                    '2.Workers/4.Reanalysed_games', info['reanalysed_games'], counter
                )
                writer.add_scalar(
                    '2.Workers/5.Training_steps_per_self_played_step_ratio',
                    info['training_step'] / max(1, info['played_steps']),
                    counter
                )
                writer.add_scalar('2.Workers/6.Learning_rate', info['lr'], counter)
                writer.add_scalar('3.Loss/1.Total_weighted_loss', info['loss'], info['training_step'])
                writer.add_scalar('3.Loss/Value_loss', info['value_loss'], info['training_step'])
                writer.add_scalar('3.Loss/Reward_loss', info['reward_loss'], info['training_step'])
                writer.add_scalar('3.Loss/Policy_loss', info['policy_loss'], info['training_step'])
                print(f'\rEpisode return: {info["episode_return"]:.2f}. '
                      + f'Training step: {info["training_step"]}/{config.training_steps}. '
                      + f'Played games: {info["played_games"]}. '
                      + f'Loss: {info["loss"]:.2f}', end="")

                if info['training_step'] > last_step and\
                    info['training_step'] % config.checkpoint_interval == 0:
                    self.save_checkpoint(
                        ray.get(shared_storage_worker.get_checkpoint.remote())
                    )
                    last_step += 1

                counter += 1
                # time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        self.save_replay_buffer(
            ray.get(replay_buffer_worker.get_buffer.remote()),
            ray.get(shared_storage_worker.get_checkpoint.remote())
        )

    def log_reward(self, rewards: List[float]) -> None:
        with open(osp.join(self.logdir, 'rewards.txt'), 'w') as f:
            f.write(','.join(map(str, rewards)) + '\n')
