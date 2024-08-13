from typing import Dict, Any, Tuple
import time

import ray
import torch
import torch.nn as nn
from torch.optim import Adam, SGD

from network import MuZeroNetwork
from network_utils import scalar_to_support, scale_gradient, update_lr, dict_to_cpu
from replay_buffer import ReplayBuffer
from shared_storage import SharedStorage
from utils import set_seed


@ray.remote
class Trainer:

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
                                     config.support_limit).to(config.device)
        self.network.set_weights(initial_checkpoint['model_state_dict'])
        self.network.train()
        if config.optimizer == 'Adam':
            self.optimizer = Adam(self.network.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
        else:
            self.optimizer = SGD(self.network.parameters(), lr=config.lr,
                                 momentum=config.momentum, weight_decay=config.weight_decay)
        if initial_checkpoint['optimizer_state_dict'] is not None:
            self.optimizer.load_state_dict(initial_checkpoint['optimizer_state_dict'])
        self.training_step = initial_checkpoint['training_step']

    def update_weights_continuously(self,
                                    shared_storage: SharedStorage,
                                    replay_buffer: ReplayBuffer) -> None:
        while ray.get(shared_storage.get_info.remote('played_games')) < 1:
            time.sleep(0.1)

        while self.training_step < self.config.training_steps\
                and not ray.get(shared_storage.get_info.remote('terminated')):
            batch = ray.get(replay_buffer.sample.remote())
            update_lr(self.config.lr, self.config.lr_decay_rate, self.config.lr_decay_steps, 
                      self.training_step, self.optimizer)
            loss, value_loss, reward_loss, policy_loss = self.update_weights(batch)

            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote({
                    'model_state_dict': dict_to_cpu(self.network.state_dict()),
                    'optimizer_state_dict': dict_to_cpu(self.optimizer.state_dict())
                })

            shared_storage.set_info.remote({
                'training_step': self.training_step,
                'lr': self.optimizer.param_groups[0]['lr'],
                'loss': loss,
                'value_loss': value_loss,
                'reward_loss': reward_loss,
                'policy_loss': policy_loss
            })

    def update_weights(
            self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ) -> Tuple[float, float, float, float]:
        observation_batch, action_batch, value_target_batch, reward_target_batch, policy_target_batch\
                            = map(lambda x: x.to(self.config.device), batch)
        value_target_batch = scalar_to_support(value_target_batch, self.config.support_limit)
        reward_target_batch = scalar_to_support(reward_target_batch, self.config.support_limit)

        policy_logits, hidden_state, value = self.network.initial_inference(observation_batch, False)
        predictions = [(1.0, value, torch.zeros_like(value, requires_grad=True), policy_logits)]

        for k in range(1, action_batch.shape[1]):
            policy_logits, hidden_state, value, reward = self.network.recurrent_inference(
                hidden_state, action_batch[:, k].unsqueeze(1), False
            )

            # Scale the gradient at the start of dynamics function by 0.5
            scale_gradient(hidden_state, 0.5)
            predictions.append((1.0 / action_batch.shape[1], value, reward, policy_logits))

        value_loss, reward_loss, policy_loss = 0, 0, 0
        for k in range(len(predictions)):
            loss_scale, value, reward, policy_logits = predictions[k]

            value_loss_k, reward_loss_k, policy_loss_k = self.loss(
                value, reward, policy_logits, value_target_batch[:, k],
                reward_target_batch[:, k], policy_target_batch[:, k]
            )
            scale_gradient(value_loss_k, loss_scale)
            scale_gradient(reward_loss_k, loss_scale)
            scale_gradient(policy_loss_k, loss_scale)

            # Ignore reward loss for the first batch step
            if k == 0:
                reward_loss_k = torch.zeros_like(value_loss_k)
            value_loss += value_loss_k
            reward_loss += reward_loss_k
            policy_loss += policy_loss_k

        loss = (value_loss * self.config.value_loss_weight + reward_loss + policy_loss).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return loss.item(), value_loss.mean().item(), reward_loss.mean().item(), policy_loss.mean().item()

    def loss(self,
             value: torch.Tensor,
             reward: torch.Tensor,
             policy_logits: torch.Tensor,
             value_target: torch.Tensor,
             reward_target: torch.Tensor,
             policy_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = nn.CrossEntropyLoss()
        value_loss = f(value, value_target)
        reward_loss = f(reward, reward_target)
        policy_loss = f(policy_logits, policy_target)
        return value_loss, reward_loss, policy_loss
