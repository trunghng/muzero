from typing import Dict, Any, Tuple

import ray
import torch
from torch.optim import Adam

from network import MuZeroNetwork
from replay_buffer import ReplayBuffer
from shared_storage import SharedStorage
from network_utils import scalar_to_support, scale_gradient, dict_to_cpu


@ray.remote
class Trainer:
    

    def __init__(self, initial_checkpoint: Dict[str, Any], config) -> None:
        self.config = config
        self.network = MuZeroNetwork(config.observation_dim,
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
                                    config.support_limit,
                                    len(config.action_space))
        self.network.set_weights(initial_checkpoint['model_state_dict'])
        self.network.train()
        self.optimizer = Adam(self.network.parameters(), lr=self.config.lr,\
                            weight_decay=self.config.weight_decay)
        if initial_checkpoint['optimizer_state_dict'] is not None:
            self.optimizer.load_state_dict(initial_checkpoint['optimizer_state_dict'])
        self.training_step = initial_checkpoint['training_step']


    def update_weights_continuously(self,
                                    shared_storage: SharedStorage,
                                    replay_buffer: ReplayBuffer) -> None:
        while self.training_step < self.config.training_steps\
                and not ray.get(shared_storage.get_info.remote('terminated'))\
                and ray.get(replay_buffer.len.remote()) >= self.config.batch_size:
            batch = replay_buffer.sample.remote()
            loss, value_loss, reward_loss, policy_loss = self.update_weights(batch)
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote({
                    'model_state_dict': dict_to_cpu(self.network.state_dict()),
                    'optimizer_state_dict': dict_to_cpu(self.optimizer.state_dict())
                })

                if self.config.save_model:
                    shared_storage.save_checkpoint.remote(self.config.save_path)

            shared_storage.set_info.remote({
                'training_step': self.training_step,
                'lr': self.optimizer.param_groups[0]['lr'],
                'loss': loss,
                'value_loss': value_loss,
                'reward_loss': reward_loss,
                'policy_loss': policy_loss
            })


    def update_weights(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor])\
                    -> Tuple[float, float, float, float]:
        observation_batch, action_batch, value_target_batch, reward_target_batch, policy_target_batch = batch
        value_target_batch = scalar_to_support(value_target_batch, self.config.support_limit)
        reward_target_batch = scalar_to_support(reward_target_batch, self.config.support_limit)

        policies_logits, hidden_state, value = self.network.initial_inference(observation_batch)
        predictions = [(1.0, value, 0, policy_logits)]

        for k in range(1, action_batch.shape[1]):
            policy_logits, hidden_state, value, reward = self.network.recurrent_inference(
                observation_batch, action_batch[:, k]
            )

            # Scale the gradient at the start of dynamics function by 0.5
            scale_gradient(hidden_state, 0.5)
            predictions.append((1.0 / action_batch.shape[1], value, reward, policy_logits))

        value_loss, reward_loss, policy_loss = 0, 0, 0
        for k in range(len(predictions)):
            loss_scale, value, reward, policy_logits = predictions[k]
            value_loss_k, reward_loss_k, policy_loss_k = map(
                scale_gradient, loss(value, reward, policy_logits, value_target_batch[:, k], 
                reward_target_batch[:, k], policy_target_batch[:, k]), [loss_scale] * 3
            )

            # Ignore reward loss for the first batch step
            if k == 0:
                reward_loss_k = torch.zeros_like(reward_loss)
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
        value_loss = nn.CrossEntropyLoss(value, value_target)
        reward_loss = nn.CrossEntropyLoss(reward, reward_target)
        policy_loss = nn.CrossEntropyLoss(policy_logits, policy_target)
        return value_loss, reward_loss, policy_loss
