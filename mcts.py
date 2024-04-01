from typing import Tuple, List, Callable
import math
import random
from copy import deepcopy

import numpy as np
import torch

from game import ActType, PlayerType
from network import MuZeroNetwork


class Node:

    def __init__(self, prior: float):
        self.visit_count = 0 # N
        self.value_sum = 0
        self.prior = prior # P
        self.reward = 0 # R
        self.children = dict() # {action (ActType): child (Node)}
        self.hidden_state = None # S
        self.to_play = -1


    def expanded(self) -> bool:
        return len(self.children) > 0


    def value(self) -> float: # Q
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


    def expand(self,
            reward: float,
            hidden_state: torch.Tensor,
            policy: torch.Tensor,
            to_play: PlayerType,
            actions: List[ActType]) -> None:
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        for i, a in enumerate(actions):
            self.children[a] = Node(policy[i])


class MinMaxStats:
    """A class holding the min, max value of the tree"""

    def __init__(self) -> None:
        self.max = -float('inf')
        self.min = float('inf')


    def update(self, value: float) -> None:
        self.max = max(self.max, value)
        self.min = min(self.min, value)


    def normalize(self, value: float) -> float:
        if self.max > self.min:
            return (value - self.min) / (self.max - self.min)
        return value


class MCTS:

    def __init__(self, config) -> None:
        self.config = config


    def add_exploration_noise(self, node: Node) -> None:
        """
        Dirichlet noise injection into prior probabilities. For each a:
            P(s,a) = (1 - exploration_frac) * P(s,a) + exploration_frac * noise
        """
        actions = node.children.keys()
        noises = np.random.dirichlet(self.config.dirichlet_alpha, len(actions))
        for a, n in zip(actions, noises):
            node.children[a].prior = (1 - self.config.exploration_frac) * node.children[a].prior\
                                        + self.config.exploration_frac * n


    def ucb_score(self,
                parent: Node,
                child: Node,
                min_max_stats: MinMaxStats) -> float:
        """
        Compute UCB score according to a variant of PUCT
            ucb_score = Q(s,a) + U(s,a)
            U(s,a) = C(s) * P(s,a) + sqrt(N(s)) / (1 + N(s,a))
            C(s) = c_init + log((N(s) + c_base + 1) / c_base)
        """
        if child.visit_count > 0:
            q = min_max_stats.normalize(child.reward + self.config.gamma * child.value())
        else:
            q = child.value()
        c = self.config.c_init + math.log((parent.visit_count + self.config.c_base + 1) / self.config.c_base)
        u = c * child.prior + math.sqrt(parent.visit_count / (1 + child.visit_count))
        return q + u


    def select_child(self, node: Node) -> Tuple[ActType, Node]:
        """Select the child node with highest UCB"""
        ucb_scores = {action: self.ucb_score(node, child) for action, child in node.children.items()}
        action, child = max(ucb_scores.items(), key=lambda k: k[1])
        return action, child


    def backpropagate(self,
                    search_path: List[Node],
                    value: float,
                    to_play: PlayerType,
                    min_max_stats: MinMaxStats) -> None:
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
            min_max_stats.update(node.value())

            value = node.reward + self.config.gamma * value


    def select_action(self, root: Node, training_step: int) -> ActType:
        """
        Select action according to
            pi(a|s_0) = N(s_0,a)^(1/tau) / sum_b(N(s_0,b)^(1/t))
        If t = 0, pi is deterministic, returns the action with max #visits

        :param root: root node
        :param training_step: current training step
        """
        visit_counts = {action: child.visit_count for action, child in root.children.items()}
        t = self.config.visit_softmax_temperature_fn(self.config.training_steps, training_step)

        if t == 0:
            max_visits = max(visit_counts.values())
            action = random.choice([a for a, v in visit_counts.items() if v == max_visits])
        else:
            total_visits = float(sum(visit_counts.values()))
            idx = np.random.choice(range(len(visit_counts.keys())), p=[v / total_visits for v in visit_counts.values()])
            action = list(visit_counts.keys())[idx]
        return action


    def action_probabilities(self, root: Node) -> List[float]:
        total_visits = float(sum([child.visit_count for child in root.children.values()]))
        action_probabilities = [
            root.children[self.idx_to_action(idx)].visit_count / total_visits \
            if self.idx_to_action(idx) in root.children else 0 \
            for idx in range(len(self.config.action_space))
        ]


    def search(self,
            network: MuZeroNetwork,
            observation: np.ndarray,
            legal_actions: List[ActType],
            action_history: List[ActType],
            action_encoder: Callable,
            to_play: PlayerType) -> Node:
        """
        Run a MCTS search

        :param network: MuZero model
        :param observation: past observations
        :param legal_actions: legal actions of the current game state
        :param action_encoder: action encoder, varies on each game
        :param to_play: current player to play
        """
        # s_t^0 = h(o_1,...,o_t)
        # p_t^0, v_t^0 = f(s_t^0)
        root = Node(0)
        policy, hidden_state, value = network.initial_inference(torch.as_tensor(observation))
        root.expand(reward, policy, hidden_state, to_play, legal_actions)
        self.add_exploration_noise(root)

        min_max_stats = MinMaxStats()

        for _ in range(self.config.simulations):
            node = root
            search_path = [node]
            history = deepcopy(action_history)

            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)
                history.append(action)
            
            '''
            r^l, s^l = g(s^{l-1}, a^l)
            '''
            parent = search_path[-2]
            policy, hidden_state, value, reward = network.recurrent_inference(
                parent.hidden_state,
                torch.as_tensor(action_encoder(history[-1])).unsqueeze(0).unsqueeze(0)
            )
    
            to_play = len(history) % self.config.players
            node.expand(reward, policy, hidden_state, to_play, self.config.action_space)

            self.backpropagate(search_path, value, to_play, min_max_stats)
        return root
    