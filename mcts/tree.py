from dataclasses import dataclass
from typing import ClassVar, Tuple

import numpy as np


@dataclass(frozen=True)
class Tree:
    """
    Search tree, with N = #nodes

    node_visits: (N,) the visit counts for each node
    node_values: (N,) the cumulative search value for each node
    node_raw_values: (N,) the value computed by the policy network for each node
    parents: (N,) the node index for the parents for each node
    action_from_parent: (N,) action to take from the parent to reach each node
    children_index: (N, n_actions) the node index of the children for each action
    children_prior_logits: (N, n_actions) the action prior logits of each node
    children_visits: (N, n_actions) the visit counts for children for each action
    children_rewards: (N, n_actions) the immediate reward for each action
    children_values: (N, n_actions) the value of the next node after the action
    hidden_states: (N,) encoded states of the environment
    to_play: (N,) current player to play for each node
    root_legal_actions: (n_actions,)
    """
    node_visits: np.ndarray  # (N,)
    node_values: np.ndarray  # (N,)
    node_raw_values: np.ndarray  # (N,)
    parents: np.ndarray  # (N,)
    action_from_parent: np.ndarray  # (N,)
    children_index: np.ndarray  # (N, n_actions)
    children_prior_logits: np.ndarray  # (N, n_actions)
    children_visits: np.ndarray  # (N, n_actions)
    children_rewards: np.ndarray  # (N, n_actions)
    children_values: np.ndarray  # (N, n_actions)
    hidden_states: np.ndarray  # (N,)
    to_play: np.ndarray  # (N,)
    root_legal_actions: np.ndarray  # (n_actions,)
    discount: float
    gumbel_noise: np.ndarray  # (n_actions,)

    ROOT_INDEX: ClassVar[int] = 0
    NO_PARENT: ClassVar[int] = -1
    UNVISITED: ClassVar[int] = -1
    P1: ClassVar[int] = 0

    @property
    def n_actions(self) -> int:
        return self.children_index.shape[-1]

    @property
    def n_simulations(self) -> int:
        return self.node_visits.shape[-1] - 1

    def qvalues(self, node_index: int) -> np.ndarray:
        """Computes search value estimation of a node"""
        return self.children_rewards[node_index] + self.discount\
            * self.children_values[node_index]  # (n_actions,)

    def summary(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        root_index = Tree.ROOT_INDEX
        visit_counts = self.children_visits[root_index]  # (n_actions,)
        total_visits = np.sum(visit_counts, keepdims=True)  # (1,)
        visit_probs = visit_counts / np.maximum(total_visits, 1)  # (n_actions,)
        # uniform probs for unvisited node
        visit_probs = np.where(total_visits > 0, visit_probs, 1 / self.n_actions)
        qvalues = self.qvalues(root_index)  # (n_actions,)
        return visit_counts, visit_probs, qvalues
