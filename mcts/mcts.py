from dataclasses import replace
from functools import partial
from typing import Callable, Tuple
import math

import numpy as np
import pygraphviz

from mcts import action_selection
from mcts import qtransforms
from mcts.tree import Tree
from network import MuZeroNetwork
from utils.game_utils import turn_swapper, mask_illegal_action_logits
from utils.utils import ftensor, to_np, softmax, probs_to_logits


class MCTS:

    def __init__(self, config) -> None:
        self.config = config

    def add_exploration_noise(self, prior_logits: np.ndarray) -> np.ndarray:
        """
        Dirichlet noise injection into prior probabilities. For each a:
            P(s,a) = (1 - exploration_frac) * P(s,a) + exploration_frac * noise
        """
        noises = np.random.dirichlet([self.config.dirichlet_alpha] * len(prior_logits))
        probs = softmax(prior_logits)
        noised_probs = (1 - self.config.exploration_frac) * probs +\
            self.config.exploration_frac * noises
        return probs_to_logits(noised_probs)

    def update(self, attr, v, index):
        attr[index] = v
        return attr

    def update_tree_node(
        self,
        tree: Tree,
        node_index: int,
        prior_logits: np.ndarray,
        hidden_state: np.ndarray,
        value: np.ndarray,
        to_play: int
    ) -> Tree:
        tree = replace(
            tree,
            children_prior_logits=self.update(
                tree.children_prior_logits, prior_logits, node_index),
            node_visits=self.update(
                tree.node_visits, tree.node_visits[node_index] + 1, node_index),
            node_values=self.update(tree.node_values, value, node_index),
            node_raw_values=self.update(tree.node_raw_values, value, node_index),
            hidden_states=self.update(tree.hidden_states, hidden_state, node_index),
            to_play=self.update(tree.to_play, to_play, node_index)
        )
        return tree

    def initialize_tree(
        self,
        prior_logits: np.ndarray,
        hidden_state: np.ndarray,
        value: np.ndarray,
        to_play: int,
        root_legal_actions: np.ndarray,
        gumbel_noise: np.ndarray
    ) -> Tree:
        n_nodes = self.config.simulations + 1
        n_actions = self.config.n_actions

        # Initialize a fresh tree
        tree = Tree(
            node_visits=np.zeros(n_nodes, dtype=np.int32),
            node_values=np.zeros(n_nodes, dtype=value.dtype),
            node_raw_values=np.zeros(n_nodes, dtype=value.dtype),
            parents=np.full(n_nodes, Tree.NO_PARENT, dtype=np.int32),
            action_from_parent=np.full(
                n_nodes, Tree.NO_PARENT, dtype=np.int32),
            children_index=np.full(
                (n_nodes, n_actions), Tree.UNVISITED, dtype=np.int32),
            children_prior_logits=np.zeros(
                (n_nodes, n_actions), dtype=prior_logits.dtype),
            children_visits=np.zeros((n_nodes, n_actions), dtype=np.int32),
            children_rewards=np.zeros((n_nodes, n_actions), dtype=value.dtype),
            children_values=np.zeros((n_nodes, n_actions), dtype=value.dtype),
            hidden_states=np.zeros(
                (n_nodes, ) + hidden_state.shape, dtype=hidden_state.dtype),
            to_play=np.full(n_nodes, Tree.P1, dtype=np.int32),
            root_legal_actions=root_legal_actions,
            discount=self.config.gamma,
            gumbel_noise=gumbel_noise
        )
        # Add root info to the tree
        self.update_tree_node(
            tree, Tree.ROOT_INDEX, prior_logits, hidden_state, value, to_play)
        return tree

    def simulate(
        self,
        tree: Tree,
        action_selection_func: Callable
    ) -> Tuple[int, int, int]:
        """Traverves the tree until reaching an unvisited node"""
        next_node_index = Tree.ROOT_INDEX
        next_to_play = Tree.P1
        depth = 0

        while next_node_index != Tree.UNVISITED:
            node_index = next_node_index
            action = action_selection_func(tree, node_index, depth)
            next_node_index = tree.children_index[node_index, action]
            next_to_play = turn_swapper(self.config.players, next_to_play)
            depth += 1
        return node_index, action, next_to_play

    def expand(
        self,
        tree: Tree,
        node_index: int,
        action: int,
        next_node_index: int,
        prior_logits: np.ndarray,
        hidden_state: np.ndarray,
        reward: np.ndarray,
        value: np.ndarray,
        next_to_play: int
    ) -> Tree:
        """Create and evaluate child nodes from given nodes and unvisited actions"""
        tree = self.update_tree_node(
            tree, next_node_index, prior_logits, hidden_state, value, next_to_play
        )
        return replace(
            tree,
            parents=self.update(tree.parents, node_index, next_node_index),
            action_from_parent=self.update(
                tree.action_from_parent, action, next_node_index),
            children_index=self.update(
                tree.children_index, next_node_index, (node_index, action)),
            children_rewards=self.update(
                tree.children_rewards, reward, (node_index, action))
        )

    def backward(self, tree: Tree, leaf_index: int, to_play: int) -> Tree:
        leaf_value = tree.node_values[leaf_index]
        while leaf_index != Tree.ROOT_INDEX:
            parent_index = tree.parents[leaf_index]
            action = tree.action_from_parent[leaf_index]
            reward = tree.children_rewards[parent_index, action]
            parent_visits = tree.node_visits[parent_index]
            parent_value = (tree.node_values[parent_index] * parent_visits + (
                leaf_value if tree.to_play[parent_index] == to_play else -leaf_value
            )) / (parent_visits + 1)
            tree = replace(
                tree,
                node_values=self.update(tree.node_values, parent_value, parent_index),
                node_visits=self.update(tree.node_visits, parent_visits + 1, parent_index),
                children_values=self.update(tree.children_values,
                    tree.node_values[leaf_index], (parent_index, action)),
                children_visits=self.update(tree.children_visits,
                    tree.children_visits[parent_index, action] + 1, (parent_index, action))
            )
            leaf_value = reward + self.config.gamma * leaf_value
            leaf_index = parent_index
        return tree

    def search(
        self,
        network: MuZeroNetwork,
        observation: np.ndarray,
        legal_actions: np.ndarray,
        action_encoder: Callable,
        to_play: int,
        temperature: float = 0
    ) -> Tuple[int, float, np.ndarray]:
        # (B, n_actions), (B, channels, h/16, w/16)|(B, channels, h, w)|(B, encoding_size), (B,)
        prior_logits, hidden_state, value = to_np(network.initial_inference(
            ftensor(observation).unsqueeze(0)
        ))
        prior_logits = prior_logits.squeeze()
        value = value.squeeze()
        if self.config.gumbel:
            root_action_selection_func = partial(
                action_selection.gumbel_muzero_root_action_selection,
                discount=self.config.gamma,
                simulations=self.config.simulations,
                max_considered_actions=self.config.max_considered_actions,
                gumbel_scale=self.config.gumbel_scale,
                qtransform=partial(
                    qtransforms.qtransform_completed_by_mix_value,
                    c_visit=self.config.c_visit,
                    c_scale=self.config.c_scale
                )
            )
            interior_action_selection_func = partial(
                action_selection.gumbel_muzero_interior_action_selection,
                discount=self.config.gamma,
                qtransform=partial(
                    qtransforms.qtransform_completed_by_mix_value,
                    c_visit=self.config.c_visit,
                    c_scale=self.config.c_scale
                )
            )
            action_selection_to_act_func = partial(
                action_selection.gumbel_muzero_action_selection_to_act,
                qtransform=partial(
                    qtransforms.qtransform_completed_by_mix_value,
                    c_visit=self.config.c_visit,
                    c_scale=self.config.c_scale
                )
            )
            gumbel_noise = np.random.gumbel(size=prior_logits.shape)
        else:
            root_action_selection_func = partial(
                action_selection.muzero_action_selection,
                discount=self.config.gamma,
                c_base=self.config.c_base,
                c_init=self.config.c_init,
                qtransform=qtransforms.qtransform_by_parent_and_siblings
            )
            interior_action_selection_func = root_action_selection_func
            action_selection_to_act_func = partial(
                action_selection.muzero_action_selection_to_act,
                temperature=temperature,
            )
            # Add Dirichlet noise for MuZero
            prior_logits = self.add_exploration_noise(prior_logits)
            gumbel_noise = None

        action_selection_func = action_selection.switching_action_selection_wrapper(
            root_action_selection_func, interior_action_selection_func
        )
        tree = self.initialize_tree(
            mask_illegal_action_logits(prior_logits, legal_actions),
            hidden_state, value, to_play, legal_actions, gumbel_noise
        )
        for sim in range(self.config.simulations):
            node_index, action, next_to_play = self.simulate(tree, action_selection_func)
            next_node_index = tree.children_index[node_index, action]
            next_node_index = next_node_index if next_node_index != Tree.UNVISITED else sim + 1
            prior_logits, hidden_state, value, reward = to_np(network.recurrent_inference(
                ftensor(tree.hidden_states[node_index]),
                ftensor(action_encoder(action)).unsqueeze(0)
            ))
            tree = self.expand(
                tree, node_index, action, next_node_index, prior_logits,
                hidden_state, reward, value, next_to_play
            )
            tree = self.backward(tree, next_node_index, next_to_play)
        action, visit_probs, qvalues = action_selection_to_act_func(tree)
        # Use the q-value of the selected action to estimate the root value
        # return action, qvalues[action], visit_probs
        return action, tree.node_values[tree.ROOT_INDEX], visit_probs

    def visualize_tree(self, tree: Tree, output_file: str = None) -> None:
        def node_to_str(node_i, reward=0):
            return (f'{node_i}\n'
                    f'Reward: {reward:.2f}\n'
                    f'Value: {tree.node_values[node_i]:.2f}\n'
                    f'Visits: {tree.node_visits[node_i]}\n'
                    f'To play: {tree.to_play[node_i]}')

        def edge_to_str(node_i, a):
            probs = softmax(tree.children_prior_logits[node_i])
            return (f'{a}\n'
                    f'Q: {tree.qvalues(node_i)[a]:.2f}\n'
                    f'p: {probs[a]:.2f}')

        graph = pygraphviz.AGraph(directed=True)
        graph.add_node(0, label=f'{node_to_str(node_i=0)}', color='green')
        for node_i in range(tree.n_simulations):
            for a in range(tree.n_actions):
                children_i = tree.children_index[node_i, a]
                if children_i >= 0:
                    graph.add_node(
                        children_i,
                        label=node_to_str(
                            node_i=children_i,
                            reward=tree.children_rewards[node_i, a]
                        ),
                        color='red'
                    )
                    graph.add_edge(node_i, children_i, label=edge_to_str(node_i, a))
        output_file = output_file if output_file is not None else '/tmp/search_tree.png'
        graph.draw(output_file, prog="dot")
