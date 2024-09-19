import math
from typing import Callable, Tuple

import numpy as np

from mcts.tree import Tree
from utils.game_utils import mask_illegal_action_logits
from utils.utils import probs_to_logits, softmax


def switching_action_selection_wrapper(
    root_action_selection_func: Callable,
    interior_action_selection_func: Callable
) -> Callable:
    def switching_action_selection(
        tree: Tree,
        node_index: int,
        depth: int
    ) -> int:
        return root_action_selection_func(tree, node_index, depth) if depth == 0\
            else interior_action_selection_func(tree, node_index, depth)
    return switching_action_selection


def muzero_action_selection(
    tree: Tree,
    node_index: int,
    depth: int,
    discount: float,
    c_base: float,
    c_init: float,
    qtransform: Callable
) -> int:
    """
    Select the child node with highest UCB score
        ucb_score = Q(s,a) + U(s,a)
        U(s,a) = C(s) * P(s,a) + sqrt(N(s)) / (1 + N(s,a))
        C(s) = c_init + log((N(s) + c_base + 1) / c_base)
    """
    node_visits = tree.node_visits[node_index]  # (1,)
    children_visits = tree.children_visits[node_index]  # (n_actions,)
    children_prior_logits = tree.children_prior_logits[node_index]  # (n_actions,)
    c = c_init + np.log((node_visits + c_base + 1) / c_base)
    u = c * children_prior_logits + np.sqrt(node_visits / (1 + children_visits))  # (n_actions,)
    q = qtransform(tree, node_index)  # (n_actions,)
    tie_break_noise = 1e-7 * np.random.uniform(size=tree.n_actions)
    ucb_scores = u + q + tie_break_noise  # (n_actions,)
    root_legal_actions = tree.root_legal_actions if depth == 0 else None
    return masked_argmax(ucb_scores, root_legal_actions)


def muzero_action_selection_to_act(
    tree: Tree,
    temperature: float
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Selects action according to the number of visits:
        pi(a|s_0) = N(s_0,a)^(1/tau) / sum_b(N(s_0,b)^(1/t))
    If t = 0, pi is deterministic, returns the action with max #visits (best action)
    """
    _, visit_probs, qvalues = tree.summary()  # (n_actions,), (n_actions,)
    visit_logits = probs_to_logits(visit_probs)
    visit_logits = visit_logits - np.max(visit_logits)
    # tiny = np.finfo(visit_logits.dtype).tiny
    # visit_logits = visit_logits / np.maximum(tiny, temperature)
    # Use 1e-27 instead to avoid overflow warning
    visit_logits = visit_logits / np.maximum(1e-27, temperature)
    visit_probs = softmax(visit_logits)
    action = np.random.choice(tree.n_actions, p=visit_probs)
    return action, visit_probs, qvalues


def gumbel_muzero_root_action_selection(
    tree: Tree,
    node_index: int,
    depth: int,
    discount: float,
    simulations: int,
    max_considered_actions: int,
    gumbel_scale: float,
    qtransform: Callable
) -> int:
    del depth
    completed_qvalues = qtransform(tree, node_index)  # (n_actions,)
    visit_counts = tree.children_visits[node_index]
    n_legal_actions = np.sum(tree.root_legal_actions)
    n_considered = np.minimum(max_considered_actions, n_legal_actions).astype(np.int32)
    considered_visits = get_considered_visits(n_considered, simulations)  # (simulations,)
    simulation_index = np.sum(visit_counts)
    considered_visit = considered_visits[simulation_index]  # (1,)

    prior_logits = tree.children_prior_logits[node_index]  # (n_actions,)
    gumbel = tree.gumbel_noise  # (n_actions,)
    to_argmax = gumbel_score(
        considered_visit, gumbel, prior_logits, completed_qvalues, visit_counts)
    return masked_argmax(to_argmax, tree.root_legal_actions)


def gumbel_muzero_interior_action_selection(
    tree: Tree,
    node_index: int,
    depth: int,
    discount: float,
    qtransform: Callable
) -> int:
    """
    Selects action according to:
        action = argmax_a(pi'(a) - N(a) / (sum_b(N(b))))
    where:
        pi' = softmax(logits + sigma(completed_qvalues))
    """
    del depth
    completed_qvalues = qtransform(tree, node_index)  # (n_actions,)
    prior_logits = tree.children_prior_logits[node_index]  # (n_actions,)
    visit_counts = tree.children_visits[node_index]  # (n_actions,)
    action_probs = softmax(prior_logits + completed_qvalues)  # (n_actions,)
    return np.argmax(action_probs - visit_counts / (
        1 + np.sum(visit_counts, keepdims=True)))


def gumbel_muzero_action_selection_to_act(
    tree: Tree,
    qtransform: Callable
) -> Tuple[int, np.ndarray, np.ndarray]:
    visit_counts, _, _ = tree.summary()
    considered_visit = np.max(visit_counts)
    gumbel = tree.gumbel_noise
    prior_logits = tree.children_prior_logits[tree.ROOT_INDEX]
    completed_qvalues = qtransform(tree, tree.ROOT_INDEX)
    to_argmax = gumbel_score(
        considered_visit, gumbel, prior_logits, completed_qvalues, visit_counts)
    legal_actions = tree.root_legal_actions
    action = masked_argmax(to_argmax, legal_actions)
    # Computes the new improved policy: pi' = softmax(logits + sigma(completed_qvalues))
    improved_action_probs = softmax(
        mask_illegal_action_logits(prior_logits + completed_qvalues, legal_actions)
    )
    return action, improved_action_probs, completed_qvalues  # or qvalues?


def masked_argmax(to_argmax: np.ndarray, legal_actions: np.ndarray) -> int:
    if legal_actions is not None:
        to_argmax = np.where(legal_actions == 0, -np.inf, to_argmax)
    return np.argmax(to_argmax, axis=-1)


def gumbel_score(
    considered_visit: int,
    gumbel: np.ndarray,
    logits: np.ndarray,
    normalized_qvalues: np.ndarray,
    visit_counts: np.ndarray
) -> np.ndarray:
    """Returns the gumbel score used for argmax: g + logits + qvalues"""
    logits = logits - np.max(logits, keepdims=True)  # (n_actions,)
    penalty = np.where(visit_counts == considered_visit, 0.0, -np.inf)
    return np.maximum(-1e9, gumbel + logits + normalized_qvalues) + penalty


def get_considered_visits(max_considered_actions: int, simulations: int):
    if max_considered_actions <= 1:
        return tuple(range(simulations))
    log2max = int(math.ceil(math.log2(max_considered_actions)))
    sequence = []
    visits = [0] * max_considered_actions
    n_considered = max_considered_actions
    while len(sequence) < simulations:
        n_extra_visits = max(1, int(simulations / (log2max * n_considered)))
        for _ in range(n_extra_visits):
            sequence.extend(visits[:n_considered])
            for i in range(n_considered):
                visits[i] += 1
        # Halving the number of considered actions.
        n_considered = max(2, n_considered // 2)
    return tuple(sequence[:simulations])
