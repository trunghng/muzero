import numpy as np

from mcts.tree import Tree
from utils.utils import softmax


def qtransform_by_parent_and_siblings(
    tree: Tree,
    node_index: int,
    epsilon: float = 1e-8
) -> np.ndarray:
    qvalues = tree.qvalues(node_index)
    visit_counts = tree.children_visits[node_index]
    node_value = tree.node_values[node_index]
    safe_qvalues = np.where(visit_counts > 0, qvalues, node_value)
    min_value = np.minimum(node_value, np.min(safe_qvalues))
    max_value = np.maximum(node_value, np.max(safe_qvalues))

    completed_by_min = np.where(visit_counts > 0, qvalues, min_value)
    normalized = (completed_by_min - min_value) /\
        np.maximum(max_value - min_value, epsilon)
    return normalized


def qtransform_completed_by_mix_value(
    tree: Tree,
    node_index: int,
    c_visit: int,
    c_scale: float,
    epsilon: float = 1e-8
) -> np.ndarray:
    qvalues = tree.qvalues(node_index)
    visit_counts = tree.children_visits[node_index]
    mixed_value = compute_mixed_value(
        tree.node_raw_values[node_index],
        qvalues,
        visit_counts,
        tree.children_prior_logits[node_index]
    )  # (1,)

    completed_qvalues = np.where(visit_counts > 0, qvalues, mixed_value)  # (n_actions,)
    # Normalizes the Q-values to fit in [0, 1]
    min_value = np.min(completed_qvalues, keepdims=True)
    max_value = np.max(completed_qvalues, keepdims=True)
    completed_qvalues = (completed_qvalues - min_value) /\
        np.maximum(max_value - min_value, epsilon)  # (n_actions,)
    max_visit = np.max(visit_counts)
    return (c_visit + max_visit) * c_scale * completed_qvalues


def compute_mixed_value(
    raw_value: np.ndarray,
    qvalues: np.ndarray,
    visit_counts: np.ndarray,
    prior_logits: np.ndarray
) -> np.ndarray:
    """
    Computes an approximation of the state value:
        v_mix = (v_hat + sum_b(N(b)) * weighted_q) / (sum_b(N(b)) + 1)

    where weighted_q is the weighted average of the available Q-values,
    i.e. from visited actions

    :param raw_value: (1,) the estimated value produced by the policy network
    :param qvalues: (n_action,) Q-values for all actions
    :param visit_counts: (n_actions,) the number of visits for all actions
    :param prior_logits: (n_actions,) the action logits produced by the policy network
    """
    visit_sum = np.sum(visit_counts)
    prior_probs = softmax(prior_logits)  # (n_actions,)
    # NaN avoiding trick for weighted_q, even if visitied actions have zero
    # prior probability
    prior_probs = np.maximum(np.finfo(prior_probs.dtype).tiny, prior_probs)
    prob_sum = np.sum(np.where(visit_counts > 0, prior_probs, 0.0))  # (1,)
    weighted_q = np.sum(np.where(visit_counts > 0, prior_probs * qvalues, 0.0))\
        / np.where(visit_counts > 0, prob_sum, 1.0)  # (1,)
    return (raw_value + visit_sum * weighted_q) / (visit_sum + 1)  # (1,)
