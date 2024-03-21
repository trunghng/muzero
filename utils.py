from typing import Tuple
import random

import numpy as np
import torch


class VisitSoftmaxTemperatureFn:

    def tictactoe(self, training_steps: int, trained_steps: int) -> float:
        return 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cell2idx(cell: Tuple[int, int], board_size: int) -> int:
    return cell[0] * board_size + cell[1]


def idx2cell(idx: int, board_size: int) -> Tuple[int, int]:
    return idx // board_size, idx % board_size