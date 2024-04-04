import string
from typing import Tuple, Dict

import numpy as np


class VisitSoftmaxTemperatureFn:

    def tictactoe(self, n_training_steps: int, training_step: int) -> float:
        """
        :param n_training_steps: number of training steps
        :param training_step: current training step
        """
        return 1.0


def cell_to_idx(cell: Tuple[int, int], board_size: int) -> int:
    return cell[0] * board_size + cell[1]


def cellstr_to_idx(cell: str, board_size: int) -> int:
    row_indices = string.ascii_lowercase
    col_indices = range(1, len(row_indices) + 1)
    row = row_indices.index(cell[0])
    col = col_indices.index(int(cell[1]))
    return cell_to_idx((row, col), board_size)


def idx_to_cell(idx: int, board_size: int) -> Tuple[int, int]:
    return idx // board_size, idx % board_size


def draw_board(board: np.ndarray, pieces: Dict[int, str]) -> None:
    """
    Draw n-players board game
    
    :param board: board state
    :param pieces: dictionary of symbols, empty cell included,
            E.g., Tictactoe: {-1: 'X', 1: 'O', 0: ' '}
    """
    row_indices = string.ascii_lowercase
    col_indices = range(1, len(row_indices) + 1)
    col_indices_txt = '  '
    rows, cols = board.shape[0], board.shape[1]

    for i in range(cols):
        col_indices_txt += '  ' + str(col_indices[i]) + ' '

    print(col_indices_txt)
    print('  +' + '---+' * cols)
    for i in range(rows):
        board_row = row_indices[i] + ' | '
        for j in range(cols):
            piece = pieces[board[i, j]]
            board_row += piece + ' | '
        print(board_row)
        print('  +' + '---+' * cols)
