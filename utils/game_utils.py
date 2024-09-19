import string
from typing import Tuple, Dict

import numpy as np


def cell_to_action(cell: Tuple[int, int], board_size: int) -> int:
    """Convert from e.g. (1, 1) to 0"""
    return cell[0] * board_size + cell[1]


def cellstr_to_action(cell: str, board_size: int) -> int:
    """Convert from e.g. a1 to 0"""
    row_indices = string.ascii_lowercase
    col_indices = range(1, len(row_indices) + 1)
    row = row_indices.index(cell[0])
    col = col_indices.index(int(cell[1]))
    return cell_to_action((row, col), board_size)


def action_to_cell(idx: int, board_size: int) -> Tuple[int, int]:
    """Convert from e.g. 0 to (1, 1)"""
    return idx // board_size, idx % board_size


def action_to_cellstr(idx: int, board_size: int) -> str:
    """Convert from e.g. 0 to a1"""
    row_indices = string.ascii_lowercase
    col_indices = range(1, len(row_indices) + 1)
    x, y = action_to_cell(idx, board_size)
    return row_indices[x] + str(col_indices[y])


def draw_board(board: np.ndarray, pieces: Dict[int, str]) -> None:
    """
    Draw n-player board game

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


def turn_swapper(players: int, to_play: int) -> int:
    assert players == 1 or players == 2, 'The project supports 1P and 2P games only.'
    return to_play if players == 1 else 1 - to_play


def mask_illegal_actions(action_mask: np.ndarray) -> np.ndarray:
    """Returns legal actions only"""
    return np.argwhere(action_mask==1).squeeze(-1)


def mask_illegal_action_logits(
    action_logits: np.ndarray,
    legal_actions: np.ndarray
) -> np.ndarray:
    """Returns logits with zero mass to illegal actions"""
    action_logits = action_logits - np.max(action_logits, keepdims=True)
    min_logit = np.finfo(action_logits.dtype).min
    return np.where(legal_actions, action_logits, min_logit)
