"""Tic Tac Toe game engine — fast, numpy-based."""

import numpy as np
from .base import Game


class TicTacToe(Game):
    """Standard Tic Tac Toe on a configurable n×n board."""

    def __init__(self, board_size: int = 3, win_length: int | None = None):
        self.board_size = board_size
        self.win_length = win_length or board_size

    @property
    def name(self) -> str:
        return f"TicTacToe-{self.board_size}x{self.board_size}"

    @property
    def action_space(self) -> int:
        return self.board_size ** 2

    @property
    def state_shape(self) -> tuple:
        return (self.board_size, self.board_size)

    def reset(self) -> np.ndarray:
        return np.zeros(self.state_shape, dtype=np.int8)

    def get_valid_actions(self, state: np.ndarray) -> np.ndarray:
        return state.ravel() == 0

    def step(self, state: np.ndarray, action: int, player: int) -> tuple:
        r, c = divmod(action, self.board_size)
        new_state = state.copy()
        new_state[r, c] = player
        winner = self._check_winner(new_state, r, c, player)
        draw = winner == 0 and not np.any(new_state == 0)
        done = winner != 0 or draw
        return new_state, done, winner

    def _check_winner(self, state: np.ndarray, r: int, c: int, player: int) -> int:
        """Check if the last move at (r,c) won the game."""
        n, w = self.board_size, self.win_length
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag, anti-diag
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                nr, nc = r + sign * dr, c + sign * dc
                while 0 <= nr < n and 0 <= nc < n and state[nr, nc] == player:
                    count += 1
                    nr += sign * dr
                    nc += sign * dc
            if count >= w:
                return player
        return 0

    def render(self, state: np.ndarray) -> str:
        symbols = {0: "·", 1: "X", -1: "O"}
        rows = []
        n = self.board_size
        # Column headers
        rows.append("  " + " ".join(str(i) for i in range(n)))
        for r in range(n):
            row_str = " ".join(symbols[int(state[r, c])] for c in range(n))
            rows.append(f"{r} {row_str}")
        return "\n".join(rows)
