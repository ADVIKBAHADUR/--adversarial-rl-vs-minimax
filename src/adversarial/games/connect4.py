"""Connect 4 game engine — fast, numpy-based with gravity."""

import numpy as np
from .base import Game


class Connect4(Game):
    """Connect 4 on a configurable rows×cols board with gravity."""

    def __init__(self, rows: int = 6, cols: int = 7, win_length: int = 4):
        self.rows = rows
        self.cols = cols
        self.win_length = win_length

    @property
    def name(self) -> str:
        return f"Connect4-{self.rows}x{self.cols}"

    @property
    def action_space(self) -> int:
        return self.cols

    @property
    def state_shape(self) -> tuple:
        return (self.rows, self.cols)

    def reset(self) -> np.ndarray:
        return np.zeros(self.state_shape, dtype=np.int8)

    def get_valid_actions(self, state: np.ndarray) -> np.ndarray:
        """A column is valid if the top row is empty."""
        return state[0, :] == 0

    def get_drop_row(self, state: np.ndarray, col: int) -> int:
        """Return the lowest empty row in the given column, or -1 if full."""
        empty = np.where(state[:, col] == 0)[0]
        return int(empty[-1]) if len(empty) > 0 else -1

    def step(self, state: np.ndarray, action: int, player: int) -> tuple:
        row = self.get_drop_row(state, action)
        new_state = state.copy()
        new_state[row, action] = player
        winner = self._check_winner(new_state, row, action, player)
        draw = winner == 0 and not np.any(new_state == 0)
        done = winner != 0 or draw
        return new_state, done, winner

    def _check_winner(self, state: np.ndarray, r: int, c: int, player: int) -> int:
        """Check if the last move at (r,c) won the game."""
        w = self.win_length
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                nr, nc = r + sign * dr, c + sign * dc
                while 0 <= nr < self.rows and 0 <= nc < self.cols and state[nr, nc] == player:
                    count += 1
                    nr += sign * dr
                    nc += sign * dc
            if count >= w:
                return player
        return 0

    def state_to_key(self, state: np.ndarray) -> tuple:
        """Convert board to a canonical hashable key for the Q-table."""
        # Horizontal mirror symmetry reduction
        norm_state = self.normalize_state(state)
        return tuple(norm_state.ravel())

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Return the canonical representation of the state (horizontal mirror)."""
        mirrored = np.flip(state, axis=1)
        # Flatten and compare to pick a stable canonical version
        if tuple(state.ravel()) <= tuple(mirrored.ravel()):
            return state
        return mirrored

    def render(self, state: np.ndarray) -> str:
        symbols = {0: "·", 1: "X", -1: "O"}
        rows = []
        # Column headers
        rows.append("  " + " ".join(str(i) for i in range(self.cols)))
        for r in range(self.rows):
            row_str = " ".join(symbols[int(state[r, c])] for c in range(self.cols))
            rows.append(f"{r} {row_str}")
        return "\n".join(rows)
