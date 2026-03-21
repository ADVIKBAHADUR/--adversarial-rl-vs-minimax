"""Default semi-intelligent opponent: wins if possible, blocks if needed, else heuristic."""

import numpy as np
from .base import Agent


class DefaultAgent(Agent):
    """Rule-based opponent (better than random).

    Priority: 1) Win  2) Block opponent win  3) Centre  4) Random.
    Works with any Game that implements the standard interface.
    """

    def __init__(self, game=None):
        self._game = game

    @property
    def name(self) -> str:
        return "Default"

    def set_game(self, game):
        """Bind the agent to a game (needed for win simulation)."""
        self._game = game

    def select_action(self, state: np.ndarray, valid_actions: np.ndarray) -> int:
        game = self._game
        actions = np.where(valid_actions)[0]

        # 1) Check for winning move
        for a in actions:
            _, done, winner = game.step(state, a, 1)
            if done and winner == 1:
                return int(a)

        # 2) Check for blocking move (opponent would win)
        for a in actions:
            _, done, winner = game.step(state, a, -1)
            if done and winner == -1:
                return int(a)

        # 3) Prefer centre
        centre = self._get_centre_actions(state)
        centre_valid = [a for a in centre if valid_actions[a]]
        if centre_valid:
            return int(centre_valid[0])

        # 4) Random
        return int(np.random.choice(actions))

    def _get_centre_actions(self, state: np.ndarray) -> list:
        """Return centre action(s) depending on board shape."""
        shape = state.shape
        if len(shape) == 2:
            rows, cols = shape
            mid_c = cols // 2
            if rows == cols:
                # Square board (TTT): centre cell
                mid_r = rows // 2
                return [mid_r * cols + mid_c]
            else:
                # Rectangular (Connect 4): centre column
                return [mid_c]
        return []
