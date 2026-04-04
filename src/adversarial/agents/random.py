"""Random agent for baseline testing and curriculum learning."""

import numpy as np
from .base import Agent


class RandomAgent(Agent):
    """An agent that selects a random valid action."""

    def __init__(self, game=None, config=None):
        self._game = game
        self.cfg = config

    @property
    def name(self) -> str:
        return "Random"

    def set_game(self, game):
        self._game = game

    def select_action(self, state: np.ndarray, valid_actions: np.ndarray) -> int:
        """Select a random true index from the valid_actions mask."""
        actions = np.where(valid_actions)[0]
        return int(np.random.choice(actions))
