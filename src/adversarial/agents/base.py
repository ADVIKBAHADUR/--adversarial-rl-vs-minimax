"""Abstract base class for game-playing agents."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class Agent(ABC):
    """Interface that all agents must implement.

    Convention:
      - The agent always plays as player +1.
      - The game engine flips the board (multiply by -1) when it's player -1's turn,
        so agents always see themselves as +1.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name."""

    @abstractmethod
    def select_action(self, state: np.ndarray, valid_actions: np.ndarray) -> int:
        """Choose an action given the current board state.

        Args:
            state: Board array where +1 = this agent, -1 = opponent.
            valid_actions: Boolean mask of legal actions.

        Returns:
            Integer action index.
        """

    def reset(self):
        """Called at the start of each game (optional override)."""

    def save(self, path: str):
        """Persist the agent to disk (optional override)."""

    def load(self, path: str):
        """Load a previously saved agent (optional override)."""

    def __repr__(self) -> str:
        return self.name
