"""Abstract base class for adversarial two-player games."""

from abc import ABC, abstractmethod
import numpy as np


class Game(ABC):
    """Interface that all games must implement.

    Convention:
      - Player 1 = +1, Player 2 = -1
      - State is always a np.ndarray (int8)
      - Actions are integers in [0, action_space)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable game name."""

    @property
    @abstractmethod
    def action_space(self) -> int:
        """Total number of possible actions (including invalid ones)."""

    @property
    @abstractmethod
    def state_shape(self) -> tuple:
        """Shape of the board numpy array."""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Return a fresh initial board state."""

    @abstractmethod
    def get_valid_actions(self, state: np.ndarray) -> np.ndarray:
        """Return a boolean mask of valid actions (shape = (action_space,))."""

    @abstractmethod
    def step(self, state: np.ndarray, action: int, player: int) -> tuple:
        """Apply action for player, return (new_state, done, winner).

        winner: +1, -1, or 0 (draw / game ongoing).
        """

    @abstractmethod
    def render(self, state: np.ndarray) -> str:
        """Return a string representation of the board."""

    def state_to_key(self, state: np.ndarray) -> tuple:
        """Hash the board state deterministically for Q-table lookups.
        
        We return a tuple of the flattened array instead of `hash()` because
        Python's built-in `hash()` is randomized per-process, meaning saved 
        models would break upon reloading.
        """
        return tuple(state.ravel().tolist())

    def clone(self, state: np.ndarray) -> np.ndarray:
        """Return a deep copy of the state."""
        return state.copy()
