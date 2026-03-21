"""Tabular Q-Learning agent.

STUB: Training loop and Q-update are placeholders.
The Q-table structure, epsilon-greedy policy, and save/load are wired up.
"""

import pickle
import numpy as np
from .base import Agent
from ..config import QLearningConfig


class QLearningAgent(Agent):
    """Tabular Q-learning with ε-greedy exploration."""

    def __init__(self, game=None, config: QLearningConfig | None = None):
        self.cfg = config or QLearningConfig()
        self._game = game
        self.q_table: dict[int, np.ndarray] = {}  # state_key -> Q-values array
        self.epsilon = self.cfg.epsilon_start
        self._training = False

    @property
    def name(self) -> str:
        return "Q-Learning"

    def set_game(self, game):
        self._game = game
        self._action_space = game.action_space

    def select_action(self, state: np.ndarray, valid_actions: np.ndarray) -> int:
        """ε-greedy action selection."""
        actions = np.where(valid_actions)[0]

        # Explore
        if self._training and np.random.random() < self.epsilon:
            return int(np.random.choice(actions))

        # Exploit
        q_values = self._get_q_values(state)
        # Mask invalid actions to -inf
        masked = np.full_like(q_values, -np.inf)
        masked[valid_actions] = q_values[valid_actions]
        return int(np.argmax(masked))

    def _get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get or initialise Q-values for a state."""
        key = self._game.state_to_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self._action_space, dtype=np.float32)
        return self.q_table[key]

    def train(self, game, opponent, episodes: int | None = None,
              callback=None) -> dict:
        """Train via self-play against an opponent.

        TODO: Implement the Q-learning training loop.
        - Play episodes against opponent
        - Update Q-values: Q[s,a] += α * (r + γ * max(Q[s']) - Q[s,a])
        - Decay epsilon
        - Track win rate, Q-table size, episode rewards

        Args:
            game: Game instance to train on.
            opponent: Agent to train against.
            episodes: Override config episodes.
            callback: Optional fn(episode, metrics_dict) called each episode.

        Returns:
            Dict of training metrics.
        """
        raise NotImplementedError("Q-learning training not yet implemented")

    def save(self, path: str):
        data = {"q_table": self.q_table, "config": self.cfg, "epsilon": self.epsilon}
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data.get("epsilon", self.cfg.epsilon_end)
        self._training = False
