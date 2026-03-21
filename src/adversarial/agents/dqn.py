"""Deep Q-Network (DQN) agent.

STUB: Network, replay buffer, and training loop are placeholders.
The architecture, action masking, and experience replay interfaces are defined.
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from .base import Agent
from ..config import DQNConfig


class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Simple MLP Q-network."""

    def __init__(self, input_size: int, output_size: int, hidden_sizes: list[int]):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(Agent):
    """Deep Q-Network agent with experience replay and target network."""

    def __init__(self, game=None, config: DQNConfig | None = None):
        self.cfg = config or DQNConfig()
        self._game = game
        self.epsilon = self.cfg.epsilon_start
        self._training = False

        # These are initialised when set_game is called
        self.q_net: QNetwork | None = None
        self.target_net: QNetwork | None = None
        self.optimiser = None
        self.buffer: ReplayBuffer | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return "DQN"

    def set_game(self, game):
        self._game = game
        input_size = int(np.prod(game.state_shape))
        output_size = game.action_space
        self.q_net = QNetwork(input_size, output_size, self.cfg.hidden_sizes).to(self.device)
        self.target_net = QNetwork(input_size, output_size, self.cfg.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimiser = torch.optim.Adam(self.q_net.parameters(), lr=self.cfg.learning_rate)
        self.buffer = ReplayBuffer(self.cfg.buffer_size)

    def select_action(self, state: np.ndarray, valid_actions: np.ndarray) -> int:
        """ε-greedy with invalid action masking."""
        actions = np.where(valid_actions)[0]

        if self._training and np.random.random() < self.epsilon:
            return int(np.random.choice(actions))

        with torch.no_grad():
            state_t = torch.FloatTensor(state.ravel()).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t).cpu().numpy().ravel()

        # Mask invalid actions
        masked = np.full_like(q_values, -np.inf)
        masked[valid_actions] = q_values[valid_actions]
        return int(np.argmax(masked))

    def train(self, game, opponent, episodes: int | None = None,
              callback=None) -> dict:
        """Train via self-play against an opponent.

        TODO: Implement the DQN training loop.
        - Play episodes, store transitions in replay buffer
        - Sample mini-batches and update Q-network
        - Periodically sync target network
        - Decay epsilon
        - Track loss, win rate, epsilon

        Args:
            game: Game instance.
            opponent: Agent to play against.
            episodes: Override config episodes.
            callback: Optional fn(episode, metrics_dict) per episode.

        Returns:
            Dict of training metrics.
        """
        raise NotImplementedError("DQN training not yet implemented")

    def save(self, path: str):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "config": self.cfg,
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if self.q_net is None:
            raise RuntimeError("Call set_game() before load() to initialise the network")
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["q_net"])
        self.epsilon = checkpoint.get("epsilon", self.cfg.epsilon_end)
        self._training = False
