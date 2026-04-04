"""Default configuration for all games and algorithms.

Override any value by passing kwargs or a config dict to the relevant class.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List


# ── Game configs ──────────────────────────────────────────────────────────────

@dataclass
class TicTacToeConfig:
    board_size: int = 3
    win_length: int = 3


@dataclass
class Connect4Config:
    rows: int = 6
    cols: int = 7
    win_length: int = 4


# ── Algorithm configs ─────────────────────────────────────────────────────────

@dataclass
class MinimaxConfig:
    max_depth: Optional[int] = None   # None = unlimited
    use_alpha_beta: bool = False
    move_ordering: bool = True        # centre-first ordering for α-β


@dataclass
class QLearningConfig:
    learning_rate: float = 0.2
    discount: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.9995
    episodes: int = 50_000


@dataclass
class DQNConfig:
    learning_rate: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000       # Large buffer retains diverse history across phases
    target_update_freq: int = 200   # sync target net every 200 episodes
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05       # Keep 5% exploration to avoid greedy collapse
    epsilon_decay: float = 0.999    # Exponential decay: hit floor ~ep 3000 for 20k run
    lr_decay: float = 1.0           # 1.0 = no decay
    lr_min: float = 1e-6            # Floor for the learning rate
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    episodes: int = 20_000
    grad_steps_per_episode: int = 5 # ~1 update per agent action (TTT avg ~5 agent turns)
    win_reward: float = 1.0
    draw_reward: float = 0.3        # draws are good — reward them properly
    loss_reward: float = -1.0


# ── Tournament/experiment configs ─────────────────────────────────────────────

@dataclass
class TournamentConfig:
    n_games: int = 100
    swap_sides: bool = True       # run each matchup with swapped P1/P2
    verbose: bool = False
    seed: Optional[int] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_dict(cfg) -> dict:
    """Convert any dataclass config to a plain dict."""
    return asdict(cfg)
