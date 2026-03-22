"""Minimax agent — supports both vanilla and alpha-beta pruning variants.

STUB: Core _minimax and _alphabeta methods are placeholders.
The interface, depth limiting, node counting, and evaluation function
hooks are all wired up and ready.
"""

import time
import numpy as np
from .base import Agent
from ..config import MinimaxConfig


class MinimaxAgent(Agent):
    """Minimax game-playing agent with optional alpha-beta pruning."""

    def __init__(self, game=None, config: MinimaxConfig | None = None):
        self.cfg = config or MinimaxConfig()
        self._game = game
        self.stats = {"nodes_visited": 0, "time_per_move": 0.0}

    @property
    def name(self) -> str:
        variant = "αβ" if self.cfg.use_alpha_beta else "vanilla"
        depth = self.cfg.max_depth or "∞"
        return f"Minimax({variant}, d={depth})"

    def set_game(self, game):
        self._game = game

    def select_action(self, state: np.ndarray, valid_actions: np.ndarray) -> int:
        self.stats["nodes_visited"] = 0
        t0 = time.perf_counter()

        actions = np.where(valid_actions)[0]
        if self.cfg.move_ordering:
            actions = self._order_moves(actions)

        best_action = actions[0]
        best_score = -np.inf

        for a in actions:
            new_state, done, winner = self._game.step(state, a, 1)
            if done:
                score = self._terminal_score(winner)
            elif self.cfg.use_alpha_beta:
                score = self._alphabeta(new_state, self.cfg.max_depth, -np.inf, np.inf, False)
            else:
                score = self._minimax(new_state, self.cfg.max_depth, False)

            if score > best_score:
                best_score = score
                best_action = a

        self.stats["time_per_move"] = time.perf_counter() - t0
        return int(best_action)

    # ── STUB: implement these ─────────────────────────────────────────────────

    def _minimax(self, state: np.ndarray, depth: int | None, is_maximising: bool) -> float:
        """Vanilla minimax recursive search.

        TODO: Implement full recursive minimax.
        - Base cases: terminal state or depth == 0
        - Recursive case: iterate valid actions, recurse, track best score
        """
        self.stats["nodes_visited"] += 1

        actions = np.where(self._game.get_valid_actions(state))[0]
        if len(actions) == 0:
            return 0.0

        best_score = -np.inf if is_maximising else np.inf
        current_player = 1 if is_maximising else -1
        for action in actions:
            new_state, done, winner = self._game.step(state, action, current_player)
            if done:
                score = self._terminal_score(winner)
            else:
                score = self._minimax(new_state, None, not is_maximising)
            
            if is_maximising:
                best_score = max(best_score, score)
            else:
                best_score = min(best_score, score)
        return best_score

    def _alphabeta(self, state: np.ndarray, depth: int | None,
                   alpha: float, beta: float, is_maximising: bool) -> float:
        """Alpha-beta pruning minimax search.

        TODO: Implement alpha-beta with pruning.
        - Same as minimax but prune when alpha >= beta
        - Track pruned branches in stats
        """
        self.stats["nodes_visited"] += 1
        raise NotImplementedError("Alpha-beta search not yet implemented")

    # ── Helpers (ready to use) ────────────────────────────────────────────────

    def _terminal_score(self, winner: int) -> float:
        """Score a terminal state: +1 for our win, -1 for loss, 0 for draw."""
        return float(winner)  # +1, -1, or 0

    def evaluate(self, state: np.ndarray) -> float:
        """Heuristic evaluation for depth-limited search.

        TODO: Implement for Connect 4 (streak counting + centre control).
        For Tic Tac Toe with unlimited depth, this is never called.
        """
        return 0.0

    def _order_moves(self, actions: np.ndarray) -> np.ndarray:
        """Order moves centre-first for better alpha-beta pruning."""
        if self._game is None:
            return actions
        mid = self._game.action_space // 2
        return np.array(sorted(actions, key=lambda a: abs(a - mid)))
