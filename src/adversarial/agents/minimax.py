"""Minimax agent — supports both vanilla and alpha-beta pruning variants.

STUB: Core _minimax and _alphabeta methods are placeholders.
The interface, depth limiting, node counting, and evaluation function
hooks are all wired up and ready.
"""

import time
import numpy as np
from .base import Agent
from ..config import MinimaxConfig


# Cap the transposition table. With 86GB+ RAM we can afford a large cache.
# Each entry ~200 bytes, so 5M entries ≈ 1GB.
_MAX_CACHE_SIZE = 5_000_000


class MinimaxAgent(Agent):
    """Minimax game-playing agent with optional alpha-beta pruning."""

    def __init__(self, game=None, config: MinimaxConfig | None = None):
        self.cfg = config or MinimaxConfig()
        self._game = game
        self.stats = {"nodes_visited": 0, "time_per_move": 0.0}
        self._cache = {}

    @property
    def name(self) -> str:
        variant = "αβ" if self.cfg.use_alpha_beta else "vanilla"
        depth = self.cfg.max_depth or "∞"
        return f"Minimax({variant}, d={depth})"

    def set_game(self, game):
        self._game = game
        self._cache.clear()

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
        """Vanilla minimax recursive search."""
        key = (state.tobytes(), depth, is_maximising)
        if key in self._cache:
            return self._cache[key]
            
        self.stats["nodes_visited"] += 1

        if depth is not None and depth <= 0:
            res = self.evaluate(state)
            self._cache[key] = res
            return res

        actions = np.where(self._game.get_valid_actions(state))[0]
        if len(actions) == 0:
            self._cache[key] = 0.0
            return 0.0

        best_score = -np.inf if is_maximising else np.inf
        current_player = 1 if is_maximising else -1
        
        for action in actions:
            new_state, done, winner = self._game.step(state, action, current_player)
            if done:
                score = self._terminal_score(winner)
            else:
                next_depth = depth - 1 if depth is not None else None
                score = self._minimax(new_state, next_depth, not is_maximising)
            
            if is_maximising:
                best_score = max(best_score, score)
            else:
                best_score = min(best_score, score)
                
        if len(self._cache) >= _MAX_CACHE_SIZE:
            self._cache.clear()
        self._cache[key] = best_score
        return best_score

    def _alphabeta(self, state: np.ndarray, depth: int | None,
                   alpha: float, beta: float, is_maximising: bool) -> float:
        """Alpha-beta pruning minimax search."""
        key = (state.tobytes(), depth, is_maximising)
        if key in self._cache:
            entry = self._cache[key]
            if entry['flag'] == 'EXACT':
                return entry['value']
            elif entry['flag'] == 'LOWERBOUND' and entry['value'] > alpha:
                alpha = entry['value']
            elif entry['flag'] == 'UPPERBOUND' and entry['value'] < beta:
                beta = entry['value']
            if alpha >= beta:
                return entry['value']
                
        self.stats["nodes_visited"] += 1

        if depth is not None and depth <= 0:
            res = self.evaluate(state)
            self._cache[key] = {'value': res, 'flag': 'EXACT'}
            return res

        actions = np.where(self._game.get_valid_actions(state))[0]
        if len(actions) == 0:
            self._cache[key] = {'value': 0.0, 'flag': 'EXACT'}
            return 0.0

        if self.cfg.move_ordering:
            actions = self._order_moves(actions)

        best_score = -np.inf if is_maximising else np.inf
        current_player = 1 if is_maximising else -1
        
        orig_alpha = alpha
        orig_beta = beta

        for action in actions:
            new_state, done, winner = self._game.step(state, action, current_player)
            if done:
                score = self._terminal_score(winner)
            else:
                next_depth = depth - 1 if depth is not None else None
                score = self._alphabeta(new_state, next_depth, alpha, beta, not is_maximising)
            
            if is_maximising:
                best_score = max(best_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            else:
                best_score = min(best_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
                    
        if best_score <= orig_alpha:
            flag = 'UPPERBOUND'
        elif best_score >= orig_beta:
            flag = 'LOWERBOUND'
        else:
            flag = 'EXACT'

        if len(self._cache) >= _MAX_CACHE_SIZE:
            self._cache.clear()
        self._cache[key] = {'value': best_score, 'flag': flag}
        return best_score

    # ── Helpers (ready to use) ────────────────────────────────────────────────

    def _terminal_score(self, winner: int) -> float:
        """Score a terminal state: +1 for our win, -1 for loss, 0 for draw."""
        return float(winner)  # +1, -1, or 0

    def evaluate(self, state: np.ndarray) -> float:
        """Fast vectorized heuristic evaluation for Connect 4.
        
        Uses numpy slicing to count 3-in-a-row windows instantly without loops.
        A sum of 3 in a 4-cell window means three 1s and one 0.
        """
        if state.shape == (3, 3):
            return 0.0

        _, cols = state.shape
        score = 0.0
        
        # 1. Center column preference (small baseline bonus)
        center = cols // 2
        score += np.sum(state[:, center]) * 3
        
        # 2. Horizontal & Vertical 3-in-a-rows
        # A sum of exactly 3 means we have 3 pieces and 1 empty space.
        # A sum of -3 means the opponent has 3 pieces and 1 empty space.
        h_windows = state[:, :-3] + state[:, 1:-2] + state[:, 2:-1] + state[:, 3:]
        score += (np.sum(h_windows == 3) - np.sum(h_windows == -3)) * 10
        
        v_windows = state[:-3, :] + state[1:-2, :] + state[2:-1, :] + state[3:, :]
        score += (np.sum(v_windows == 3) - np.sum(v_windows == -3)) * 10
        
        return float(score)
    def _order_moves(self, actions: np.ndarray) -> np.ndarray:
        """Order moves centre-first for better alpha-beta pruning."""
        if self._game is None:
            return actions
        mid = self._game.action_space // 2
        return np.array(sorted(actions, key=lambda a: abs(a - mid)))
