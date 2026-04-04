"""Tabular Q-Learning agent.

STUB: Training loop and Q-update are placeholders.
The Q-table structure, epsilon-greedy policy, and save/load are wired up.
"""

import pickle
import numpy as np
from .base import Agent
from ..config import QLearningConfig


class QLearningAgent(Agent):
    """Afterstate tabular Q-learning with ε-greedy exploration.
    
    Instead of Q(s, a), we learn V(s_after) where s_after is the state 
    resulting from an action. This is more efficient for games where 
    different (s, a) lead to the same board.
    """

    def __init__(self, game=None, config: QLearningConfig | None = None):
        self.cfg = config or QLearningConfig()
        self._game = game
        self.v_table: dict[tuple, float] = {}  # state_key -> Value
        self.epsilon = self.cfg.epsilon_start
        self.episodes_trained: int = 0          # cumulative across all curriculum stages
        self._training = False

    @property
    def name(self) -> str:
        return "Q-Learning"

    def set_game(self, game):
        self._game = game
        self._action_space = game.action_space

    def select_action(self, state: np.ndarray, valid_actions: np.ndarray) -> int:
        """ε-greedy action selection using afterstates."""
        actions = np.where(valid_actions)[0]

        # Explore
        if self._training and np.random.random() < self.epsilon:
            return int(np.random.choice(actions))

        # Exploit: find action leading to best afterstate
        best_val = -float('inf')
        best_actions = []

        for action in actions:
            # Predict the next state (agent is always viewed as player 1 in its perspective)
            next_state, _, _ = self._game.step(state, action, 1)
            val = self._get_v_value(next_state)
            
            if val > best_val:
                best_val = val
                best_actions = [action]
            elif val == best_val:
                best_actions.append(action)

        return int(np.random.choice(best_actions))

    def _get_v_value(self, state: np.ndarray) -> float:
        """Get or initialise Value for a state."""
        key = self._game.state_to_key(state)
        val = self.v_table.get(key, 0.0)
        # Coerce to plain Python float — old pickles may have stored numpy arrays/scalars
        try:
            return float(np.asarray(val).flat[0])
        except Exception:
            return 0.0


    def train(self, game, opponent, episodes: int | None = None,
              callback=None, start_ep: int = 0, total_eps: int | None = None) -> dict:
        """Train via self-play against an opponent using afterstates."""
        self._training = True
        episodes = episodes or self.cfg.episodes
        total_eps = total_eps or self.cfg.episodes
        
        metrics = {"p1_wins": 0, "p2_wins": 0, "draws": 0}
        episodes_this_call = 0
        
        for i in range(episodes):
            global_ep = start_ep + i
            state = game.reset()
            opponent.reset()
            
            # Agent side: 1 or -1
            agent_side = 1 if global_ep % 2 == 0 else -1
            
            player = 1
            last_afterstate_key = None
            
            while True:
                # Get current perspective (1 = us, -1 = opponent)
                perspective = state * agent_side
                valid = game.get_valid_actions(state)
                
                if player == agent_side:
                    # Agent's turn
                    action = self.select_action(perspective, valid)
                    
                    # Transition to AFTERSTATE (result of our move)
                    next_state, done, winner = game.step(state, action, player)
                    afterstate_perspective = next_state * agent_side
                    current_afterstate_key = game.state_to_key(afterstate_perspective)
                    
                    # Update transition: last_afterstate -> current_afterstate
                    if last_afterstate_key is not None:
                        # V(s_last) = V(s_last) + alpha * (gamma * V(s_current) - V(s_last))
                        # Note: no reward here as it's a non-terminal transition
                        target = self.cfg.discount * self.v_table.get(current_afterstate_key, 0.0)
                        old_v = self.v_table.get(last_afterstate_key, 0.0)
                        self.v_table[last_afterstate_key] = old_v + self.cfg.learning_rate * (target - old_v)

                    last_afterstate_key = current_afterstate_key
                    state = next_state
                else:
                    # Opponent's turn
                    opp_perspective = state * -agent_side
                    action = opponent.select_action(opp_perspective, valid)
                    state, done, winner = game.step(state, action, player)

                if done:
                    # Terminal state reached
                    if winner == agent_side:
                        reward = 1.0  # Agent won
                    elif winner == -agent_side:
                        reward = -1.0 # Agent lost
                    else:
                        reward = 0.5  # Draw
                        
                    # Final update for the last afterstate reached by the agent
                    if last_afterstate_key is not None:
                        old_v = self.v_table.get(last_afterstate_key, 0.0)
                        # Target is just the immediate reward (no future afterstate)
                        self.v_table[last_afterstate_key] = old_v + self.cfg.learning_rate * (reward - old_v)
                        
                    if winner == agent_side: metrics["p1_wins"] += 1
                    elif winner == -agent_side: metrics["p2_wins"] += 1
                    else: metrics["draws"] += 1
                    break
                    
                player *= -1

            episodes_this_call += 1

            # Simple linear epsilon decay
            explore_duration = total_eps * 0.9
            if global_ep < explore_duration:
                self.epsilon = self.cfg.epsilon_start - (self.cfg.epsilon_start - self.cfg.epsilon_end) * (global_ep / explore_duration)
            else:
                self.epsilon = self.cfg.epsilon_end
            
            if callback:
                callback(global_ep, metrics)
                
        self._training = False
        self.episodes_trained += episodes_this_call
        metrics["v_table_size"] = len(self.v_table)
        return metrics

    def save(self, path: str):
        data = {
            "v_table": self.v_table,
            "config": self.cfg,
            "epsilon": self.epsilon,
            "episodes_trained": self.episodes_trained,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Handle old checkpoints that used 'q_table' key before the rename to 'v_table'
        self.v_table = data.get("v_table") or data.get("q_table", {})
        self.epsilon = data.get("epsilon", self.cfg.epsilon_end)
        self.episodes_trained = data.get("episodes_trained", 0)
        self._training = False

