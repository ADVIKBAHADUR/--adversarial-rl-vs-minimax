"""Human interactive agent — reads moves from terminal input."""

import numpy as np
from .base import Agent


class HumanAgent(Agent):
    """Lets a human play via terminal input."""

    @property
    def name(self) -> str:
        return "Human"

    def select_action(self, state: np.ndarray, valid_actions: np.ndarray) -> int:
        actions = np.where(valid_actions)[0]
        while True:
            try:
                raw = input(f"Your move (valid: {actions}): ").strip()
                action = int(raw)
                if valid_actions[action]:
                    return action
                print(f"Action {action} is not valid. Try again.")
            except (ValueError, IndexError):
                print("Enter a valid integer action.")
