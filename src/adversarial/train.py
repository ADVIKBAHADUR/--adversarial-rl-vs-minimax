"""Training CLI — train RL agents and save models."""

import argparse
import os
from .games import TicTacToe, Connect4
from .agents import DefaultAgent, QLearningAgent, DQNAgent


def _make_game(name: str):
    if name == "tictactoe":
        return TicTacToe()
    elif name == "connect4":
        return Connect4()
    raise ValueError(f"Unknown game: {name}")


def main():
    parser = argparse.ArgumentParser(description="Train an RL agent")
    parser.add_argument("--game", choices=["tictactoe", "connect4"], default="tictactoe")
    parser.add_argument("--algo", choices=["qlearning", "dqn"], default="qlearning")
    parser.add_argument("--episodes", type=int, default=None, help="Override training episodes")
    parser.add_argument("--opponent", choices=["default", "random"], default="default")
    parser.add_argument("--output", type=str, default=None, help="Model save path")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epsilon-decay", type=float, default=None)
    args = parser.parse_args()

    game = _make_game(args.game)

    # Create opponent
    if args.opponent == "default":
        opponent = DefaultAgent(game)
        opponent.set_game(game)
    else:
        # Random opponent (DefaultAgent without game = falls back to random)
        from .agents.base import Agent
        import numpy as np

        class RandomAgent(Agent):
            @property
            def name(self): return "Random"
            def select_action(self, state, valid_actions):
                return int(np.random.choice(np.where(valid_actions)[0]))
        opponent = RandomAgent()

    # Create agent
    if args.algo == "qlearning":
        from .config import QLearningConfig
        cfg = QLearningConfig()
        if args.episodes: cfg.episodes = args.episodes
        if args.lr: cfg.learning_rate = args.lr
        if args.epsilon_decay: cfg.epsilon_decay = args.epsilon_decay
        agent = QLearningAgent(game, cfg)
    else:
        from .config import DQNConfig
        cfg = DQNConfig()
        if args.episodes: cfg.episodes = args.episodes
        if args.lr: cfg.learning_rate = args.lr
        if args.epsilon_decay: cfg.epsilon_decay = args.epsilon_decay
        agent = DQNAgent(game, cfg)

    agent.set_game(game)

    print(f"🎓 Training {agent.name} on {game.name}")
    print(f"   Opponent: {opponent.name}")
    print(f"   Episodes: {cfg.episodes}")
    print()

    # Train
    metrics = agent.train(game, opponent, cfg.episodes)

    # Save
    save_path = args.output or f"models/{args.algo}_{args.game}.{'pkl' if args.algo == 'qlearning' else 'pt'}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    print(f"\n💾 Model saved to {save_path}")


if __name__ == "__main__":
    main()
