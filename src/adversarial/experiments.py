"""Experiment sweeps — automated parameter scanning with CSV output."""

import argparse
import itertools
import pandas as pd
from .games import TicTacToe, Connect4
from .agents import DefaultAgent, MinimaxAgent, QLearningAgent, DQNAgent
from .tournament import run_match
from .config import MinimaxConfig


def _make_game(name):
    return TicTacToe() if name == "tictactoe" else Connect4()


def _parse_sweep(sweep_str: str) -> tuple[str, list[str]]:
    """Parse 'param=v1,v2,v3' into (param, [v1, v2, v3])."""
    param, values = sweep_str.split("=")
    return param, values.split(",")


def run_sweep(game_name: str, algo: str, sweeps: list[str],
              n_games: int = 100, opponent: str = "default") -> pd.DataFrame:
    """Run a parameter sweep experiment."""
    game = _make_game(game_name)

    # Parse sweep params
    sweep_params = [_parse_sweep(s) for s in sweeps]
    param_names = [p[0] for p in sweep_params]
    param_values = [p[1] for p in sweep_params]

    results = []
    for combo in itertools.product(*param_values):
        config = dict(zip(param_names, combo))
        print(f"\n📊 Running: {config}")

        # Create agent with config
        agent = _create_agent(algo, game, config)

        # Create opponent
        opp = DefaultAgent(game)
        opp.set_game(game)

        # Run match
        result = run_match(game, agent, opp, n_games)
        result.update(config)
        results.append(result)

    return pd.DataFrame(results)


def _create_agent(algo: str, game, config_overrides: dict):
    """Create an agent with specific config overrides."""
    if algo in ("minimax", "minimax_ab"):
        cfg = MinimaxConfig()
        cfg.use_alpha_beta = algo == "minimax_ab"
        if "max_depth" in config_overrides:
            cfg.max_depth = int(config_overrides["max_depth"])
        agent = MinimaxAgent(game, cfg)
    elif algo == "qlearning":
        from .config import QLearningConfig
        cfg = QLearningConfig()
        for k, v in config_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, type(getattr(cfg, k))(v))
        agent = QLearningAgent(game, cfg)
    elif algo == "dqn":
        from .config import DQNConfig
        cfg = DQNConfig()
        for k, v in config_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, type(getattr(cfg, k))(v))
        agent = DQNAgent(game, cfg)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    agent.set_game(game)
    return agent


def main():
    parser = argparse.ArgumentParser(description="Run parameter sweep experiments")
    parser.add_argument("--game", choices=["tictactoe", "connect4"], default="tictactoe")
    parser.add_argument("--algo", required=True,
                        choices=["minimax", "minimax_ab", "qlearning", "dqn"])
    parser.add_argument("--sweep", nargs="+", required=True,
                        help="Params to sweep: param=v1,v2,v3")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    df = run_sweep(args.game, args.algo, args.sweep, args.games)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\n💾 Results saved to {args.output}")
    else:
        print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
