"""Tournament runner — batch game evaluation between agents."""

import time
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from .games import TicTacToe, Connect4
from .agents import DefaultAgent, MinimaxAgent, QLearningAgent, DQNAgent


def play_game(game, agent1, agent2, verbose: bool = False) -> dict:
    """Play a single game between two agents.

    Agents always see the board from their perspective (+1 = self).
    """
    state = game.reset()
    agents = {1: agent1, -1: agent2}
    player = 1
    moves = 0
    times = {1: [], -1: []}

    agent1.reset()
    agent2.reset()

    while True:
        agent = agents[player]
        # Flip board so agent always sees itself as +1
        perspective = state * player
        valid = game.get_valid_actions(state)

        t0 = time.perf_counter()
        action = agent.select_action(perspective, valid)
        times[player].append(time.perf_counter() - t0)

        state, done, winner = game.step(state, action, player)
        moves += 1

        if verbose:
            print(game.render(state))
            print()

        if done:
            break
        player *= -1

    return {
        "winner": int(winner),
        "moves": moves,
        "p1_avg_time": np.mean(times[1]) if times[1] else 0,
        "p2_avg_time": np.mean(times[-1]) if times[-1] else 0,
    }


def run_match(game, agent1, agent2, n_games: int = 100,
              swap_sides: bool = True, verbose: bool = False,
              seed: int | None = None) -> dict:
    """Run a full match (multiple games) between two agents.

    If swap_sides=True, half the games have agent1 as P1 and half as P2.
    """
    if seed is not None:
        np.random.seed(seed)

    results = {"p1_wins": 0, "p2_wins": 0, "draws": 0,
               "total_moves": 0, "p1_times": [], "p2_times": []}
    games_per_side = n_games // 2 if swap_sides else n_games
    game_count = 0
    total_games = games_per_side * (2 if swap_sides else 1)

    for swapped in ([False, True] if swap_sides else [False]):
        a1, a2 = (agent2, agent1) if swapped else (agent1, agent2)
        for _ in tqdm(range(games_per_side), desc=f"{a1.name} vs {a2.name}",
                      leave=False, disable=not verbose):
            result = play_game(game, a1, a2)
            game_count += 1
            print(f"\r    game {game_count}/{total_games}", end="", flush=True)
            w = result["winner"]
            if swapped:
                w = -w  # Flip perspective back
            if w == 1:
                results["p1_wins"] += 1
            elif w == -1:
                results["p2_wins"] += 1
            else:
                results["draws"] += 1
            results["total_moves"] += result["moves"]
            if swapped:
                results["p1_times"].append(result["p2_avg_time"])
                results["p2_times"].append(result["p1_avg_time"])
            else:
                results["p1_times"].append(result["p1_avg_time"])
                results["p2_times"].append(result["p2_avg_time"])

    total = results["p1_wins"] + results["p2_wins"] + results["draws"]
    return {
        "game": game.name,
        "p1": agent1.name,
        "p2": agent2.name,
        "n_games": total,
        "p1_wins": results["p1_wins"],
        "p2_wins": results["p2_wins"],
        "draws": results["draws"],
        "p1_win_pct": results["p1_wins"] / total * 100,
        "p2_win_pct": results["p2_wins"] / total * 100,
        "draw_pct": results["draws"] / total * 100,
        "avg_moves": results["total_moves"] / total,
        "p1_avg_time": np.mean(results["p1_times"]),
        "p2_avg_time": np.mean(results["p2_times"]),
    }


def run_tournament(game, agents: list, n_games: int = 100,
                   swap_sides: bool = True) -> pd.DataFrame:
    """Run all pairwise matchups between agents, return results DataFrame."""
    rows = []
    for i, a1 in enumerate(agents):
        for j, a2 in enumerate(agents):
            if i >= j:
                continue
            result = run_match(game, a1, a2, n_games, swap_sides)
            rows.append(result)
            print(f"  {result['p1']} vs {result['p2']}: "
                  f"{result['p1_win_pct']:.1f}% / {result['p2_win_pct']:.1f}% / "
                  f"{result['draw_pct']:.1f}% draws")
    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, path: str):
    """Save results to CSV."""
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _make_game(name: str):
    if name == "tictactoe":
        return TicTacToe()
    elif name == "connect4":
        return Connect4()
    raise ValueError(f"Unknown game: {name}")


def _make_agent(name: str, game, depth: int | None = None, model_path: str | None = None):
    from .config import MinimaxConfig
    agents_map = {
        "default": lambda: DefaultAgent(game),
        "random": lambda: RandomAgent(game),
        "minimax": lambda: MinimaxAgent(game),
        "minimax_ab": lambda: MinimaxAgent(game, MinimaxConfig(use_alpha_beta=True)),
        "qlearning": lambda: QLearningAgent(game),
        "dqn": lambda: DQNAgent(game),
    }
    if name not in agents_map:
        raise ValueError(f"Unknown agent: {name}. Available: {list(agents_map.keys())}")
    agent = agents_map[name]()
    agent.set_game(game)

    # Apply depth if supported
    if depth is not None and isinstance(agent, MinimaxAgent):
        agent.cfg.max_depth = depth

    if model_path is not None and hasattr(agent, "load"):
        agent.load(model_path)
        print(f"Loaded {name} model from {model_path}")

    return agent


def main_cli():
    parser = argparse.ArgumentParser(description="Run a tournament between agents")
    parser.add_argument("--game", choices=["tictactoe", "connect4"], default="tictactoe")
    parser.add_argument("--agents", nargs="+", default=["default"],
                        help="Agents to include in tournament")
    parser.add_argument("--models", nargs="*", default=[],
                        help="Parallel list of model paths (use 'none' for agents without models)")
    parser.add_argument("--games", type=int, default=100, help="Games per matchup")
    parser.add_argument("--depth", type=int, default=None, help="Max depth for Minimax agents")
    parser.add_argument("--output", type=str, default=None, help="CSV output path")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    game = _make_game(args.game)
    
    models = args.models + ["none"] * max(0, len(args.agents) - len(args.models))
    agents = []
    for name, m_path in zip(args.agents, models):
        path = m_path if m_path.lower() != "none" else None
        agents.append(_make_agent(name, game, args.depth, path))

    if len(agents) < 2:
        print("Need at least 2 agents for a tournament.")
        return

    print(f"\n🏆 Tournament: {game.name}")
    print(f"   Agents: {[a.name for a in agents]}")
    print(f"   Games per matchup: {args.games}\n")

    df = run_tournament(game, agents, args.games)

    if args.output:
        save_results(df, args.output)
    else:
        print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main_cli()
