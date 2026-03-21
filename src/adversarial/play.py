"""Interactive play CLI — human vs agent or agent vs agent with visualisation."""

import argparse
import numpy as np
from .games import TicTacToe, Connect4
from .agents import DefaultAgent, MinimaxAgent, QLearningAgent, DQNAgent, HumanAgent


def _make_game(name: str):
    games = {"tictactoe": TicTacToe, "connect4": Connect4}
    return games[name]()


def _make_agent(name: str, game, model_path: str | None = None):
    if name == "human":
        return HumanAgent()

    agents = {
        "default": lambda: DefaultAgent(game),
        "minimax": lambda: MinimaxAgent(game),
        "qlearning": lambda: QLearningAgent(game),
        "dqn": lambda: DQNAgent(game),
    }
    agent = agents[name]()
    agent.set_game(game)
    if model_path:
        agent.load(model_path)
    return agent


def play_interactive(game, agent1, agent2):
    """Play a single interactive game with terminal rendering."""
    state = game.reset()
    agents = {1: agent1, -1: agent2}
    player = 1

    print(f"\n{'='*40}")
    print(f"  {game.name}: {agent1.name} (X) vs {agent2.name} (O)")
    print(f"{'='*40}\n")
    print(game.render(state))
    print()

    while True:
        agent = agents[player]
        symbol = "X" if player == 1 else "O"
        print(f"--- {agent.name}'s turn ({symbol}) ---")

        perspective = state * player
        valid = game.get_valid_actions(state)
        action = agent.select_action(perspective, valid)

        state, done, winner = game.step(state, action, player)
        print(game.render(state))
        print()

        if done:
            if winner == 0:
                print("🤝 It's a draw!")
            else:
                w_name = agents[winner].name
                print(f"🏆 {w_name} wins!")
            break

        player *= -1

    return winner


def main():
    parser = argparse.ArgumentParser(description="Interactive game play")
    parser.add_argument("--game", choices=["tictactoe", "connect4"], default="tictactoe")
    parser.add_argument("--p1", default="human", help="Player 1 agent")
    parser.add_argument("--p2", default="default", help="Player 2 agent")
    parser.add_argument("--model1", default=None, help="Model path for P1")
    parser.add_argument("--model2", default=None, help="Model path for P2")
    parser.add_argument("--rounds", type=int, default=1, help="Number of games")
    args = parser.parse_args()

    game = _make_game(args.game)
    agent1 = _make_agent(args.p1, game, args.model1)
    agent2 = _make_agent(args.p2, game, args.model2)

    scores = {1: 0, -1: 0, 0: 0}
    for i in range(args.rounds):
        if args.rounds > 1:
            print(f"\n--- Game {i+1}/{args.rounds} ---")
        winner = play_interactive(game, agent1, agent2)
        scores[winner] += 1

    if args.rounds > 1:
        print(f"\n{'='*40}")
        print(f"  Final Score ({args.rounds} games)")
        print(f"  {agent1.name}: {scores[1]}  |  {agent2.name}: {scores[-1]}  |  Draws: {scores[0]}")
        print(f"{'='*40}")


if __name__ == "__main__":
    main()
