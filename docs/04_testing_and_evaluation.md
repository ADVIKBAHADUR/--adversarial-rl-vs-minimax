# Testing & Evaluation Framework

## Design Philosophy

The framework must support **two modes**:

1. **Automated batch mode** — run thousands of games, collect stats, produce plots
2. **Interactive mode** — play manually against agents, observe behaviour

Both modes share the same game engines and agents; only the "driver" differs.

---

## Tournament Runner (Automated)

### Architecture
```
TournamentRunner
├── config: YAML/dict with matchups, game counts, seeds
├── run() → results DataFrame
├── agents: dict of name → Agent
└── games: dict of name → Game
```

### Matchup Matrix

For each game (TTT, Connect 4), run all pairwise matchups:

| Player 1 | Player 2 | Games |
|---|---|---|
| Minimax | Default | 100+ |
| Minimax+αβ | Default | 100+ |
| Q-Learning | Default | 1,000+ |
| DQN | Default | 1,000+ |
| Minimax | Q-Learning | 100+ |
| Minimax | DQN | 100+ |
| Minimax+αβ | Q-Learning | 100+ |
| Minimax+αβ | DQN | 100+ |
| Q-Learning | DQN | 1,000+ |

Note: For deterministic agents (Minimax), fewer games needed. For stochastic agents (RL), more games for statistical significance.

### Running both sides
Each matchup runs with Player 1 going first AND second (to account for first-move advantage).

### Output
- CSV/JSON per experiment run
- Columns: `game, p1, p2, p1_wins, p2_wins, draws, avg_moves, p1_time_avg, p2_time_avg`

---

## Metrics

### Per-Algorithm Metrics

| Metric | Minimax | RL |
|---|---|---|
| Win rate vs opponent | ✓ | ✓ |
| Avg. game length (moves) | ✓ | ✓ |
| Avg. time per move | ✓ | ✓ |
| Nodes visited per move | ✓ | — |
| Pruned nodes (α-β only) | ✓ | — |
| Training episodes | — | ✓ |
| Training time | — | ✓ |
| Q-table size / Model params | — | ✓ |
| Convergence curve | — | ✓ |

### Aggregate Metrics
- Win/loss/draw percentage
- 95% confidence intervals (for stochastic results)
- Elo-style relative ratings (optional)

---

## Visualisation / Plotting

### Required Plots
1. **Win rate bar chart** — all algorithms vs default opponent
2. **Head-to-head matrix** — heatmap of algo vs algo win rates
3. **Training curve** — RL win rate over episodes
4. **Node count comparison** — Minimax vs α-β
5. **Time per move** — bar chart across algorithms
6. **Scalability** — Connect 4 minimax nodes vs depth limit

### Implementation
- Use `matplotlib` with a consistent style theme
- Large fonts (14pt+ labels, 12pt+ ticks) for paper readability
- Export as both PNG (for paper) and SVG (for scaling)
- Plotting utility: `src/adversarial/plotting.py`

---

## Scalability Testing

### Connect 4 Minimax Infeasibility Proof
1. Run full Minimax for 30 minutes on standard 6×7 board
2. Record: nodes visited, depth reached, % of game tree explored
3. Run with α-β pruning for comparison
4. Report both in the document

### Depth-Limited Experiments
- Run depth 3, 5, 7, 9 and measure:
  - Win rate vs default
  - Time per move
  - Quality of play
- Find the sweet spot of depth vs time

### RL Training Scalability
- Plot win rate vs training episodes at checkpoints
- Compare Q-table size growth for tabular
- Compare training time for DQN vs tabular

### Automated Experiment Sweeps
```bash
# Example: sweep depth limits for Connect 4 Minimax
python -m adversarial.experiments --game connect4 --algo minimax_ab \
    --sweep max_depth=3,5,7,9 --games 100 --output results/depth_sweep.csv

# Example: sweep epsilon decay for Q-Learning
python -m adversarial.experiments --game tictactoe --algo qlearning \
    --sweep epsilon_decay=0.999,0.9995,0.9999 --episodes 50000
```

---

## Interactive Mode

### CLI Interface
```bash
# Play against trained Q-learning agent
python -m adversarial.play --game tictactoe --opponent qlearning --model models/q_ttt.pkl

# Watch two agents play
python -m adversarial.play --game connect4 --p1 minimax_ab --p2 dqn --model2 models/dqn_c4.pt
```

### GUI (Pygame/Tkinter)
- Simple board display with large, clear pieces
- Current player indicator
- Move history sidebar
- Win/draw announcement
- Font sizes: 24pt+ for board labels, 18pt+ for info text
