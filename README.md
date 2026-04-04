# Adversarial RL vs Minimax

**CS7IS2 Assignment 3** — Comparing Minimax and Reinforcement Learning algorithms for playing Tic Tac Toe and Connect 4.

## 📊 Project Status

| Component | Status | Notes |
|---|---|---|
| **Tic Tac Toe engine** | ✅ Done | Configurable n×n board, numpy-based |
| **Connect 4 engine** | ✅ Done | 6×7 with gravity, configurable |
| **Default opponent** | ✅ Done | Win/block/centre/random priorities |
| **Minimax (vanilla)** | ✅ Done | Interface ready, search not implemented |
| **Minimax (α-β pruning)** | ✅ Done | Interface ready, pruning not implemented |
| **Q-Learning (tabular)** | 🔲 Stub | Q-table structure ready, training loop not implemented |
| **DQN** | 🔲 Stub | Network + replay buffer defined, training not implemented |
| **Tournament runner** | ✅ Done | Batch matchups, CSV export, side-swapping |
| **Experiment sweeps** | ✅ Done | CLI parameter scanning |
| **Interactive play** | ✅ Done | Terminal-based human vs agent |
| **GUI** | ✅ Done | Pygame, big fonts, paper-ready |
| **Plotting** | ✅ Done | Paper-ready matplotlib (14pt+ fonts) |

## 🚀 Installation

```bash
# Clone the repo
git clone <repo-url>
cd adversarial-rl-vs-minimax

# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## 🎮 Usage

### Interactive Play (Terminal)

```bash
# Human vs Default opponent in Tic Tac Toe
adv-play --game tictactoe --p1 human --p2 default

# Human vs Default in Connect 4
adv-play --game connect4 --p1 human --p2 default

# Watch Default vs Default
adv-play --game tictactoe --p1 default --p2 default --rounds 5
```

### GUI (Pygame)

```bash
# Human vs Default in Tic Tac Toe
adv-gui --game tictactoe --p1 human --p2 default

# Human vs Default in Connect 4
adv-gui --game connect4 --p1 human --p2 default

# Watch agent vs agent
adv-gui --game connect4 --p1 default --p2 default
```

**GUI Controls:** `R` = restart, `Q` = quit, click cells to play.

### Training RL Agents

```bash
# Train Q-Learning on Tic Tac Toe
adv-train --game tictactoe --algo qlearning --episodes 50000

# Train DQN on Connect 4 against random opponent
adv-train --game connect4 --algo dqn --episodes 100000 --opponent random
```

### Tournament (Batch Evaluation)

```bash
# Run a tournament between multiple agents
adv-tournament --game tictactoe --agents default minimax qlearning --games 100 --output results/ttt_tournament.csv
```

### Experiment Sweeps

```bash
# Sweep Minimax depth limits for Connect 4
adv-experiments --game connect4 --algo minimax_ab --sweep max_depth=3,5,7,9 --games 100

# Sweep Q-Learning epsilon decay
adv-experiments --game tictactoe --algo qlearning --sweep epsilon_decay=0.999,0.9995,0.9999
```

## 📁 Project Structure

```
src/adversarial/
├── games/
│   ├── base.py          # Abstract Game interface
│   ├── tictactoe.py     # Tic Tac Toe engine
│   └── connect4.py      # Connect 4 engine
├── agents/
│   ├── base.py          # Abstract Agent interface
│   ├── default.py       # Rule-based default opponent
│   ├── minimax.py       # Minimax (both variants) [STUB]
│   ├── qlearning.py     # Tabular Q-learning [STUB]
│   ├── dqn.py           # Deep Q-Network [STUB]
│   └── human.py         # Human interactive agent
├── tournament.py        # Batch experiment runner
├── experiments.py       # Parameter sweep CLI
├── train.py             # Training CLI
├── play.py              # Interactive play CLI
├── plotting.py          # Paper-ready visualisation
├── gui.py               # Pygame GUI
└── config.py            # Default configurations
```

## 📚 Documentation

See the `docs/` folder for detailed plans:
- `01_project_overview.md` — High-level architecture & goals
- `02_game_specifications.md` — Game interfaces, parameters, win detection
- `03_algorithm_plans.md` — Detailed algorithm pseudocode & parameters
- `04_testing_and_evaluation.md` — Metrics, tournament design, scalability testing
- `05_project_structure.md` — Module layout & dependency graph

## 🔑 Key Design Decisions

1. **Game-agnostic agents** — all agents use the same `Game` interface
2. **NumPy-first** — board states are `np.int8` arrays for fast vectorised ops
3. **Config-driven** — all parameters are dataclasses, overridable via CLI
4. **Dual-mode** — automated batch experiments AND interactive play
5. **Paper-ready visuals** — large fonts, clean themes, PNG + SVG export
