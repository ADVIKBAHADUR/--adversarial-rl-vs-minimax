# Project Structure

```
adversarial-rl-vs-minimax/
├── pyproject.toml                 # pip-installable project config
├── README.md                      # Status, usage, experiments guide
├── docs/
│   ├── 01_project_overview.md
│   ├── 02_game_specifications.md
│   ├── 03_algorithm_plans.md
│   ├── 04_testing_and_evaluation.md
│   └── 05_project_structure.md
│
├── src/
│   └── adversarial/
│       ├── __init__.py
│       ├── games/
│       │   ├── __init__.py
│       │   ├── base.py            # Abstract Game interface
│       │   ├── tictactoe.py       # TTT implementation
│       │   └── connect4.py        # Connect 4 implementation
│       │
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── base.py            # Abstract Agent interface
│       │   ├── default.py         # Rule-based default opponent
│       │   ├── minimax.py         # Minimax (both variants)
│       │   ├── qlearning.py       # Tabular Q-learning
│       │   └── dqn.py             # Deep Q-Network
│       │
│       ├── tournament.py          # Batch experiment runner
│       ├── experiments.py         # CLI for sweeps/experiments
│       ├── play.py                # Interactive play CLI
│       ├── plotting.py            # Paper-ready visualisation
│       ├── gui.py                 # Simple GUI for demo
│       └── config.py              # Default configs & param schemas
│
├── models/                        # Saved trained models
├── results/                       # Experiment CSV/JSON outputs
└── figures/                       # Generated plots
```

## Module Dependency Graph

```
config.py (no deps)
    ↓
games/base.py (no deps)
    ↓
games/tictactoe.py, games/connect4.py (→ base, numpy)
    ↓
agents/base.py (→ games/base)
    ↓
agents/default.py, agents/minimax.py, agents/qlearning.py, agents/dqn.py
    ↓
tournament.py (→ agents, games)
    ↓
experiments.py, play.py (→ tournament, agents, games)
    ↓
plotting.py (→ results data)
gui.py (→ games, agents)
```

## Key Design Decisions

1. **Game-agnostic agents**: All agents use the same `Game` interface, so they work with both TTT and C4 without modification
2. **NumPy-first**: Board states are always `np.ndarray`, enabling vectorised operations
3. **Config-driven**: All parameters come from `config.py` defaults, overridable via CLI args
4. **Separation of concerns**: Games know nothing about agents; agents know nothing about GUI
