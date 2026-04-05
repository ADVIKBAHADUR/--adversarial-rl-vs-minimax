# 🤖 Adversarial RL Project: Agent Handover & Context

This document provides all necessary context for an AI assistant to take over the **Adversarial RL vs. Minimax** project. The goal is to finalize the training and evaluation of DQN and Q-Learning agents for Tic-Tac-Toe and Connect 4.

## 🏗️ Project Architecture
- **Core Library**: `src/adversarial/` (Game engines, Agents, Training logic).
- **Agents**: DQNAgent (PyTorch), QLearningAgent (Tabular), MinimaxAgent (Alpha-Beta).
- **Reporting**: `scripts/` contains TeX generators for professional Win/Draw/Loss tables.
- **Evaluation**: `evaluate_all.py` runs the full round-robin tournament for the report.

## 🏆 Current State of Progress
1.  **Tic-Tac-Toe (Perfected)**:
    - DQN has reached a **0.0% loss rate** against the perfect Minimax ($d=9$) and Default engines.
    - Model: `models/tictactoe/dqn/optimized_200k_best.pt`
2.  **Connect 4 (In Progress)**:
    - We have a baseline model (`model_500k_tuned_v11_rescue_best.pt`) but were training a "Validity Proof" model to show current learning capacity.
    - Architecture: `[256, 256, 256, 256]` (Optimized for deep search).
3.  **Reporting Infrastructure**:
    - **15 LaTeX Tables** are automatically generated from tournament CSVs.
    - Current results are stored in `results/latex_tables/`.

## 🚀 How to Start Training Immediately (CUDA-ready)
The `train.py` script automatically detects CUDA/XPU. Use these high-velocity commands:

### 🎮 Connect 4 DQN (High Capacity)
Master the "Default" board engine with an optimized curriculum (Random -> Default):
```bash
env PYTHONPATH=src python3 -m adversarial.train \
    --game connect4 --algo dqn \
    --episodes 500000 \
    --curriculum-1 3000 \
    --curriculum-2 3000 \
    --gate-1 0.75 \
    --epsilon-decay 0.99985 \
    --output models/connect4/dqn/cuda_master
```

### 🎮 Tic-Tac-Toe DQN (Expert Hardening)
Refine the perfect draws ( Nash Equilibrium):
```bash
env PYTHONPATH=src python3 -m adversarial.train \
    --game tictactoe --algo dqn \
    --episodes 200000 \
    --curriculum-1 5000 \
    --curriculum-2 20000 \
    --gate-1 0.85 \
    --gate-2 0.8 \
    --output models/tictactoe/dqn/cuda_master
```

## 📊 Finalizing the Report
After any major training run, regenerate the official performance tables:
1.  **Run Tournament**: `python3 evaluate_all.py --games 200`
2.  **Generate Tables**: 
    - `python3 scripts/generate_latex_tables.py`
    - `python3 scripts/generate_comparison_tables.py`
    - `python3 scripts/generate_summary_tables.py`

## 📂 Key File Locations
- **Final Models**: `models/**/best.pt` (White-listed in `.gitignore`).
- **Game Engines**: `src/adversarial/games/`.
- **Training Logs**: `results/training_log_*.csv`.
- **Walkthrough**: `walkthrough.md` contains historical performance benchmarks.

---
> [!IMPORTANT]
> The current `.gitignore` allows only the five finalized "best" models. If you train a new model that you wish to keep, update the `.gitignore` negation rules accordingly.
