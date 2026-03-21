# Project Overview: Adversarial RL vs Minimax

## Objective

Compare **Minimax** (with and without alpha-beta pruning) against **Reinforcement Learning** (tabular Q-learning and Deep DQN) across two adversarial games:

1. **Tic Tac Toe** (3×3, ~5,478 valid states)
2. **Connect 4** (6×7, ~4.5 trillion states)

## Assignment Deliverables

| Deliverable | Description |
|---|---|
| Game engines | Tic Tac Toe + Connect 4 |
| Algorithms | Minimax, Minimax + α-β, Q-Learning, DQN — for **each** game |
| Default opponent | Semi-intelligent (blocks wins, takes wins) |
| Comparisons | Algo vs default opponent, algo vs algo — for both games |
| Demo video | ≤ 5 min screen-grab |
| Report | Performance analysis with graphs |

## Key Scalability Constraints

- **Connect 4 Minimax**: Full search infeasible → need **depth-limited** search + evaluation function
- **Connect 4 RL**: Needs many training episodes → use random opponent for training
- Must run **batch experiments** (hundreds/thousands of games) for statistical significance
- Must also support **interactive play** for demo

## Architecture Principles

1. **Lightweight & structured** — no bloat, clear module boundaries
2. **Vectorised where possible** — NumPy arrays for board states, batch operations
3. **Fast algorithms** — bitboard representations, efficient alpha-beta, vectorised Q-table lookups
4. **Dual-mode** — automated batch experiments AND interactive play
5. **Paper-ready GUI** — big fonts, clean visuals, screenshot-friendly
