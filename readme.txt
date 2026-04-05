# Adversarial RL vs Minimax - Execution Guide

This document provides the command lines required to run and verify the 6 algorithms implemented for this assignment (Tic-Tac-Toe and Connect 4).

## Prerequisites
Ensure the package is installed in editable mode:
```bash
pip install -e .
```

---

## 1. Tic-Tac-Toe (3x3)

### Appendix 1: Minimax (Vanilla)
Runs the exhaustive Minimax search (depth-unlimited).
```bash
adv-play --game tictactoe --p1 minimax --p2 human
```

### Appendix 2: Minimax (Alpha-Beta Pruning)
Runs Minimax with Alpha-Beta pruning enabled.
```bash
adv-play --game tictactoe --p1 minimax --ab --p2 human
```

### Appendix 3: RL (Tabular Q-Learning)
Runs the trained tabular Q-learning agent.
```bash
adv-play --game tictactoe --p1 qlearning --model1 models/tictactoe/qlearning/model_best.pkl --p2 human
```

---

## 2. Connect 4 (6x7)

### Appendix 4: Minimax (Vanilla, Depth-Limited)
Runs depth-limited Minimax search (Depth 3).
```bash
adv-play --game connect4 --p1 minimax --depth 3 --p2 human
```

### Appendix 5: Minimax (Alpha-Beta + Heuristic)
Runs depth-limited Alpha-Beta search (Depth 5) with heuristic evaluation.
```bash
adv-play --game connect4 --p1 minimax --ab --depth 5 --p2 human
```

### Appendix 6: RL (Deep Q-Network)
Runs the trained DQN agent.
```bash
adv-play --game connect4 --p1 dqn --model1 models/connect4/dqn/model_validity_best.pt --p2 human
```

---

## Batch Evaluation & Training

### Full Tournament
To reproduce all performance metrics and heatmaps used in the report:
```bash
python3 evaluate_all.py --games 100
```

### Training Commands
Should you wish to retrain the agents from scratch:

# Tic-Tac-Toe Q-Learning
adv-train --game tictactoe --algo qlearning --episodes 50000

# Connect 4 DQN
adv-train --game connect4 --algo dqn --episodes 200000 --opponent default
