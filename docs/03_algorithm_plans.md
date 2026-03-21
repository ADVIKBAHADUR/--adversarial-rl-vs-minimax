# Algorithm Implementation Plans

All algorithms implement a common `Agent` interface:

```python
class Agent:
    def select_action(state, valid_actions) -> int   # Choose action
    def train(env, episodes, opponent) -> dict        # Train (RL only), return metrics
    def save(path) / load(path)                       # Persistence
    def name -> str                                   # For logging
```

---

## 1. Minimax (without Alpha-Beta Pruning)

### How it works
- Exhaustive recursive search of the entire game tree
- At MAX nodes (our turn): pick action maximising score
- At MIN nodes (opponent turn): pick action minimising score
- Terminal states: +1 (win), -1 (loss), 0 (draw)

### Configurable Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_depth` | `None` (unlimited) | Depth limit for Connect 4 |
| `eval_fn` | `None` | Evaluation function for non-terminal cutoffs |

### For Tic Tac Toe
- Full search, no depth limit needed
- Return exact move in ~ms

### For Connect 4
- First run **unlimited** to demonstrate infeasibility (count nodes visited in 30 min)
- Then switch to **depth-limited** (depth 5-7) with evaluation function:
  - Score based on: number of 2-in-a-row, 3-in-a-row, centre control
  - `eval(state) = w1 * our_streaks - w2 * opp_streaks + w3 * centre_control`

### Node Counting
- Track `nodes_visited` counter for reporting

---

## 2. Minimax with Alpha-Beta Pruning

### How it works
- Same as Minimax but maintains α (best for MAX) and β (best for MIN)
- Prune branches where `α ≥ β`
- Same result as Minimax, fewer nodes visited

### Configurable Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_depth` | `None` | Depth limit |
| `eval_fn` | `None` | Evaluation function |
| `move_ordering` | `True` | Order moves (centre-first) for better pruning |

### Move Ordering Heuristic
- Try centre columns first → more pruning
- For Connect 4 col order: `[3, 2, 4, 1, 5, 0, 6]`

### Metrics to Track
- `nodes_visited` — compare vs vanilla Minimax
- `pruned_branches` — how many cutoffs
- `time_per_move`

---

## 3. Q-Learning (Tabular)

### How it works
- State-action value table `Q[state_key][action] -> float`
- Update: `Q[s,a] += α * (r + γ * max(Q[s']) - Q[s,a])`
- ε-greedy exploration

### Configurable Parameters

| Parameter | Default TTT | Default C4 | Description |
|---|---|---|---|
| `learning_rate` (α) | 0.1 | 0.1 | Step size |
| `discount` (γ) | 0.95 | 0.95 | Future reward weight |
| `epsilon_start` | 1.0 | 1.0 | Initial exploration |
| `epsilon_end` | 0.01 | 0.05 | Final exploration |
| `epsilon_decay` | 0.9995 | 0.9999 | Decay per episode |
| `episodes` | 50,000 | 200,000 | Training episodes |

### State Representation
- Use `state_to_key()` (hash of board) as dict key
- For Connect 4: may need to limit via symmetry or board reduction

### Training Loop
```
for episode:
    state = env.reset()
    while not done:
        action = ε-greedy from Q[state]
        next_state, reward, done = env.step(action)
        Q[s,a] += α * (r + γ * max(Q[s']) - Q[s,a])
        state = next_state
    decay ε
```

### Metrics to Track
- Win rate over rolling window (e.g., last 1000 games)
- Q-table size over time
- Episode rewards
- Convergence curve

---

## 4. Deep Q-Network (DQN)

### How it works
- Neural network approximates Q(s, a) for all actions
- Experience replay buffer for stable training
- Target network updated periodically

### Network Architecture
- Input: flattened board state (9 for TTT, 42 for C4)
- Hidden: 2-3 layers, 128-256 neurons, ReLU
- Output: Q-value per action (9 for TTT, 7 for C4)
- Mask invalid actions with `-inf` before argmax

### Configurable Parameters

| Parameter | Default TTT | Default C4 | Description |
|---|---|---|---|
| `learning_rate` | 1e-3 | 1e-4 | Adam LR |
| `batch_size` | 64 | 128 | Replay batch |
| `buffer_size` | 10,000 | 50,000 | Replay capacity |
| `target_update` | 500 | 1,000 | Steps between target net sync |
| `gamma` | 0.99 | 0.99 | Discount factor |
| `epsilon_start` | 1.0 | 1.0 | Exploration start |
| `epsilon_end` | 0.01 | 0.05 | Exploration end |
| `epsilon_decay` | 0.9995 | 0.99995 | Decay rate |
| `hidden_sizes` | [128, 128] | [256, 256] | Layer sizes |
| `episodes` | 20,000 | 100,000 | Train episodes |

### Training Loop
```
for episode:
    state = env.reset()
    while not done:
        action = ε-greedy from Q_network(state)
        next_state, reward, done = env.step(action)
        buffer.push(s, a, r, s', done)
        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            loss = MSE(Q(s,a), r + γ * max(Q_target(s')))
            optimiser.step()
        if steps % target_update == 0:
            Q_target.load(Q.params)
    decay ε
```

### Metrics to Track
- Training loss curve
- Win rate rolling average
- Epsilon decay curve

---

## Evaluation Function for Depth-Limited Connect 4

```python
def evaluate(state, player):
    score = 0
    # Count streaks of 2, 3 in each direction
    for window in all_windows_of_4(state):
        mine = count(window, player)
        theirs = count(window, -player)
        empty = count(window, 0)
        if mine == 3 and empty == 1: score += 50
        if mine == 2 and empty == 2: score += 10
        if theirs == 3 and empty == 1: score -= 80  # defensive
    # Centre control bonus
    centre_col = state[:, cols//2]
    score += np.count_nonzero(centre_col == player) * 6
    return score
```
