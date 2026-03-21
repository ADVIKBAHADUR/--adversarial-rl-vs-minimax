# Game Specifications

## Game Interface Contract

Both games implement the same abstract interface so all algorithms are game-agnostic:

```python
class Game:
    def reset() -> State                     # Return initial state
    def get_valid_actions(state) -> np.array  # 1D mask of legal moves
    def step(state, action) -> (State, done, winner)
    def render(state) -> str                 # Terminal-printable board
    def state_to_key(state) -> int           # Hash for Q-table lookup
    def clone_state(state) -> State
```

State is always a **NumPy int8 array** (0 = empty, 1 = player 1, -1 = player 2).

---

## Tic Tac Toe

### Board Representation
- **Shape**: `(3, 3)` int8 array
- **Encoding**: `0` = empty, `1` = X, `-1` = O

### Configurable Parameters

| Parameter | Default | Description |
|---|---|---|
| `board_size` | 3 | Board dimension (3 = standard, configurable for experiments) |
| `win_length` | 3 | Consecutive marks to win (always = board_size for standard) |

### Actions
- Integer `0..8` mapping to `(row, col)` via `divmod(action, board_size)`
- Valid actions = positions where `board[pos] == 0`

### Win Detection
- Check rows, columns, and diagonals using `np.sum` along axes
- A player wins if any line sums to `±board_size`

### Complexity
- Max 9 moves, ~5,478 unique valid states
- Full Minimax is trivially feasible

---

## Connect 4

### Board Representation
- **Shape**: `(rows, cols)` int8 array, default `(6, 7)`
- **Encoding**: `0` = empty, `1` = player 1, `-1` = player 2
- Pieces drop to the lowest available row in a column (gravity)

### Configurable Parameters

| Parameter | Default | Description |
|---|---|---|
| `rows` | 6 | Board height |
| `cols` | 7 | Board width |
| `win_length` | 4 | Consecutive pieces to win |

### Actions
- Integer `0..cols-1` = column to drop piece into
- Valid actions = columns where `board[0, col] == 0` (top row not full)

### Win Detection
- Use **convolution-based** approach: create 4 kernels (horizontal, vertical, 2 diagonals) and check for sum of ±4
- Alternative: sliding window with `np.lib.stride_tricks`

### Gravity
- For column `c`, find lowest row `r` where `board[r, c] == 0`
- Efficiently: `r = np.max(np.where(board[:, c] == 0))`

### Complexity
- Max 42 moves, ~4.5 × 10¹² states
- Full Minimax **infeasible** → depth-limited + evaluation function
- Reduced board sizes (e.g., 5×4) useful for testing

---

## Default Opponent

A rule-based agent, better than random:

### Priority Rules (in order)
1. **Win**: If a winning move exists, play it
2. **Block**: If opponent has a winning move, block it
3. **Centre preference**: Prefer centre column (Connect 4) or centre cell (TTT)
4. **Random**: Otherwise pick random valid action

### Implementation
- Scan all valid actions, simulate each, check for win/block
- O(valid_actions × win_check) per move — very fast
