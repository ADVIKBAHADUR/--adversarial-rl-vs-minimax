"""Deep Q-Network (DQN) agent.

Implementation with parallel experience collection, XPU training, and RAM-efficient replay buffer.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import time
from functools import partial
from .base import Agent
from ..config import DQNConfig


class ReplayBuffer:
    """Fixed-size experience replay buffer using pre-allocated Numpy arrays for RAM efficiency."""

    def __init__(self, capacity: int, state_shape: tuple, action_size: int):
        self.capacity = capacity
        self.idx = 0
        self.full = False
        
        # Pre-allocate contiguous memory to avoid Python object overhead
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_valids = np.zeros((capacity, action_size), dtype=bool)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, next_valid, done):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.next_valids[self.idx] = next_valid
        self.dones[self.idx] = done
        
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int):
        high = self.capacity if self.full else self.idx
        indices = np.random.randint(0, high, size=batch_size)
            
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.next_valids[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.capacity if self.full else self.idx


class QNetwork(nn.Module):
    """Simple MLP Q-network."""

    def __init__(self, input_size: int, output_size: int, hidden_sizes: list[int]):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _select_action_logic(q_net, game, state, valid_actions, epsilon, training, device):
    """Encapsulated action selection logic for both main agent and workers."""
    actions = np.where(valid_actions)[0]

    if training and np.random.random() < epsilon:
        return int(np.random.choice(actions))

    with torch.no_grad():
        norm_state = game.normalize_state(state) if hasattr(game, "normalize_dqn_state") else state
        state_t = torch.FloatTensor(norm_state.ravel()).unsqueeze(0).to(device)
        q_values = q_net(state_t).cpu().numpy().ravel()

    # Mask invalid actions
    masked = np.full_like(q_values, -np.inf)
    masked[valid_actions] = q_values[valid_actions]
    return int(np.argmax(masked))


def _worker_play_episode(game, state_dict, opponent, epsilon, agent_side, config):
    """Standalone worker function to play one episode and return transitions."""
    # Prevent PyTorch from utilizing all CPU cores on background threads, causing massive contention
    torch.set_num_threads(1)
    
    input_size = int(np.prod(game.state_shape))
    output_size = game.action_space
    # Workers use CPU for inference to avoid multi-process contention on XPU/CUDA
    device = torch.device("cpu")
    
    net = QNetwork(input_size, output_size, config.hidden_sizes).to(device)
    net.load_state_dict(state_dict)
    net.eval()
    
    state = game.reset()
    opponent.reset()
    
    player = 1
    prev_state = None
    prev_action = None
    transitions = []
    
    while True:
        perspective = state * player
        norm_perspective = perspective
        valid = game.get_valid_actions(state)
        
        if player == agent_side:
            if prev_state is not None:
                transitions.append((prev_state, prev_action, 0.0, norm_perspective, valid, False))
            
            action = _select_action_logic(net, game, perspective, valid, epsilon, True, device)
            prev_state = norm_perspective.copy()
            prev_action = action
        else:
            opp_perspective = state * -agent_side
            action = opponent.select_action(opp_perspective, valid)
            
        state, done, winner = game.step(state, action, player)
        
        if done:
            if winner == agent_side: reward = config.win_reward
            elif winner == -agent_side: reward = config.loss_reward
            else: reward = config.draw_reward
                
            if prev_state is not None:
                next_perspective = state * agent_side
                next_valid = game.get_valid_actions(state)
                transitions.append((prev_state, prev_action, reward, next_perspective, next_valid, True))
            
            return transitions, winner
            
        player *= -1


class DQNAgent(Agent):
    """Deep Q-Network agent with parallel experience collection."""

    def __init__(self, game=None, config: DQNConfig | None = None):
        self.cfg = config or DQNConfig()
        self._game = game
        self.epsilon = self.cfg.epsilon_start
        self._training = False

        self.q_net: QNetwork | None = None
        self.target_net: QNetwork | None = None
        self.optimiser = None
        self.scheduler = None
        self._optim_stepped = False
        self.buffer: ReplayBuffer | None = None
        self.device = (
            torch.device("xpu")   if hasattr(torch, "xpu")  and torch.xpu.is_available()  else
            torch.device("cuda")  if torch.cuda.is_available() else
            torch.device("cpu")
        )

    @property
    def name(self) -> str:
        return "DQN"

    def set_game(self, game):
        self._game = game
        input_size = int(np.prod(game.state_shape))
        output_size = game.action_space
        self.q_net = QNetwork(input_size, output_size, self.cfg.hidden_sizes).to(self.device)
        self.target_net = QNetwork(input_size, output_size, self.cfg.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimiser = torch.optim.Adam(self.q_net.parameters(), lr=self.cfg.learning_rate)
        # Using ExponentialLR as per optimization request
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimiser, gamma=self.cfg.lr_decay)
        self._optim_stepped = False
        self.buffer = ReplayBuffer(self.cfg.buffer_size, game.state_shape, output_size)

    def select_action(self, state: np.ndarray, valid_actions: np.ndarray) -> int:
        return _select_action_logic(self.q_net, self._game, state, valid_actions, 
                                   self.epsilon, self._training, self.device)

    def train(self, game, opponent, episodes: int | None = None,
              callback=None, start_ep: int = 0, total_eps: int | None = None,
              num_workers: int = 1) -> dict:
        """High-performance training loop utilizing parallel CPU cores for simulation."""
        self._training = True
        episodes = episodes or self.cfg.episodes
        
        metrics = {"p1_wins": 0, "p2_wins": 0, "draws": 0, "loss": []}
        loss_fn = nn.MSELoss()
        
        # Determine chunk size: gather enough data to justify a training wave
        chunk_size = 1000 
        num_chunks = (episodes + chunk_size - 1) // chunk_size
        
        # Parallel setup
        ctx = mp.get_context('spawn') if self.device.type != 'cpu' else mp.get_context('fork')
        
        with ctx.Pool(processes=num_workers) as pool:
            for c in range(num_chunks):
                # ── Per-Block Performance Tracking ──
                if c % (1000 // chunk_size) == 0:
                    t_block_start = time.time()
                    metrics_block = {"sim": 0, "buf": 0, "train": 0}

                t_chunk_start = time.time()
                eps_in_chunk = min(chunk_size, episodes - c * chunk_size)
                
                # Snapshot main net for workers
                state_dict = {k: v.cpu() for k, v in self.q_net.state_dict().items()}
                
                # Prepare parallel tasks
                tasks = []
                for i in range(eps_in_chunk):
                    global_ep = start_ep + c * chunk_size + i
                    agent_side = 1 if global_ep % 2 == 0 else -1
                    tasks.append((game, state_dict, opponent, self.epsilon, agent_side, self.cfg))
                
                # ── Parallel Experience Collection ──
                t_sim_start = time.time()
                results = pool.starmap(_worker_play_episode, tasks)
                t_sim = time.time() - t_sim_start
                metrics_block["sim"] += t_sim
                
                # ── Fill Buffer & Track Stats ──
                t_buf_start = time.time()
                for transitions, winner in results:
                    if winner == 1: metrics["p1_wins"] += 1
                    elif winner == -1: metrics["p2_wins"] += 1
                    else: metrics["draws"] += 1
                    
                    for t in transitions:
                        self.buffer.push(*t)
                t_buf = time.time() - t_buf_start
                metrics_block["buf"] += t_buf
                
                # ── Training Steps (XPU) ──
                t_train_start = time.time()
                train_steps = eps_in_chunk * self.cfg.grad_steps_per_episode
                
                # Prevent catastrophic overfitting by waiting for buffer to fill somewhat
                warmup_size = min(10000, self.cfg.buffer_size // 4)
                if len(self.buffer) >= warmup_size:
                    for _ in range(train_steps):
                        s_batch, a_batch, r_batch, s_next_batch, next_valids_batch, d_batch = self.buffer.sample(self.cfg.batch_size)
                        
                        s_tensor = torch.FloatTensor(s_batch).to(self.device).view(self.cfg.batch_size, -1)
                        a_tensor = torch.LongTensor(a_batch).to(self.device).unsqueeze(1)
                        r_tensor = torch.FloatTensor(r_batch).to(self.device).unsqueeze(1)
                        s_next_tensor = torch.FloatTensor(s_next_batch).to(self.device).view(self.cfg.batch_size, -1)
                        d_tensor = torch.FloatTensor(d_batch).to(self.device).unsqueeze(1)
                        
                        current_q = self.q_net(s_tensor).gather(1, a_tensor)
                        
                        with torch.no_grad():
                            next_q_values = self.q_net(s_next_tensor)
                            next_valid_t = torch.BoolTensor(next_valids_batch).to(self.device)
                            next_q_values[~next_valid_t] = -float('inf')
                            
                            best_actions = next_q_values.argmax(1).unsqueeze(1)
                            max_next_q = self.target_net(s_next_tensor).gather(1, best_actions)
                            target_q = r_tensor + (self.cfg.gamma * max_next_q * (1 - d_tensor))
                        
                        loss = loss_fn(current_q, target_q)
                        self.optimiser.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
                        self.optimiser.step()
                        self._optim_stepped = True
                        metrics["loss"].append(loss.item())
                    
                    # Target network update (every N episodes)
                    global_chunk_start = start_ep + c * chunk_size
                    if global_chunk_start % max(100, self.cfg.target_update_freq) < chunk_size:
                        self.target_net.load_state_dict(self.q_net.state_dict())
                    
                    # Scheduler and Epsilon decay (applied per episode)
                    if self.scheduler:
                        for _ in range(eps_in_chunk):
                            self.scheduler.step()
                        for pg in self.optimiser.param_groups:
                            pg['lr'] = max(self.cfg.lr_min, pg['lr'])
                    
                    t_train = time.time() - t_train_start
                    metrics_block["train"] += t_train
                    
                    # ── Performance Reporting (Every 1000 episodes) ──
                    global_ep = start_ep + (c + 1) * chunk_size
                    if global_ep % 1000 == 0:
                        t_block_total = time.time() - t_block_start
                        gps = 1000 / t_block_total
                        print(f"\n⏱️ Profiling (Block {global_ep-1000}-{global_ep}):")
                        print(f"   - Simulation: {metrics_block['sim']:.1f}s ({(metrics_block['sim']/t_block_total)*100:.1f}%)")
                        print(f"   - Buffering:  {metrics_block['buf']:.1f}s ({(metrics_block['buf']/t_block_total)*100:.1f}%)")
                        print(f"   - XPU Training: {metrics_block['train']:.1f}s ({(metrics_block['train']/t_block_total)*100:.1f}%)")
                        print(f"   - Total Time: {t_block_total:.1f}s | Velocity: {gps:.1f} games/sec\n")
                    
                    for _ in range(eps_in_chunk):
                        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)
                
                if callback:
                    last_global = start_ep + (c + 1) * chunk_size
                    callback(last_global, metrics)
                    
        self._training = False
        return metrics

    def save(self, path: str):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "config": self.cfg,
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "config" in checkpoint:
            self.cfg = checkpoint["config"]
            if self._game is not None:
                self.set_game(self._game)
        
        self.q_net.load_state_dict(checkpoint["q_net"])
        if hasattr(self, "target_net") and self.target_net is not None:
            self.target_net.load_state_dict(checkpoint["q_net"])
        self.epsilon = checkpoint.get("epsilon", self.cfg.epsilon_end)
        self._training = False
