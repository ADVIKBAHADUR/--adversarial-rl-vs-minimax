import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from adversarial.games import Connect4
from adversarial.agents import DQNAgent, RandomAgent, DefaultAgent
from adversarial.tournament import run_match
from adversarial.plotting import plot_training_curve
from adversarial.config import DQNConfig

# Allow loading DQNAgent/Config from checkpoints
torch.serialization.add_safe_globals([DQNConfig])

def get_episode_from_path(path):
    """Extract episode number from model_X.pt"""
    try:
        base = os.path.basename(path)
        if "best" in base: return -1
        return int(base.split("_")[1].split(".")[0])
    except:
        return -1

def evaluate_checkpoint(args):
    """Worker function to evaluate a single checkpoint."""
    model_path, seeds, game_name = args
    episode = get_episode_from_path(model_path)
    if episode == -1: return None
    
    # Initialise game and agent
    game = Connect4()
    
    # Load model
    try:
        # Initialise DQNAgent properly
        agent = DQNAgent(game, DQNConfig())
        agent.device = torch.device("cpu") # Force CPU for workers
        agent.set_game(game) # Initialise networks
        agent.load(model_path)
        agent.epsilon = 0.0 # Strict evaluation
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

    # Determine opponent phase (Curriculum = 10k)
    opponent_name = "Random" if episode < 10000 else "Default"
    opponent = RandomAgent(game) if opponent_name == "Random" else DefaultAgent(game)
    
    # Run games
    wins, draws, losses = 0, 0, 0
    from adversarial.tournament import play_game
    
    for i, seed in enumerate(seeds):
        # Set seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Switch sides every other game to ensure fairness
        if i % 2 == 0:
            res = play_game(game, agent, opponent)
            winner = res["winner"] # +1 is agent, -1 is opponent
        else:
            res = play_game(game, opponent, agent)
            winner = -res["winner"] # -1 becomes +1 if agent wins as P2
            
        if winner == 1: wins += 1
        elif winner == -1: losses += 1
        else: draws += 1
        
    total = len(seeds)
    return {
        "episode": episode,
        "win_pct": (wins / total) * 100,
        "draw_pct": (draws / total) * 100,
        "loss_pct": (losses / total) * 100,
        "opponent": opponent_name
    }

def main():
    model_dir = Path("models/connect4/dqn")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Discover and sort checkpoints
    model_paths = sorted(
        [str(p) for p in model_dir.glob("model_*.pt")],
        key=get_episode_from_path
    )
    # Remove duplicates or 'best' if we want just the timeline
    model_paths = [p for p in model_paths if get_episode_from_path(p) >= 0]
    
    print(f"🔍 Found {len(model_paths)} checkpoints.")
    
    # 2. Generate 100 fixed seeds
    np.random.seed(42)
    seeds = np.random.randint(0, 100000, size=100).tolist()
    
    # 3. Parallel Evaluation
    n_procs = max(1, cpu_count() - 1)
    print(f"🚀 Processing with {n_procs} workers (100 games per model)...")
    
    results = []
    task_args = [(p, seeds, "connect4") for p in model_paths]
    
    with Pool(n_procs) as pool:
        for res in tqdm(pool.imap(evaluate_checkpoint, task_args), total=len(model_paths)):
            if res:
                results.append(res)
    if not results:
        print("❌ No matching results found! Check your checkpoint filenames and paths.")
        return
        
    # 4. Save CSV
    df = pd.DataFrame(results)
    df = df.sort_values("episode")
    csv_path = results_dir / "training_log_dqn_connect4-6x7.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved rebuilt log to {csv_path}")
    
    # 5. Plot
    print("📈 Generating high-resolution curriculum plot...")
    plot_training_curve(
        episodes=df["episode"].tolist(),
        metrics=df["win_pct"].tolist(),
        opponent_history=df["opponent"].tolist()
    )
    import matplotlib.pyplot as plt
    plt.savefig("figures/training_curve_dqn_rebuilt.png")
    print("✅ Done! Plot saved to figures/training_curve_dqn_rebuilt.png")

if __name__ == "__main__":
    main()
