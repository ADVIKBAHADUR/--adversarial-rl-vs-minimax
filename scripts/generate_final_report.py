import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from adversarial.plotting import plot_training_curve

def main():
    csv_path = Path("results/training_log_dqn_connect4-6x7.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run rebuild_log.py first.")
        return

    # 1. Load data
    df = pd.read_csv(csv_path)
    df = df.sort_values("episode")
    
    # 2. Prepare metrics
    episodes = df["episode"].tolist()
    opponent_history = df["opponent"].tolist()
    epsilon = df.get("epsilon", [0.0]*len(df)).tolist()
    
    metrics = {
        "win_pct": df["win_pct"].tolist(),
        "draw_pct": df["draw_pct"].tolist(),
    }
    
    # Validation metrics extraction
    if "val_win_pct" in df.columns:
        val_df = df[df["val_win_pct"].notna()]
        metrics["val_wins"] = val_df["val_win_pct"].tolist()
        metrics["val_episodes"] = val_df["episode"].tolist()
    
    # 3. Generate high-res plot
    print("📈 Generating final report plot (High-Res, MiniMax Win Rate + Epsilon)...")
    plot_training_curve(
        episodes=episodes,
        metrics=metrics,
        opponent_history=opponent_history,
        smooth=False,
        epsilon_history=epsilon
    )
    
    # Save the figure
    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_path = fig_dir / "connect4_dqn_final_report.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Final report saved to: {save_path}")

if __name__ == "__main__":
    main()
