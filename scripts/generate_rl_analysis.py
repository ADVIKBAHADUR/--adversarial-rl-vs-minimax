import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src to sys.path to import adversarial modules
sys.path.append(os.getcwd() + "/src")
from adversarial.plotting import plot_training_curve, apply_style, save_fig

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

def generate_ttt_comparison():
    """Compare DQN and Q-Learning on Tic-Tac-Toe."""
    dqn_path = f"{RESULTS_DIR}/training_log_dqn_tictactoe-3x3.csv"
    ql_path = f"{RESULTS_DIR}/training_log_q-learning_tictactoe-3x3.csv"
    
    if not os.path.exists(dqn_path) or not os.path.exists(ql_path):
        print("Missing TTT logs for comparison.")
        return

    dqn_df = pd.read_csv(dqn_path)
    ql_df = pd.read_csv(ql_path)

    # Plot DQN
    plot_training_curve(
        episodes=dqn_df['episode'].tolist(),
        metrics={
            "win_pct": dqn_df['win_pct'].tolist(),
            "draw_pct": dqn_df['draw_pct'].tolist(),
            "loss_pct": dqn_df['loss_pct'].tolist()
        },
        opponent_history=dqn_df['opponent'].tolist(),
        epsilon_history=dqn_df['epsilon'].tolist(),
        algo_name="DQN (Tic-Tac-Toe)",
        title="DQN Learning Dynamics: Tic-Tac-Toe Expert Curriculum",
        filename="analysis_dqn_ttt_full"
    )

    # Plot Q-Learning
    plot_training_curve(
        episodes=ql_df['episode'].tolist(),
        metrics={
            "win_pct": ql_df['win_pct'].tolist(),
            "draw_pct": ql_df['draw_pct'].tolist(),
            "loss_pct": ql_df['loss_pct'].tolist()
        },
        opponent_history=ql_df['opponent'].tolist(),
        epsilon_history=ql_df['epsilon'].tolist(),
        vtable_history=ql_df['vtable_size'].tolist(),
        algo_name="Q-Learning (Tic-Tac-Toe)",
        title="Tabular Q-Learning: State Space Exploration vs. Performance",
        filename="analysis_ql_ttt_full"
    )

def generate_c4_dqn_analysis():
    """Deep dive into Connect 4 DQN performance."""
    c4_path = f"{RESULTS_DIR}/training_log_dqn_connect4-6x7.csv"
    if not os.path.exists(c4_path):
        print("Missing Connect 4 logs.")
        return

    df = pd.read_csv(c4_path)
    
    # Connect 4 often has very noisy training early on, let's plot with smoothing
    plot_training_curve(
        episodes=df['episode'].tolist(),
        metrics={
            "win_pct": df['win_pct'].tolist(),
            "draw_pct": df['draw_pct'].tolist(),
            "loss_pct": df['loss_pct'].tolist()
        },
        opponent_history=df['opponent'].tolist(),
        epsilon_history=df['epsilon'].tolist(),
        algo_name="DQN (Connect 4)",
        title="DQN Convergence: Mastering Connect 4 Board Engines",
        filename="analysis_dqn_c4_full",
        window=10
    )

def generate_efficiency_comparison():
    """Plot sample efficiency metrics across agents."""
    # We will use the 'agent_summary.csv' for this
    summary_path = f"{RESULTS_DIR}/agent_summary.csv"
    if not os.path.exists(summary_path):
        return
        
    df = pd.read_csv(summary_path)
    # Filter for RL agents
    rl_df = df[df['agent'].isin(['DQN', 'Q-Learning'])].copy()
    
    import matplotlib.pyplot as plt
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # We will plot episodes_trained vs win rate as a scatter plot
    # But for a better "Efficiency" view, we use the LaTeX logic: first ep where win >= 90
    # Since we don't have that in agent_summary, we'll manually extract peaks
    
    agents = []
    peaks = []
    
    for _, row in rl_df.iterrows():
        algo = row['agent']
        game = row['game']
        slug = f"{algo} ({game})"
        agents.append(slug)
        peaks.append(row['avg_win_pct'])
        
    ax.barh(agents, peaks, color=['#3498db', '#2ecc71', '#9b59b6', '#e67e22'])
    ax.set_xlabel("Peak Win Rate (%)")
    ax.set_title("Peak Performance Across RL Architectures")
    ax.set_xlim(0, 100)
    
    save_fig(fig, "rl_peak_comparison")

if __name__ == "__main__":
    print("🚀 Generating Advanced RL Analysis Plots...")
    generate_ttt_comparison()
    generate_c4_dqn_analysis()
    generate_efficiency_comparison()
    print("✅ All plots saved to 'figures/' directory.")
