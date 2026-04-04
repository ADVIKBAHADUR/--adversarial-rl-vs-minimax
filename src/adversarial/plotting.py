"""Paper-ready plotting utilities — large fonts, clean themes, export-friendly."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path

# ── Global style for paper-readiness ──────────────────────────────────────────

STYLE = {
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 20,
    "lines.linewidth": 2.5,
    "axes.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "figure.figsize": (10, 7),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def apply_style():
    """Apply paper-ready matplotlib style."""
    matplotlib.rcParams.update(STYLE)


def save_fig(fig, name: str, output_dir: str = "figures"):
    """Save figure as PNG and SVG."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/{name}.png")
    fig.savefig(f"{output_dir}/{name}.svg")
    print(f"📊 Saved {output_dir}/{name}.png and .svg")


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_win_rates(df: pd.DataFrame, title: str = "Win Rates vs Default Opponent",
                   output_dir: str = "figures"):
    """Bar chart of win/draw/loss rates for each algorithm vs default."""
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    algos = df["p1"].unique()
    x = np.arange(len(algos))
    width = 0.25

    ax.bar(x - width, df["p1_win_pct"], width, label="Win %", color="#2ecc71")
    ax.bar(x, df["draw_pct"], width, label="Draw %", color="#f39c12")
    ax.bar(x + width, df["p2_win_pct"], width, label="Loss %", color="#e74c3c")

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=15)
    ax.legend()
    ax.set_ylim(0, 100)

    fig.tight_layout()
    save_fig(fig, "win_rates", output_dir)
    return fig


def plot_head_to_head(df: pd.DataFrame, title: str = "Head-to-Head Results",
                      output_dir: str = "figures"):
    """Heatmap of pairwise win rates."""
    apply_style()
    agents = sorted(set(df["p1"].tolist() + df["p2"].tolist()))
    n = len(agents)
    matrix = np.full((n, n), np.nan)

    for _, row in df.iterrows():
        i = agents.index(row["p1"])
        j = agents.index(row["p2"])
        matrix[i, j] = row["p1_win_pct"]
        matrix[j, i] = row["p2_win_pct"]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(agents, rotation=45, ha="right")
    ax.set_yticklabels(agents)
    ax.set_title(title)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.0f}%", ha="center", va="center",
                        fontsize=14, fontweight="bold")

    fig.colorbar(im, label="Win %")
    fig.tight_layout()
    save_fig(fig, "head_to_head", output_dir)
    return fig


def plot_training_curve(episodes: list, metrics: dict | list, 
                        opponent_history: list | None = None,
                        algo_name: str = "DQN", output_dir: str = "figures",
                        window: int = 20, smooth: bool = True,
                        epsilon_history: list | None = None,
                        vtable_history: list | None = None,
                        title: str | None = None,
                        filename: str | None = None):
    """Training convergence curve with optional V-table tracking and smoothing."""
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    if isinstance(metrics, list):
        metrics = {"Non-Loss Rate": metrics}

    # Enhanced colour palette for final report
    colours = {
        "wins": "#2ecc71",      # Green
        "draws": "#f39c12",     # Orange
        "losses": "#e74c3c",    # Red
        "win_pct": "#2ecc71",   # Consistency for CSV keys
        "draw_pct": "#f39c12",
        "loss_pct": "#e74c3c",
        "Non-Loss Rate": "#000000", # Black
        "non_loss_pct": "#000000"
    }
    labels = {
        "wins": "Wins %", "draws": "Draws %", "losses": "Losses %",
        "win_pct": "Wins %", "draw_pct": "Draws %", "loss_pct": "Losses %",
        "Non-Loss Rate": "Non-Loss Rate (Win + Draw)",
        "non_loss_pct": "Non-Loss Rate (Win + Draw)"
    }

    # ── Background shading for Curriculum phases ──────────────────────────────
    if opponent_history and len(opponent_history) == len(episodes):
        # Accessible phase colors (Light, distinct hues)
        phase_colors = {
            "Random": "#e0f7fa",  # Very light Cyan
            "Default": "#fff8e1", # Very light Amber
            "Minimax": "#fce4ec"  # Very light Pink (as requested)
        }
        
        # Text colors for better readability against shading
        phase_text_colors = {
            "Random": "#006064",
            "Default": "#ff8f00",
            "Minimax": "#880e4f"
        }
        
        current_opp = opponent_history[0]
        start_idx = 0
        
        # Add labels and shading
        for i in range(1, len(opponent_history)):
            if opponent_history[i] != current_opp or i == len(opponent_history) - 1:
                label = f"VS {current_opp.upper()}"
                shade_color = phase_colors.get(current_opp, "#cccccc")
                text_color = phase_text_colors.get(current_opp, "#666666")
                
                start_x = episodes[start_idx]
                end_x = episodes[i]
                
                ax.axvspan(start_x, end_x, alpha=0.4, color=shade_color)
                # Clean up phase name for label (e.g. Minimax(αβ, d=2) -> MINIMAX)
                clean_name = current_opp.split('(')[0].upper()
                msg = f"VS {clean_name}"
                ax.text((start_x + end_x) / 2, 102, msg, ha="center", va="bottom",
                         fontsize=11, fontweight="bold", color=text_color, alpha=0.9)
                
                current_opp = opponent_history[i]
                start_idx = i

    # ── Trend: Gold Standard Validation (Curriculum-Independent) ─────────────
    val_episodes = metrics.get("val_episodes", [])
    val_wins = metrics.get("val_wins", [])
    if val_episodes and val_wins:
        ax.plot(val_episodes, val_wins, label="MiniMax win rate", 
                color='#f1c40f', linestyle='--', linewidth=3, marker='o', 
                markersize=8, alpha=1.0, zorder=10)

    # ── Plot standard metrics ───────────────────────────────────────────────
    for key, values in metrics.items():
        if key.startswith("val_"): continue
        if key in ["vtable_size", "epsilon"]: continue # Handled by secondary axis
        if not values: continue
            
        color = colours.get(key, "#3498db")
        label = labels.get(key, str(key).replace("_", " ").capitalize())
        linewidth = 2.5 if "non_loss" in key.lower() or "non-loss" in key.lower() else 1.5

        if smooth and len(values) > window:
            smoothed = pd.Series(values).rolling(window, min_periods=1).mean()
            ax.plot(episodes, smoothed, label=label, color=color, linewidth=linewidth)
            ax.plot(episodes, values, alpha=0.1, color=color, linestyle='-')
        else:
            ax.plot(episodes, values, label=label, color=color, linewidth=linewidth)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Percentage (%)")
    display_title = title if title else f"{algo_name} Training Convergence"
    ax.set_title(display_title, fontweight="bold")
    
    # ── Secondary Axis: Epsilon or V-Table ──────────────────────────────────
    if (epsilon_history and len(epsilon_history) == len(episodes)) or \
       (vtable_history and len(vtable_history) == len(episodes)):
        
        ax2 = ax.twinx()
        handles, labels_list = ax.get_legend_handles_labels()

        if vtable_history and any(v is not None for v in vtable_history):
            # Fill Nones for plotting if mixed
            v_plot = [v if v is not None else 0 for v in vtable_history]
            line, = ax2.plot(episodes, v_plot, color='#9b59b6', linestyle='--', 
                             alpha=0.7, linewidth=2, label='V-Table States')
            ax2.set_ylabel("Unique States Explored", color='#8e44ad', fontsize=12, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='#8e44ad')
            handles.append(line); labels_list.append('V-Table States')
        elif epsilon_history and len(epsilon_history) == len(episodes):
            line, = ax2.plot(episodes, epsilon_history, color='#666666', linestyle='-', 
                             alpha=0.6, linewidth=1.5, label='Epsilon')
            ax2.set_ylabel("Exploration Rate (Epsilon)", color='black', fontsize=12, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='black')
            ax2.set_ylim(0, 1.1)
            handles.append(line); labels_list.append('Epsilon')

        ax.legend(handles, labels_list, 
                  loc="upper center", bbox_to_anchor=(0.5, -0.12),
                  ncol=len(handles), frameon=False)
    else:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
                  ncol=len(ax.get_lines()), frameon=False)
    ax.set_ylim(-5, 115) # Leave room for labels

    fig.tight_layout()
    final_filename = filename if filename else f"training_curve_{algo_name.lower().replace(' ', '_')}"
    save_fig(fig, final_filename, output_dir)
    return fig


def plot_node_comparison(labels: list, nodes: list, title: str = "Nodes Visited per Move",
                         output_dir: str = "figures"):
    """Bar chart comparing node counts (Minimax vs Alpha-Beta)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    colours = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
    ax.bar(labels, nodes, color=colours[:len(labels)])
    ax.set_ylabel("Nodes Visited")
    ax.set_title(title)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 3))

    fig.tight_layout()
    save_fig(fig, "node_comparison", output_dir)
    return fig


def plot_time_per_move(df: pd.DataFrame, title: str = "Average Time per Move",
                       output_dir: str = "figures"):
    """Bar chart of average time per move for each algorithm."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    algos = df["p1"].unique()
    times = df["p1_avg_time"].values * 1000  # ms

    ax.bar(algos, times, color="#3498db")
    ax.set_ylabel("Time per Move (ms)")
    ax.set_title(title)
    ax.set_xlabel("Algorithm")

    fig.tight_layout()
    save_fig(fig, "time_per_move", output_dir)
    return fig
