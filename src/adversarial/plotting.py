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
    "figure.figsize": (10, 6),
    "savefig.dpi": 150,
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


def plot_training_curve(episodes: list, win_rates: list, algo_name: str = "RL",
                        window: int = 100, output_dir: str = "figures"):
    """Training convergence curve (rolling win rate)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    # Rolling average
    if len(win_rates) > window:
        smoothed = pd.Series(win_rates).rolling(window).mean()
        ax.plot(episodes, smoothed, label=f"Rolling avg (window={window})", color="#3498db")
        ax.plot(episodes, win_rates, alpha=0.15, color="#3498db")
    else:
        ax.plot(episodes, win_rates, color="#3498db")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate")
    ax.set_title(f"{algo_name} Training Convergence")
    ax.legend()

    fig.tight_layout()
    save_fig(fig, f"training_curve_{algo_name.lower().replace(' ', '_')}", output_dir)
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
