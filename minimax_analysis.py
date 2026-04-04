"""
Minimax analysis script — generates nodes visited, time, and pruning efficiency
tables for both TicTacToe and Connect 4, then plots them.
"""
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

from adversarial.games import TicTacToe, Connect4
from adversarial.agents import MinimaxAgent
from adversarial.config import MinimaxConfig


# ── Style ─────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "lines.linewidth": 2.5, "axes.grid": True, "grid.alpha": 0.3,
    "figure.figsize": (10, 6), "savefig.dpi": 150, "savefig.bbox": "tight",
})


def collect_stats(game, vanilla_depths, ab_extra_depths=None, timeout_s=30.0):
    """
    Collect nodes + time for vanilla and alpha-beta minimax.
    - vanilla_depths: run vanilla minimax at these depths (skip if it takes too long)
    - ab_extra_depths: additional depths to run alpha-beta only (too slow for vanilla)
    - timeout_s: skip vanilla at a depth if it takes longer than this
    """
    ab_extra_depths = ab_extra_depths or []
    all_depths = sorted(set(vanilla_depths) | set(ab_extra_depths))
    rows = []

    state = game.reset()
    valid = game.get_valid_actions(state)

    for d in all_depths:
        print(f"  depth={d} ...", end="", flush=True)
        row = {"Game": game.name, "Depth": d,
               "Vanilla Nodes": None, "Vanilla Time (s)": None,
               "Alpha-Beta Nodes": None, "Alpha-Beta Time (s)": None}

        # Vanilla — only if in vanilla_depths
        if d in vanilla_depths:
            cfg = MinimaxConfig(max_depth=d, use_alpha_beta=False, move_ordering=False)
            agent = MinimaxAgent(game, cfg)
            agent.set_game(game)
            t0 = time.perf_counter()
            agent.select_action(state, valid)
            elapsed = time.perf_counter() - t0
            if elapsed > timeout_s:
                print(f" [vanilla timed out at {elapsed:.1f}s — skipping]", flush=True)
                # Still record what we got
            row["Vanilla Nodes"] = agent.stats["nodes_visited"]
            row["Vanilla Time (s)"] = elapsed
            print(f" vanilla={row['Vanilla Nodes']:,} ({elapsed:.3f}s)", end="", flush=True)

        # Alpha-Beta
        ab_cfg = MinimaxConfig(max_depth=d, use_alpha_beta=True, move_ordering=True)
        ab_agent = MinimaxAgent(game, ab_cfg)
        ab_agent.set_game(game)
        t0 = time.perf_counter()
        ab_agent.select_action(state, valid)
        ab_elapsed = time.perf_counter() - t0
        row["Alpha-Beta Nodes"] = ab_agent.stats["nodes_visited"]
        row["Alpha-Beta Time (s)"] = ab_elapsed
        print(f"  ab={row['Alpha-Beta Nodes']:,} ({ab_elapsed:.3f}s)", flush=True)

        # Derived stats (only when vanilla ran)
        if row["Vanilla Nodes"] is not None and row["Vanilla Nodes"] > 0:
            pruned = 1.0 - (row["Alpha-Beta Nodes"] / row["Vanilla Nodes"])
            row["Nodes Pruned (%)"] = round(pruned * 100, 1)
            if row["Alpha-Beta Time (s)"] > 0:
                row["Speedup (×)"] = round(row["Vanilla Time (s)"] / row["Alpha-Beta Time (s)"], 2)

        rows.append(row)

    return pd.DataFrame(rows)


def save_fig(fig, name, output_dir="figures"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/{name}.png")
    fig.savefig(f"{output_dir}/{name}.svg")
    print(f"📊 Saved figures/{name}.png + .svg")


def plot_nodes(ttt_df, c4_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, df, title in [
        (axes[0], ttt_df, "Tic-Tac-Toe"),
        (axes[1], c4_df, "Connect 4"),
    ]:
        ax.semilogy(df["Depth"], df["Vanilla Nodes"], "o-", color="#e74c3c", label="Vanilla Minimax")
        ax.semilogy(df["Depth"], df["Alpha-Beta Nodes"], "s--", color="#2ecc71", label="Alpha-Beta")
        ax.fill_between(df["Depth"], df["Alpha-Beta Nodes"], df["Vanilla Nodes"],
                        alpha=0.12, color="#3498db", label="Nodes pruned")
        ax.set_title(f"Nodes Visited — {title}")
        ax.set_xlabel("Search Depth")
        ax.set_ylabel("Nodes Visited (log scale)")
        ax.legend()

    fig.suptitle("Minimax vs Alpha-Beta Pruning: Search Tree Size", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "minimax_nodes_comparison")
    return fig


def plot_time(ttt_df, c4_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, df, title in [
        (axes[0], ttt_df, "Tic-Tac-Toe"),
        (axes[1], c4_df, "Connect 4"),
    ]:
        ax.semilogy(df["Depth"], df["Vanilla Time (s)"] * 1000, "o-", color="#e74c3c", label="Vanilla Minimax")
        ax.semilogy(df["Depth"], df["Alpha-Beta Time (s)"] * 1000, "s--", color="#2ecc71", label="Alpha-Beta")
        ax.set_title(f"Time per Move — {title}")
        ax.set_xlabel("Search Depth")
        ax.set_ylabel("Time per Move (ms, log scale)")
        ax.legend()

    fig.suptitle("Minimax vs Alpha-Beta Pruning: Wall-Clock Time", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "minimax_time_comparison")
    return fig


def plot_pruning_efficiency(ttt_df, c4_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, df, title in [
        (axes[0], ttt_df, "Tic-Tac-Toe"),
        (axes[1], c4_df, "Connect 4"),
    ]:
        bars = ax.bar(df["Depth"].astype(str), df["Nodes Pruned (%)"], color="#9b59b6", edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, df["Nodes Pruned (%)"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_title(f"Nodes Pruned by Alpha-Beta — {title}")
        ax.set_xlabel("Search Depth")
        ax.set_ylabel("% Nodes Pruned vs Vanilla")

    fig.suptitle("Alpha-Beta Pruning Efficiency", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "minimax_pruning_efficiency")
    return fig


def plot_speedup(ttt_df, c4_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, df, title in [
        (axes[0], ttt_df, "Tic-Tac-Toe"),
        (axes[1], c4_df, "Connect 4"),
    ]:
        ax.plot(df["Depth"], df["Speedup (×)"], "D-", color="#f39c12", markersize=8)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="No speedup (1×)")
        for x, y in zip(df["Depth"], df["Speedup (×)"]):
            ax.annotate(f"{y:.1f}×", (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=11, fontweight="bold")
        ax.set_title(f"Speedup Factor — {title}")
        ax.set_xlabel("Search Depth")
        ax.set_ylabel("Speedup (×) vs Vanilla")
        ax.legend()

    fig.suptitle("Alpha-Beta vs Vanilla Minimax: Wall-Clock Speedup", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "minimax_speedup")
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("🤖  Tic-Tac-Toe Minimax Analysis")
    print("=" * 60)
    ttt = TicTacToe()
    # TTT: run vanilla d=1..9 (full game depth). α-β is near-instant so all depths.
    # Vanilla d=9 can take ~10-30s from the initial empty board.
    ttt_df = collect_stats(
        ttt,
        vanilla_depths=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        ab_extra_depths=[],
        timeout_s=60.0,
    )

    print("\n" + "=" * 60)
    print("📦  Connect 4 Minimax Analysis (vanilla capped at d=4)")
    print("=" * 60)
    c4 = Connect4()
    # C4: vanilla d=4 takes ~60-120s; we run vanilla d=1..3 and α-β up to d=5
    c4_df = collect_stats(
        c4,
        vanilla_depths=[1, 2, 3, 4],
        ab_extra_depths=[5],
        timeout_s=120.0,
    )

    print("\n\n📋  TicTacToe Summary Table")
    display_cols = ["Depth", "Vanilla Nodes", "Alpha-Beta Nodes", "Nodes Pruned (%)", "Speedup (×)"]
    print(ttt_df[[c for c in display_cols if c in ttt_df.columns]].to_string(index=False))

    print("\n📋  Connect 4 Summary Table")
    print(c4_df[[c for c in display_cols if c in c4_df.columns]].to_string(index=False))

    print("\n📈 Generating plots...")
    plot_nodes(ttt_df, c4_df)
    plot_time(ttt_df, c4_df)
    plot_pruning_efficiency(ttt_df, c4_df)
    plot_speedup(ttt_df, c4_df)

    print("\n✅ Done! All figures saved to figures/")
