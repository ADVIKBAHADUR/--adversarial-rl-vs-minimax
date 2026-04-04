#!/usr/bin/env python3
"""
evaluate_all.py — Comprehensive tournament & analysis for Adversarial RL vs Minimax.

Usage:
    python evaluate_all.py                    # TTT + C4
    python evaluate_all.py --game tictactoe
    python evaluate_all.py --game connect4
    python evaluate_all.py --games 200        # more games per matchup
"""

import argparse
import os
import sys
import time
import pickle
import gc
import traceback
from pathlib import Path
from functools import wraps

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adversarial.games import TicTacToe, Connect4
from adversarial.agents import MinimaxAgent, QLearningAgent, DQNAgent, DefaultAgent, RandomAgent
from adversarial.config import MinimaxConfig
from adversarial.tournament import run_match

# ── Global style ──────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "lines.linewidth": 2.5, "axes.grid": True, "grid.alpha": 0.3,
    "figure.figsize": (12, 7), "savefig.dpi": 150, "savefig.bbox": "tight",
})

COLORS = {
    "win":  "#2ecc71",
    "draw": "#f39c12",
    "loss": "#e74c3c",
    "blue": "#3498db",
    "purple": "#9b59b6",
    "teal": "#1abc9c",
}

OUT_DIR = Path("figures")
RES_DIR = Path("results")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — AGENT ROSTER
# ═══════════════════════════════════════════════════════════════════════════════

def _timed(agent):
    """Wrap an agent's select_action to collect per-move timings."""
    agent._move_times = []
    orig = agent.select_action

    @wraps(orig)
    def wrapper(state, valid_actions):
        t0 = time.perf_counter()
        action = orig(state, valid_actions)
        agent._move_times.append(time.perf_counter() - t0)
        return action

    agent.select_action = wrapper
    return agent


def _file_kb(path):
    """Return file size in KB, or 0 if file doesn't exist."""
    p = Path(path)
    return p.stat().st_size / 1024 if p.exists() else 0.0


def _ram_est_kb(agent):
    """Estimate peak RAM for an agent (V-table or DQN params)."""
    if hasattr(agent, "v_table"):
        # Each entry: tuple key (~50 bytes) + float (~8 bytes)
        return len(agent.v_table) * 58 / 1024
    if hasattr(agent, "_cache"):
        # Transposition table entries: tuple (~50 bytes) + dict or float (~100 bytes)
        return len(agent._cache) * 150 / 1024
    if hasattr(agent, "q_net") and agent.q_net is not None:
        params = sum(p.numel() for p in agent.q_net.parameters())
        return params * 4 / 1024  # float32
    return 0.0


def build_roster(game_name: str) -> list[dict]:
    """
    Return a list of agent dicts for the given game.
    Each dict: {name, agent, model_path (or None), skip_reason (or None)}
    """
    roster = []
    g = _game(game_name)

    def add(name, agent_factory, model_path=None):
        """Build an agent safely, mark skip if model missing or error."""
        skip = None
        agent = None
        try:
            agent = agent_factory()
            agent.set_game(g)          # always set_game FIRST
            if model_path:
                if not Path(model_path).exists():
                    skip = f"model not found: {model_path}"
                else:
                    agent.load(model_path)
        except Exception as e:
            skip = f"{type(e).__name__}: {e}"
            agent = None
        roster.append({"name": name, "agent": agent, "model_path": model_path, "skip": skip})

    # ── Always-available agents ───────────────────────────────────────────────
    add("Random",  lambda: RandomAgent(g))
    add("Default", lambda: DefaultAgent(g))

    if game_name == "tictactoe":
        add("Minimax(vanilla,d=9)",
            lambda: MinimaxAgent(g, MinimaxConfig(max_depth=9, use_alpha_beta=False, move_ordering=False)))
        add("Minimax(αβ,d=9)",
            lambda: MinimaxAgent(g, MinimaxConfig(max_depth=9, use_alpha_beta=True, move_ordering=True)))
        add("Q-Learning",
            lambda: QLearningAgent(g),
            model_path="models/tictactoe/qlearning/model_best.pkl")
        add("DQN",
            lambda: DQNAgent(g),
            model_path="models/tictactoe/dqn/optimized_200k_best.pt")

    elif game_name == "connect4":
        add("Minimax(vanilla,d=3)",
            lambda: MinimaxAgent(g, MinimaxConfig(max_depth=3, use_alpha_beta=False, move_ordering=False)))
        add("Minimax(αβ,d=5)",
            lambda: MinimaxAgent(g, MinimaxConfig(max_depth=5, use_alpha_beta=True, move_ordering=True)))
        add("Q-Learning",
            lambda: QLearningAgent(g),
            model_path="models/connect4/qlearning/model_best.pkl")
        add("DQN",
            lambda: DQNAgent(g),
            model_path="models/connect4/dqn/model_validity_best.pt")

    return roster


def _game(name: str):
    return TicTacToe() if name == "tictactoe" else Connect4()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TOURNAMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_tournament(roster: list[dict], game_name: str, n_games: int, max_workers: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full round-robin. Returns:
      - match_df: one row per (p1, p2) matchup with win/draw/loss/time
      - agent_df: one row per agent with aggregate stats + memory
    """
    game = _game(game_name)
    active = [r for r in roster if r["skip"] is None]
    n = len(active)

    print(f"\n  Active agents ({n}): {[r['name'] for r in active]}")
    skipped = [r for r in roster if r["skip"]]
    for s in skipped:
        print(f"  ⚠️  Skipping '{s['name']}': {s['skip']}")

    for r in active:
        _timed(r["agent"])

    rows = []
    total_matchups = n * (n - 1) // 2
    done = 0

    for i in range(n):
        for j in range(i + 1, n):
            a1, a2 = active[i], active[j]
            a1["agent"]._move_times.clear()
            a2["agent"]._move_times.clear()

            print(f"  [{done+1}/{total_matchups}] {a1['name']} vs {a2['name']}")
            t0 = time.perf_counter()
            result = run_match(game, a1["agent"], a2["agent"],
                               n_games=n_games, swap_sides=True, verbose=False)
            elapsed = time.perf_counter() - t0

            t1 = np.mean(a1["agent"]._move_times) * 1000 if a1["agent"]._move_times else 0.0
            t2 = np.mean(a2["agent"]._move_times) * 1000 if a2["agent"]._move_times else 0.0

            rows.append({
                "game": game_name, "p1": a1["name"], "p2": a2["name"],
                "p1_win_pct": result["p1_win_pct"],
                "p2_win_pct": result["p2_win_pct"],
                "draw_pct":   result["draw_pct"],
                "p1_ms_per_move": round(t1, 3),
                "p2_ms_per_move": round(t2, 3),
                "n_games": result["n_games"],
            })
            print(f"\r    {result['p1_win_pct']:.0f}% / {result['draw_pct']:.0f}% / {result['p2_win_pct']:.0f}%  ({elapsed:.1f}s)")
            done += 1

    match_df = pd.DataFrame(rows)

    # ── Per-agent aggregate stats ─────────────────────────────────────────────
    agent_rows = []
    for r in active:
        ag = r["agent"]
        name = r["name"]
        # Win pct averaged across all matchups this agent appeared in
        as_p1 = match_df[match_df["p1"] == name]
        as_p2 = match_df[match_df["p2"] == name]

        wins   = list(as_p1["p1_win_pct"]) + list(as_p2["p2_win_pct"])
        draws  = list(as_p1["draw_pct"])   + list(as_p2["draw_pct"])
        losses = list(as_p1["p2_win_pct"]) + list(as_p2["p1_win_pct"])
        times  = list(as_p1["p1_ms_per_move"]) + list(as_p2["p2_ms_per_move"])

        agent_rows.append({
            "game":             game_name,
            "agent":            name,
            "avg_win_pct":      round(np.mean(wins),   1) if wins   else 0.0,
            "avg_draw_pct":     round(np.mean(draws),  1) if draws  else 0.0,
            "avg_loss_pct":     round(np.mean(losses), 1) if losses else 0.0,
            "ms_per_move":      round(np.mean(times),  3) if times  else 0.0,
            "model_kb":         round(_file_kb(r["model_path"]) if r["model_path"] else 0.0, 1),
            "ram_est_kb":       round(_ram_est_kb(ag), 1),
            "vtable_states":    len(ag.v_table) if hasattr(ag, "v_table") else None,
            "episodes_trained": getattr(ag, "episodes_trained", None),
        })

    agent_df = pd.DataFrame(agent_rows)
    return match_df, agent_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png")
    fig.savefig(OUT_DIR / f"{name}.svg")
    print(f"  📊 figures/{name}.png + .svg")
    plt.close(fig)


def plot_heatmap(match_df: pd.DataFrame, game_name: str):
    """Win-rate heatmap for every pairwise matchup."""
    agents = []
    for r in [match_df["p1"], match_df["p2"]]:
        for a in r:
            if a not in agents:
                agents.append(a)

    n = len(agents)
    matrix = np.full((n, n), np.nan)
    for _, row in match_df.iterrows():
        i, j = agents.index(row["p1"]), agents.index(row["p2"])
        matrix[i, j] = row["p1_win_pct"]
        matrix[j, i] = row["p2_win_pct"]

    fig, ax = plt.subplots(figsize=(max(8, n * 1.4), max(7, n * 1.2)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(agents, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(agents, fontsize=10)
    ax.set_xlabel("Opponent (P2)", fontsize=12)
    ax.set_ylabel("Agent (P1)", fontsize=12)
    ax.set_title(f"Head-to-Head Win Rate (%) — {game_name.title()}", fontsize=14, fontweight="bold")

    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                v = matrix[i, j]
                color = "white" if (v < 20 or v > 80) else "black"
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)
            elif i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=True, color="#dddddd", zorder=2))
                ax.text(j, i, "—", ha="center", va="center", fontsize=14, color="#999")

    plt.colorbar(im, ax=ax, label="P1 Win %", shrink=0.8)
    fig.tight_layout()
    _save(fig, f"heatmap_{game_name}")


def plot_summary_vs_default(match_df: pd.DataFrame, agent_df: pd.DataFrame, game_name: str):
    """Grouped Win/Draw/Loss bars vs Default opponent."""
    vs_def = match_df[match_df["p2"] == "Default"].copy()
    # Also add rows where Default was P1, flip perspective
    vs_def2 = match_df[match_df["p1"] == "Default"].copy()
    vs_def2 = vs_def2.rename(columns={
        "p2": "agent", "p2_win_pct": "win", "draw_pct": "draw", "p1_win_pct": "loss"
    })
    vs_def["agent"] = vs_def["p1"]
    vs_def["win"]   = vs_def["p1_win_pct"]
    vs_def["draw"]  = vs_def["draw_pct"]
    vs_def["loss"]  = vs_def["p2_win_pct"]

    combined = pd.concat([vs_def[["agent","win","draw","loss"]],
                          vs_def2[["agent","win","draw","loss"]]], ignore_index=True)
    combined = combined[combined["agent"] != "Default"].drop_duplicates("agent")

    agents = combined["agent"].tolist()
    x = np.arange(len(agents))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(agents) * 1.6), 6))
    ax.bar(x - w, combined["win"],  w, label="Win %",  color=COLORS["win"])
    ax.bar(x,     combined["draw"], w, label="Draw %", color=COLORS["draw"])
    ax.bar(x + w, combined["loss"], w, label="Loss %", color=COLORS["loss"])

    ax.set_xticks(x); ax.set_xticklabels(agents, rotation=20, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 110)
    ax.set_title(f"Performance vs Default Opponent — {game_name.title()}", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _save(fig, f"summary_vs_default_{game_name}")


def plot_move_time(all_agent_dfs: list[pd.DataFrame]):
    """Log-scale bar chart of move time per agent, both games side by side."""
    combined = pd.concat(all_agent_dfs, ignore_index=True)
    games = combined["game"].unique()
    agents_all = combined["agent"].unique().tolist()

    fig, ax = plt.subplots(figsize=(14, 6))
    game_colors = {"tictactoe": COLORS["blue"], "connect4": COLORS["purple"]}
    width = 0.35
    agent_set = list(dict.fromkeys(combined["agent"].tolist()))  # preserve order, dedupe

    for gi, g in enumerate(games):
        sub = combined[combined["game"] == g].set_index("agent")
        positions = []
        heights = []
        labels = []
        for ai, ag in enumerate(agent_set):
            if ag in sub.index:
                positions.append(ai + gi * width)
                heights.append(max(sub.loc[ag, "ms_per_move"], 0.001))
                labels.append(ag)

        bars = ax.bar(positions, heights, width=width,
                      label=g.title(), color=game_colors[g], alpha=0.85, edgecolor="white")
        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.1,
                    f"{h:.2f}" if h < 10 else f"{h:.0f}",
                    ha="center", va="bottom", fontsize=8, rotation=45)

    xs = np.arange(len(agent_set))
    ax.set_xticks(xs + width / 2)
    ax.set_xticklabels(agent_set, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Avg Move Time (ms, log scale)")
    ax.set_title("Move Time per Agent — Both Games", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _save(fig, "move_time_comparison")


def plot_memory(all_agent_dfs: list[pd.DataFrame]):
    """Stacked bar: disk size + estimated RAM per agent."""
    combined = pd.concat(all_agent_dfs, ignore_index=True)
    # Deduplicate by agent name — keep max (connect4 models are bigger)
    combined = combined.sort_values("model_kb", ascending=False).drop_duplicates("agent")
    combined = combined.sort_values("model_kb", ascending=True)

    agents = combined["agent"].tolist()
    disk   = combined["model_kb"].fillna(0).tolist()
    ram    = combined["ram_est_kb"].fillna(0).tolist()
    x = np.arange(len(agents))

    fig, ax = plt.subplots(figsize=(max(10, len(agents) * 1.5), 6))
    ax.bar(x, disk, label="Model file (KB)", color=COLORS["blue"], alpha=0.85)
    ax.bar(x, ram,  bottom=disk, label="Est. RAM (KB)", color=COLORS["teal"],
           alpha=0.75, hatch="//")

    ax.set_xticks(x); ax.set_xticklabels(agents, rotation=25, ha="right")
    ax.set_ylabel("Memory (KB)")
    ax.set_title("Memory Footprint per Agent", fontweight="bold")
    ax.set_yscale("log")
    ax.legend()

    for xi, (d, r) in enumerate(zip(disk, ram)):
        total = d + r
        if total > 0:
            ax.text(xi, total * 1.1, f"{total:,.0f}KB",
                    ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _save(fig, "memory_footprint")


def plot_scalability(ttt_agent_df: pd.DataFrame, c4_agent_df: pd.DataFrame):
    """Lines per algorithm showing win% vs Default as game complexity increases."""
    # Find agents that appear in BOTH games
    ttt_vs = ttt_agent_df.set_index("agent")
    c4_vs  = c4_agent_df.set_index("agent")

    # We want: win% vs Default specifically. Use avg_win_pct as proxy (dominated by Default matchup).
    shared = [a for a in ttt_vs.index if a in c4_vs.index
              and a not in ("Random", "Default")]

    if len(shared) < 1:
        print("  ⚠️  No agents common to both games — skipping scalability plot")
        return

    agent_colors = {
        "Q-Learning":         COLORS["blue"],
        "Minimax(vanilla,d=9)": "#e74c3c",
        "Minimax(αβ,d=9)":    "#c0392b",
        "Minimax(vanilla,d=4)": "#e74c3c",
        "Minimax(αβ,d=5)":    "#c0392b",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    game_x = [0, 1]
    game_labels = ["Tic-Tac-Toe", "Connect 4"]

    for ag in shared:
        ttt_win = ttt_vs.loc[ag, "avg_win_pct"]
        c4_win  = c4_vs.loc[ag, "avg_win_pct"]
        color = agent_colors.get(ag, COLORS["purple"])
        ax.plot(game_x, [ttt_win, c4_win], "o-", label=ag, color=color, markersize=9)
        ax.annotate(f"{ttt_win:.0f}%", (0, ttt_win), textcoords="offset points",
                    xytext=(-28, 0), ha="right", fontsize=10, color=color)
        ax.annotate(f"{c4_win:.0f}%",  (1, c4_win),  textcoords="offset points",
                    xytext=(6, 0),   ha="left",  fontsize=10, color=color)

    ax.set_xticks(game_x); ax.set_xticklabels(game_labels, fontsize=13)
    ax.set_ylabel("Avg Win % (across all opponents)")
    ax.set_ylim(-5, 105)
    ax.set_title("Scalability: Performance Across Games", fontweight="bold")
    ax.legend(loc="lower left")
    fig.tight_layout()
    _save(fig, "scalability_cliff")


def plot_master(game_results: dict):
    """
    2×2 composite killer figure:
      TL: TTT heatmap  |  TR: C4 heatmap
      BL: Move time    |  BR: Scalability
    """
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    def _draw_heatmap(ax, match_df, title):
        agents = []
        for col in [match_df["p1"], match_df["p2"]]:
            for a in col:
                if a not in agents: agents.append(a)
        n = len(agents)
        matrix = np.full((n, n), np.nan)
        for _, row in match_df.iterrows():
            i, j = agents.index(row["p1"]), agents.index(row["p2"])
            matrix[i, j] = row["p1_win_pct"]
            matrix[j, i] = row["p2_win_pct"]
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(agents, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(agents, fontsize=8)
        ax.set_title(title, fontsize=12, fontweight="bold")
        for i in range(n):
            for j in range(n):
                if not np.isnan(matrix[i, j]):
                    v = matrix[i, j]
                    c = "white" if (v < 20 or v > 80) else "black"
                    ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8, color=c, fontweight="bold")
                elif i == j:
                    ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,fill=True,color="#ddd",zorder=2))
                    ax.text(j,i,"—",ha="center",va="center",fontsize=10,color="#999")
        plt.colorbar(im, ax=ax, shrink=0.85, label="Win %")

    # TL — TTT heatmap
    if "tictactoe" in game_results:
        ax_tl = fig.add_subplot(gs[0, 0])
        _draw_heatmap(ax_tl, game_results["tictactoe"]["match_df"], "Win Rate — Tic-Tac-Toe")

    # TR — C4 heatmap
    if "connect4" in game_results:
        ax_tr = fig.add_subplot(gs[0, 1])
        _draw_heatmap(ax_tr, game_results["connect4"]["match_df"], "Win Rate — Connect 4")

    # BL — Move time
    ax_bl = fig.add_subplot(gs[1, 0])
    all_agent_dfs = [v["agent_df"] for v in game_results.values()]
    combined = pd.concat(all_agent_dfs, ignore_index=True)
    games = combined["game"].unique()
    agent_set = list(dict.fromkeys(combined["agent"].tolist()))
    game_colors = {"tictactoe": COLORS["blue"], "connect4": COLORS["purple"]}
    width = 0.35
    for gi, g in enumerate(games):
        sub = combined[combined["game"] == g].set_index("agent")
        xs, hs = [], []
        for ai, ag in enumerate(agent_set):
            if ag in sub.index:
                xs.append(ai + gi * width)
                hs.append(max(sub.loc[ag, "ms_per_move"], 0.001))
        ax_bl.bar(xs, hs, width=width, label=g.title(),
                  color=game_colors[g], alpha=0.85, edgecolor="white")
    ax_bl.set_xticks(np.arange(len(agent_set)) + width/2)
    ax_bl.set_xticklabels(agent_set, rotation=30, ha="right", fontsize=8)
    ax_bl.set_yscale("log")
    ax_bl.set_ylabel("Move Time (ms, log)")
    ax_bl.set_title("Move Time per Agent", fontweight="bold")
    ax_bl.legend(fontsize=9)

    # BR — Scalability
    ax_br = fig.add_subplot(gs[1, 1])
    if "tictactoe" in game_results and "connect4" in game_results:
        ttt_df = game_results["tictactoe"]["agent_df"].set_index("agent")
        c4_df  = game_results["connect4"]["agent_df"].set_index("agent")
        shared = [a for a in ttt_df.index if a in c4_df.index
                  and a not in ("Random", "Default")]
        cmap = plt.cm.Set2(np.linspace(0, 1, max(len(shared),1)))
        for idx, ag in enumerate(shared):
            ax_br.plot([0,1], [ttt_df.loc[ag,"avg_win_pct"], c4_df.loc[ag,"avg_win_pct"]],
                       "o-", label=ag, color=cmap[idx], markersize=8)
        ax_br.set_xticks([0,1]); ax_br.set_xticklabels(["Tic-Tac-Toe","Connect 4"], fontsize=11)
        ax_br.set_ylabel("Avg Win %"); ax_br.set_ylim(-5, 105)
        ax_br.set_title("Scalability Across Games", fontweight="bold")
        ax_br.legend(fontsize=9)

    fig.suptitle("Adversarial RL vs Minimax — Complete Evaluation", fontsize=16, fontweight="bold", y=1.01)
    _save(fig, "master_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CSV SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def save_csv(all_match_dfs, all_agent_dfs):
    RES_DIR.mkdir(parents=True, exist_ok=True)
    match_path = RES_DIR / "matchup_results.csv"
    agent_path = RES_DIR / "agent_summary.csv"
    pd.concat(all_match_dfs).to_csv(match_path, index=False)
    pd.concat(all_agent_dfs).to_csv(agent_path, index=False)
    print(f"\n  💾 {match_path}")
    print(f"  💾 {agent_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Full tournament & analysis")
    parser.add_argument("--game", choices=["tictactoe", "connect4", "all"], default="all")
    parser.add_argument("--games", type=int, default=100,
                        help="Games per matchup (default 100, use ≥200 for stable results)")
    parser.add_argument("--c4-games", type=int, default=None,
                        help="Override games per matchup for Connect 4 only (default: same as --games)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel matchup threads (default 4)")
    args = parser.parse_args()

    games_to_run = (["tictactoe", "connect4"] if args.game == "all"
                    else [args.game])

    game_results = {}
    all_match_dfs = []
    all_agent_dfs = []

    for game_name in games_to_run:
        n = args.c4_games if (game_name == "connect4" and args.c4_games) else args.games
        print(f"{'='*60}")
        print(f"🎮  {game_name.upper()} TOURNAMENT  ({n} games/matchup)")
        print(f"{'='*60}")

        roster = build_roster(game_name)
        match_df, agent_df = run_tournament(roster, game_name, n, max_workers=args.workers)

        game_results[game_name] = {"match_df": match_df, "agent_df": agent_df}
        all_match_dfs.append(match_df)
        all_agent_dfs.append(agent_df)

        print(f"\n📋 {game_name.title()} — Agent Summary:")
        print(agent_df[["agent","avg_win_pct","avg_draw_pct","avg_loss_pct",
                         "ms_per_move","model_kb","ram_est_kb"]].to_string(index=False))

    print(f"\n\n{'='*60}")
    print("📈  Generating plots...")
    print(f"{'='*60}")

    if "tictactoe" in game_results:
        plot_heatmap(game_results["tictactoe"]["match_df"], "tictactoe")
        plot_summary_vs_default(game_results["tictactoe"]["match_df"],
                                game_results["tictactoe"]["agent_df"], "tictactoe")

    if "connect4" in game_results:
        plot_heatmap(game_results["connect4"]["match_df"], "connect4")
        plot_summary_vs_default(game_results["connect4"]["match_df"],
                                game_results["connect4"]["agent_df"], "connect4")

    plot_move_time(all_agent_dfs)
    plot_memory(all_agent_dfs)

    if "tictactoe" in game_results and "connect4" in game_results:
        plot_scalability(game_results["tictactoe"]["agent_df"],
                         game_results["connect4"]["agent_df"])

    plot_master(game_results)

    print("\n\n📊  Saving CSV results...")
    save_csv(all_match_dfs, all_agent_dfs)

    print(f"\n✅  Done!  Figures → figures/   Results → results/")


if __name__ == "__main__":
    main()
