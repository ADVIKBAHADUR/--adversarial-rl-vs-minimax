"""generate_summary_tables.py — Quantitative and qualitative summary tables.

Reads agent_summary.csv to emit:
  1. summary_benchmarks.tex   — Win/Draw/Loss + efficiency for all agents
  2. summary_tradeoffs.tex    — Architectural trade-offs matrix
"""

import pandas as pd
import numpy as np
import os

OUT_DIR = "results/latex_tables"
os.makedirs(OUT_DIR, exist_ok=True)

agent_df = pd.read_csv("results/agent_summary.csv")


def fmt_mem_mb(kb):
    if kb == 0 or (isinstance(kb, float) and np.isnan(kb)):
        return "---"
    return f"{kb/1024:.2f}MB" if kb >= 1024 else f"{kb/1024:.3f}MB"


def generate_quantitative_summary():
    """
    Full benchmark table: Win%, Draw%, Loss%, Latency, Model size, RAM.
    """
    df = agent_df[~agent_df['agent'].isin(['Random', 'Default'])].copy()
    df['game_order'] = df['game'].map({'tictactoe': 0, 'connect4': 1})
    df = df.sort_values(['game_order', 'avg_win_pct'], ascending=[True, False])

    tex = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Comparative Performance Benchmarks across games.}",
        "\\label{tab:summary-benchmarks}",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\small",
        "\\begin{tabularx}{\\linewidth}{"
        ">{\\raggedright\\arraybackslash}p{1.6cm} "
        ">{\\raggedright\\arraybackslash}p{3.0cm} "
        ">{\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X "
        ">{\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X "
        ">{\\centering\\arraybackslash}X}",
        "\\toprule",
        "\\textbf{Game} & \\textbf{Algorithm} & "
        "\\textbf{Win\\%} & \\textbf{Draw\\%} & \\textbf{Loss\\%} & "
        "\\textbf{Latency} & \\textbf{RAM} \\\\",
        "\\midrule",
    ]

    current_game = None
    for _, row in df.iterrows():
        game_label = "Tic-Tac-Toe" if row['game'] == "tictactoe" else "Connect 4"
        if row['game'] != current_game:
            if current_game is not None:
                tex.append("\\midrule")
            current_game = row['game']
            disp = game_label
        else:
            disp = ""

        algo = row['agent'].replace("vanilla", "Vanilla").replace("αβ", r"$\alpha\beta$")
        tex.append(
            f"{disp} & {algo} & "
            f"{row['avg_win_pct']:.1f}\\% & "
            f"{row['avg_draw_pct']:.1f}\\% & "
            f"{row['avg_loss_pct']:.1f}\\% & "
            f"{row['ms_per_move']:.2f}ms & "
            f"{fmt_mem_mb(row['ram_est_kb'])} \\\\"
        )

    tex += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]

    with open(f"{OUT_DIR}/summary_benchmarks.tex", "w") as f:
        f.write("\n".join(tex))
    print("  ✅ summary_benchmarks.tex")


def generate_qualitative_summary():
    """
    Architectural trade-off matrix — qualitative comparison.
    """
    tradeoffs = [
        ["Vanilla Minimax",    "Heuristic Search", r"Exponential ($b^d$)",  r"Recursion stack",        "Optimal; no learning; slow at depth"],
        ["Alpha-Beta",         "Heuristic Search", r"Pruned ($b^{d/2}$)",   r"Recursion stack",        "Optimal; faster; still no learning"],
        ["Tabular Q-Learning", "Model-Free RL",    r"$O(1)$ lookup",        r"State table $|S|$",      "Fast inference; state-space explosion"],
        ["DQN",                "Deep RL",           r"$O(\theta)$ forward",  r"Fixed neural weights",   "Scalable; needs many episodes; GPU-friendly"],
    ]

    tex = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Architectural trade-offs and scalability matrix.}",
        "\\label{tab:summary-tradeoffs}",
        "\\renewcommand{\\arraystretch}{1.25}",
        "\\small",
        "\\begin{tabularx}{\\linewidth}{"
        ">{\\raggedright\\arraybackslash}p{2.8cm} "        # Algorithm
        ">{\\centering\\arraybackslash}p{2.2cm} "         # Paradigm
        ">{\\centering\\arraybackslash}X "                # Complexity
        ">{\\centering\\arraybackslash}X "                # Memory Growth
        ">{\\raggedright\\arraybackslash}X}",             # Key Trade-off
        "\\toprule",
        "\\textbf{Algorithm} & \\textbf{Paradigm} & "
        "\\textbf{Time Complexity} & \\textbf{Memory Growth} & "
        "\\textbf{Key Trade-off} \\\\",
        "\\midrule",
    ]

    for r in tradeoffs:
        tex.append(" & ".join(r) + " \\\\")

    tex += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]

    with open(f"{OUT_DIR}/summary_tradeoffs.tex", "w") as f:
        f.write("\n".join(tex))
    print("  ✅ summary_tradeoffs.tex")


if __name__ == "__main__":
    print("Generating summary tables...")
    generate_quantitative_summary()
    generate_qualitative_summary()
    print(f"\nWritten to {OUT_DIR}/")
