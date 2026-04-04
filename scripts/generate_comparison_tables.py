"""generate_comparison_tables.py — 5 assignment comparison LaTeX tables.

Reads matchup_results.csv and agent_summary.csv to emit:
  1. ttt_vs_default.tex         — Algorithms vs Default in Tic-Tac-Toe
  2. c4_vs_default.tex          — Algorithms vs Default in Connect 4
  3. ttt_head_to_head.tex       — Pairwise win matrix in Tic-Tac-Toe
  4. c4_head_to_head.tex        — Pairwise win matrix in Connect 4
  5. overall_comparison.tex     — Cross-game overall comparison
"""

import pandas as pd
import numpy as np
import os

OUT_DIR = "results/latex_tables"
os.makedirs(OUT_DIR, exist_ok=True)

match_df = pd.read_csv("results/matchup_results.csv")
agent_df = pd.read_csv("results/agent_summary.csv")

SKIP = {"Random", "Default"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def algo_label(name: str) -> str:
    """Shorten agent names for table display."""
    return (name.replace("αβ", r"$\alpha\beta$")
                .replace("vanilla", "Vanilla")
                .replace("Minimax(", "Minimax(")
                .replace(",d=", ", d="))


def pct(v) -> str:
    return f"{v:.1f}\\%"


# ── Table 1 & 2: vs Default ───────────────────────────────────────────────────

def table_vs_default(game: str, out_file: str, caption: str, label: str):
    """Win/Draw/Loss for each algorithm against the Default opponent."""
    gdf = match_df[match_df["game"] == game]

    rows = []
    # Agent is P1, Default is P2
    sub1 = gdf[gdf["p2"] == "Default"].copy()
    sub1["algo"]   = sub1["p1"]
    sub1["win"]    = sub1["p1_win_pct"]
    sub1["draw"]   = sub1["draw_pct"]
    sub1["loss"]   = sub1["p2_win_pct"]
    sub1["ms"]     = sub1["p1_ms_per_move"]

    # Agent is P2, Default is P1 — flip perspective
    sub2 = gdf[gdf["p1"] == "Default"].copy()
    sub2["algo"]   = sub2["p2"]
    sub2["win"]    = sub2["p2_win_pct"]
    sub2["draw"]   = sub2["draw_pct"]
    sub2["loss"]   = sub2["p1_win_pct"]
    sub2["ms"]     = sub2["p2_ms_per_move"]

    combined = pd.concat([sub1[["algo","win","draw","loss","ms"]],
                          sub2[["algo","win","draw","loss","ms"]]],
                         ignore_index=True)
    combined = combined[~combined["algo"].isin(SKIP)].drop_duplicates("algo")

    # Sort: best win rate first
    combined = combined.sort_values("win", ascending=False)

    tex = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\small",
        "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}X "
        ">{"
        "\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X "
        ">{"
        "\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X}",
        "\\toprule",
        "\\textbf{Algorithm} & \\textbf{Win \\%} & \\textbf{Draw \\%} & "
        "\\textbf{Loss \\%} & \\textbf{Latency (ms)} \\\\",
        "\\midrule",
    ]

    for _, r in combined.iterrows():
        tex.append(
            f"{algo_label(r['algo'])} & {pct(r['win'])} & {pct(r['draw'])} "
            f"& {pct(r['loss'])} & {r['ms']:.2f} \\\\"
        )

    tex += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]
    with open(f"{OUT_DIR}/{out_file}", "w") as f:
        f.write("\n".join(tex))
    print(f"  ✅ {out_file}")


# ── Table 3 & 4: Head-to-Head matrix ──────────────────────────────────────────

def table_head_to_head(game: str, out_file: str, caption: str, label: str):
    """Pairwise win-rate matrix (rows=P1, cols=P2)."""
    gdf = match_df[match_df["game"] == game]

    # Build ordered agent list, excluding Random/Default
    ordered = []
    for col in ["p1", "p2"]:
        for a in gdf[col].unique():
            if a not in ordered and a not in SKIP:
                ordered.append(a)

    n = len(ordered)
    # Matrix[i][j] = P1 (ordered[i]) win% when playing ordered[j]
    matrix = {}
    for _, row in gdf.iterrows():
        p1, p2 = row["p1"], row["p2"]
        if p1 not in SKIP and p2 not in SKIP:
            matrix[(p1, p2)] = (row["p1_win_pct"], row["draw_pct"], row["p2_win_pct"])
            matrix[(p2, p1)] = (row["p2_win_pct"], row["draw_pct"], row["p1_win_pct"])

    short_labels = [algo_label(a) for a in ordered]

    # Column spec: algo name col + n data cols
    col_spec = ">{\\raggedright\\arraybackslash}p{3.5cm} " + " ".join(
        [">{\\centering\\arraybackslash}X"] * n
    )

    tex = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\small",
        f"\\begin{{tabularx}}{{\\linewidth}}{{{col_spec}}}",
        "\\toprule",
        "\\textbf{P1 $\\downarrow$ / P2 $\\rightarrow$} & "
        + " & ".join([f"\\textbf{{{l}}}" for l in short_labels]) + " \\\\",
        "\\midrule",
    ]

    note_lines = []
    for i, a1 in enumerate(ordered):
        cells = [algo_label(a1)]
        for j, a2 in enumerate(ordered):
            if i == j:
                cells.append("---")
            elif (a1, a2) in matrix:
                w, d, l = matrix[(a1, a2)]
                cells.append(f"{w:.0f}/{d:.0f}/{l:.0f}")
            else:
                cells.append("N/A")
        tex.append(" & ".join(cells) + " \\\\")

    tex += [
        "\\bottomrule",
        "\\multicolumn{" + str(n + 1) + "}{l}{\\footnotesize "
        "Format: Win\\% / Draw\\% / Loss\\% (P1 perspective)} \\\\",
        "\\end{tabularx}",
        "\\end{table}",
    ]

    with open(f"{OUT_DIR}/{out_file}", "w") as f:
        f.write("\n".join(tex))
    print(f"  ✅ {out_file}")


# ── Table 5: Overall comparison ───────────────────────────────────────────────

def table_overall(out_file: str, caption: str, label: str):
    """Cross-game performance overview — avg Win/Draw/Loss + efficiency metrics."""
    df = agent_df[~agent_df["agent"].isin(SKIP)].copy()
    df["game_order"] = df["game"].map({"tictactoe": 0, "connect4": 1})

    # Sort: TTT first, then C4; within each game by avg win rate desc
    df = df.sort_values(["game_order", "avg_win_pct"], ascending=[True, False])

    def fmt_mem(kb):
        if kb == 0 or (isinstance(kb, float) and np.isnan(kb)):
            return "---"
        return f"{kb/1024:.2f}MB" if kb >= 1024 else f"{int(kb)}KB"

    tex = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\small",
        "\\begin{tabularx}{\\linewidth}{"
        ">{\\raggedright\\arraybackslash}p{1.8cm} "
        ">{"
        "\\raggedright\\arraybackslash}p{3.2cm} "
        ">{\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X "
        ">{"
        "\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X "
        ">{"
        "\\centering\\arraybackslash}X}",
        "\\toprule",
        "\\textbf{Game} & \\textbf{Algorithm} & "
        "\\textbf{Win\\%} & \\textbf{Draw\\%} & \\textbf{Loss\\%} & "
        "\\textbf{Latency} & \\textbf{RAM} \\\\",
        "\\midrule",
    ]

    current_game = None
    for _, row in df.iterrows():
        game_label = ("Tic-Tac-Toe" if row["game"] == "tictactoe" else "Connect 4")
        if row["game"] != current_game:
            if current_game is not None:
                tex.append("\\midrule")
            current_game = row["game"]
            disp_game = game_label
        else:
            disp_game = ""

        algo = algo_label(row["agent"])
        win  = pct(row["avg_win_pct"])
        draw = pct(row["avg_draw_pct"])
        loss = pct(row["avg_loss_pct"])
        lat  = f"{row['ms_per_move']:.2f}ms"
        ram  = fmt_mem(row["ram_est_kb"])

        tex.append(f"{disp_game} & {algo} & {win} & {draw} & {loss} & {lat} & {ram} \\\\")

    tex += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]

    with open(f"{OUT_DIR}/{out_file}", "w") as f:
        f.write("\n".join(tex))
    print(f"  ✅ {out_file}")


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating comparison tables...")

    table_vs_default(
        game="tictactoe",
        out_file="ttt_vs_default.tex",
        caption="Algorithm performance vs. Default opponent in Tic-Tac-Toe.",
        label="tab:ttt-vs-default",
    )
    table_vs_default(
        game="connect4",
        out_file="c4_vs_default.tex",
        caption="Algorithm performance vs. Default opponent in Connect 4.",
        label="tab:c4-vs-default",
    )
    table_head_to_head(
        game="tictactoe",
        out_file="ttt_head_to_head.tex",
        caption="Head-to-head pairwise results in Tic-Tac-Toe (Win\\% / Draw\\% / Loss\\%).",
        label="tab:ttt-head-to-head",
    )
    table_head_to_head(
        game="connect4",
        out_file="c4_head_to_head.tex",
        caption="Head-to-head pairwise results in Connect 4 (Win\\% / Draw\\% / Loss\\%).",
        label="tab:c4-head-to-head",
    )
    table_overall(
        out_file="overall_comparison.tex",
        caption="Overall algorithm comparison across both games.",
        label="tab:overall-comparison",
    )

    print(f"\nAll tables written to {OUT_DIR}/")
