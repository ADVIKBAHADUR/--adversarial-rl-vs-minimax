"""Generate per-agent LaTeX tables from agent_summary.csv.

Playing Strength section: Win Rate | Draw Rate | Loss Rate
Resource section:         Latency  | Weights   | RAM (Est) | Meta
Learning Dynamics (RL):  Sample Efficiency | Episodes Trained | Exploration Decay
"""

import pandas as pd
import numpy as np
import os

OUT_DIR = "results/latex_tables"
os.makedirs(OUT_DIR, exist_ok=True)

agent_df = pd.read_csv("results/agent_summary.csv")


def get_rl_dynamics(agent_name, game):
    """
    Parse the training log for an RL agent to find:
    - Sample Efficiency: First episode where win_pct >= 90
    - Exploration Decay: Final epsilon value
    """
    algo_slug = "q-learning" if "Q-Learning" in agent_name else "dqn"
    game_slug = "tictactoe-3x3" if game == "tictactoe" else "connect4-6x7"
    log_path = f"results/training_log_{algo_slug}_{game_slug}.csv"

    if not os.path.exists(log_path):
        return "N/A", "N/A"

    try:
        log_df = pd.read_csv(log_path)
        # Sample Efficiency (90% threshold)
        over_90 = log_df[log_df['win_pct'] >= 90.0]
        if not over_90.empty:
            sample_eff = f"{over_90.iloc[0]['episode']:,}"
        else:
            max_win = log_df['win_pct'].max()
            sample_eff = f"> 100k (Peak {max_win:.1f}\\%)"

        # Exploration Decay — final epsilon
        min_eps = log_df['epsilon'].min()
        exp_decay = f"{min_eps:.3f}"

        return sample_eff, exp_decay
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
        return "N/A", "N/A"


def fmt_mem(kb):
    if kb == 0:
        return "---"
    return f"{kb/1024:.2f} MB" if kb >= 1024 else f"{kb:.0f} KB"


def fmt_episodes(n):
    """Format an episode count nicely."""
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    n = int(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n//1000}k"
    return str(n)


def format_table(agent_name, game, metrics):
    """Format a compact 4-column LaTeX table for a single agent."""
    algo_clean = agent_name.replace("αβ", "Alpha-Beta").replace("vanilla", "Vanilla")
    game_clean = "Tic-Tac-Toe" if game == "tictactoe" else "Connect 4"

    tex = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{Performance metrics for {algo_clean} in {game_clean}.}}",
        f"\\label{{tab:metrics-{algo_clean.lower().replace(' ', '-')}-{game}}}",
        "\\renewcommand{\\arraystretch}{1.0}",
        "\\small",
        "\\begin{tabularx}{\\linewidth}{>{"
        "\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X "
        ">{"
        "\\centering\\arraybackslash}X >{\\centering\\arraybackslash}X}",
        "\\toprule",
    ]

    # ── Playing Strength ──────────────────────────────────────────────────────
    tex += [
        "\\multicolumn{4}{c}{\\textbf{Playing Strength \\& Strategy}} \\\\",
        "\\midrule",
        "Win Rate & Draw Rate & Loss Rate & Latency (ms) \\\\",
        f"{metrics['win_rate']:.1%} & {metrics['draw_rate']:.1%} & "
        f"{metrics['loss_rate']:.1%} & {metrics['time_ms']:.2f} \\\\",
        "\\midrule",
    ]

    # ── Resource & Storage ────────────────────────────────────────────────────
    is_rl = "Q-Learning" in agent_name or "DQN" in agent_name

    tex += [
        "\\multicolumn{4}{c}{\\textbf{Resource \\& Storage Efficiency}} \\\\",
        "\\midrule",
        "\\multicolumn{2}{c}{Weights (disk)} & \\multicolumn{2}{c}{RAM (Est.)} \\\\",
        f"\\multicolumn{{2}}{{c}}{{{metrics['weights_footprint']}}} & "
        f"\\multicolumn{{2}}{{c}}{{{metrics['ram_footprint']}}} \\\\",
    ]

    if not is_rl:
        meta = "Alpha-Beta Pruning" if "Alpha-Beta" in algo_clean else "Vanilla (no pruning)"
        tex += [
            "\\midrule",
            f"\\multicolumn{{4}}{{c}}{{\\textit{{Search strategy: {meta}}}}} \\\\",
        ]

    # ── Learning Dynamics (RL only) ───────────────────────────────────────────
    if is_rl:
        tex += [
            "\\midrule",
            "\\multicolumn{4}{c}{\\textbf{Learning Dynamics (RL Only)}} \\\\",
            "\\midrule",
            "Sample Eff. & Episodes & Expl. Decay & Type \\\\",
            f"{metrics['sample_eff']} & {metrics['episodes_trained']} & "
            f"{metrics['exp_decay']} & {metrics['rl_type']} \\\\",
        ]

        # Extra line: V-table size for Q-Learning
        if "Q-Learning" in agent_name:
            v_den = metrics.get('visit_density', 'N/A')
            density_str = f"{v_den:.4e}" if isinstance(v_den, float) else str(v_den)
            tex += [
                "\\midrule",
                f"\\multicolumn{{4}}{{c}}{{\\textit{{V-table density: "
                f"{density_str} (explored / total states)}}}} \\\\",
            ]

    tex += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]
    return "\n".join(tex)


# ── Parse and generate ────────────────────────────────────────────────────────
total_states_c4  = 4_531_985_219_092
total_states_ttt = 5_478

generated = 0
for idx, row in agent_df.iterrows():
    agent = row['agent']
    game  = row['game']

    if agent in ["Random", "Default"]:
        continue

    metrics = {
        'win_rate':          row['avg_win_pct']  / 100.0,
        'draw_rate':         row['avg_draw_pct'] / 100.0,
        'loss_rate':         row['avg_loss_pct'] / 100.0,
        'time_ms':           row['ms_per_move'],
        'weights_footprint': fmt_mem(row['model_kb']),
        'ram_footprint':     fmt_mem(row['ram_est_kb']),
    }

    if "Q-Learning" in agent or "DQN" in agent:
        sample_eff, exp_decay = get_rl_dynamics(agent, game)
        metrics['sample_eff']      = sample_eff
        metrics['exp_decay']       = exp_decay
        metrics['rl_type']         = "Tabular Q" if "Q-Learning" in agent else "Deep Q (DQN)"

        # Episodes trained — from agent_summary if present
        ep_trained = row.get('episodes_trained', None)
        if pd.isna(ep_trained) if isinstance(ep_trained, float) else ep_trained is None:
            ep_trained = None
        metrics['episodes_trained'] = fmt_episodes(ep_trained)

        if "Q-Learning" in agent:
            vstates = row.get('vtable_states', None)
            if pd.notna(vstates) and vstates:
                total = total_states_ttt if game == 'tictactoe' else total_states_c4
                metrics['visit_density'] = int(vstates) / total

    tex_str = format_table(agent, game, metrics)

    safe = (agent.replace("(", "_").replace(")", "")
                 .replace("=", "_").replace(",", "_")
                 .replace("αβ", "ab").replace(" ", "_"))
    path = f"{OUT_DIR}/{game}_{safe}.tex"
    with open(path, "w") as f:
        f.write(tex_str)
    generated += 1

print(f"Generated {generated} LaTeX files in {OUT_DIR}")
