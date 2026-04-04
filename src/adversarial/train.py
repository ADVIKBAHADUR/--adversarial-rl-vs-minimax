"""Training CLI — train RL agents and save models."""

import argparse
import os
from .games import TicTacToe, Connect4
from .agents import DefaultAgent, QLearningAgent, DQNAgent, RandomAgent
import torch
import pandas as pd
from pathlib import Path

def _make_game(name: str):
    if name == "tictactoe":
        return TicTacToe()
    elif name == "connect4":
        return Connect4()
    raise ValueError(f"Unknown game: {name}")


def main():
    parser = argparse.ArgumentParser(description="Train an RL agent")
    parser.add_argument("--game", choices=["tictactoe", "connect4"], default="tictactoe")
    parser.add_argument("--algo", choices=["qlearning", "dqn"], default="qlearning")
    parser.add_argument("--episodes", type=int, default=None, help="Override training episodes")
    parser.add_argument("--opponent", choices=["default", "random"], default="default")
    parser.add_argument("--output", type=str, default=None, help="Model save path")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epsilon-decay", type=float, default=None)
    parser.add_argument("--epsilon-end", type=float, default=None,
                        help="Minimum epsilon after decay (default 0.05)")
    parser.add_argument("--draw-reward", type=float, default=None, help="Reward for a draw (DQN only)")
    parser.add_argument("--eval-freq", type=int, default=None,
                        help="Evaluate every N episodes (default: auto)")
    parser.add_argument("--discount", type=float, default=None, help="Discount factor gamma (Q-learning)")
    
    # DQN Optimized Defaults
    parser.add_argument("--batch-size", type=int, default=512, help="Balanced for speed and XPU saturation")
    parser.add_argument("--grad-steps", type=int, default=8, help="High-velocity training intensity (4x faster)")
    parser.add_argument("--hidden-sizes", type=str, default="256,256,256,256", help="Wider network for XPU")
    parser.add_argument("--buffer-size", type=int, default=100000, help="More efficient array buffer")
    parser.add_argument("--curriculum-1", type=int, default=5000, help="Min episodes before switching from random to default")
    parser.add_argument("--curriculum-2", type=int, default=20000, help="Min episodes before switching from default to minimax")
    parser.add_argument("--gate-1", type=float, default=0.85, help="Non-loss threshold to pass Stage 1 (Random)")
    parser.add_argument("--gate-2", type=float, default=0.80, help="Non-loss threshold to pass Stage 2 (Default)")
    parser.add_argument("--max-stage", type=int, default=3, help="Maximum curriculum stage to reach (1=Random, 2=Default, 3=Minimax)")
    parser.add_argument("--val-freq", type=int, default=5000, help="Frequency of Gold Standard Minimax validation")
    
    parser.add_argument("--lr-decay", type=float, default=None, help="Learning rate decay per episode")
    parser.add_argument("--lr-min", type=float, default=None, help="Minimum learning rate")
    parser.add_argument("--num-workers", type=int, default=6, help="Parallel CPU workers for simulation (default: 6 for 7-core VM)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--epsilon", type=float, default=None, help="Manual epsilon override")
    parser.add_argument("--target-update", type=int, default=None, help="Target network update frequency")
    parser.add_argument("--no-curriculum", action="store_true", help="Stay vs Default opponent entire time")
    args = parser.parse_args()

    game = _make_game(args.game)

    # Create opponent
    if args.opponent == "default":
        opponent = DefaultAgent(game)
    else:
        opponent = RandomAgent(game)
    opponent.set_game(game)

    # Create agent
    if args.algo == "qlearning":
        from .config import QLearningConfig
        cfg = QLearningConfig()
        if args.episodes: cfg.episodes = args.episodes
        if args.lr: cfg.learning_rate = args.lr
        if args.epsilon_decay: cfg.epsilon_decay = args.epsilon_decay
        if args.discount is not None: cfg.discount = args.discount
        agent = QLearningAgent(game, cfg)
    else:
        from .config import DQNConfig
        cfg = DQNConfig()
        if args.episodes:    cfg.episodes = args.episodes
        if args.lr:          cfg.learning_rate = args.lr
        if args.lr_decay:    cfg.lr_decay = args.lr_decay
        if args.lr_min:      cfg.lr_min = args.lr_min
        if args.epsilon_decay: cfg.epsilon_decay = args.epsilon_decay
        if args.epsilon_end is not None: cfg.epsilon_end = args.epsilon_end
        if args.draw_reward is not None: cfg.draw_reward = args.draw_reward
        
        # 🧪 Optimized High-Intensity Defaults
        cfg.buffer_size = args.buffer_size
        cfg.batch_size = args.batch_size
        cfg.grad_steps_per_episode = args.grad_steps
        
        if args.game == "tictactoe":
            cfg.hidden_sizes = [256, 256, 128]
            cfg.draw_reward = 0.5
            cfg.episodes = 200000
            cfg.grad_steps_per_episode = 1
            cfg.learning_rate = 1e-3
            print("🎮 TTT Optimized DQN engaged: [256, 256, 128], Draw Reward=0.5, Eps=200k, GradSteps=1")
        else:
            cfg.hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]
            cfg.grad_steps_per_episode = args.grad_steps
        
        if args.draw_reward is not None: cfg.draw_reward = args.draw_reward
        if args.episodes is not None: cfg.episodes = args.episodes
        if args.lr: cfg.learning_rate = args.lr

        agent = DQNAgent(game, cfg)
        if args.target_update:
            agent.cfg.target_update_freq = args.target_update
            print(f"💡 Target update frequency set to {args.target_update}")

    agent.set_game(game)

    print(f"🎓 Training {agent.name} on {game.name}")
    print(f"   Opponent: {opponent.name}")
    print(f"   Episodes: {cfg.episodes}")
    if args.algo == "dqn":
        print(f"   Rewards: Win=1.0, Draw={cfg.draw_reward}, Loss=-1.0")
    print()

    if args.output:
        base_save_path = args.output
    else:
        # Organise into models/{game}/{algo}/model
        base_save_path = f"models/{args.game}/{args.algo}/model"
        
    ext = "pkl" if args.algo == "qlearning" else "pt"
    os.makedirs(os.path.dirname(base_save_path), exist_ok=True)

    # Save initial untrained model
    agent.save(f"{base_save_path}_0.{ext}")
    print(f"💾 Saved initial checkpoint: {base_save_path}_0.{ext}")

    # chunk_size = how many episodes between each evaluation.
    default_chunk = max(500, cfg.episodes // 100)
    chunk_size = args.eval_freq if args.eval_freq else min(default_chunk, 5000)
    
    # Resume from checkpoint if provided
    episodes_run = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=agent.device, weights_only=False)
        agent.load(args.resume)
        episodes_run = checkpoint.get("episodes_total", 0)
        
        # Fallback to filename parsing if episodes_total is missing
        if episodes_run == 0:
            filename = os.path.basename(args.resume)
            try:
                parts = filename.replace(".", "_").split("_")
                for p in parts:
                    if p.isdigit():
                        episodes_run = int(p)
                        break
            except: pass
        print(f"🔄 Resuming from {args.resume} (Episode {episodes_run})")
        if args.epsilon is not None:
            agent.epsilon = args.epsilon
            print(f"💡 Epsilon manually set to {agent.epsilon}")

    metrics_history = {
        "wins": [], "draws": [], "losses": [], "epsilon": [],
        "val_wins": [], "val_draws": [], "val_losses": [], "val_episodes": [],
        "vtable_size": []
    }
    eval_episodes = []
    opponent_history = []

    # 📊 Resume metrics history if available
    results_dir = Path("results")
    csv_path = results_dir / f"training_log_{agent.name.lower()}_{game.name.lower()}.csv"
    if args.resume and csv_path.exists():
        try:
            df_old = pd.read_csv(csv_path)
            df_old = df_old[df_old["episode"] <= episodes_run]
            eval_episodes = df_old["episode"].tolist()
            metrics_history["wins"] = df_old["win_pct"].tolist()
            metrics_history["draws"] = df_old["draw_pct"].tolist()
            metrics_history["losses"] = df_old["loss_pct"].tolist()
            metrics_history["epsilon"] = df_old["epsilon"].tolist()
            if "vtable_size" in df_old.columns:
                metrics_history["vtable_size"] = df_old["vtable_size"].tolist()
            opponent_history = df_old["opponent"].tolist()
            if "val_win_pct" in df_old.columns:
                val_rows = df_old[df_old["val_win_pct"].notna()]
                metrics_history["val_wins"] = val_rows["val_win_pct"].tolist()
                metrics_history["val_episodes"] = val_rows["episode"].tolist()
            print(f"📊 Resumed {len(eval_episodes)} metric points from CSV")
        except Exception as e:
            print(f"⚠️ Could not resume metrics from CSV: {e}")
    
    from .agents import MinimaxAgent
    from .config import MinimaxConfig
    m_cfg_val = MinimaxConfig(max_depth=2, use_alpha_beta=True)
    val_opponent = MinimaxAgent(game, m_cfg_val)
    val_opponent.set_game(game)
    
    from .tournament import run_match
    from .plotting import plot_training_curve
    import time

    t0 = time.perf_counter()

    best_score = (-1.0, -1.0)   # (non_loss_pct, win_pct)
    best_ep = 0
    best_save_path = f"{base_save_path}_best.{ext}"

    current_stage = 1
    last_eval_non_loss = 0.0

    while episodes_run < cfg.episodes:
        # ── Performance-Gated Curriculum Logic ──
        if not args.no_curriculum:
            # Stage 1 -> 2: vs Random until 98% non-loss AND at least curriculum_1 episodes
            if current_stage == 1:
                if episodes_run >= args.curriculum_1 and last_eval_non_loss >= args.gate_1:
                    print(f"\n✅ passed stage 1 ({last_eval_non_loss:.1%} non-loss)")
                    current_stage = 2
                
                if not isinstance(opponent, RandomAgent):
                    print(f"\n🎓 Stage 1: VS RANDOM")
                    opponent = RandomAgent(game)
                    opponent.set_game(game)

            # Stage 2 -> 3: vs Default until 95% non-loss AND at least curriculum_2 episodes
            if current_stage == 2:
                if episodes_run >= args.curriculum_2 and last_eval_non_loss >= args.gate_2 and args.max_stage >= 3:
                    print(f"\n✅ passed stage 2 ({last_eval_non_loss:.1%} non-loss)")
                    current_stage = 3
                
                if not isinstance(opponent, DefaultAgent):
                    print(f"\n🎓 Stage 2: VS DEFAULT")
                    opponent = DefaultAgent(game)
                    opponent.set_game(game)

            if current_stage == 3:
                from .agents import MinimaxAgent
                from .config import MinimaxConfig
                if not isinstance(opponent, MinimaxAgent) or opponent.cfg.max_depth != 9:
                    print(f"\n🎓 Stage 3: VS PERFECT MINIMAX (AB, d=9)")
                    m_cfg = MinimaxConfig(max_depth=9, use_alpha_beta=True)
                    opponent = MinimaxAgent(game, m_cfg)
                    opponent.set_game(game)
        else:
            # Fixed opponent if no curriculum
            if episodes_run == 0 or not isinstance(opponent, DefaultAgent):
                opponent = DefaultAgent(game)
                opponent.set_game(game)
            
        current_chunk = min(chunk_size, cfg.episodes - episodes_run)
        
        # Train for a chunk of episodes
        # Pass the global episode count so epsilon decays properly
        agent.episodes_total = episodes_run # Track globally for saving
        def progress_callback(ep, metrics):
            print(".", end="", flush=True)
            
        # Use parallel workers if specified (currently only optimized for DQN)
        train_kwargs = {"num_workers": args.num_workers, "callback": progress_callback} if args.algo == "dqn" else {}
        agent.train(game, opponent, current_chunk, start_ep=episodes_run, total_eps=cfg.episodes, **train_kwargs)
        print(" Done.") # Newline after chunk of dots
        episodes_run += current_chunk
        
        # Temporarily disable exploration for evaluation
        original_training = getattr(agent, "_training", False)
        original_epsilon = getattr(agent, "epsilon", 0.0)
        
        if hasattr(agent, "_training"): agent._training = False
        if hasattr(agent, "epsilon"): agent.epsilon = 0.0
            
        print(f"\n🧪 Evaluating at episode {episodes_run}...")
        eval_res = run_match(game, agent, opponent, n_games=100, swap_sides=True, verbose=False)
        
        last_eval_non_loss = (eval_res["p1_win_pct"] + eval_res["draw_pct"]) / 100.0
        
        metrics_history["wins"].append(eval_res["p1_win_pct"])
        metrics_history["draws"].append(eval_res["draw_pct"])
        metrics_history["losses"].append(eval_res["p2_win_pct"])
        metrics_history["epsilon"].append(original_epsilon)
        
        # Track V-table size if applicable
        if hasattr(agent, "v_table"):
            metrics_history["vtable_size"].append(len(agent.v_table))
        else:
            metrics_history["vtable_size"].append(None)
            
        opponent_history.append(opponent.name)
        eval_episodes.append(episodes_run)
        
        # Restore training state
        if hasattr(agent, "_training"): agent._training = original_training
        if hasattr(agent, "epsilon"): agent.epsilon = original_epsilon
            
        current_lr = 0.0
        if args.algo == "dqn" and hasattr(agent, "optimiser"):
            current_lr = agent.optimiser.param_groups[0]['lr']
            
        print(f"   Win: {eval_res['p1_win_pct']:.1f}%, Draw: {eval_res['draw_pct']:.1f}%, Loss: {eval_res['p2_win_pct']:.1f}%, Epsilon: {original_epsilon:.3f}", end="")
        
        # 🧪 Gold Standard Validation (Curriculum-Independent)
        if episodes_run % args.val_freq == 0:
            print(f"\n🏆 Gold Standard Validation (vs Minimax AB d=2)...")
            val_res = run_match(game, agent, val_opponent, n_games=50, swap_sides=True, verbose=False)
            metrics_history["val_wins"].append(val_res["p1_win_pct"])
            metrics_history["val_draws"].append(val_res["draw_pct"])
            metrics_history["val_losses"].append(val_res["p2_win_pct"])
            metrics_history["val_episodes"].append(episodes_run)
            print(f"   Val Win: {val_res['p1_win_pct']:.1f}%, Val Draw: {val_res['draw_pct']:.1f}%, Val Loss: {val_res['p2_win_pct']:.1f}%")
        if args.algo == "dqn":
            print(f", LR: {current_lr:g}", end="")
        if args.algo == "qlearning" and hasattr(agent, "v_table"):
            print(f", V-table: {len(agent.v_table):,} states", end="")
        print()
        
        # Best-checkpoint tracking: (non_loss, win_pct) tuple — win_pct breaks ties
        non_loss = eval_res["p1_win_pct"] + eval_res["draw_pct"]
        win_pct = eval_res["p1_win_pct"]
        score = (non_loss, win_pct)
        if score > best_score:
            best_score = score
            best_ep = episodes_run
            agent.save(best_save_path)
            print(f"   ⭐ New best! ({non_loss:.1f}% non-loss, {win_pct:.1f}% win) → saved {best_save_path}")
        
        # Save periodic checkpoint
        save_path = f"{base_save_path}_{episodes_run}.{ext}"
        agent.save(save_path)

        # 📄 Save incremental training metrics to CSV
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / f"training_log_{agent.name.lower()}_{game.name.lower()}.csv"
        
        df_log_data = {
            "episode": eval_episodes,
            "win_pct": metrics_history["wins"],
            "draw_pct": metrics_history["draws"],
            "loss_pct": metrics_history["losses"],
            "epsilon": metrics_history["epsilon"],
            "vtable_size": metrics_history["vtable_size"],
            "opponent": opponent_history
        }
        # Add validation if available
        if metrics_history["val_episodes"]:
            # We align validation by padding/matching episodes (simplified for CSV storage)
            df_log_data["val_win_pct"] = [(metrics_history["val_wins"][metrics_history["val_episodes"].index(e)] if e in metrics_history["val_episodes"] else None) for e in eval_episodes]

        df_log = pd.DataFrame(df_log_data)
        df_log.to_csv(csv_path, index=False)

    print(f"\n✅ Training complete in {time.perf_counter() - t0:.1f}s")
    print(f"🏆 Best model: episode {best_ep} with {best_score[0]:.1f}% non-loss, {best_score[1]:.1f}% win rate → {best_save_path}")
    
    # Save training metrics to CSV
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"training_log_{agent.name.lower()}_{game.name.lower()}.csv"
    
    df_log = pd.DataFrame({
        "episode": eval_episodes,
        "win_pct": metrics_history["wins"],
        "draw_pct": metrics_history["draws"],
        "loss_pct": metrics_history["losses"],
        "epsilon": metrics_history["epsilon"],
        "vtable_size": metrics_history["vtable_size"],
        "opponent": opponent_history
    })
    df_log.to_csv(csv_path, index=False)

    # 📈 Generate convergence plot with Epsilon/V-table tracking
    clean_game_name = game.name.split('-')[0].replace('3x3', '').replace('6x7', '').strip()
    if "tictactoe" in clean_game_name.lower():
        game_title_name = "Tic-Tac-Toe"
    elif "connect4" in clean_game_name.lower():
        game_title_name = "Connect 4"
    else:
        game_title_name = clean_game_name.title()

    plot_title = f"{game_title_name} {agent.name.replace('-', ' ').title()} Training Convergence"
    algo_filename = f"training_curve_{agent.name.lower().replace(' ', '_')}_{game.name.lower()}"
    
    # Use raw data (no smoothing) for Q-Learning as requested
    is_q = agent.name.lower() == "q-learning"
    
    plot_training_curve(eval_episodes, metrics_history, opponent_history, 
                        algo_name=agent.name, 
                        epsilon_history=metrics_history["epsilon"],
                        vtable_history=metrics_history.get("vtable_size"),
                        title=plot_title,
                        smooth=not is_q,
                        filename=algo_filename)
    print(f"📄 Saved training log and plot to {csv_path} and figures/{algo_filename}.png")




if __name__ == "__main__":
    main()
