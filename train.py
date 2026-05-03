import csv
import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from env.sumo_env import SumoEnv
from agent.dqn import D3QNAgent
from agent.replay_buffer import PrioritizedReplayBuffer
from utils.seed import set_global_seed

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def create_run_dir(base_dir="runs"):
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_path / f"exp_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = base_path / f"exp_{timestamp}_{suffix:02d}"
        suffix += 1

    run_dir.mkdir(parents=True)
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train a D3QN traffic-light agent.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for this run.")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=300,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help="Allow nondeterministic backend algorithms for speed.",
    )
    return parser.parse_args()


def save_hyperparameters(run_dir, hparams):
    def to_jsonable(value):
        if isinstance(value, dict):
            return {str(k): to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [to_jsonable(v) for v in value]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        return value

    with open(run_dir / "hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(hparams), f, indent=2, sort_keys=True)


def save_reward_plot(run_dir, episode_rewards, window_size):
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, label="Raw Reward", alpha=0.3, color="blue")

    if len(episode_rewards) >= window_size:
        smoothed_rewards = np.convolve(
            episode_rewards,
            np.ones(window_size) / window_size,
            mode="valid",
        )
        plt.plot(
            range(window_size - 1, len(episode_rewards)),
            smoothed_rewards,
            label=f"Moving Average (Window={window_size})",
            color="red",
            linewidth=2,
        )

    plt.xlabel("Episode", fontsize=12, fontweight="bold")
    plt.ylabel("Total Reward", fontsize=12, fontweight="bold")
    plt.title("D3QN Training Reward Convergence", fontsize=14, pad=15)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(run_dir / "reward_convergence_plot.png", dpi=300)
    plt.close()


def train(args=None):
    if args is None:
        args = parse_args()

    run_dir = create_run_dir()

    hparams = {
        "seed": args.seed,
        "deterministic": not args.no_deterministic,
        "sumocfg_file": "sumo/scenario.sumocfg",
        "use_gui": False,
        "max_steps": 7200,
        "reward_gamma": 0.99,
        "gamma": 0.99,
        "learning_rate": 5e-4,
        "batch_size": 128,
        "target_update_freq": 5,
        "num_episodes": args.num_episodes,
        "learning_starts": 2000,
        "train_every": 1,
        "replay_buffer_size": 100000,
        "per_alpha": 0.6,
        "per_beta_start": 0.4,
        "epsilon_start": 1.0,
        "epsilon_end": 0.03,
        "epsilon_decay_target_fraction": 0.9,
        "estimated_action_interval_seconds": 12.0,
        "checkpoint_freq": 20,
        "grad_clip_norm": 1.0,
        "reward_plot_window": 10,
    }

    hparams["epsilon_decay"] = (
        hparams["epsilon_end"] / hparams["epsilon_start"]
    ) ** (1.0 / (hparams["num_episodes"] * hparams["epsilon_decay_target_fraction"]))

    max_steps_per_ep = hparams["max_steps"] / hparams["estimated_action_interval_seconds"]
    total_expected_updates = max(
        1,
        int(max_steps_per_ep * hparams["num_episodes"] - hparams["learning_starts"]),
    )
    hparams["per_beta_increment"] = (
        1.0 - hparams["per_beta_start"]
    ) / total_expected_updates

    save_hyperparameters(run_dir, hparams)

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard")) if SummaryWriter else None

    set_global_seed(hparams["seed"], deterministic=hparams["deterministic"])
    env = SumoEnv(
        sumocfg_file=hparams["sumocfg_file"],
        use_gui=hparams["use_gui"],
        max_steps=hparams["max_steps"],
        reward_gamma=hparams["reward_gamma"],
    )
    env.action_space.seed(hparams["seed"])

    raw_state, _ = env.reset(seed=hparams["seed"])
    num_lanes = len(env.controller.lanes)
    num_phases = env.controller.num_phases
    action_dim = env.action_space.n

    hparams.update({
        "num_lanes": num_lanes,
        "num_phases": num_phases,
        "action_dim": action_dim,
        "run_dir": str(run_dir),
    })
    save_hyperparameters(run_dir, hparams)

    agent = D3QNAgent(
        num_lanes=num_lanes,
        num_phases=num_phases,
        action_dim=action_dim,
        lr=hparams["learning_rate"],
        gamma=hparams["gamma"],
    )

    buffer = PrioritizedReplayBuffer(
        max_size=hparams["replay_buffer_size"],
        alpha=hparams["per_alpha"],
        beta=hparams["per_beta_start"],
        beta_increment_per_sampling=hparams["per_beta_increment"],
    )

    log_path = run_dir / "train_log.csv"
    episode_rewards = []
    epsilon = hparams["epsilon_start"]

    print(f"Run directory: {run_dir}")
    print(f"Training on device: {agent.device}")

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            "episode",
            "steps",
            "sim_seconds",
            "total_reward",
            "epsilon",
            "average_loss",
            "updates",
            "buffer_size",
        ])

        for episode in range(1, hparams["num_episodes"] + 1):
            raw_state, _ = env.reset(seed=hparams["seed"] + episode)
            state = agent.preprocess_state(raw_state)

            total_reward = 0.0
            step_count = 0
            update_count = 0
            episode_losses = []
            done = False

            while not done:
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = agent.policy_net(state)
                        action = q_values.argmax().item()

                raw_next_state, reward, terminated, truncated, info = env.step(action)
                duration = info["action_duration"]
                done = terminated or truncated

                next_state = agent.preprocess_state(raw_next_state)
                buffer.add(state, action, reward, next_state, done, duration)

                state = next_state
                total_reward += reward
                step_count += 1

                if (
                    buffer.tree.n_entries >= hparams["learning_starts"]
                    and step_count % hparams["train_every"] == 0
                ):
                    batch, idxs, is_weights = buffer.sample(hparams["batch_size"])
                    states, actions, rewards, next_states, dones, durations = batch

                    states_t = torch.cat(states, dim=0)
                    next_states_t = torch.cat(next_states, dim=0)
                    actions_t = torch.LongTensor(actions).to(agent.device)
                    rewards_t = torch.FloatTensor(rewards).to(agent.device)
                    dones_t = torch.FloatTensor(dones).to(agent.device)
                    durations_t = torch.FloatTensor(durations).to(agent.device)
                    is_weights_t = torch.FloatTensor(is_weights).to(agent.device)

                    loss, td_errors = agent.compute_loss(
                        states_t,
                        actions_t,
                        rewards_t,
                        next_states_t,
                        dones_t,
                        durations_t,
                        is_weights_t,
                    )

                    agent.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.policy_net.parameters(),
                        max_norm=hparams["grad_clip_norm"],
                    )
                    agent.optimizer.step()
                    buffer.update_priorities(idxs, td_errors)

                    episode_losses.append(loss.item())
                    update_count += 1

            epsilon = max(hparams["epsilon_end"], epsilon * hparams["epsilon_decay"])
            avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            episode_rewards.append(total_reward)

            if episode % hparams["target_update_freq"] == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

            log_writer.writerow([
                episode,
                step_count,
                env.current_step,
                f"{total_reward:.6f}",
                f"{epsilon:.6f}",
                f"{avg_loss:.8f}",
                update_count,
                buffer.tree.n_entries,
            ])
            log_file.flush()

            if writer:
                writer.add_scalar("train/total_reward", total_reward, episode)
                writer.add_scalar("train/epsilon", epsilon, episode)
                writer.add_scalar("train/avg_loss", avg_loss, episode)
                writer.add_scalar("train/steps", step_count, episode)
                writer.add_scalar("train/buffer_size", buffer.tree.n_entries, episode)

            print(
                f"Episode {episode:03d} | Steps: {step_count:04d} | "
                f"Reward: {total_reward:9.2f} | AvgLoss: {avg_loss:.6f} | "
                f"Epsilon: {epsilon:.3f}"
            )

            if episode % hparams["checkpoint_freq"] == 0:
                checkpoint_path = run_dir / f"dueling_dqn_ep{episode}.pth"
                torch.save(agent.policy_net.state_dict(), checkpoint_path)

    final_model_path = run_dir / "dueling_dqn_final.pth"
    torch.save(agent.policy_net.state_dict(), final_model_path)

    save_reward_plot(run_dir, episode_rewards, hparams["reward_plot_window"])

    if writer:
        writer.close()
    env.close()

    print("Training complete.")
    print(f"Final model: {final_model_path}")
    print(f"Train log: {log_path}")
    print(f"Hyperparameters: {run_dir / 'hyperparameters.json'}")


if __name__ == "__main__":
    train(parse_args())
