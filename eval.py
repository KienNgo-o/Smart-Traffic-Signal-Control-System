import argparse
import csv
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import torch

from env.sumo_env import SumoEnv
from agent.dqn import D3QNAgent
from utils.seed import set_global_seed
from utils.metrics import evaluate_tripinfo_advanced, evaluate_queue


if "LIBSUMO_AS_TRACI" in os.environ:
    import libsumo as traci
else:
    import traci


SIM_DURATION = 7200.0
MASTER_RESULTS_FILE = Path("master_evaluation_results.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one D3QN experiment folder against Webster baseline."
    )
    parser.add_argument(
        "experiment_dir",
        nargs="?",
        default=None,
        help="Path to a run folder, e.g. runs/exp_20260502_143000",
    )
    parser.add_argument("--seed", type=int, default=42, help="SUMO/evaluation seed.")
    parser.add_argument("--base-cfg", default="sumo/scenario.sumocfg", help="Base SUMO config.")
    parser.add_argument(
        "--model-name",
        default="dueling_dqn_final.pth",
        help="Model filename inside the experiment folder.",
    )
    return parser.parse_args()


def resolve_experiment_dir(experiment_dir):
    if experiment_dir:
        run_dir = Path(experiment_dir)
    else:
        runs = sorted(Path("runs").glob("exp_*"))
        if not runs:
            sys.exit("ERROR: No runs/exp_* folders found. Pass an experiment folder explicitly.")
        run_dir = runs[-1]

    if not run_dir.exists() or not run_dir.is_dir():
        sys.exit(f"ERROR: Experiment folder not found: {run_dir}")

    return run_dir


def create_webster_sumocfg(base_cfg_path, webster_cfg_path):
    """Create a SUMO config that swaps the RL traffic-light plan for Webster."""
    tree = ET.parse(base_cfg_path)
    root = tree.getroot()
    for input_tag in root.findall("input"):
        for add_tag in input_tag.findall("additional-files"):
            val = add_tag.get("value")
            if val:
                add_tag.set("value", val.replace("tls.add.xml", "tls_webster.add.xml"))
    tree.write(webster_cfg_path)


def run_webster(seed, cfg_path, eval_dir):
    """Run the Webster fixed-time baseline and return advanced metrics."""
    set_global_seed(seed)

    tripinfo_file = eval_dir / "tripinfo_webster.xml"
    cmd = [
        "sumo", "-c", cfg_path,
        "--tripinfo-output", str(tripinfo_file.resolve()),
        "--tripinfo-output.write-unfinished", "true",
        "--device.emissions.probability", "1.0",
        "--time-to-teleport", "300",
        "--no-step-log", "true",
        "--seed", str(seed),
    ]

    traci.start(cmd)
    while traci.simulation.getTime() < SIM_DURATION and traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
    traci.close()

    detector_file = eval_dir / "detector_webster.xml"
    if os.path.exists("sumo/detector_output.xml"):
        shutil.copy("sumo/detector_output.xml", detector_file)

    metrics = evaluate_tripinfo_advanced(str(tripinfo_file), sim_duration=SIM_DURATION)
    metrics["avg_queue_length"] = evaluate_queue(str(detector_file))
    return metrics


def run_rl(model_path, seed, cfg_path, eval_dir):
    """Run the trained D3QN agent and return advanced metrics."""
    set_global_seed(seed)

    env = SumoEnv(cfg_path, use_gui=False, max_steps=int(SIM_DURATION))
    tripinfo_file = eval_dir / "tripinfo_rl.xml"
    raw_state, _ = env.reset(seed=seed, options={"tripinfo": str(tripinfo_file.resolve())})

    num_lanes = len(env.controller.lanes)
    num_phases = env.controller.num_phases
    action_dim = env.action_space.n

    agent = D3QNAgent(num_lanes, num_phases, action_dim)
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.policy_net.eval()

    state = agent.preprocess_state(raw_state)
    done = False

    while not done:
        with torch.no_grad():
            q_values = agent.policy_net(state)
            action = q_values.argmax().item()

        raw_next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = agent.preprocess_state(raw_next_state)

    env.close()

    detector_file = eval_dir / "detector_rl.xml"
    if os.path.exists("sumo/detector_output.xml"):
        shutil.copy("sumo/detector_output.xml", detector_file)

    metrics = evaluate_tripinfo_advanced(str(tripinfo_file), sim_duration=SIM_DURATION)
    metrics["avg_queue_length"] = evaluate_queue(str(detector_file))
    return metrics


def percent_change(baseline, candidate, lower_is_better=True):
    if abs(baseline) < 1e-9:
        return 0.0
    if lower_is_better:
        return ((baseline - candidate) / baseline) * 100.0
    return ((candidate - baseline) / baseline) * 100.0


def save_metrics_csv(webster_metrics, rl_metrics, filename):
    keys = sorted(set(webster_metrics) | set(rl_metrics))
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "webster", "d3qn"])
        for key in keys:
            writer.writerow([key, webster_metrics.get(key, 0.0), rl_metrics.get(key, 0.0)])
    print(f"[+] Saved per-run metrics CSV: {filename}")


def append_master_results(run_dir, model_path, seed, webster_metrics, rl_metrics):
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": run_dir.name,
        "experiment_dir": str(run_dir),
        "model_path": str(model_path),
        "seed": seed,
        "avg_wait_webster": webster_metrics.get("avg_waiting_time", 0.0),
        "avg_wait_d3qn": rl_metrics.get("avg_waiting_time", 0.0),
        "avg_wait_improvement_pct": percent_change(
            webster_metrics.get("avg_waiting_time", 0.0),
            rl_metrics.get("avg_waiting_time", 0.0),
            lower_is_better=True,
        ),
        "p95_wait_webster": webster_metrics.get("p95_waiting_time", 0.0),
        "p95_wait_d3qn": rl_metrics.get("p95_waiting_time", 0.0),
        "p95_wait_improvement_pct": percent_change(
            webster_metrics.get("p95_waiting_time", 0.0),
            rl_metrics.get("p95_waiting_time", 0.0),
            lower_is_better=True,
        ),
        "avg_time_loss_webster": webster_metrics.get("avg_time_loss", 0.0),
        "avg_time_loss_d3qn": rl_metrics.get("avg_time_loss", 0.0),
        "avg_time_loss_improvement_pct": percent_change(
            webster_metrics.get("avg_time_loss", 0.0),
            rl_metrics.get("avg_time_loss", 0.0),
            lower_is_better=True,
        ),
        "avg_queue_webster": webster_metrics.get("avg_queue_length", 0.0),
        "avg_queue_d3qn": rl_metrics.get("avg_queue_length", 0.0),
        "avg_queue_improvement_pct": percent_change(
            webster_metrics.get("avg_queue_length", 0.0),
            rl_metrics.get("avg_queue_length", 0.0),
            lower_is_better=True,
        ),
        "throughput_webster": webster_metrics.get("throughput", 0),
        "throughput_d3qn": rl_metrics.get("throughput", 0),
        "throughput_improvement_pct": percent_change(
            webster_metrics.get("throughput", 0.0),
            rl_metrics.get("throughput", 0.0),
            lower_is_better=False,
        ),
        "throughput_per_hour_webster": webster_metrics.get("throughput_per_hour", 0.0),
        "throughput_per_hour_d3qn": rl_metrics.get("throughput_per_hour", 0.0),
        "jain_fairness_webster": webster_metrics.get("jain_fairness", 0.0),
        "jain_fairness_d3qn": rl_metrics.get("jain_fairness", 0.0),
        "co2_per_vehicle_webster": webster_metrics.get("avg_co2_mg_per_completed_vehicle", 0.0),
        "co2_per_vehicle_d3qn": rl_metrics.get("avg_co2_mg_per_completed_vehicle", 0.0),
        "fuel_per_vehicle_webster": webster_metrics.get("avg_fuel_mg_per_completed_vehicle", 0.0),
        "fuel_per_vehicle_d3qn": rl_metrics.get("avg_fuel_mg_per_completed_vehicle", 0.0),
        "nox_per_vehicle_webster": webster_metrics.get("avg_nox_mg_per_completed_vehicle", 0.0),
        "nox_per_vehicle_d3qn": rl_metrics.get("avg_nox_mg_per_completed_vehicle", 0.0),
        "unfinished_webster": webster_metrics.get("unfinished_trips", 0),
        "unfinished_d3qn": rl_metrics.get("unfinished_trips", 0),
    }

    file_exists = MASTER_RESULTS_FILE.exists()
    with open(MASTER_RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[+] Appended master result: {MASTER_RESULTS_FILE}")


def print_summary(webster_metrics, rl_metrics):
    summary_metrics = [
        ("avg_waiting_time", "Avg waiting time", "s", True),
        ("p95_waiting_time", "P95 waiting time", "s", True),
        ("avg_time_loss", "Avg time loss", "s", True),
        ("avg_queue_length", "Avg queue length", "veh", True),
        ("throughput", "Completed throughput", "veh", False),
        ("throughput_per_hour", "Throughput rate", "veh/h", False),
        ("jain_fairness", "Jain fairness", "", False),
        ("avg_co2_mg_per_completed_vehicle", "CO2 per completed vehicle", "mg", True),
        ("avg_fuel_mg_per_completed_vehicle", "Fuel per completed vehicle", "mg", True),
        ("avg_nox_mg_per_completed_vehicle", "NOx per completed vehicle", "mg", True),
        ("unfinished_trips", "Unfinished trips", "veh", True),
    ]

    print("\n" + "=" * 78)
    print("ADVANCED EVALUATION SUMMARY")
    print("=" * 78)
    for key, label, unit, lower_is_better in summary_metrics:
        w_val = webster_metrics.get(key, 0.0)
        r_val = rl_metrics.get(key, 0.0)
        delta = percent_change(w_val, r_val, lower_is_better)
        unit_suffix = f" {unit}" if unit else ""
        print(
            f"{label:34s} | Webster: {w_val:10.3f}{unit_suffix:6s} | "
            f"D3QN: {r_val:10.3f}{unit_suffix:6s} | Improvement: {delta:8.2f}%"
        )


def plot_comparison(webster_metrics, rl_metrics, output_path):
    labels = ["Webster", "D3QN"]
    plot_metrics = [
        ("avg_waiting_time", "Avg Wait", "s", ".1f"),
        ("p95_waiting_time", "P95 Wait", "s", ".1f"),
        ("avg_time_loss", "Time Loss", "s", ".1f"),
        ("avg_queue_length", "Queue", "veh", ".1f"),
        ("throughput_per_hour", "Throughput", "veh/h", ".0f"),
        ("jain_fairness", "Fairness", "", ".3f"),
        ("avg_co2_mg_per_completed_vehicle", "CO2 / Vehicle", "mg", ".0f"),
        ("avg_fuel_mg_per_completed_vehicle", "Fuel / Vehicle", "mg", ".0f"),
        ("avg_nox_mg_per_completed_vehicle", "NOx / Vehicle", "mg", ".1f"),
        ("unfinished_trips", "Unfinished", "veh", ".0f"),
    ]

    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Traffic Light Control: D3QN vs Webster Advanced Metrics", fontsize=16, fontweight="bold")
    colors = ["tab:red", "tab:green"]

    for ax, (key, title, unit, fmt) in zip(axs.flat, plot_metrics):
        data = [webster_metrics.get(key, 0.0), rl_metrics.get(key, 0.0)]
        bars = ax.bar(labels, data, color=colors, alpha=0.85, width=0.55)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(unit)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        max_val = max(max(data), 1.0)
        ax.set_ylim(0, max_val * 1.18)
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval + max_val * 0.03,
                f"{yval:{fmt}}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved evaluation dashboard: {output_path}")


def main():
    args = parse_args()
    run_dir = resolve_experiment_dir(args.experiment_dir)
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / args.model_name
    if not model_path.exists():
        sys.exit(f"ERROR: Model not found: {model_path}")

    webster_cfg = "sumo/scenario_webster.sumocfg"
    create_webster_sumocfg(args.base_cfg, webster_cfg)

    print(f"Experiment: {run_dir}")
    print(f"Model: {model_path}")
    print("=" * 50)
    print("STEP 1: Evaluating Webster baseline...")
    webster_metrics = run_webster(args.seed, webster_cfg, eval_dir)

    print("\nSTEP 2: Evaluating D3QN agent...")
    rl_metrics = run_rl(model_path, args.seed, args.base_cfg, eval_dir)

    print_summary(webster_metrics, rl_metrics)
    save_metrics_csv(webster_metrics, rl_metrics, eval_dir / "advanced_metrics.csv")
    plot_comparison(webster_metrics, rl_metrics, eval_dir / "evaluation_dashboard.png")
    append_master_results(run_dir, model_path, args.seed, webster_metrics, rl_metrics)


if __name__ == "__main__":
    main()
