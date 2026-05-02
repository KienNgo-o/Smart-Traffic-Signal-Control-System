import os
import sys
import shutil
import csv
import torch
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from env.sumo_env import SumoEnv
from agent.dqn import D3QNAgent
from utils.seed import set_global_seed
from utils.metrics import evaluate_tripinfo_advanced, evaluate_queue


if 'LIBSUMO_AS_TRACI' in os.environ:
    import libsumo as traci
else:
    import traci


SIM_DURATION = 7200.0


def create_webster_sumocfg(base_cfg_path, webster_cfg_path):
    """Create a SUMO config that swaps the RL traffic-light plan for Webster."""
    tree = ET.parse(base_cfg_path)
    root = tree.getroot()
    for input_tag in root.findall('input'):
        for add_tag in input_tag.findall('additional-files'):
            val = add_tag.get('value')
            if val:
                add_tag.set('value', val.replace('tls.add.xml', 'tls_webster.add.xml'))
    tree.write(webster_cfg_path)


def run_webster(seed, cfg_path):
    """Run the Webster fixed-time baseline and return advanced metrics."""
    set_global_seed(seed)

    tripinfo_file = "sumo/tripinfo_webster.xml"
    cmd = [
        "sumo", "-c", cfg_path,
        "--tripinfo-output", tripinfo_file,
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

    detector_file = "sumo/detector_webster.xml"
    if os.path.exists("sumo/detector_output.xml"):
        shutil.copy("sumo/detector_output.xml", detector_file)

    metrics = evaluate_tripinfo_advanced(tripinfo_file, sim_duration=SIM_DURATION)
    metrics["avg_queue_length"] = evaluate_queue(detector_file)
    return metrics


def run_rl(model_path, seed, cfg_path):
    """Run the trained D3QN agent and return advanced metrics."""
    set_global_seed(seed)

    env = SumoEnv(cfg_path, use_gui=False, max_steps=int(SIM_DURATION))
    tripinfo_file = "sumo/tripinfo_rl.xml"
    raw_state, _ = env.reset(seed=seed, options={"tripinfo": tripinfo_file})

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

    detector_file = "sumo/detector_rl.xml"
    if os.path.exists("sumo/detector_output.xml"):
        shutil.copy("sumo/detector_output.xml", detector_file)

    metrics = evaluate_tripinfo_advanced(tripinfo_file, sim_duration=SIM_DURATION)
    metrics["avg_queue_length"] = evaluate_queue(detector_file)
    return metrics


def percent_change(baseline, candidate, lower_is_better=True):
    if abs(baseline) < 1e-9:
        return 0.0
    if lower_is_better:
        return ((baseline - candidate) / baseline) * 100.0
    return ((candidate - baseline) / baseline) * 100.0


def save_metrics_csv(webster_metrics, rl_metrics, filename="eval/advanced_metrics.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    keys = sorted(set(webster_metrics) | set(rl_metrics))
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "webster", "d3qn"])
        for key in keys:
            writer.writerow([key, webster_metrics.get(key, 0.0), rl_metrics.get(key, 0.0)])
    print(f"[+] Saved advanced metrics CSV: {filename}")


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
        print(f"{label:34s} | Webster: {w_val:10.3f}{unit_suffix:6s} | D3QN: {r_val:10.3f}{unit_suffix:6s} | Improvement: {delta:8.2f}%")


def plot_comparison(webster_metrics, rl_metrics):
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

    os.makedirs("eval", exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig("eval/evaluation_dashboard.png", dpi=300)
    print("[+] Saved evaluation dashboard: eval/evaluation_dashboard.png")


if __name__ == "__main__":
    SEED = 42
    BASE_CFG = "sumo/scenario.sumocfg"
    WEBSTER_CFG = "sumo/scenario_webster.sumocfg"
    MODEL_PATH = "models/d3qn_per_final.pth"

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"ERROR: Model not found at {MODEL_PATH}. Please run train.py first.")

    create_webster_sumocfg(BASE_CFG, WEBSTER_CFG)

    print("=" * 50)
    print("STEP 1: Evaluating Webster baseline...")
    webster_metrics = run_webster(SEED, WEBSTER_CFG)

    print("\nSTEP 2: Evaluating D3QN agent...")
    rl_metrics = run_rl(MODEL_PATH, SEED, BASE_CFG)

    print_summary(webster_metrics, rl_metrics)
    save_metrics_csv(webster_metrics, rl_metrics)
    plot_comparison(webster_metrics, rl_metrics)
