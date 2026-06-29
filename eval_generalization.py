import argparse
import csv
import math
import os
import sys
import traceback
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

import eval as eval_module
from eval import (
    run_webster,
    run_rl,
    run_actuated,
    create_webster_sumocfg,
    create_actuated_sumocfg,
    percent_change,
    resolve_experiment_dir,
)
from utils.seed import set_global_seed


SCENARIO_CONFIGS = {
    "symmetric": "sumo/scenario.sumocfg",
    "asymmetric": "sumo/scenario_asymmetric.sumocfg",
    "incident": "sumo/scenario_incident.sumocfg",
    "high_demand": "sumo/scenario_high_demand.sumocfg",
    "low_demand": "sumo/scenario_low_demand.sumocfg",
}

SCENARIO_ROUTE_FILES = {
    "symmetric": "sumo/routes.rou.xml",
    "asymmetric": "sumo/routes_asymmetric.rou.xml",
    "incident": "sumo/routes_incident.rou.xml",
    "high_demand": "sumo/routes_high_demand.rou.xml",
    "low_demand": "sumo/routes_low_demand.rou.xml",
}

METRICS_TO_PLOT = [
    ("avg_waiting_time", "Avg Wait Time (s)"),
    ("p95_waiting_time", "P95 Wait Time (s)"),
    ("avg_time_loss", "Avg Time Loss (s)"),
    ("avg_queue_length", "Avg Queue (veh)"),
]

METRIC_TO_RESULT_COLUMN = {
    "avg_waiting_time": "avg_wait",
    "p95_waiting_time": "p95_wait",
    "avg_time_loss": "avg_time_loss",
    "avg_queue_length": "avg_queue",
    "throughput": "throughput",
    "jain_fairness": "jain_fairness",
}

RESULT_FIELDS = [
    "scenario",
    "seed",
    "controller",
    "avg_wait",
    "p95_wait",
    "avg_time_loss",
    "avg_queue",
    "throughput",
    "jain_fairness",
    "switch_rate",
]

SUMMARY_METRICS = [
    "avg_wait",
    "p95_wait",
    "avg_time_loss",
    "avg_queue",
    "throughput",
    "jain_fairness",
    "switch_rate",
]

CONTROLLERS = ["webster", "actuated", "d3qn"]
CONTROLLER_LABELS = {
    "webster": "Webster",
    "actuated": "Actuated",
    "d3qn": "D3QN",
}
CONTROLLER_COLORS = {
    "webster": "tab:red",
    "actuated": "tab:orange",
    "d3qn": "tab:green",
}


def parse_args():
    """Parse command-line arguments for cross-scenario evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate D3QN generalization across multiple traffic scenarios."
    )
    parser.add_argument(
        "--exp-dir",
        required=True,
        help="Path to experiment folder, e.g. runs/exp_20260503_104131",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[101, 102, 103, 104, 105],
        help="Evaluation seeds (different from training seeds 41-46).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(SCENARIO_CONFIGS.keys()),
        choices=list(SCENARIO_CONFIGS.keys()),
        help="Which scenarios to evaluate.",
    )
    parser.add_argument("--model-name", default="dueling_dqn_final.pth")
    parser.add_argument(
        "--output-dir",
        default="generalization_results",
        help="Directory to save all outputs.",
    )
    return parser.parse_args()


def as_posix_abs(path):
    """Return an absolute path string with forward slashes for SUMO config files."""
    return Path(path).resolve().as_posix()


def write_sumo_config(route_file, output_path):
    """Write a temporary SUMO config with absolute network, route, and additional paths."""
    xsi = "http://www.w3.org/2001/XMLSchema-instance"
    ET.register_namespace("xsi", xsi)
    root = ET.Element(
        "configuration",
        {f"{{{xsi}}}noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/sumoConfiguration.xsd"},
    )

    input_tag = ET.SubElement(root, "input")
    ET.SubElement(input_tag, "net-file", {"value": as_posix_abs("sumo/network.net.xml")})
    ET.SubElement(input_tag, "route-files", {"value": as_posix_abs(route_file)})
    ET.SubElement(
        input_tag,
        "additional-files",
        {
            "value": ",".join(
                [
                    Path("sumo/tls.add.xml").resolve().as_posix(),
                    as_posix_abs("sumo/traffic.add.xml"),
                ]
            )
        },
    )

    time_tag = ET.SubElement(root, "time")
    ET.SubElement(time_tag, "begin", {"value": "0"})
    ET.SubElement(time_tag, "end", {"value": "7200"})

    processing_tag = ET.SubElement(root, "processing")
    ET.SubElement(processing_tag, "time-to-teleport", {"value": "300"})
    ET.SubElement(processing_tag, "waiting-time-memory", {"value": "10000"})

    report_tag = ET.SubElement(root, "report")
    ET.SubElement(report_tag, "verbose", {"value": "false"})
    ET.SubElement(report_tag, "no-step-log", {"value": "true"})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)


def create_scenario_sumocfg(scenario, output_path):
    """Create a temporary base SUMO config for one named traffic scenario."""
    route_file = SCENARIO_ROUTE_FILES[scenario]
    write_sumo_config(route_file, output_path)
    return Path(output_path)


def create_eval_configs(scenario, seed, output_dir):
    """Create temporary RL, Webster, and Actuated configs for a scenario and seed."""
    config_dir = Path(output_dir) / "_configs"
    base_cfg = config_dir / f"{scenario}_seed{seed}.sumocfg"
    webster_cfg = config_dir / f"{scenario}_seed{seed}_webster.sumocfg"
    actuated_cfg = config_dir / f"{scenario}_seed{seed}_actuated.sumocfg"

    create_scenario_sumocfg(scenario, base_cfg)
    create_webster_sumocfg(str(base_cfg), str(webster_cfg))
    create_actuated_sumocfg(str(base_cfg), str(actuated_cfg))

    return {
        "base": base_cfg,
        "webster": webster_cfg,
        "actuated": actuated_cfg,
    }


def validate_inputs(args, model_path):
    """Validate experiment, scenario config, and route file inputs before evaluation."""
    if not model_path.exists():
        sys.exit(f"ERROR: Model not found: {model_path}")

    missing = []
    for scenario in args.scenarios:
        cfg_path = Path(SCENARIO_CONFIGS[scenario])
        route_path = Path(SCENARIO_ROUTE_FILES[scenario])
        if not cfg_path.exists():
            missing.append(str(cfg_path))
        if not route_path.exists():
            missing.append(str(route_path))

    if missing:
        sys.exit("ERROR: Missing scenario files:\n" + "\n".join(f"  - {path}" for path in missing))


def metrics_to_row(scenario, seed, controller, metrics, switch_rate=""):
    """Convert metrics returned by eval.py into one raw CSV result row."""
    return {
        "scenario": scenario,
        "seed": seed,
        "controller": controller,
        "avg_wait": metrics.get("avg_waiting_time", np.nan),
        "p95_wait": metrics.get("p95_waiting_time", np.nan),
        "avg_time_loss": metrics.get("avg_time_loss", np.nan),
        "avg_queue": metrics.get("avg_queue_length", np.nan),
        "throughput": metrics.get("throughput", np.nan),
        "jain_fairness": metrics.get("jain_fairness", np.nan),
        "switch_rate": switch_rate,
    }


def failed_row(scenario, seed, controller):
    """Create a placeholder result row for a failed controller run."""
    return {
        "scenario": scenario,
        "seed": seed,
        "controller": controller,
        "avg_wait": np.nan,
        "p95_wait": np.nan,
        "avg_time_loss": np.nan,
        "avg_queue": np.nan,
        "throughput": np.nan,
        "jain_fairness": np.nan,
        "switch_rate": "",
    }


def log_error(error_log, scenario, seed, controller, exc):
    """Append one evaluation failure with traceback to the error log."""
    with open(error_log, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"scenario={scenario} seed={seed} controller={controller}\n")
        f.write(f"{type(exc).__name__}: {exc}\n")
        f.write(traceback.format_exc())
        f.write("\n")


def safe_close_traci():
    """Best-effort cleanup after a failed SUMO run."""
    try:
        eval_module.traci.close()
    except Exception:
        pass


def run_controller(controller, seed, configs, model_path, eval_dir):
    """Run one controller for a scenario/seed pair and return its metrics."""
    if controller == "webster":
        return run_webster(seed, str(configs["webster"]), eval_dir)
    if controller == "actuated":
        return run_actuated(seed, str(configs["actuated"]), eval_dir)
    if controller == "d3qn":
        return run_rl(model_path, seed, str(configs["base"]), eval_dir)
    raise ValueError(f"Unknown controller: {controller}")


def evaluate_scenario_seed(scenario, seed, model_path, output_dir, error_log):
    """Evaluate all controllers for a single scenario and seed."""
    configs = create_eval_configs(scenario, seed, output_dir)
    eval_dir = Path(output_dir) / scenario / f"seed_{seed}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for controller in CONTROLLERS:
        try:
            set_global_seed(seed)
            print(f"[+] {scenario} | seed={seed} | controller={controller}")
            metrics = run_controller(controller, seed, configs, model_path, eval_dir)
            rows.append(metrics_to_row(scenario, seed, controller, metrics))
        except Exception as exc:
            print(f"[!] Failed: {scenario} seed={seed} controller={controller}: {exc}")
            log_error(error_log, scenario, seed, controller, exc)
            safe_close_traci()
            rows.append(failed_row(scenario, seed, controller))

    return rows


def save_generalization_csv(results, output_dir):
    """Save raw cross-scenario results to generalization_results.csv."""
    output_path = Path(output_dir) / "generalization_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    return output_path


def numeric_values(records, scenario, controller, metric):
    """Collect finite numeric values for one scenario/controller/metric group."""
    values = []
    for row in records:
        if row.get("scenario") != scenario or row.get("controller") != controller:
            continue
        value = row.get(metric, np.nan)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(value):
            values.append(value)
    return values


def records_from_df(df):
    """Convert either a pandas DataFrame or a list of dictionaries into records."""
    if pd is not None and hasattr(df, "to_dict"):
        return df.to_dict("records")
    return list(df)


def compute_summary(df):
    """Compute mean and standard deviation per scenario/controller."""
    records = records_from_df(df)
    summary_rows = []
    scenarios = [scenario for scenario in SCENARIO_CONFIGS if any(row.get("scenario") == scenario for row in records)]

    for scenario in scenarios:
        for controller in CONTROLLERS:
            out = {"scenario": scenario, "controller": controller}
            for metric in SUMMARY_METRICS:
                values = numeric_values(records, scenario, controller, metric)
                if values:
                    out[f"{metric}_mean"] = float(np.mean(values))
                    ddof = 1 if len(values) > 1 else 0
                    out[f"{metric}_std"] = float(np.std(values, ddof=ddof))
                else:
                    out[f"{metric}_mean"] = np.nan
                    out[f"{metric}_std"] = np.nan
            summary_rows.append(out)

    return summary_rows


def save_summary_csv(summary_rows, output_dir):
    """Save aggregated generalization summary statistics to CSV."""
    output_path = Path(output_dir) / "generalization_summary.csv"
    fieldnames = ["scenario", "controller"]
    for metric in SUMMARY_METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return output_path


def get_summary_value(summary_rows, scenario, controller, metric, suffix):
    """Fetch one summary mean or std value from summary rows."""
    key = f"{metric}_{suffix}"
    for row in summary_rows:
        if row.get("scenario") == scenario and row.get("controller") == controller:
            try:
                return float(row.get(key, np.nan))
            except (TypeError, ValueError):
                return np.nan
    return np.nan


def present_scenarios(summary_rows):
    """Return evaluated scenarios in the canonical order."""
    available = {row.get("scenario") for row in summary_rows}
    return [scenario for scenario in SCENARIO_CONFIGS if scenario in available]


def plot_generalization_dashboard(summary_df, output_dir):
    """Plot grouped bar charts for key metrics across scenarios and controllers."""
    summary_rows = records_from_df(summary_df)
    scenarios = present_scenarios(summary_rows)
    x = np.arange(len(scenarios))
    width = 0.24

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(
        "D3QN Generalization: Performance Across Traffic Scenarios",
        fontsize=16,
        fontweight="bold",
    )

    for ax, (metric_key, title) in zip(axes.flat, METRICS_TO_PLOT):
        metric = METRIC_TO_RESULT_COLUMN[metric_key]
        for idx, controller in enumerate(CONTROLLERS):
            offset = (idx - 1) * width
            means = [
                get_summary_value(summary_rows, scenario, controller, metric, "mean")
                for scenario in scenarios
            ]
            stds = [
                get_summary_value(summary_rows, scenario, controller, metric, "std")
                for scenario in scenarios
            ]
            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=4,
                label=CONTROLLER_LABELS[controller],
                color=CONTROLLER_COLORS[controller],
                alpha=0.85,
            )

        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.94))
    plt.tight_layout(rect=(0, 0, 1, 0.90))

    output_path = Path(output_dir) / "generalization_dashboard.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def format_value(value):
    """Format numeric table values while keeping missing values readable."""
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:.2f}"


def format_pct(value):
    """Format percentage deltas for terminal output."""
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:+.1f}%"


def print_summary_table(summary_df):
    """Print terminal summary comparing D3QN against Webster and Actuated."""
    summary_rows = records_from_df(summary_df)
    scenarios = present_scenarios(summary_rows)

    print("\nScenario       | Metric        | Webster | Actuated | D3QN   | vs W    | vs A")
    print("-" * 78)
    for scenario in scenarios:
        for metric_key, _ in METRICS_TO_PLOT:
            metric = METRIC_TO_RESULT_COLUMN[metric_key]
            label = metric.replace("avg_", "avg_").replace("_time_loss", "_loss")
            w_val = get_summary_value(summary_rows, scenario, "webster", metric, "mean")
            a_val = get_summary_value(summary_rows, scenario, "actuated", metric, "mean")
            d_val = get_summary_value(summary_rows, scenario, "d3qn", metric, "mean")
            vs_w = percent_change(w_val, d_val, lower_is_better=True)
            vs_a = percent_change(a_val, d_val, lower_is_better=True)
            print(
                f"{scenario:14s} | {label:13s} | {format_value(w_val):>7s} | "
                f"{format_value(a_val):>8s} | {format_value(d_val):>6s} | "
                f"{format_pct(vs_w):>7s} | {format_pct(vs_a):>7s}"
            )


def main():
    """Run the generalization evaluation workflow."""
    args = parse_args()
    run_dir = resolve_experiment_dir(args.exp_dir)
    model_path = run_dir / args.model_name
    validate_inputs(args, model_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    error_log = output_dir / "generalization_errors.log"
    if error_log.exists():
        error_log.unlink()

    print(f"Experiment: {run_dir}")
    print(f"Model: {model_path}")
    print(f"Scenarios: {', '.join(args.scenarios)}")
    print(f"Seeds: {', '.join(str(seed) for seed in args.seeds)}")
    print(f"Output: {output_dir}")

    results = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            results.extend(evaluate_scenario_seed(scenario, seed, model_path, output_dir, error_log))

    results_csv = save_generalization_csv(results, output_dir)
    df = pd.DataFrame(results) if pd is not None else results
    summary_rows = compute_summary(df)
    summary_csv = save_summary_csv(summary_rows, output_dir)
    dashboard = plot_generalization_dashboard(summary_rows, output_dir)
    print_summary_table(summary_rows)

    print("\nSaved outputs:")
    print(f"  - {results_csv}")
    print(f"  - {summary_csv}")
    print(f"  - {dashboard}")
    if error_log.exists():
        print(f"  - {error_log}")


if __name__ == "__main__":
    main()
