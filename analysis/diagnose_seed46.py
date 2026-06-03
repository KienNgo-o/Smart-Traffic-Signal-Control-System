"""Diagnostic analysis for the seed 46 over-switching hypothesis.

This module checks whether seed 46 has a higher action switch rate than the
other training seeds and whether that behavior is associated with worse
time-loss improvements in evaluation.

Usage:
    python -m analysis.diagnose_seed46
    python -m analysis.diagnose_seed46 --runs-dir runs --eval-csv master_evaluation_results.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


TARGET_SEED = 46


def _read_seed_from_hyperparameters(run_dir: Path) -> int | None:
    """Read the seed value from a run's hyperparameters.json file."""
    hyperparameters_path = run_dir / "hyperparameters.json"
    if not hyperparameters_path.exists():
        return None

    try:
        with hyperparameters_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    seed_value = data.get("seed")
    if seed_value is None:
        return None
    try:
        return int(seed_value)
    except (TypeError, ValueError):
        return None


def load_train_logs(runs_dir="runs"):
    """Load train_log.csv from all exp_* folders and add a seed column."""
    runs_path = Path(runs_dir)
    records = []

    for train_log_path in sorted(runs_path.glob("exp_*/train_log.csv")):
        run_dir = train_log_path.parent
        seed_value = _read_seed_from_hyperparameters(run_dir)
        if seed_value is None:
            continue

        try:
            train_df = pd.read_csv(train_log_path)
        except OSError:
            continue

        train_df = train_df.copy()
        train_df["seed"] = seed_value
        train_df["experiment"] = run_dir.name
        train_df["run_dir"] = str(run_dir)
        records.append(train_df)

    if not records:
        return pd.DataFrame()

    combined = pd.concat(records, ignore_index=True)
    numeric_columns = [column for column in combined.columns if column not in {"experiment", "run_dir"}]
    for column in numeric_columns:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    return combined


def plot_switch_rate_comparison(df, output_path):
    """Create a box plot of switch_rate by seed and highlight seed 46."""
    if df.empty:
        raise ValueError("The training log dataframe is empty.")
    if "switch_rate" not in df.columns or "seed" not in df.columns:
        raise ValueError("The dataframe must contain 'seed' and 'switch_rate' columns.")

    plot_df = df[["seed", "switch_rate"]].dropna().copy()
    plot_df["seed"] = pd.to_numeric(plot_df["seed"], errors="coerce")
    plot_df["switch_rate"] = pd.to_numeric(plot_df["switch_rate"], errors="coerce")
    plot_df = plot_df.dropna()
    if plot_df.empty:
        raise ValueError("No numeric switch_rate data available for plotting.")

    seeds = sorted(plot_df["seed"].astype(int).unique().tolist())
    grouped_values = [plot_df.loc[plot_df["seed"] == seed_value, "switch_rate"].to_numpy() for seed_value in seeds]

    fig, ax = plt.subplots(figsize=(11, 6))
    boxplot = ax.boxplot(grouped_values, labels=[str(seed_value) for seed_value in seeds], patch_artist=True, showmeans=True)

    for patch, seed_value in zip(boxplot["boxes"], seeds):
        if seed_value == TARGET_SEED:
            patch.set_facecolor("#d62728")
            patch.set_alpha(0.75)
        else:
            patch.set_facecolor("#8fb3d9")
            patch.set_alpha(0.55)

    for median in boxplot["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.5)

    for mean_line in boxplot["means"]:
        mean_line.set_marker("o")
        mean_line.set_markerfacecolor("#111111")
        mean_line.set_markeredgecolor("#111111")
        mean_line.set_markersize(5)

    ax.set_title("Action Switch Rate Distribution by Seed")
    ax.set_xlabel("Seed ID")
    ax.set_ylabel("Switch Rate (actions/step)")
    ax.grid(axis="y", alpha=0.25)

    if TARGET_SEED in seeds:
        target_index = seeds.index(TARGET_SEED)
        target_values = grouped_values[target_index]
        if target_values.size:
            target_mean = float(np.mean(target_values))
            ax.annotate(
                f"Seed {TARGET_SEED}",
                xy=(target_index + 1, target_mean),
                xytext=(target_index + 1.15, target_mean + 0.015),
                arrowprops={"arrowstyle": "->", "color": "#d62728"},
                color="#d62728",
                fontsize=10,
                weight="bold",
            )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _aggregate_eval_by_seed(eval_df):
    """Aggregate evaluation results to one row per seed using mean improvement values."""
    if eval_df.empty:
        return pd.DataFrame()

    required_columns = {"seed", "avg_wait_improvement_pct", "avg_time_loss_improvement_pct"}
    missing_columns = required_columns - set(eval_df.columns)
    if missing_columns:
        raise ValueError(f"Evaluation CSV is missing required columns: {sorted(missing_columns)}")

    working = eval_df.copy()
    working["seed"] = pd.to_numeric(working["seed"], errors="coerce")
    working["avg_wait_improvement_pct"] = pd.to_numeric(working["avg_wait_improvement_pct"], errors="coerce")
    working["avg_time_loss_improvement_pct"] = pd.to_numeric(working["avg_time_loss_improvement_pct"], errors="coerce")
    working = working.dropna(subset=["seed", "avg_wait_improvement_pct", "avg_time_loss_improvement_pct"])
    if working.empty:
        return pd.DataFrame()

    return (
        working.groupby("seed", as_index=False)
        .agg(
            avg_wait_improvement_pct=("avg_wait_improvement_pct", "mean"),
            avg_time_loss_improvement_pct=("avg_time_loss_improvement_pct", "mean"),
            n_rows=("seed", "size"),
        )
        .sort_values("seed")
        .reset_index(drop=True)
    )


def plot_wait_vs_timeloss_scatter(eval_df, output_path):
    """Create a scatter plot of wait improvement versus time-loss improvement by seed."""
    aggregated = _aggregate_eval_by_seed(eval_df)
    if aggregated.empty:
        raise ValueError("No numeric evaluation data available for the scatter plot.")

    fig, ax = plt.subplots(figsize=(8.5, 7))

    for _, row in aggregated.iterrows():
        seed_value = int(row["seed"])
        x_value = float(row["avg_wait_improvement_pct"])
        y_value = float(row["avg_time_loss_improvement_pct"])
        if seed_value == TARGET_SEED:
            ax.scatter(x_value, y_value, s=170, color="#d62728", edgecolor="#7f1d1d", linewidth=1.2, zorder=4)
        else:
            ax.scatter(x_value, y_value, s=75, color="#4c78a8", alpha=0.85, zorder=3)
        ax.text(x_value + 0.15, y_value + 0.15, str(seed_value), fontsize=9, color="#222222")

    ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    ax.axvline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    ax.set_title("Wait vs Time-Loss Improvement per Seed")
    ax.set_xlabel("avg_wait_improvement_pct")
    ax.set_ylabel("avg_time_loss_improvement_pct")
    ax.grid(alpha=0.25)

    if TARGET_SEED in aggregated["seed"].astype(int).tolist():
        target_row = aggregated.loc[aggregated["seed"].astype(int) == TARGET_SEED].iloc[0]
        target_wait = float(target_row["avg_wait_improvement_pct"])
        target_time_loss = float(target_row["avg_time_loss_improvement_pct"])
        if target_wait > 0 and target_time_loss < 0:
            ax.annotate(
                "Seed 46: wait improved, time-loss degraded",
                xy=(target_wait, target_time_loss),
                xytext=(target_wait + 0.8, target_time_loss - 0.8),
                arrowprops={"arrowstyle": "->", "color": "#d62728"},
                fontsize=10,
                color="#d62728",
                weight="bold",
            )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_switch_rate_stats(df):
    """Compute mean, standard deviation, and coefficient of variation for each seed."""
    if df.empty:
        return pd.DataFrame(columns=["seed", "mean_switch_rate", "std_switch_rate", "cv"])
    if "seed" not in df.columns or "switch_rate" not in df.columns:
        raise ValueError("The dataframe must contain 'seed' and 'switch_rate' columns.")

    working = df[["seed", "switch_rate"]].copy()
    working["seed"] = pd.to_numeric(working["seed"], errors="coerce")
    working["switch_rate"] = pd.to_numeric(working["switch_rate"], errors="coerce")
    working = working.dropna()
    if working.empty:
        return pd.DataFrame(columns=["seed", "mean_switch_rate", "std_switch_rate", "cv"])

    stats_df = (
        working.groupby("seed", as_index=False)
        .agg(
            mean_switch_rate=("switch_rate", "mean"),
            std_switch_rate=("switch_rate", "std"),
            n_episodes=("switch_rate", "size"),
        )
        .sort_values("seed")
        .reset_index(drop=True)
    )
    stats_df["std_switch_rate"] = stats_df["std_switch_rate"].fillna(0.0)
    stats_df["cv"] = np.where(
        stats_df["mean_switch_rate"].abs() > np.finfo(float).eps,
        stats_df["std_switch_rate"] / stats_df["mean_switch_rate"].abs(),
        np.nan,
    )
    return stats_df[["seed", "mean_switch_rate", "std_switch_rate", "cv", "n_episodes"]]


def _safe_pearsonr(x_values, y_values):
    """Compute Pearson correlation with graceful handling for short inputs."""
    if len(x_values) < 2 or len(y_values) < 2:
        return np.nan, np.nan
    try:
        return stats.pearsonr(x_values, y_values)
    except Exception:
        return np.nan, np.nan


def _safe_spearmanr(x_values, y_values):
    """Compute Spearman correlation with graceful handling for short inputs."""
    if len(x_values) < 2 or len(y_values) < 2:
        return np.nan, np.nan
    try:
        return stats.spearmanr(x_values, y_values)
    except Exception:
        return np.nan, np.nan


def generate_diagnosis_report(switch_stats, eval_df):
    """Generate a text report that evaluates the seed 46 over-switching hypothesis."""
    if switch_stats.empty:
        return "No switch-rate statistics available."
    if eval_df.empty:
        return "No evaluation data available."

    switch_stats = switch_stats.copy()
    eval_df = eval_df.copy()
    switch_stats["seed"] = pd.to_numeric(switch_stats["seed"], errors="coerce")
    eval_df["seed"] = pd.to_numeric(eval_df["seed"], errors="coerce")

    eval_grouped = _aggregate_eval_by_seed(eval_df)
    merged = switch_stats.merge(eval_grouped, on="seed", how="inner")

    lines = []
    lines.append("=" * 80)
    lines.append("SEED 46 DIAGNOSIS REPORT")
    lines.append("=" * 80)

    if merged.empty:
        lines.append("No overlapping seed-level data could be constructed from the inputs.")
        return "\n".join(lines)

    target_row = merged.loc[merged["seed"].astype(int) == TARGET_SEED]
    other_rows = merged.loc[merged["seed"].astype(int) != TARGET_SEED]

    lines.append(f"Seeds analyzed: {', '.join(str(int(seed)) for seed in merged['seed'].astype(int).tolist())}")
    lines.append(f"Episode-level samples: {int(switch_stats['n_episodes'].sum()) if 'n_episodes' in switch_stats.columns else 'n/a'}")
    lines.append("")

    lines.append("Switch-rate summary by seed")
    for _, row in switch_stats.sort_values("seed").iterrows():
        seed_value = int(row["seed"])
        lines.append(
            f"  Seed {seed_value}: mean={row['mean_switch_rate']:.6f}, std={row['std_switch_rate']:.6f}, cv={row['cv']:.3f}"
        )

    if target_row.empty:
        lines.append("")
        lines.append(f"Seed {TARGET_SEED} was not found in the available training logs.")
        return "\n".join(lines)

    target_mean_switch = float(target_row.iloc[0]["mean_switch_rate"])
    other_mean_switch = float(other_rows["mean_switch_rate"].mean()) if not other_rows.empty else np.nan
    other_std_switch = float(other_rows["mean_switch_rate"].std(ddof=1)) if len(other_rows) > 1 else 0.0
    target_rank = _format_rank(switch_stats, TARGET_SEED)
    seed_count = int(switch_stats.shape[0])

    lines.append("")
    lines.append("Seed 46 comparison")
    lines.append(f"  Seed 46 mean switch_rate: {target_mean_switch:.6f}")
    lines.append(f"  Other seeds mean switch_rate: {other_mean_switch:.6f}")
    lines.append(f"  Difference: {target_mean_switch - other_mean_switch:+.6f}")
    if np.isfinite(other_mean_switch) and other_mean_switch != 0.0:
        lines.append(f"  Relative change: {(target_mean_switch / other_mean_switch - 1.0) * 100.0:+.2f}%")
    else:
        lines.append("  Relative change: n/a")
    lines.append(f"  Other seeds std: {other_std_switch:.6f}")
    lines.append(f"  Seed 46 rank by mean switch_rate: {target_rank} / {seed_count}")

    x_values = merged["mean_switch_rate"].to_numpy(dtype=float)
    y_values = merged["avg_time_loss_improvement_pct"].to_numpy(dtype=float)
    pearson_r, pearson_p = _safe_pearsonr(x_values, y_values)
    spearman_rho, spearman_p = _safe_spearmanr(x_values, y_values)

    lines.append("")
    lines.append("Correlation between switch_rate and time-loss improvement")
    lines.append(f"  Pearson r:   {pearson_r:.4f} (p={pearson_p:.6f})")
    lines.append(f"  Spearman rho:{spearman_rho:.4f} (p={spearman_p:.6f})")

    target_wait = float(target_row.iloc[0]["avg_wait_improvement_pct"])
    target_timeloss = float(target_row.iloc[0]["avg_time_loss_improvement_pct"])
    lines.append("")
    lines.append("Seed 46 evaluation profile")
    lines.append(f"  avg_wait_improvement_pct: {target_wait:.3f}")
    lines.append(f"  avg_time_loss_improvement_pct: {target_timeloss:.3f}")
    if target_wait > 0 and target_timeloss < 0:
        lines.append("  Pattern: wait improved, time-loss degraded.")
    else:
        lines.append("  Pattern: does not match the expected over-switching signature.")

    seed46_high_switch = np.isfinite(other_mean_switch) and target_mean_switch > other_mean_switch
    negative_correlation = np.isfinite(pearson_r) and pearson_r < 0 and np.isfinite(pearson_p) and pearson_p < 0.05
    degraded_time_loss = target_timeloss < float(other_rows["avg_time_loss_improvement_pct"].mean()) if not other_rows.empty else False

    confirmed = seed46_high_switch and (negative_correlation or degraded_time_loss)

    lines.append("")
    lines.append("Conclusion")
    if confirmed:
        lines.append("  CONFIRMED: The evidence supports the over-switching hypothesis for seed 46.")
        lines.append("  Recommendation: inspect switch_penalty and related reward shaping terms.")
    else:
        lines.append("  NOT CONFIRMED: The current data is not strong enough to support the hypothesis.")
        lines.append("  Recommendation: collect more runs or inspect other seeds for the same pattern.")

    return "\n".join(lines)


def _format_rank(seed_stats, target_seed):
    """Format the rank position of a seed by descending mean switch rate."""
    ordered = seed_stats.sort_values("mean_switch_rate", ascending=False).reset_index(drop=True)
    matching = ordered.index[ordered["seed"].astype(int) == target_seed].tolist()
    if not matching:
        return None
    return int(matching[0] + 1)


def main():
    """Command-line entry point for the seed 46 diagnosis workflow."""
    parser = argparse.ArgumentParser(description="Diagnose seed 46 over-switching behavior.")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing exp_* training runs")
    parser.add_argument("--eval-csv", default="master_evaluation_results.csv", help="Evaluation CSV with improvement columns")
    parser.add_argument("--output-dir", default="analysis", help="Directory for plots and report")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_train_logs(args.runs_dir)
    if train_df.empty:
        raise FileNotFoundError("No training logs were found under the runs directory.")

    switch_stats = compute_switch_rate_stats(train_df)
    eval_df = pd.read_csv(args.eval_csv)

    switch_plot_path = output_dir / "seed46_switch_rate_comparison.png"
    scatter_plot_path = output_dir / "seed46_wait_vs_timeloss.png"
    report_path = output_dir / "seed46_diagnosis_report.txt"

    plot_switch_rate_comparison(train_df, switch_plot_path)
    plot_wait_vs_timeloss_scatter(eval_df, scatter_plot_path)

    report_text = generate_diagnosis_report(switch_stats, eval_df)
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"[OK] Saved plot: {switch_plot_path}")
    print(f"[OK] Saved plot: {scatter_plot_path}")
    print(f"[OK] Saved report: {report_path}")


if __name__ == "__main__":
    main()