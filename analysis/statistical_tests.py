"""Statistical validation module for D3QN vs baselines.

Usage:
    python -m analysis.statistical_tests master_evaluation_results.csv
    python -m analysis.statistical_tests generalization_results.csv --group-by scenario

Output:
    statistical_validation_results.csv
    statistical_validation_report.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats


METRICS_LOWER_BETTER = {
    "avg_wait": ("avg_wait_webster", "avg_wait_d3qn"),
    "p95_wait": ("p95_wait_webster", "p95_wait_d3qn"),
    "avg_time_loss": ("avg_time_loss_webster", "avg_time_loss_d3qn"),
    "avg_queue": ("avg_queue_webster", "avg_queue_d3qn"),
}
BASELINE_LABELS = ("webster", "actuated")
CLIFF_THRESHOLDS = (
    (0.147, "negligible"),
    (0.33, "small"),
    (0.474, "medium"),
    (1.0, "large"),
)
PAIR_KEY_CANDIDATES = ("experiment", "scenario", "seed")


def normalize_controller_name(value):
    """Normalize controller names to a lower-case label used by the analysis."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def safe_std(values):
    """Compute a sample standard deviation with a safe fallback for short arrays."""
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return 0.0
    return float(np.std(array, ddof=1))


def cliffs_delta(treatment, control):
    """Compute Cliff's Delta for two independent samples.

    A negative value indicates the treatment tends to be smaller than the control.
    """
    treatment = np.asarray(treatment, dtype=float)
    control = np.asarray(control, dtype=float)
    n_treatment = treatment.size
    n_control = control.size
    if n_treatment == 0 or n_control == 0:
        return np.nan

    more = 0
    less = 0
    for treatment_value in treatment:
        for control_value in control:
            if treatment_value > control_value:
                more += 1
            elif treatment_value < control_value:
                less += 1
    return float((more - less) / (n_treatment * n_control))


def classify_effect(delta):
    """Classify the absolute magnitude of Cliff's Delta."""
    if np.isnan(delta):
        return "unknown"

    abs_delta = abs(float(delta))
    for threshold, label in CLIFF_THRESHOLDS:
        if abs_delta <= threshold:
            return label
    return "large"


def bootstrap_improvement_ci(treatment_vals, control_vals, n_boot=10000, alpha=0.05, seed=42):
    """Estimate mean percentage improvement and its bootstrap confidence interval."""
    treatment = np.asarray(treatment_vals, dtype=float)
    control = np.asarray(control_vals, dtype=float)
    if treatment.size == 0 or control.size == 0:
        return np.nan, np.nan, np.nan

    safe_control = np.where(control != 0, control, np.nan)
    improvements = (safe_control - treatment) / safe_control * 100.0
    improvements = improvements[np.isfinite(improvements)]
    if improvements.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boot_means = np.array(
        [np.mean(rng.choice(improvements, size=improvements.size, replace=True)) for _ in range(n_boot)]
    )
    return (
        float(np.mean(improvements)),
        float(np.percentile(boot_means, 100 * alpha / 2.0)),
        float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0))),
    )


def cohens_d(treatment, control):
    """Compute Cohen's d as a reference effect size."""
    treatment = np.asarray(treatment, dtype=float)
    control = np.asarray(control, dtype=float)
    if treatment.size == 0 or control.size == 0:
        return np.nan

    pooled_std = np.sqrt((np.std(treatment, ddof=1) ** 2 + np.std(control, ddof=1) ** 2) / 2.0)
    if not np.isfinite(pooled_std) or pooled_std == 0:
        return 0.0
    return float((np.mean(control) - np.mean(treatment)) / pooled_std)


def wilcoxon_paired(treatment, control, alternative="less"):
    """Run the paired Wilcoxon signed-rank test on matched samples."""
    treatment = np.asarray(treatment, dtype=float)
    control = np.asarray(control, dtype=float)
    if treatment.size == 0 or control.size == 0:
        return np.nan, np.nan

    if treatment.size != control.size:
        raise ValueError("Wilcoxon paired test requires equal-length paired samples.")

    if treatment.size < 2:
        return np.nan, np.nan

    if np.allclose(treatment, control, equal_nan=False):
        return 0.0, 1.0

    try:
        statistic, p_value = stats.wilcoxon(
            treatment,
            control,
            alternative=alternative,
            zero_method="wilcox",
            correction=False,
            mode="auto",
        )
    except TypeError:
        statistic, p_value = stats.wilcoxon(
            treatment,
            control,
            alternative=alternative,
            zero_method="wilcox",
            correction=False,
        )

    statistic_value = float(np.asarray(statistic).reshape(-1)[0])
    p_value_value = float(np.asarray(p_value).reshape(-1)[0])
    return statistic_value, p_value_value


def pair_arrays_from_wide(df, treatment_col, control_col):
    """Extract paired arrays from a wide-format dataframe using two metric columns."""
    required = [treatment_col, control_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        return np.array([]), np.array([])

    subset = df[[treatment_col, control_col]].dropna()
    return subset[treatment_col].to_numpy(dtype=float), subset[control_col].to_numpy(dtype=float)


def detect_data_mode(df):
    """Detect whether the input file is wide-format or long-format results."""
    has_wide_columns = any(
        f"{metric}_{baseline}" in df.columns
        for metric in METRICS_LOWER_BETTER
        for baseline in BASELINE_LABELS
    )
    if has_wide_columns:
        return "wide"
    if {"controller", "seed"}.issubset(df.columns):
        return "long"
    raise ValueError(
        "Unable to detect result format. Expected either wide metric columns or a long-format dataframe with 'controller' and 'seed'."
    )


def detect_available_baselines(df):
    """Return the baseline controllers that can be compared with D3QN."""
    mode = detect_data_mode(df)
    if mode == "wide":
        baselines = [
            baseline
            for baseline in BASELINE_LABELS
            if any(f"{metric}_{baseline}" in df.columns for metric in METRICS_LOWER_BETTER)
        ]
    else:
        controllers = {normalize_controller_name(value) for value in df["controller"].dropna().unique()}
        baselines = [baseline for baseline in BASELINE_LABELS if baseline in controllers]
    return baselines


def pair_arrays_from_long(df, metric, baseline, pair_keys):
    """Extract paired arrays from a long-format dataframe using controller rows."""
    required = set(pair_keys) | {"controller", metric}
    missing = [column for column in required if column not in df.columns]
    if missing:
        return np.array([]), np.array([])

    working = df[list(pair_keys) + ["controller", metric]].copy()
    working = working.dropna(subset=[metric])
    working["controller"] = working["controller"].map(normalize_controller_name)
    pivot = working.pivot_table(index=list(pair_keys), columns="controller", values=metric, aggfunc="mean")

    if "d3qn" not in pivot.columns or baseline not in pivot.columns:
        return np.array([]), np.array([])

    paired = pivot[["d3qn", baseline]].dropna()
    return paired["d3qn"].to_numpy(dtype=float), paired[baseline].to_numpy(dtype=float)


def infer_pair_keys(df, group_by=None):
    """Infer the columns used to align paired measurements in long-format inputs."""
    candidate_keys = [column for column in PAIR_KEY_CANDIDATES if column in df.columns]
    if group_by and group_by in candidate_keys:
        candidate_keys = [column for column in candidate_keys if column != group_by]
    return candidate_keys


def summarize_metric_pair(treatment_vals, control_vals, metric, baseline, alpha, n_tests):
    """Build the statistical summary for one metric and one baseline comparison."""
    treatment_vals = np.asarray(treatment_vals, dtype=float)
    control_vals = np.asarray(control_vals, dtype=float)
    pair_count = int(min(treatment_vals.size, control_vals.size))
    if pair_count == 0:
        return None

    treatment_vals = treatment_vals[:pair_count]
    control_vals = control_vals[:pair_count]

    statistic, p_value = wilcoxon_paired(treatment_vals, control_vals, alternative="less")
    delta = cliffs_delta(treatment_vals, control_vals)
    effect_label = classify_effect(delta)
    mean_improvement, ci_lower, ci_upper = bootstrap_improvement_ci(treatment_vals, control_vals)
    cohen_d = cohens_d(treatment_vals, control_vals)

    return {
        "metric": metric,
        "baseline": baseline,
        "n_pairs": pair_count,
        "d3qn_mean": float(np.mean(treatment_vals)),
        "d3qn_std": safe_std(treatment_vals),
        f"{baseline}_mean": float(np.mean(control_vals)),
        f"{baseline}_std": safe_std(control_vals),
        "wilcoxon_stat": statistic,
        "wilcoxon_p": p_value,
        "bonferroni_alpha": alpha / n_tests if n_tests else np.nan,
        "significant": bool(np.isfinite(p_value) and p_value < (alpha / n_tests if n_tests else np.nan)),
        "cliffs_delta": delta,
        "effect_size": effect_label,
        "cohens_d": cohen_d,
        "mean_improvement_pct": mean_improvement,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
    }


def run_pairwise_tests(df, alpha=0.05, group_label=None, group_by=None):
    """Run the full statistical battery for all supported metrics and baselines."""
    mode = detect_data_mode(df)
    available_baselines = detect_available_baselines(df)
    if not available_baselines:
        return pd.DataFrame()

    if mode == "wide":
        n_tests = len(METRICS_LOWER_BETTER) * len(available_baselines)
        rows = []
        for metric, (d3qn_col, _) in METRICS_LOWER_BETTER.items():
            for baseline in available_baselines:
                baseline_col = f"{metric}_{baseline}"
                treatment_vals, control_vals = pair_arrays_from_wide(df, d3qn_col, baseline_col)
                result = summarize_metric_pair(treatment_vals, control_vals, metric, baseline, alpha, n_tests)
                if result is not None:
                    rows.append(result)
        result_df = pd.DataFrame(rows)
        if group_label is not None and not result_df.empty:
            result_df.insert(0, group_by or "group", group_label)
        return result_df

    pair_keys = infer_pair_keys(df, group_by=group_by)
    if not pair_keys:
        raise ValueError(
            "Long-format input requires at least one pairing key such as 'scenario' or 'seed' to align D3QN with each baseline."
        )

    n_tests = len(METRICS_LOWER_BETTER) * len(available_baselines)
    rows = []
    for metric in METRICS_LOWER_BETTER:
        for baseline in available_baselines:
            treatment_vals, control_vals = pair_arrays_from_long(df, metric, baseline, pair_keys)
            result = summarize_metric_pair(treatment_vals, control_vals, metric, baseline, alpha, n_tests)
            if result is not None:
                rows.append(result)

    result_df = pd.DataFrame(rows)
    if group_label is not None and not result_df.empty:
        result_df.insert(0, group_by or "group", group_label)
    return result_df


def format_metric_block(row):
    """Format one metric comparison row for the human-readable report."""
    baseline = row["baseline"]
    significant = "SIGNIFICANT" if row["significant"] else "NOT SIGNIFICANT"
    baseline_mean = row.get(f"{baseline}_mean", np.nan)
    baseline_std = row.get(f"{baseline}_std", np.nan)
    return [
        f"{row['metric'].upper()} vs {baseline.upper()}",
        f"  Baseline:      {baseline_mean:.3f} +/- {baseline_std:.3f}",
        f"  D3QN:          {row['d3qn_mean']:.3f} +/- {row['d3qn_std']:.3f}",
        f"  Wilcoxon p:    {row['wilcoxon_p']:.6f} ({significant})",
        f"  Cliff's delta: {row['cliffs_delta']:.3f} ({row['effect_size']})",
        f"  Improvement:   {row['mean_improvement_pct']:.2f}% [95% CI: {row['ci_95_lower']:.2f}%, {row['ci_95_upper']:.2f}%]",
    ]


def build_report(results_df, alpha=0.05, group_by=None):
    """Render the statistical validation report as plain text."""
    if results_df.empty:
        return "No valid paired comparisons were found."

    lines = ["=" * 80]
    lines.append("STATISTICAL VALIDATION REPORT - D3QN vs baselines")
    if group_by and group_by in results_df.columns:
        lines.append(f"Grouped by: {group_by}")
    lines.append(f"Total comparisons: {len(results_df)}")
    lines.append("Primary test: Wilcoxon signed-rank, one-tailed (treatment < baseline)")
    lines.append("Effect size: Cliff's Delta; reference effect size: Cohen's d")
    lines.append(f"Bonferroni alpha per row is stored in the results table (family-wise alpha = {alpha:.3f})")
    lines.append("=" * 80)

    if group_by and group_by in results_df.columns:
        group_order = results_df[group_by].dropna().drop_duplicates().tolist()
        for group_value in group_order:
            group_df = results_df[results_df[group_by] == group_value]
            lines.append(f"\n[{group_by} = {group_value}]")
            for _, row in group_df.iterrows():
                lines.extend(format_metric_block(row))
    else:
        for _, row in results_df.iterrows():
            lines.extend(format_metric_block(row))

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def run_analysis(csv_path, output_dir="analysis", group_by=None, alpha=0.05):
    """Run the complete analysis workflow and write the CSV and text report."""
    df = pd.read_csv(csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if group_by and group_by in df.columns:
        grouped_results = []
        for group_value, group_df in df.groupby(group_by, dropna=False):
            result = run_pairwise_tests(group_df, alpha=alpha, group_label=group_value, group_by=group_by)
            if not result.empty:
                grouped_results.append(result)
        results_df = pd.concat(grouped_results, ignore_index=True) if grouped_results else pd.DataFrame()
    else:
        results_df = run_pairwise_tests(df, alpha=alpha)

    results_csv = output_path / "statistical_validation_results.csv"
    results_df.to_csv(results_csv, index=False)

    report_text = build_report(results_df, alpha=alpha, group_by=group_by)
    report_path = output_path / "statistical_validation_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"[OK] Results saved: {results_csv}")
    print(f"[OK] Report saved: {report_path}")
    return results_df, report_text, results_csv, report_path


def parse_args(argv=None):
    """Parse command-line arguments for the statistical validation module."""
    parser = argparse.ArgumentParser(
        prog="analysis.statistical_tests",
        description="Statistical tests for D3QN evaluation results.",
    )
    parser.add_argument("csv_path", nargs="?", default="master_evaluation_results.csv", help="Input results CSV file.")
    parser.add_argument("--output-dir", default="analysis", help="Directory for the generated outputs.")
    parser.add_argument(
        "--group-by",
        default=None,
        help="Optional column to group by before testing, for example 'scenario'.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Family-wise significance level used for Bonferroni correction.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Entry point used by python -m analysis.statistical_tests."""
    args = parse_args(argv)
    run_analysis(args.csv_path, output_dir=args.output_dir, group_by=args.group_by, alpha=args.alpha)


if __name__ == "__main__":
    main()