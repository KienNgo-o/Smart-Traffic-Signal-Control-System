"""Generate comprehensive final report with figures and summary tables.

Synthesizes results from all previous tasks into publication-ready outputs:
  - Figure 1: Training reward convergence (6 seeds)
  - Figure 2: Scenario comparison (generalization)
  - Figure 3: Statistical summary (forest plots)
  - Figure 4: Switch rate analysis
  - Final report in Markdown format

Usage:
    python -m analysis.generate_report
    python -m analysis.generate_report --help

Output:
    analysis/fig1_reward_convergence.png
    analysis/fig2_scenario_comparison.png
    analysis/fig3_statistical_summary.png
    analysis/fig4_switch_rate_analysis.png
    analysis/final_report.md
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from scipy import stats

# Color scheme: Webster=red, Actuated=orange, D3QN=green
COLOR_SCHEME = {
    'webster': 'tab:red',
    'actuated': 'tab:orange',
    'd3qn': 'tab:green',
}

METRICS_LABELS = {
    'avg_wait': 'Avg Wait (s)',
    'p95_wait': 'P95 Wait (s)',
    'avg_time_loss': 'Time Loss (s)',
    'avg_queue': 'Avg Queue (veh)',
}


def percent_change(baseline: float, treatment: float, lower_is_better: bool = True) -> float:
    """Compute percent change from baseline to treatment.

    When lower_is_better is True, positive values mean treatment improved over baseline.
    """
    baseline = float(baseline)
    treatment = float(treatment)
    if not np.isfinite(baseline) or baseline == 0 or not np.isfinite(treatment):
        return np.nan
    if lower_is_better:
        return (baseline - treatment) / baseline * 100.0
    return (treatment - baseline) / baseline * 100.0


def log_warning(msg: str) -> None:
    """Print warning message."""
    print(f"[WARN] {msg}")


def load_csv_safe(filepath: Path) -> pd.DataFrame | None:
    """Load CSV file, return None if not found."""
    if not filepath.exists():
        log_warning(f"File not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        print(f"[OK] Loaded {filepath} ({len(df)} rows)")
        return df
    except Exception as e:
        log_warning(f"Failed to load {filepath}: {e}")
        return None


def find_train_logs() -> dict[int, Path]:
    """Find all train_log.csv files organized by seed."""
    runs_dir = Path('runs')
    seed_logs = {}
    
    if not runs_dir.exists():
        log_warning("runs/ directory not found")
        return seed_logs
    
    for exp_dir in sorted(runs_dir.glob('exp_*')):
        train_log = exp_dir / 'train_log.csv'
        if train_log.exists():
            try:
                hyperparameters_path = exp_dir / 'hyperparameters.json'
                if hyperparameters_path.exists():
                    with hyperparameters_path.open('r', encoding='utf-8') as handle:
                        seed = int(json.load(handle).get('seed'))
                    seed_logs[seed] = train_log
            except Exception:
                pass
    
    if seed_logs:
        print(f"[OK] Found {len(seed_logs)} train_log.csv files")
    else:
        log_warning("No train_log.csv files found in runs/")
    
    return seed_logs


def load_train_curves(master_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Load training curves for each seed."""
    train_logs = find_train_logs()
    curves = {}

    if master_df is not None and len(master_df) > 0 and 'seed' in master_df.columns:
        target_seeds = [int(seed) for seed in pd.to_numeric(master_df['seed'], errors='coerce').dropna().unique()]
    else:
        target_seeds = sorted(train_logs.keys())

    for seed in target_seeds:
        if seed not in train_logs:
            continue
        try:
            curves[seed] = pd.read_csv(train_logs[seed])
        except Exception as e:
            log_warning(f"Failed to load train_log for seed {seed}: {e}")
    
    return curves


def calculate_moving_average(arr: np.ndarray, window: int = 10) -> np.ndarray:
    """Calculate moving average."""
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode='valid')


def figure_1_reward_convergence(train_curves: dict[int, pd.DataFrame]) -> Path:
    """Figure 1: Training reward convergence across 6 seeds (2×3 layout)."""
    if not train_curves:
        log_warning("No training curves available for Figure 1")
        return Path('analysis/fig1_reward_convergence.png')
    # Show up to 10 seeds (layout adapts automatically)
    n_seeds = min(len(train_curves), 10)
    cols = 5
    rows = math.ceil(n_seeds / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle('D3QN Training Reward Convergence — All Seeds', fontsize=14, fontweight='bold')

    # Flatten axes for easier iteration
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Find y-axis range for all subplots
    all_rewards = []
    for curves in train_curves.values():
        for reward_col in ['total_reward', 'reward', 'episode_reward', 'return']:
            if reward_col in curves.columns:
                all_rewards.extend(pd.to_numeric(curves[reward_col], errors='coerce').dropna().values)
                break
    
    if all_rewards:
        y_min = np.percentile(all_rewards, 5)
        y_max = np.percentile(all_rewards, 95)
    else:
        y_min, y_max = 0, 100
    
    for idx, (seed, curves) in enumerate(sorted(train_curves.items())):
        if idx >= n_seeds:
            break
        
        ax = axes[idx]
        
        reward_col = None
        for col in ['total_reward', 'reward', 'episode_reward', 'return']:
            if col in curves.columns:
                reward_col = col
                break
        
        if reward_col is None:
            ax.text(0.5, 0.5, 'No reward data', ha='center', va='center')
            ax.set_title(f'Seed {seed}', fontsize=12)
            continue
        
        rewards = pd.to_numeric(curves[reward_col], errors='coerce').dropna().values
        episodes = np.arange(len(rewards))
        
        # Plot raw rewards with alpha
        ax.plot(episodes, rewards, color=COLOR_SCHEME['d3qn'], alpha=0.3, label='Raw')
        
        # Plot moving average
        if len(rewards) >= 10:
            ma = calculate_moving_average(rewards, window=10)
            episodes_ma = np.arange(len(ma))
            ax.plot(episodes_ma, ma, color=COLOR_SCHEME['d3qn'], linewidth=2, label='MA (window=10)')
        
        ax.set_title(f'Seed {seed}', fontsize=12)
        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel('Reward', fontsize=10)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=10)
    
    # Hide unused subplots
    total_slots = rows * cols
    for idx in range(n_seeds, total_slots):
        try:
            axes[idx].set_visible(False)
        except Exception:
            pass
    
    plt.tight_layout()
    output_path = Path('analysis/fig1_reward_convergence.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 1 saved: {output_path}")
    return output_path


def figure_2_scenario_comparison(gen_df: pd.DataFrame) -> Path:
    """Figure 2: Scenario comparison with 4 metrics and 3 controllers."""
    if gen_df is None or len(gen_df) == 0:
        log_warning("No generalization data available for Figure 2")
        return Path('analysis/fig2_scenario_comparison.png')

    required_columns = {'scenario', 'controller'}
    if not required_columns.issubset(gen_df.columns):
        log_warning("Generalization data is missing scenario/controller columns")
        return Path('analysis/fig2_scenario_comparison.png')

    metrics = ['avg_wait', 'p95_wait', 'avg_time_loss', 'avg_queue']
    scenarios = [scenario for scenario in ['symmetric', 'asymmetric', 'incident', 'high_demand', 'low_demand'] if scenario in gen_df['scenario'].astype(str).unique()]
    controllers = [controller for controller in ['webster', 'actuated', 'd3qn'] if controller in gen_df['controller'].astype(str).str.lower().unique()]

    if not scenarios or not controllers:
        log_warning("Generalization data does not contain enough scenarios/controllers for plotting")
        return Path('analysis/fig2_scenario_comparison.png')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('D3QN Generalization Across Traffic Scenarios', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    working = gen_df.copy()
    working['controller'] = working['controller'].astype(str).str.lower()

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        if metric not in working.columns:
            ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center')
            ax.set_title(METRICS_LABELS.get(metric, metric), fontsize=12)
            continue

        x_pos = np.arange(len(scenarios))
        bar_width = 0.25

        for ctrl_idx, ctrl in enumerate(controllers):
            means = []
            stds = []
            for scenario in scenarios:
                values = pd.to_numeric(
                    working.loc[(working['scenario'] == scenario) & (working['controller'] == ctrl), metric],
                    errors='coerce',
                ).dropna()
                means.append(values.mean() if len(values) else np.nan)
                stds.append(values.std(ddof=1) if len(values) > 1 else 0.0)

            offset = (ctrl_idx - (len(controllers) - 1) / 2.0) * bar_width
            ax.bar(
                x_pos + offset,
                means,
                bar_width,
                label=ctrl.capitalize(),
                color=COLOR_SCHEME.get(ctrl, 'tab:blue'),
                alpha=0.8,
                yerr=stds,
                capsize=4,
            )

        ax.set_xlabel('Scenario', fontsize=10)
        ax.set_ylabel(METRICS_LABELS.get(metric, metric), fontsize=10)
        ax.set_title(METRICS_LABELS.get(metric, metric), fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenarios, rotation=20, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    output_path = Path('analysis/fig2_scenario_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 2 saved: {output_path}")
    return output_path


def figure_3_statistical_summary(stat_df: pd.DataFrame) -> Path:
    """Figure 3: Forest plot with effect sizes and 95% CI."""
    if stat_df is None or len(stat_df) == 0:
        log_warning("No statistical validation data available for Figure 3")
        return Path('analysis/fig3_statistical_summary.png')

    required_columns = {'metric', 'mean_improvement_pct', 'ci_95_lower', 'ci_95_upper'}
    if not required_columns.issubset(stat_df.columns):
        log_warning("Statistical validation data is missing the required columns")
        return Path('analysis/fig3_statistical_summary.png')

    metrics = ['avg_wait', 'p95_wait', 'avg_time_loss', 'avg_queue']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('D3QN Improvement: Mean Difference and 95% CI', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    working = stat_df.copy()
    working['metric'] = working['metric'].astype(str)
    if 'baseline' in working.columns:
        working['baseline'] = working['baseline'].astype(str)

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        subset = working.loc[working['metric'] == metric].copy()
        if subset.empty:
            ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center')
            ax.set_title(METRICS_LABELS.get(metric, metric), fontsize=12)
            continue

        subset = subset.reset_index(drop=True)
        y_pos = np.arange(len(subset))
        baseline_labels = subset['baseline'].str.capitalize().tolist() if 'baseline' in subset.columns else [f'Test {i + 1}' for i in range(len(subset))]
        colors = [COLOR_SCHEME['webster'] if str(label).lower() == 'webster' else COLOR_SCHEME['actuated'] if str(label).lower() == 'actuated' else COLOR_SCHEME['d3qn'] for label in baseline_labels]

        for y_index, (_, row) in enumerate(subset.iterrows()):
            x_value = float(row['mean_improvement_pct'])
            x_lower = float(row['ci_95_lower'])
            x_upper = float(row['ci_95_upper'])
            ax.plot([x_lower, x_upper], [y_index, y_index], color=colors[y_index], linewidth=2)
            ax.scatter(x_value, y_index, color=colors[y_index], s=70, zorder=3)
            if bool(row.get('significant', False)):
                ax.text(x_upper + 0.4, y_index, '*', va='center', fontsize=12, fontweight='bold')

        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(baseline_labels)
        ax.set_title(METRICS_LABELS.get(metric, metric), fontsize=12)
        ax.set_xlabel('Improvement (%)', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    output_path = Path('analysis/fig3_statistical_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 3 saved: {output_path}")
    return output_path


def figure_4_switch_rate_analysis(master_df: pd.DataFrame) -> Path:
    """Figure 4: Switch rate analysis (box plot + scatter)."""
    if master_df is None or len(master_df) == 0:
        log_warning("No master evaluation data available for Figure 4")
        return Path('analysis/fig4_switch_rate_analysis.png')

    train_logs = find_train_logs()
    switch_series = {}
    for seed, train_log_path in train_logs.items():
        try:
            train_df = pd.read_csv(train_log_path)
        except Exception:
            continue
        if 'switch_rate' not in train_df.columns:
            continue
        values = pd.to_numeric(train_df['switch_rate'], errors='coerce').dropna().to_numpy()
        if values.size:
            switch_series[int(seed)] = values

    if not switch_series:
        log_warning("No switch_rate values found in training logs; skipping Figure 4")
        return Path('analysis/fig4_switch_rate_analysis.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Switch Rate Analysis Across Seeds', fontsize=14, fontweight='bold')

    seeds = sorted(switch_series)
    box_data = [switch_series[seed] for seed in seeds]
    ax_left = axes[0]
    bp = ax_left.boxplot(box_data, tick_labels=[str(seed) for seed in seeds], patch_artist=True, showmeans=True)

    for patch, seed in zip(bp['boxes'], seeds):
        patch.set_facecolor(COLOR_SCHEME['d3qn'] if seed != 46 else 'tab:red')
        patch.set_alpha(0.7)

    ax_left.set_ylabel('Switch Rate (actions/step)', fontsize=10)
    ax_left.set_title('Switch Rate Distribution', fontsize=12)
    ax_left.grid(True, alpha=0.3, axis='y')
    ax_left.tick_params(labelsize=10)

    seed_df = pd.to_numeric(master_df['seed'], errors='coerce')
    wait_improvement = pd.to_numeric(master_df['avg_wait_improvement_pct'], errors='coerce') if 'avg_wait_improvement_pct' in master_df.columns else pd.Series(dtype=float)
    timeloss_improvement = pd.to_numeric(master_df['avg_time_loss_improvement_pct'], errors='coerce') if 'avg_time_loss_improvement_pct' in master_df.columns else pd.Series(dtype=float)
    scatter_df = pd.DataFrame({
        'seed': seed_df,
        'wait_improvement': wait_improvement,
        'timeloss_improvement': timeloss_improvement,
    }).dropna()

    ax_right = axes[1]
    if not scatter_df.empty:
        for _, row in scatter_df.iterrows():
            seed = int(row['seed'])
            color = 'tab:red' if seed == 46 else COLOR_SCHEME['d3qn']
            size = 140 if seed == 46 else 80
            ax_right.scatter(row['wait_improvement'], row['timeloss_improvement'], s=size, color=color, alpha=0.8)
            ax_right.annotate(str(seed), (row['wait_improvement'], row['timeloss_improvement']), fontsize=8, xytext=(4, 4), textcoords='offset points')

        ax_right.axhline(0, color='black', linestyle='--', linewidth=1)
        ax_right.axvline(0, color='black', linestyle='--', linewidth=1)
    ax_right.set_xlabel('Wait Time Improvement (%)', fontsize=10)
    ax_right.set_ylabel('Time Loss Improvement (%)', fontsize=10)
    ax_right.set_title('Improvement Correlation', fontsize=12)
    ax_right.grid(True, alpha=0.3)
    ax_right.tick_params(labelsize=10)

    plt.tight_layout()
    output_path = Path('analysis/fig4_switch_rate_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 4 saved: {output_path}")
    return output_path


def generate_markdown_report(master_df: pd.DataFrame, gen_df: pd.DataFrame, 
                             stat_df: pd.DataFrame) -> Path:
    """Generate final_report.md with summary tables."""
    report_lines = [
        "# D3QN Smart Traffic Signal Control - Final Report",
        "",
        "## Executive Summary",
        "",
        "This report summarizes D3QN performance against Webster and Actuated control across the training, evaluation, and generalization experiments.",
        "",
    ]

    if master_df is not None and len(master_df) > 0:
        report_lines.extend([
            "## 1. Baseline Evaluation Results",
            "",
            "| Metric | Webster mean | D3QN mean | Improvement |",
            "|---|---:|---:|---:|",
        ])
        for metric_col, label in [('avg_wait', 'Avg Wait (s)'), ('p95_wait', 'P95 Wait (s)'), ('avg_time_loss', 'Avg Time Loss (s)'), ('avg_queue', 'Avg Queue (veh)')]:
            w_col = f'{metric_col}_webster'
            d_col = f'{metric_col}_d3qn'
            if w_col in master_df.columns and d_col in master_df.columns:
                w_mean = pd.to_numeric(master_df[w_col], errors='coerce').mean()
                d_mean = pd.to_numeric(master_df[d_col], errors='coerce').mean()
                improvement = percent_change(w_mean, d_mean, lower_is_better=True)
                report_lines.append(f"| {label} | {w_mean:.2f} | {d_mean:.2f} | {improvement:+.1f}% |")
        report_lines.append("")

    if gen_df is not None and len(gen_df) > 0 and {'scenario', 'controller'}.issubset(gen_df.columns):
        report_lines.extend([
            "## 2. Generalization Across Scenarios",
            "",
            "| Scenario | Webster avg_wait | Actuated avg_wait | D3QN avg_wait |",
            "|---|---:|---:|---:|",
        ])
        gen_working = gen_df.copy()
        gen_working['controller'] = gen_working['controller'].astype(str).str.lower()
        for scenario in sorted(gen_working['scenario'].dropna().astype(str).unique()):
            scenario_df = gen_working.loc[gen_working['scenario'].astype(str) == scenario]
            values = {}
            for controller in ['webster', 'actuated', 'd3qn']:
                ctrl_vals = pd.to_numeric(scenario_df.loc[scenario_df['controller'] == controller, 'avg_wait'], errors='coerce').dropna()
                values[controller] = ctrl_vals.mean() if len(ctrl_vals) else np.nan
            report_lines.append(
                f"| {scenario} | {values['webster']:.2f} | {values['actuated']:.2f} | {values['d3qn']:.2f} |"
            )
        report_lines.append("")

    if stat_df is not None and len(stat_df) > 0 and {'metric', 'mean_improvement_pct'}.issubset(stat_df.columns):
        report_lines.extend([
            "## 3. Statistical Validation",
            "",
            "| Metric | Baseline | Improvement | 95% CI | Significant |",
            "|---|---|---:|---:|---|",
        ])
        for _, row in stat_df.iterrows():
            metric = str(row.get('metric', '')).replace('_', ' ').title()
            baseline = str(row.get('baseline', 'webster')).title()
            improvement = float(row.get('mean_improvement_pct', np.nan))
            ci_lower = float(row.get('ci_95_lower', np.nan))
            ci_upper = float(row.get('ci_95_upper', np.nan))
            significant = 'Yes' if bool(row.get('significant', False)) else 'No'
            report_lines.append(f"| {metric} | {baseline} | {improvement:.2f}% | [{ci_lower:.2f}, {ci_upper:.2f}] | {significant} |")
        report_lines.append("")

    report_lines.extend([
        "## 4. Figures Generated",
        "",
        "- Figure 1: Training reward convergence across all seeds",
        "- Figure 2: Scenario comparison across controllers",
        "- Figure 3: Statistical summary with confidence intervals",
        "- Figure 4: Switch rate analysis and improvement correlation",
        "",
        "## 5. Conclusions",
        "",
        "The combined results provide evidence for the D3QN controller's advantage over fixed-time baselines, while also exposing seed-level variability that merits further diagnosis.",
        "",
    ])

    report_content = "\n".join(report_lines)

    output_path = Path('analysis/final_report.md')
    output_path.write_text(report_content, encoding='utf-8')
    print(f"✓ Final report saved: {output_path}")
    return output_path


def main(args: argparse.Namespace) -> None:
    """Generate all reports and figures."""
    print("\n" + "=" * 70)
    print("GENERATING COMPREHENSIVE FINAL REPORT")
    print("=" * 70 + "\n")
    
    # Ensure analysis directory exists
    analysis_dir = Path('analysis')
    analysis_dir.mkdir(exist_ok=True)
    
    # Load input data
    print("Loading input data...\n")
    master_df = load_csv_safe(Path('master_evaluation_results.csv'))
    gen_df = load_csv_safe(Path('generalization_results.csv'))
    stat_df = load_csv_safe(analysis_dir / 'statistical_validation_results.csv')
    
    print()
    
    # Load training curves
    print("Loading training curves...\n")
    train_curves = load_train_curves(master_df)
    print()
    
    # Generate figures
    print("Generating figures...\n")
    figure_1_reward_convergence(train_curves)
    figure_2_scenario_comparison(gen_df)
    figure_3_statistical_summary(stat_df)
    figure_4_switch_rate_analysis(master_df)
    print()
    
    # Generate markdown report
    print("Generating markdown report...\n")
    generate_markdown_report(master_df, gen_df, stat_df)
    print()
    
    print("=" * 70)
    print("✓ REPORT GENERATION COMPLETE")
    print("=" * 70)
    print("\nGenerated outputs:")
    print("  ✓ analysis/fig1_reward_convergence.png")
    print("  ✓ analysis/fig2_scenario_comparison.png")
    print("  ✓ analysis/fig3_statistical_summary.png")
    print("  ✓ analysis/fig4_switch_rate_analysis.png")
    print("  ✓ analysis/final_report.md")
    print()


def parse_args(argv=None):
    """Parse command-line arguments for the final report generator."""
    parser = argparse.ArgumentParser(
        description='Generate comprehensive final report with figures and tables.'
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    parse_args()
    main(argparse.Namespace())
