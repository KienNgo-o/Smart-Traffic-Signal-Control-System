# D3QN Smart Traffic Signal Control - Final Report

## Executive Summary

This report summarizes D3QN performance against Webster and Actuated control across the training, evaluation, and generalization experiments.

## 1. Baseline Evaluation Results

| Metric | Webster mean | D3QN mean | Improvement |
|---|---:|---:|---:|
| Avg Wait (s) | 13.18 | 11.47 | +13.0% |
| P95 Wait (s) | 37.00 | 32.00 | +13.5% |
| Avg Time Loss (s) | 30.01 | 28.51 | +5.0% |
| Avg Queue (veh) | 1.16 | 1.05 | +9.9% |

## 2. Generalization Across Scenarios

| Scenario | Webster avg_wait | Actuated avg_wait | D3QN avg_wait |
|---|---:|---:|---:|
| asymmetric | 13.20 | 7.94 | 13.53 |
| high_demand | 13.22 | 17.28 | 11.02 |
| incident | 13.21 | 16.35 | 11.45 |
| low_demand | 11.81 | 7.52 | 10.42 |
| symmetric | 13.15 | 16.17 | 11.39 |

## 3. Statistical Validation

| Metric | Baseline | Improvement | 95% CI | Significant |
|---|---|---:|---:|---|
| Avg Wait | Webster | -2.56% | [-7.31, 1.86] | No |
| Avg Wait | Actuated | -70.56% | [-80.26, -61.55] | No |
| P95 Wait | Webster | 10.61% | [8.59, 12.63] | No |
| P95 Wait | Actuated | -23.94% | [-31.33, -17.61] | No |
| Avg Time Loss | Webster | -6.65% | [-9.38, -3.56] | No |
| Avg Time Loss | Actuated | -40.39% | [-44.84, -35.75] | No |
| Avg Queue | Webster | -2.93% | [-6.75, 0.65] | No |
| Avg Queue | Actuated | -61.17% | [-69.78, -53.50] | No |
| Avg Wait | Webster | 16.60% | [15.96, 17.42] | No |
| Avg Wait | Actuated | 36.20% | [34.68, 37.21] | No |
| P95 Wait | Webster | 14.05% | [13.51, 15.14] | No |
| P95 Wait | Actuated | 41.74% | [40.50, 42.91] | No |
| Avg Time Loss | Webster | 8.02% | [7.71, 8.38] | No |
| Avg Time Loss | Actuated | 20.64% | [19.56, 21.43] | No |
| Avg Queue | Webster | 12.51% | [12.02, 13.31] | No |
| Avg Queue | Actuated | 16.59% | [14.78, 18.24] | No |
| Avg Wait | Webster | 13.33% | [12.70, 14.16] | No |
| Avg Wait | Actuated | 29.92% | [28.30, 30.98] | No |
| P95 Wait | Webster | 13.51% | [13.51, 13.51] | No |
| P95 Wait | Actuated | 37.94% | [36.74, 39.37] | No |
| Avg Time Loss | Webster | 5.60% | [5.06, 6.33] | No |
| Avg Time Loss | Actuated | 15.79% | [14.43, 16.69] | No |
| Avg Queue | Webster | 9.75% | [8.70, 10.62] | No |
| Avg Queue | Actuated | 14.46% | [13.19, 15.57] | No |
| Avg Wait | Webster | 11.73% | [6.31, 16.62] | No |
| Avg Wait | Actuated | -38.68% | [-46.85, -31.40] | No |
| P95 Wait | Webster | 7.25% | [4.68, 9.77] | No |
| P95 Wait | Actuated | -43.35% | [-50.58, -37.96] | No |
| Avg Time Loss | Webster | 4.35% | [1.25, 7.21] | No |
| Avg Time Loss | Actuated | -17.30% | [-20.25, -14.03] | No |
| Avg Queue | Webster | 11.39% | [6.36, 16.05] | No |
| Avg Queue | Actuated | -36.94% | [-43.85, -29.96] | No |
| Avg Wait | Webster | 13.32% | [11.57, 14.95] | No |
| Avg Wait | Actuated | 29.53% | [28.55, 30.44] | No |
| P95 Wait | Webster | 14.50% | [13.51, 16.46] | No |
| P95 Wait | Actuated | 37.14% | [36.17, 38.21] | No |
| Avg Time Loss | Webster | 5.55% | [4.57, 6.52] | No |
| Avg Time Loss | Actuated | 15.88% | [15.33, 16.42] | No |
| Avg Queue | Webster | 10.03% | [8.26, 11.56] | No |
| Avg Queue | Actuated | 14.96% | [13.79, 16.13] | No |

## 4. Figures Generated

- Figure 1: Training reward convergence across all seeds
- Figure 2: Scenario comparison across controllers
- Figure 3: Statistical summary with confidence intervals
- Figure 4: Switch rate analysis and improvement correlation

## 5. Conclusions

The combined results provide evidence for the D3QN controller's advantage over fixed-time baselines, while also exposing seed-level variability that merits further diagnosis.
