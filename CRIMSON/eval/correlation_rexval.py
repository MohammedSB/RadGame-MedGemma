#!/usr/bin/env python3
"""
Compute correlations between automated metrics and expert error counts
from the RExVal dataset with bootstrap confidence intervals (n=10,000).
Outputs results as CSV and JSON.

Supports multiple results directories -- when more than one is given,
metric scores are averaged across runs before computing correlations.
This mitigates variance from non-deterministic LLM inference.

Usage:
    # Single run
    python eval/correlation_rexval.py --results-dir data/rexval_results/run1

    # Multiple runs (scores averaged before correlation)
    python eval/correlation_rexval.py --results-dir data/rexval_results/run1 data/rexval_results/run2 data/rexval_results/run3
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, pearsonr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "radiology-report-expert-evaluation-rexval-dataset-1.0.0"

# Candidate → expected results filename
FILE_MAP = {
    "radgraph":  "results_gt_radgraph.csv",
    "bertscore": "results_gt_bertscore.csv",
    "bleu":      "results_gt_bleu.csv",
    "s_emb":     "results_gt_s_emb.csv",
}

# Columns to skip (not metrics)
SKIP_COLS = {
    "id", "ground_truth", "predicted", "crimson_json", "study_number",
    "crimson_false_findings", "crimson_missing_findings", "crimson_attribute_errors",
    # GREEN sub-error columns (green_total_sig_errors is kept as the aggregate)
    "green_false_finding", "green_missing_finding", "green_location_error",
    "green_severity_error", "green_false_comparison", "green_missing_comparison",
    "green_matched_findings",
}


def load_expert_errors(clinically_significant_only=False):
    """Load and aggregate expert error counts per (study_number, candidate_type)."""
    errors = pd.read_csv(DATA_DIR / "6_valid_raters_per_rater_error_categories.csv")

    if clinically_significant_only:
        errors = errors[errors["clinically_significant"] == True]

    per_rater = (
        errors.groupby(["study_number", "candidate_type", "rater_index"])["num_errors"]
        .sum()
        .reset_index()
    )

    avg = (
        per_rater.groupby(["study_number", "candidate_type"])["num_errors"]
        .mean()
        .reset_index(name="avg_errors")
    )
    return avg


def load_results_single(results_dir, candidate):
    """Load results CSV for a candidate from a single directory, return None if not found."""
    fname = FILE_MAP[candidate]
    path = results_dir / fname

    if not path.exists():
        return None

    df = pd.read_csv(path)
    # Keep only rows with numeric IDs
    df = df[pd.to_numeric(df["id"], errors="coerce").notna()].copy()
    df["id"] = df["id"].astype(int)
    df.rename(columns={"id": "study_number"}, inplace=True)
    return df


def load_results(results_dirs, candidate):
    """Load results for a candidate from one or more directories and average numeric scores.
    
    When multiple directories are provided, numeric metric columns are averaged
    per study_number across runs. Non-numeric columns are taken from the first run.
    """
    dfs = []
    for d in results_dirs:
        df = load_results_single(d, candidate)
        if df is not None:
            dfs.append(df)

    if not dfs:
        print(f"  [skip] {candidate}: file not found in any results directory")
        return None

    if len(dfs) == 1:
        return dfs[0]

    # Average numeric metric columns across runs
    print(f"  [avg] {candidate}: averaging scores across {len(dfs)} runs")
    metric_cols = [c for c in dfs[0].columns if c not in SKIP_COLS and pd.api.types.is_numeric_dtype(dfs[0][c])]
    non_metric_cols = [c for c in dfs[0].columns if c not in metric_cols or c == "study_number"]

    # Stack all runs and group by study_number to average
    for i, df in enumerate(dfs):
        df["_run"] = i
    stacked = pd.concat(dfs, ignore_index=True)
    averaged = stacked.groupby("study_number")[metric_cols].mean().reset_index()

    # Merge non-metric columns from the first run
    base = dfs[0][non_metric_cols].drop_duplicates(subset="study_number")
    merged = base.merge(averaged, on="study_number")
    return merged


def detect_metrics(df):
    """Auto-detect metric columns from the results DataFrame."""
    return [c for c in df.columns if c not in SKIP_COLS and pd.api.types.is_numeric_dtype(df[c])]


def bootstrap_correlation(x, y, corr_func, n_bootstrap=10000, ci_level=0.95):
    """Compute bootstrap confidence intervals for a correlation coefficient."""
    n = len(x)
    bootstrap_corrs = []
    
    rng = np.random.RandomState(42)  # For reproducibility
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n, size=n, replace=True)
        x_boot = x.iloc[indices].values
        y_boot = y.iloc[indices].values
        
        # Compute correlation on bootstrap sample
        if corr_func == kendalltau:
            corr, _ = corr_func(x_boot, y_boot)
        elif corr_func == spearmanr:
            corr, _ = corr_func(x_boot, y_boot)
        elif corr_func == pearsonr:
            corr, _ = corr_func(x_boot, y_boot)
        else:
            corr, _ = corr_func(x_boot, y_boot)
        
        bootstrap_corrs.append(corr)
    
    # Compute confidence intervals
    alpha = 1 - ci_level
    lower = np.percentile(bootstrap_corrs, 100 * alpha / 2)
    upper = np.percentile(bootstrap_corrs, 100 * (1 - alpha / 2))
    
    return lower, upper


def compute_correlations(merged, metrics, n_bootstrap=10000):
    """Compute Kendall, Spearman, Pearson correlations for each metric vs avg_errors.
    
    If n_bootstrap > 0, bootstrap confidence intervals are included.
    If n_bootstrap == 0, only point estimates and p-values are returned.
    """
    rows = []
    for metric in metrics:
        valid = merged[["avg_errors", metric]].dropna()
        if len(valid) < 3:
            continue
        x, y = valid["avg_errors"], valid[metric]
        
        # Compute point estimates
        kt, kp = kendalltau(x, y)
        sr, sp = spearmanr(x, y)
        pr, pp = pearsonr(x, y)
        
        row = {
            "metric": metric,
            "n": len(valid),
            "kendall_tau": round(kt, 4),
            "kendall_p": round(kp, 4),
            "spearman_r": round(sr, 4),
            "spearman_p": round(sp, 4),
            "pearson_r": round(pr, 4),
            "pearson_p": round(pp, 4),
        }
        
        if n_bootstrap > 0:
            kt_lower, kt_upper = bootstrap_correlation(x, y, kendalltau, n_bootstrap)
            sr_lower, sr_upper = bootstrap_correlation(x, y, spearmanr, n_bootstrap)
            pr_lower, pr_upper = bootstrap_correlation(x, y, pearsonr, n_bootstrap)
            row.update({
                "kendall_ci_lower": round(kt_lower, 4),
                "kendall_ci_upper": round(kt_upper, 4),
                "spearman_ci_lower": round(sr_lower, 4),
                "spearman_ci_upper": round(sr_upper, 4),
                "pearson_ci_lower": round(pr_lower, 4),
                "pearson_ci_upper": round(pr_upper, 4),
            })
        
        rows.append(row)
    return rows


def run(results_dirs, output_dir, n_bootstrap=10000):
    output_dir.mkdir(parents=True, exist_ok=True)

    n_runs = len(results_dirs)
    if n_runs > 1:
        print(f"\nAveraging scores across {n_runs} result directories:")
        for d in results_dirs:
            print(f"  - {d}")
    
    if n_bootstrap > 0:
        print(f"Bootstrap CIs enabled (n={n_bootstrap})")
    else:
        print("Bootstrap CIs disabled")

    all_records = []

    for error_mode, sig_only in [("all_errors", False), ("clinically_significant", True)]:
        avg_errors = load_expert_errors(clinically_significant_only=sig_only)
        print(f"\n{'='*60}")
        print(f"  Error mode: {error_mode}")
        print(f"{'='*60}")

        for candidate in FILE_MAP:
            df = load_results(results_dirs, candidate)
            if df is None:
                continue

            metrics = detect_metrics(df)
            if not metrics:
                print(f"  [skip] {candidate}: no numeric metric columns found")
                continue

            merged = avg_errors[avg_errors["candidate_type"] == candidate].merge(
                df, on="study_number"
            )
            if merged.empty:
                print(f"  [skip] {candidate}: no matching study numbers")
                continue

            rows = compute_correlations(merged, metrics, n_bootstrap)
            for r in rows:
                r["candidate"] = candidate
                r["error_mode"] = error_mode
            all_records.extend(rows)

            print(f"\n  --- {candidate} ({error_mode}) ---")
            display_df = pd.DataFrame(rows).set_index("metric").drop(columns=["candidate", "error_mode"])
            print(display_df.to_string())

        # --- Aggregated: stack all candidates into one DataFrame (200 rows) ---
        aggregated_frames = []
        agg_metrics = None
        for candidate in FILE_MAP:
            df = load_results(results_dirs, candidate)
            if df is None:
                continue
            df["candidate_type"] = candidate
            if agg_metrics is None:
                agg_metrics = detect_metrics(df)
            aggregated_frames.append(df)

        if aggregated_frames and agg_metrics:
            all_candidates_df = pd.concat(aggregated_frames, ignore_index=True)
            agg_merged = avg_errors.merge(
                all_candidates_df, on=["study_number", "candidate_type"]
            )
            if not agg_merged.empty:
                rows = compute_correlations(agg_merged, agg_metrics, n_bootstrap)
                for r in rows:
                    r["candidate"] = "aggregated"
                    r["error_mode"] = error_mode
                all_records.extend(rows)

                print(f"\n  --- aggregated ({error_mode}, n={len(agg_merged)}) ---")
                display_df = pd.DataFrame(rows).set_index("metric").drop(columns=["candidate", "error_mode"])
                print(display_df.to_string())

    if not all_records:
        print("\nNo results computed. Check that result CSVs exist in the results directory.")
        return

    # --- Build final DataFrames ---
    full_df = pd.DataFrame(all_records)
    col_order = ["error_mode", "candidate", "metric", "n",
                 "kendall_tau", "kendall_ci_lower", "kendall_ci_upper", "kendall_p",
                 "spearman_r", "spearman_ci_lower", "spearman_ci_upper", "spearman_p",
                 "pearson_r", "pearson_ci_lower", "pearson_ci_upper", "pearson_p"]
    full_df = full_df[[c for c in col_order if c in full_df.columns]]

    # Save detailed CSV
    csv_path = output_dir / "correlation_rexval_detailed.csv"
    full_df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed CSV: {csv_path}")

    # --- Summary pivot: one row per metric, columns = candidates, values = kendall_tau ---
    for error_mode in ["all_errors", "clinically_significant"]:
        subset = full_df[full_df["error_mode"] == error_mode]
        if subset.empty:
            continue

        pivot = subset.pivot_table(
            index="metric", columns="candidate", values="kendall_tau", aggfunc="first"
        )
        # Add mean across candidates
        pivot["mean"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("mean", ascending=False)

        pivot_path = output_dir / f"correlation_summary_kendall_{error_mode}.csv"
        pivot.round(4).to_csv(pivot_path)
        print(f"Saved summary CSV: {pivot_path}")

        print(f"\n  Kendall τ summary ({error_mode}):")
        print(pivot.round(4).to_string())

    # Save full results as JSON
    json_path = output_dir / "correlation_rexval.json"
    with open(json_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nSaved JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="RExVal correlation analysis")
    parser.add_argument(
        "--results-dir", type=str, nargs="+", required=True,
        help="One or more directories containing results CSVs. "
             "When multiple are given, scores are averaged across runs before computing correlations.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results (default: <first-results-dir>/correlations)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=10000,
        help="Number of bootstrap samples for confidence intervals. 0 disables bootstrap CIs (default: 10000).",
    )
    args = parser.parse_args()

    results_dirs = [Path(d) for d in args.results_dir]
    output_dir = Path(args.output_dir) if args.output_dir else results_dirs[0] / "correlations"

    run(results_dirs, output_dir, n_bootstrap=args.n_bootstrap)


if __name__ == "__main__":
    main()
