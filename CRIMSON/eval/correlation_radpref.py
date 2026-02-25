#!/usr/bin/env python3
"""
Compute correlations between automated metrics and radiologist preferences
from the RadPref dataset with bootstrap confidence intervals.

RadPref uses *paired* preference data: each case has two candidate reports
(C1, C2) scored against the same ground truth.  For each case we compute
    delta_R    = rating_C2  - rating_C1     (radiologist preference)
    delta_M    = metric_C2  - metric_C1     (metric preference)
and correlate these deltas across cases.

Correlations are computed *per radiologist* as well as on the
average rating across all raters.

Outputs results as CSV and JSON.

Usage:
    python eval/correlation_radpref.py \\
        --scores RadPref/preference_data.json \\
        --annotations-dir RadPref/annotations

    # Specify a custom output directory:
    python eval/correlation_radpref.py \\
        --scores RadPref/preference_data.json \\
        --annotations-dir RadPref/annotations \\
        --output-dir RadPref/correlations
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

# Metric names to detect in the scored JSON (suffix _C1 / _C2 is stripped)
# If --metrics is not given, all *_C1 keys are auto-detected.
NON_METRIC_SUFFIXED_KEYS = {
    "crimson_C1_full", "crimson_C2_full",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scores(path):
    """Load the scored JSON file produced by evaluate_radpref.py.

    Returns (list[dict], list[str]):
        - raw data list
        - detected metric names (without _C1/_C2 suffix)
    """
    with open(path) as f:
        data = json.load(f)

    # Auto-detect metrics: keys ending with _C1 that have a numeric value
    sample = data[0]
    metrics = []
    for key in sample:
        if key in NON_METRIC_SUFFIXED_KEYS:
            continue
        if key.endswith("_C1"):
            base = key[:-3]
            val = sample[key]
            if isinstance(val, (int, float)):
                metrics.append(base)

    return data, sorted(metrics)


def load_annotations_per_rater(annotations_dir):
    """Load radiologist annotations, keeping each rater separate.

    Returns dict[rater_name] -> dict[case_id] -> {"C1_rating": float, "C2_rating": float}
    """
    ann_dir = Path(annotations_dir)
    ann_files = sorted(ann_dir.glob("*.json"))
    if not ann_files:
        raise FileNotFoundError(f"No annotation JSON files found in {ann_dir}")

    per_rater = {}  # rater_name -> {case_id -> {C1_rating, C2_rating}}

    for path in ann_files:
        rater_name = path.stem  # e.g. "user_1"
        with open(path) as f:
            raw = json.load(f)
        annotations = raw.get("annotations", raw)

        rater_ratings = {}
        for case_id, ann in annotations.items():
            if ann.get("needs_review"):
                continue
            c1_rating = ann.get("C1", {}).get("rating")
            c2_rating = ann.get("C2", {}).get("rating")
            if c1_rating is None or c2_rating is None:
                continue
            rater_ratings[case_id] = {
                "C1_rating": c1_rating,
                "C2_rating": c2_rating,
            }

        if rater_ratings:
            per_rater[rater_name] = rater_ratings

    return per_rater


def average_annotations(per_rater):
    """Average ratings across all raters per case.

    Returns dict[case_id] -> {"C1_rating": float, "C2_rating": float}
    """
    accum = {}  # case_id -> {"C1": [vals], "C2": [vals]}
    for rater_ratings in per_rater.values():
        for case_id, r in rater_ratings.items():
            entry = accum.setdefault(case_id, {"C1": [], "C2": []})
            entry["C1"].append(r["C1_rating"])
            entry["C2"].append(r["C2_rating"])

    return {
        cid: {"C1_rating": np.mean(v["C1"]), "C2_rating": np.mean(v["C2"])}
        for cid, v in accum.items()
    }


def build_deltas(data, annotations, metrics):
    """Build paired delta arrays for radiologist ratings and each metric.

    Returns (delta_R, {metric: delta_M}, case_ids) where delta = C2 - C1.
    Only cases present in both data and annotations are included.
    """
    delta_R = []
    deltas = {m: [] for m in metrics}
    case_ids = []

    for sample in data:
        case_id = str(sample["id"])
        if case_id not in annotations:
            continue

        ann = annotations[case_id]
        delta_R.append(ann["C2_rating"] - ann["C1_rating"])

        for m in metrics:
            c1 = sample.get(f"{m}_C1")
            c2 = sample.get(f"{m}_C2")
            if c1 is not None and c2 is not None:
                deltas[m].append(c2 - c1)
            else:
                deltas[m].append(np.nan)

        case_ids.append(case_id)

    delta_R = np.array(delta_R, dtype=float)
    for m in metrics:
        deltas[m] = np.array(deltas[m], dtype=float)

    return delta_R, deltas, case_ids


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def bootstrap_correlation(x, y, corr_func, n_bootstrap=10000, ci_level=0.95, seed=42):
    """Compute bootstrap confidence interval for a correlation coefficient."""
    n = len(x)
    rng = np.random.RandomState(seed)
    boot_corrs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        corr, _ = corr_func(x[idx], y[idx])
        boot_corrs.append(corr)

    alpha = 1 - ci_level
    lower = np.percentile(boot_corrs, 100 * alpha / 2)
    upper = np.percentile(boot_corrs, 100 * (1 - alpha / 2))
    return lower, upper


def compute_correlations(delta_R, deltas, metrics, n_bootstrap=10000):
    """Compute Kendall τ-b, Spearman ρ, Pearson r for each metric vs radiologist deltas.

    Returns list of dicts with point estimates, p-values, bootstrap CIs,
    and directional agreement rate.
    """
    rows = []
    for m in metrics:
        dm = deltas[m]
        mask = ~np.isnan(dm) & ~np.isnan(delta_R)
        if mask.sum() < 3:
            continue

        dr = delta_R[mask]
        dm_clean = dm[mask]

        # Point estimates
        kt, kp = kendalltau(dr, dm_clean, variant="b")
        sr, sp = spearmanr(dr, dm_clean)
        pr, pp = pearsonr(dr, dm_clean)

        # Directional agreement
        agree = np.sum(np.sign(dr) == np.sign(dm_clean))
        agreement_rate = agree / len(dr)

        row = {
            "metric": m,
            "n": int(mask.sum()),
            "kendall_tau": round(kt, 4),
            "kendall_p": round(kp, 6),
            "spearman_r": round(sr, 4),
            "spearman_p": round(sp, 6),
            "pearson_r": round(pr, 4),
            "pearson_p": round(pp, 6),
            "directional_agreement": round(agreement_rate, 4),
        }

        if n_bootstrap > 0:
            kt_lo, kt_hi = bootstrap_correlation(dr, dm_clean, lambda a, b: kendalltau(a, b, variant="b"), n_bootstrap)
            sr_lo, sr_hi = bootstrap_correlation(dr, dm_clean, spearmanr, n_bootstrap)
            pr_lo, pr_hi = bootstrap_correlation(dr, dm_clean, pearsonr, n_bootstrap)
            row.update({
                "kendall_ci_lower": round(kt_lo, 4),
                "kendall_ci_upper": round(kt_hi, 4),
                "spearman_ci_lower": round(sr_lo, 4),
                "spearman_ci_upper": round(sr_hi, 4),
                "pearson_ci_lower": round(pr_lo, 4),
                "pearson_ci_upper": round(pr_hi, 4),
            })

        rows.append(row)

    # Sort by Kendall τ descending
    rows.sort(key=lambda r: r["kendall_tau"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def display_results(results, label, n_bootstrap):
    """Pretty-print correlation results for one rater."""
    print(f"\n{'='*70}")
    print(f"  {label} (delta = C2 - C1)")
    print(f"{'='*70}\n")

    if n_bootstrap > 0:
        print(f"{'Metric':<35} {'τ-b':>6} {'95% CI':>16} {'ρ':>6} {'95% CI':>16} {'r':>6} {'95% CI':>16} {'Agree':>6}")
        print("-" * 115)
        for r in results:
            print(
                f"{r['metric']:<35} "
                f"{r['kendall_tau']:>6.4f} [{r['kendall_ci_lower']:.4f}, {r['kendall_ci_upper']:.4f}] "
                f"{r['spearman_r']:>6.4f} [{r['spearman_ci_lower']:.4f}, {r['spearman_ci_upper']:.4f}] "
                f"{r['pearson_r']:>6.4f} [{r['pearson_ci_lower']:.4f}, {r['pearson_ci_upper']:.4f}] "
                f"{r['directional_agreement']:>5.1%}"
            )
    else:
        print(f"{'Metric':<35} {'τ-b':>6} {'p':>10} {'ρ':>6} {'p':>10} {'r':>6} {'p':>10} {'Agree':>6}")
        print("-" * 100)
        for r in results:
            print(
                f"{r['metric']:<35} "
                f"{r['kendall_tau']:>6.4f} {r['kendall_p']:>10.2e} "
                f"{r['spearman_r']:>6.4f} {r['spearman_p']:>10.2e} "
                f"{r['pearson_r']:>6.4f} {r['pearson_p']:>10.2e} "
                f"{r['directional_agreement']:>5.1%}"
            )


def run(scores_path, annotations_dir, output_dir, n_bootstrap=10000,
        metrics_filter=None, raters_filter=None):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data, detected_metrics = load_scores(scores_path)
    per_rater = load_annotations_per_rater(annotations_dir)

    if metrics_filter:
        metrics = [m for m in metrics_filter if m in detected_metrics]
        missing = set(metrics_filter) - set(detected_metrics)
        if missing:
            print(f"Warning: requested metrics not found in data: {missing}")
    else:
        metrics = detected_metrics

    # Filter to requested raters
    if raters_filter:
        available = set(per_rater.keys())
        missing_raters = set(raters_filter) - available
        if missing_raters:
            print(f"Warning: requested raters not found: {missing_raters}")
            print(f"Available raters: {sorted(available)}")
        per_rater = {r: per_rater[r] for r in raters_filter if r in per_rater}

    rater_names = sorted(per_rater.keys())
    total_annotated = len(set().union(*(per_rater[r].keys() for r in rater_names)))
    print(f"Loaded {len(data)} cases, {total_annotated} annotated across {len(rater_names)} raters")
    print(f"Raters: {rater_names}")
    print(f"Metrics: {metrics}")
    if n_bootstrap > 0:
        print(f"Bootstrap CIs enabled (n={n_bootstrap})")

    all_records = []  # every row gets a "rater" column
    col_order = [
        "rater", "metric", "n",
        "kendall_tau", "kendall_ci_lower", "kendall_ci_upper", "kendall_p",
        "spearman_r", "spearman_ci_lower", "spearman_ci_upper", "spearman_p",
        "pearson_r", "pearson_ci_lower", "pearson_ci_upper", "pearson_p",
        "directional_agreement",
    ]

    # ---- Per-rater correlations ----
    for rater in rater_names:
        annotations = per_rater[rater]
        delta_R, deltas, case_ids = build_deltas(data, annotations, metrics)
        if len(case_ids) == 0:
            print(f"\n[skip] {rater}: no paired cases")
            continue

        results = compute_correlations(delta_R, deltas, metrics, n_bootstrap)
        for r in results:
            r["rater"] = rater
        all_records.extend(results)

        display_results(results, f"{rater} ({len(case_ids)} cases)", n_bootstrap)

    # ---- Averaged across raters ----
    if len(rater_names) > 1:
        avg_annotations = average_annotations(per_rater)
        delta_R, deltas, case_ids = build_deltas(data, avg_annotations, metrics)
        if len(case_ids) > 0:
            results = compute_correlations(delta_R, deltas, metrics, n_bootstrap)
            for r in results:
                r["rater"] = "averaged"
            all_records.extend(results)

            display_results(results, f"Averaged ({len(case_ids)} cases)", n_bootstrap)

    if not all_records:
        print("\nNo results computed. Check that case IDs match between scores and annotations.")
        return

    # ---- Summary table: Kendall τ per (metric, rater) ----
    print(f"\n{'='*70}")
    print("  Summary: Kendall τ-b by rater")
    print(f"{'='*70}")
    summary_df = pd.DataFrame(all_records)
    pivot = summary_df.pivot_table(index="metric", columns="rater", values="kendall_tau", aggfunc="first")
    # Reorder columns: individual raters sorted, then averaged last
    rater_cols = [c for c in sorted(pivot.columns) if c != "averaged"]
    if "averaged" in pivot.columns:
        rater_cols.append("averaged")
    pivot = pivot[rater_cols]
    pivot = pivot.sort_values(rater_cols[-1], ascending=False)
    print(pivot.round(4).to_string())

    # ---- Save CSV ----
    full_df = pd.DataFrame(all_records)
    full_df = full_df[[c for c in col_order if c in full_df.columns]]

    csv_path = output_dir / "correlation_radpref.csv"
    full_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # ---- Save JSON ----
    json_path = output_dir / "correlation_radpref.json"
    with open(json_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"Saved JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="RadPref correlation analysis: metric scores vs radiologist preferences"
    )
    parser.add_argument(
        "--scores", type=str, required=True,
        help="Scored JSON file (output of evaluate_radpref.py) with *_C1/*_C2 metric columns",
    )
    parser.add_argument(
        "--annotations-dir", type=str, required=True,
        help="Directory containing radiologist annotation JSON files",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: same directory as --scores)",
    )
    parser.add_argument(
        "--metrics", type=str, nargs="+", default=None,
        help="Specific metrics to evaluate (default: auto-detect all *_C1 keys)",
    )
    parser.add_argument(
        "--raters", type=str, nargs="+", default=None,
        help="Specific raters to include, e.g. user_1 user_3 (default: all)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=10000,
        help="Bootstrap samples for CIs. 0 disables bootstrap (default: 10000)",
    )
    args = parser.parse_args()

    scores_path = Path(args.scores)
    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir) if args.output_dir else scores_path.parent / "correlations"

    run(scores_path, annotations_dir, output_dir,
        n_bootstrap=args.n_bootstrap, metrics_filter=args.metrics,
        raters_filter=args.raters)


if __name__ == "__main__":
    main()
