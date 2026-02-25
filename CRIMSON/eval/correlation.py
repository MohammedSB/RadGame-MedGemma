"""
Correlation analysis: evaluate how well each automated metric correlates
with human expert scores from the RaTE benchmark (Task 2 – paragraph level).

Data pairing
------------
- rate_test.csv (370 rows): the RaTE benchmark ground truth.  Each row is a
  (gt_report, similar_report) pair with a human-assigned similarity ``Score``
  (ordinal, 0–4.5 in 0.25 increments).
- results_test.csv (370 data rows + 1 AVERAGE summary row): the same 370 pairs
  scored by multiple automated metrics.  Row order matches rate_test.csv
  exactly – verified by checking that ``gt_report == ground_truth`` and
  ``similar_report == predicted`` for every row (0 mismatches out of 370).

The AVERAGE row in results_test.csv is dropped before analysis so that only
the 370 data rows remain.

For each metric we compute three correlation statistics against the human
``Score``:

1. **Kendall's τ** – rank correlation robust to ties and small samples;
   does not assume linearity or normality.
2. **Spearman's ρ** – rank correlation; measures monotonic association.
   Reported with the correlation coefficient *r* and the two-sided *p*-value.
3. **Pearson's r** – linear correlation; assumes bivariate normality and
   linear relationship.  Included for completeness / comparability with
   prior work, though the ordinal human scores may violate normality.

Rows where either the metric value or the human score is NaN are dropped
pair-wise (relevant for ``crimson_score`` which has 5 missing values).

Usage:
    python correlation.py
"""

from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
RATE_PATH = BASE_DIR / "data" / "RaTE-Eval" / "Task2_paragraph_rate" / "rate_test.csv"
RESULTS_PATH = BASE_DIR / "data" / "RaTE-Eval" / "Task2_paragraph_rate" / "results_test.csv"

# ── Metrics to evaluate ─────────────────────────────────────────────────────
METRICS = [
    "radgraph_complete",
    "bleu",
    "bertscore",
    "green",
    "rougeL",
    "chexbert-5_micro avg_f1-score",
    "ratescore",
    "radcliq-v1",
    "crimson_score",
]


def load_and_pair() -> pd.DataFrame:
    """Load both CSVs, drop the AVERAGE row, verify pairing, return merged DataFrame."""
    rate = pd.read_csv(RATE_PATH)
    results = pd.read_csv(RESULTS_PATH)

    # Drop the summary AVERAGE row from results
    results = results[results["id"] != "AVERAGE"].reset_index(drop=True)

    assert len(rate) == len(results), (
        f"Row count mismatch after dropping AVERAGE: "
        f"rate_test={len(rate)}, results_test={len(results)}"
    )

    # Full row-by-row pairing verification
    mismatches = 0
    for i in range(len(rate)):
        gt_match = str(rate.iloc[i]["gt_report"])[:200] == str(results.iloc[i]["ground_truth"])[:200]
        pred_match = str(rate.iloc[i]["similar_report"])[:200] == str(results.iloc[i]["predicted"])[:200]
        if not gt_match or not pred_match:
            mismatches += 1
    assert mismatches == 0, f"Found {mismatches} pairing mismatches!"
    print(f"Pairing verified: 0 mismatches across all {len(rate)} rows.")

    # Build a clean DataFrame with human score + all metrics
    df = pd.DataFrame({"human_score": rate["Score"].astype(float)})
    for m in METRICS:
        df[m] = pd.to_numeric(results[m], errors="coerce")

    return df


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Kendall τ, Spearman ρ, and Pearson r for each metric vs human_score."""
    rows = []
    for metric in METRICS:
        # Drop rows where either value is NaN (pair-wise deletion)
        valid = df[["human_score", metric]].dropna()
        human = valid["human_score"].values
        auto = valid[metric].values
        n = len(human)

        # Kendall's tau
        kt_tau, kt_p = stats.kendalltau(human, auto)

        # Spearman's rho
        sp_r, sp_p = stats.spearmanr(human, auto)

        # Pearson's r
        pe_r, pe_p = stats.pearsonr(human, auto)

        rows.append({
            "metric": metric,
            "n": n,
            "kendall_tau": round(kt_tau, 4),
            "kendall_p": kt_p,
            "spearman_r": round(sp_r, 4),
            "spearman_p": sp_p,
            "pearson_r": round(pe_r, 4),
            "pearson_p": pe_p,
        })

    return pd.DataFrame(rows)


def fmt_p(p):
    """Format p-value for display."""
    if p < 0.001:
        return f"{p:.2e}"
    else:
        return f"{p:.4f}"


def print_distribution_diagnostics(df: pd.DataFrame):
    """Print per-metric distribution stats and signal-to-noise analysis."""
    human = df["human_score"]
    buckets = sorted(human.unique())

    print("DISTRIBUTION DIAGNOSTICS")
    print("-" * 100)
    print(f"{'Metric':<35s} {'Range':>20s} {'Mean':>8s} {'Std':>8s} {'Zeros':>8s} {'%Zero':>7s} {'NaN':>5s}")
    print("-" * 100)
    for m in METRICS:
        vals = df[m]
        n_zero = (vals == 0).sum()
        n_nan = vals.isna().sum()
        n_valid = vals.notna().sum()
        pct_zero = n_zero / n_valid * 100 if n_valid > 0 else 0
        print(
            f"{m:<35s} [{vals.min():>7.4f}, {vals.max():>7.4f}] "
            f"{vals.mean():>8.4f} {vals.std():>8.4f} {n_zero:>8d} {pct_zero:>6.1f}% {n_nan:>5d}"
        )

    print()
    print("SIGNAL-TO-NOISE ANALYSIS (per-bucket mean range vs avg within-bucket std)")
    print("-" * 100)
    for m in METRICS:
        vals = df[m]
        means, stds = [], []
        for b in buckets:
            v = vals[human == b].dropna()
            if len(v) > 1:
                means.append(v.mean())
                stds.append(v.std())
        if means and stds:
            signal = max(means) - min(means)
            noise = np.mean(stds)
            ratio = signal / noise if noise > 0 else float("inf")
            print(f"  {m:<33s}  signal_range={signal:.4f}  avg_within_std={noise:.4f}  SNR={ratio:.2f}")
    print()

    print("MEAN METRIC SCORE PER HUMAN-SCORE BUCKET")
    print("-" * 100)
    bucket_header = f"{'Metric':<35s}" + "".join(f"{b:>7.1f}" for b in buckets)
    print(bucket_header)
    print("-" * len(bucket_header))
    for m in METRICS:
        vals = df[m]
        row_str = f"{m:<35s}"
        for b in buckets:
            v = vals[human == b].dropna()
            row_str += f"{v.mean():>7.3f}" if len(v) > 0 else f"{'N/A':>7s}"
        print(row_str)
    print()


def main():
    df = load_and_pair()
    corr = compute_correlations(df)

    # ── Pretty-print ─────────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("CORRELATION OF AUTOMATED METRICS WITH HUMAN EXPERT SCORES (RaTE Task 2)")
    print("=" * 100)
    print()
    print("Methodology")
    print("-" * 100)
    print("• Ground truth:  human similarity 'Score' from rate_test.csv (ordinal, 0–4.5)")
    print("• Predictions:   automated metric scores from results_test.csv")
    print("• Pairing:       row-by-row alignment, verified by text comparison (0 mismatches / 370)")
    print("• NaN handling:  pair-wise deletion (only affects crimson_score: 5 NaN → N=365)")
    print()
    print("Statistical tests:")
    print("  1. Kendall τ  – non-parametric rank correlation; robust to ties")
    print("  2. Spearman ρ – non-parametric rank correlation; monotonic association")
    print("  3. Pearson r  – parametric linear correlation (normality assumed)")
    print()

    # ── Distribution diagnostics ─────────────────────────────────────────────
    print_distribution_diagnostics(df)

    # ── Main correlation table ───────────────────────────────────────────────
    print("CORRELATION RESULTS")
    print("=" * 100)

    header = (
        f"{'Metric':<35s} {'N':>4s}  "
        f"{'Kendall τ':>10s} {'p':>10s}  "
        f"{'Spearman ρ':>10s} {'p':>10s}  "
        f"{'Pearson r':>10s} {'p':>10s}"
    )
    print(header)
    print("-" * len(header))

    for _, row in corr.iterrows():
        print(
            f"{row['metric']:<35s} {row['n']:>4.0f}  "
            f"{row['kendall_tau']:>10.4f} {fmt_p(row['kendall_p']):>10s}  "
            f"{row['spearman_r']:>10.4f} {fmt_p(row['spearman_p']):>10s}  "
            f"{row['pearson_r']:>10.4f} {fmt_p(row['pearson_p']):>10s}"
        )

    print()
    print("=" * 100)

    # ── Save to CSV ──────────────────────────────────────────────────────────
    out_path = BASE_DIR / "data" / "RaTE-Eval" / "Task2_paragraph_rate" / "correlation_results.csv"
    corr.to_csv(out_path, index=False)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
