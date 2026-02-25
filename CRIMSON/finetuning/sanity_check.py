#!/usr/bin/env python3
"""Sanity-check every entry in a CRIMSON train_data.jsonl file.

Validates:
  1. JSON parse-ability of every line
  2. Required top-level keys and types
  3. raw_evaluation structure (findings lists, errors dict)
  4. Nested item schemas (finding IDs, clinical_significance enums, etc.)
  5. Cross-referential consistency (matched IDs exist, no duplicate IDs)

Usage:
    python sanity_check.py /path/to/train_data.jsonl
    python sanity_check.py /path/to/train_data.jsonl --max-samples 1000
    python sanity_check.py /path/to/train_data.jsonl --verbose
"""

import argparse
import json
import sys
from collections import Counter, defaultdict

# ── Expected schema constants ────────────────────────────────────────
REQUIRED_TOP_KEYS = {"ground_truth", "candidate", "raw_evaluation"}
OPTIONAL_TOP_KEYS = {
    "patient_context", "context_included", "crimson_score",
    "error", "error_counts", "metrics", "weighted_error_counts",
}

REQUIRED_RAW_EVAL_KEYS = {
    "reference_findings", "predicted_findings", "matched_findings", "errors",
}
REQUIRED_ERRORS_KEYS = {"false_findings", "missing_findings", "attribute_errors"}

VALID_CLINICAL_SIGNIFICANCE = {
    "actionable_not_urgent",
    "benign_expected",
    "not_actionable_not_urgent",
    "urgent",
}

VALID_SEVERITY = {"negligible", "significant"}

VALID_ERROR_TYPES = {
    "certainty", "descriptor", "location", "measurement",
    "overinterpretation", "severity", "temporal", "uncertainty", "unspecific",
}

FINDING_REQUIRED_KEYS = {"id", "finding", "clinical_significance"}
MATCHED_REQUIRED_KEYS = {"ref_id", "pred_id"}
ATTR_ERROR_REQUIRED_KEYS = {"ref_id", "pred_id", "severity", "error_types", "explanation"}


# ── Helpers ──────────────────────────────────────────────────────────
def add_error(errors: list, line_num: int, msg: str):
    errors.append((line_num, msg))


def validate_finding(item, prefix: str, line_num: int, errors: list):
    """Validate a single reference_findings or predicted_findings item."""
    if not isinstance(item, dict):
        add_error(errors, line_num, f"{prefix}: item is {type(item).__name__}, expected dict")
        return

    missing = FINDING_REQUIRED_KEYS - set(item.keys())
    if missing:
        add_error(errors, line_num, f"{prefix}: missing keys {missing}")

    fid = item.get("id")
    if fid is not None and not isinstance(fid, str):
        add_error(errors, line_num, f"{prefix}: 'id' is {type(fid).__name__}, expected str")

    finding_text = item.get("finding")
    if finding_text is not None:
        if not isinstance(finding_text, str):
            add_error(errors, line_num, f"{prefix}: 'finding' is {type(finding_text).__name__}, expected str")
        elif not finding_text.strip():
            add_error(errors, line_num, f"{prefix}: 'finding' is empty string")

    cs = item.get("clinical_significance")
    if cs is not None and cs not in VALID_CLINICAL_SIGNIFICANCE:
        add_error(errors, line_num, f"{prefix}: invalid clinical_significance '{cs}'")


def validate_matched(item, prefix: str, line_num: int, errors: list):
    """Validate a single matched_findings item."""
    if not isinstance(item, dict):
        add_error(errors, line_num, f"{prefix}: item is {type(item).__name__}, expected dict")
        return

    missing = MATCHED_REQUIRED_KEYS - set(item.keys())
    if missing:
        add_error(errors, line_num, f"{prefix}: missing keys {missing}")

    for k in ("ref_id", "pred_id"):
        v = item.get(k)
        if v is not None and not isinstance(v, str):
            add_error(errors, line_num, f"{prefix}: '{k}' is {type(v).__name__}, expected str")


def validate_attr_error(item, prefix: str, line_num: int, errors: list):
    """Validate a single attribute_errors item."""
    if not isinstance(item, dict):
        add_error(errors, line_num, f"{prefix}: item is {type(item).__name__}, expected dict")
        return

    missing = ATTR_ERROR_REQUIRED_KEYS - set(item.keys())
    if missing:
        add_error(errors, line_num, f"{prefix}: missing keys {missing}")

    sev = item.get("severity")
    if sev is not None and sev not in VALID_SEVERITY:
        add_error(errors, line_num, f"{prefix}: invalid severity '{sev}'")

    et = item.get("error_types")
    if et is not None:
        if not isinstance(et, list):
            add_error(errors, line_num, f"{prefix}: 'error_types' is {type(et).__name__}, expected list")
        else:
            for val in et:
                if val not in VALID_ERROR_TYPES:
                    add_error(errors, line_num, f"{prefix}: invalid error_type '{val}'")

    expl = item.get("explanation")
    if expl is not None:
        if not isinstance(expl, str):
            add_error(errors, line_num, f"{prefix}: 'explanation' is {type(expl).__name__}, expected str")
        elif not expl.strip():
            add_error(errors, line_num, f"{prefix}: 'explanation' is empty")


# ── Main validation ─────────────────────────────────────────────────
def validate_entry(entry: dict, line_num: int) -> list[tuple[int, str]]:
    """Validate a single JSONL entry. Returns list of (line_num, error_msg)."""
    errors = []

    # ── Top-level ──
    if not isinstance(entry, dict):
        add_error(errors, line_num, f"entry is {type(entry).__name__}, expected dict")
        return errors

    missing_top = REQUIRED_TOP_KEYS - set(entry.keys())
    if missing_top:
        add_error(errors, line_num, f"missing required top-level keys: {missing_top}")

    unexpected = set(entry.keys()) - REQUIRED_TOP_KEYS - OPTIONAL_TOP_KEYS
    if unexpected:
        add_error(errors, line_num, f"unexpected top-level keys: {unexpected}")

    # ground_truth
    gt = entry.get("ground_truth")
    if gt is not None:
        if not isinstance(gt, str):
            add_error(errors, line_num,
                      f"ground_truth is {type(gt).__name__}, expected str")
        elif not gt.strip():
            add_error(errors, line_num, "ground_truth is empty string")

    # candidate
    cand = entry.get("candidate")
    if cand is not None:
        if not isinstance(cand, str):
            add_error(errors, line_num,
                      f"candidate is {type(cand).__name__}, expected str")
        elif not cand.strip():
            add_error(errors, line_num, "candidate is empty string")

    # patient_context (optional — dict with keys like age/sex/indication, or None)
    pc = entry.get("patient_context")
    if pc is not None and not isinstance(pc, (dict, str)):
        add_error(errors, line_num,
                  f"patient_context is {type(pc).__name__}, expected dict, str, or null")
    if isinstance(pc, dict):
        for k, v in pc.items():
            if not isinstance(v, str):
                add_error(errors, line_num,
                          f"patient_context['{k}'] is {type(v).__name__}, expected str")

    # ── raw_evaluation ──
    re_val = entry.get("raw_evaluation")
    if re_val is None:
        add_error(errors, line_num, "raw_evaluation is null")
        return errors
    if not isinstance(re_val, dict):
        add_error(errors, line_num,
                  f"raw_evaluation is {type(re_val).__name__}, expected dict")
        return errors

    missing_re = REQUIRED_RAW_EVAL_KEYS - set(re_val.keys())
    if missing_re:
        add_error(errors, line_num, f"raw_evaluation missing keys: {missing_re}")

    # ── Findings lists ──
    ref_ids, pred_ids = set(), set()

    for list_key, validator in [
        ("reference_findings", validate_finding),
        ("predicted_findings", validate_finding),
    ]:
        items = re_val.get(list_key)
        if items is None:
            continue
        if not isinstance(items, list):
            add_error(errors, line_num,
                      f"raw_evaluation['{list_key}'] is {type(items).__name__}, expected list")
            continue

        id_set = ref_ids if list_key == "reference_findings" else pred_ids
        seen_ids = []
        for j, item in enumerate(items):
            prefix = f"raw_evaluation['{list_key}'][{j}]"
            validator(item, prefix, line_num, errors)
            if isinstance(item, dict):
                fid = item.get("id")
                if fid:
                    if fid in seen_ids:
                        add_error(errors, line_num, f"{prefix}: duplicate id '{fid}'")
                    seen_ids.append(fid)
                    id_set.add(fid)

    # ── Matched findings ──
    matched = re_val.get("matched_findings")
    if matched is not None:
        if not isinstance(matched, list):
            add_error(errors, line_num,
                      f"raw_evaluation['matched_findings'] is {type(matched).__name__}, expected list")
        else:
            for j, item in enumerate(matched):
                prefix = f"raw_evaluation['matched_findings'][{j}]"
                validate_matched(item, prefix, line_num, errors)
                if isinstance(item, dict):
                    rid = item.get("ref_id")
                    pid = item.get("pred_id")
                    if rid and ref_ids and rid not in ref_ids:
                        add_error(errors, line_num,
                                  f"{prefix}: ref_id '{rid}' not in reference_findings")
                    if pid and pred_ids and pid not in pred_ids:
                        add_error(errors, line_num,
                                  f"{prefix}: pred_id '{pid}' not in predicted_findings")

    # ── Errors ──
    errs = re_val.get("errors")
    if errs is None:
        return errors
    if not isinstance(errs, dict):
        add_error(errors, line_num,
                  f"raw_evaluation['errors'] is {type(errs).__name__}, expected dict")
        return errors

    missing_errs = REQUIRED_ERRORS_KEYS - set(errs.keys())
    if missing_errs:
        add_error(errors, line_num, f"raw_evaluation['errors'] missing keys: {missing_errs}")

    # false_findings — list of pred_id strings
    ff = errs.get("false_findings")
    if ff is not None:
        if not isinstance(ff, list):
            add_error(errors, line_num,
                      f"errors['false_findings'] is {type(ff).__name__}, expected list")
        else:
            for j, item in enumerate(ff):
                if not isinstance(item, str):
                    add_error(errors, line_num,
                              f"errors['false_findings'][{j}] is {type(item).__name__}, expected str")
                elif pred_ids and item not in pred_ids:
                    add_error(errors, line_num,
                              f"errors['false_findings'][{j}]: pred_id '{item}' not in predicted_findings")

    # missing_findings — list of ref_id strings
    mf = errs.get("missing_findings")
    if mf is not None:
        if not isinstance(mf, list):
            add_error(errors, line_num,
                      f"errors['missing_findings'] is {type(mf).__name__}, expected list")
        else:
            for j, item in enumerate(mf):
                if not isinstance(item, str):
                    add_error(errors, line_num,
                              f"errors['missing_findings'][{j}] is {type(item).__name__}, expected str")
                elif ref_ids and item not in ref_ids:
                    add_error(errors, line_num,
                              f"errors['missing_findings'][{j}]: ref_id '{item}' not in reference_findings")

    # attribute_errors
    ae = errs.get("attribute_errors")
    if ae is not None:
        if not isinstance(ae, list):
            add_error(errors, line_num,
                      f"errors['attribute_errors'] is {type(ae).__name__}, expected list")
        else:
            for j, item in enumerate(ae):
                prefix = f"errors['attribute_errors'][{j}]"
                validate_attr_error(item, prefix, line_num, errors)
                if isinstance(item, dict):
                    rid = item.get("ref_id")
                    pid = item.get("pred_id")
                    if rid and ref_ids and rid not in ref_ids:
                        add_error(errors, line_num,
                                  f"{prefix}: ref_id '{rid}' not in reference_findings")
                    if pid and pred_ids and pid not in pred_ids:
                        add_error(errors, line_num,
                                  f"{prefix}: pred_id '{pid}' not in predicted_findings")

    return errors


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Validate every entry in a CRIMSON train_data.jsonl"
    )
    parser.add_argument("jsonl_path", help="Path to train_data.jsonl")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Only check the first N lines (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print every individual error (can be long)")
    args = parser.parse_args()

    all_errors: list[tuple[int, str]] = []
    error_type_counts: Counter = Counter()
    lines_with_errors = set()
    json_parse_failures = 0
    total_lines = 0

    print(f"Scanning {args.jsonl_path} ...")
    with open(args.jsonl_path) as f:
        for line_num, line in enumerate(f, start=1):
            if args.max_samples is not None and line_num > args.max_samples:
                break
            total_lines += 1
            line = line.strip()
            if not line:
                continue

            # JSON parse check
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                json_parse_failures += 1
                err_msg = f"JSON parse error: {e}"
                all_errors.append((line_num, err_msg))
                error_type_counts[err_msg] += 1
                lines_with_errors.add(line_num)
                continue

            # Structural validation
            entry_errors = validate_entry(entry, line_num)
            if entry_errors:
                lines_with_errors.add(line_num)
                all_errors.extend(entry_errors)
                for _, msg in entry_errors:
                    error_type_counts[msg.split(":")[0] if ":" in msg else msg] += 1

    # ── Report ──
    print(f"\n{'='*70}")
    print(f"  SANITY CHECK REPORT")
    print(f"{'='*70}")
    print(f"  Total lines scanned:     {total_lines:,}")
    print(f"  Lines with errors:       {len(lines_with_errors):,} "
          f"({len(lines_with_errors)/total_lines*100:.2f}%)" if total_lines else "")
    print(f"  Total error instances:   {len(all_errors):,}")
    print(f"  JSON parse failures:     {json_parse_failures:,}")
    print(f"  Clean lines:             {total_lines - len(lines_with_errors):,}")

    if all_errors:
        print(f"\n{'─'*70}")
        print("  ERROR SUMMARY (by type):")
        print(f"{'─'*70}")
        for msg, count in error_type_counts.most_common():
            print(f"  {count:>6,}x  {msg}")

        if args.verbose:
            print(f"\n{'─'*70}")
            print("  ALL ERRORS (verbose):")
            print(f"{'─'*70}")
            for line_num, msg in all_errors:
                print(f"  Line {line_num:>7,}: {msg}")

        # Show first few problematic lines as examples
        example_lines = sorted(lines_with_errors)[:5]
        print(f"\n{'─'*70}")
        print(f"  FIRST {len(example_lines)} PROBLEMATIC LINES:")
        print(f"{'─'*70}")
        for ln in example_lines:
            line_errors = [(l, m) for l, m in all_errors if l == ln]
            print(f"  Line {ln:,}:")
            for _, msg in line_errors:
                print(f"    - {msg}")

    print(f"\n{'='*70}")
    if not all_errors:
        print("  ✓ ALL LINES PASSED VALIDATION")
    else:
        print(f"  ✗ {len(lines_with_errors):,} LINES HAVE ERRORS")
    print(f"{'='*70}\n")

    sys.exit(1 if all_errors else 0)


if __name__ == "__main__":
    main()
