#!/usr/bin/env python3
"""Test that CRIMSONDataset gracefully skips bad entries and keeps good ones.

Builds a mock dataset with one valid entry and one of each error type,
then verifies the dataset loads without crashing and only includes the
valid entries.

Usage:
    python test_dataset_errors.py
"""

import json
import sys
import os

# ── Mock tokenizer (avoids loading MedGemma just for this test) ────
class MockTokenizer:
    """Minimal tokenizer stand-in for testing dataset construction."""
    eos_token_id = 1
    pad_token = "<pad>"
    padding_side = "left"

    def apply_chat_template(self, messages, **kwargs):
        import torch
        # Return a fixed-size "prompt" tensor
        ids = torch.tensor([2, 3, 4, 5, 6], dtype=torch.long)
        return {"input_ids": ids.unsqueeze(0), "attention_mask": torch.ones_like(ids).unsqueeze(0)}

    def __call__(self, text, **kwargs):
        import torch
        # Tokenize target: ~1 token per word
        n = max(1, len(text.split()))
        ids = torch.arange(10, 10 + n, dtype=torch.long)
        return {"input_ids": ids.unsqueeze(0)}


# ── Test data ─────────────────────────────────────────────────────
def make_valid_entry():
    """A fully correct entry."""
    return {
        "ground_truth": "Normal heart size. Clear lungs.",
        "candidate": "Heart size normal. Lungs are clear bilaterally.",
        "patient_context": {"age": "065Y", "sex": "M", "indication": "Chest pain"},
        "raw_evaluation": {
            "reference_findings": [
                {"id": "R1", "finding": "Normal heart size", "clinical_significance": "benign_expected"},
                {"id": "R2", "finding": "Clear lungs", "clinical_significance": "benign_expected"},
            ],
            "predicted_findings": [
                {"id": "P1", "finding": "Heart size normal", "clinical_significance": "benign_expected"},
                {"id": "P2", "finding": "Lungs clear bilaterally", "clinical_significance": "benign_expected"},
            ],
            "matched_findings": [
                {"ref_id": "R1", "pred_id": "P1"},
                {"ref_id": "R2", "pred_id": "P2"},
            ],
            "errors": {
                "false_findings": [],
                "missing_findings": [],
                "attribute_errors": [],
            },
        },
    }


BAD_ENTRIES = {
    # ── Top-level problems ──
    "not_a_dict": "this is a string, not a dict",

    "missing_ground_truth": {
        "candidate": "Some report.",
        "raw_evaluation": make_valid_entry()["raw_evaluation"],
    },

    "missing_candidate": {
        "ground_truth": "Some report.",
        "raw_evaluation": make_valid_entry()["raw_evaluation"],
    },

    "missing_raw_evaluation": {
        "ground_truth": "Some report.",
        "candidate": "Some report.",
    },

    "empty_ground_truth": {
        "ground_truth": "",
        "candidate": "Some report.",
        "raw_evaluation": make_valid_entry()["raw_evaluation"],
    },

    "empty_candidate": {
        "ground_truth": "Some report.",
        "candidate": "   ",
        "raw_evaluation": make_valid_entry()["raw_evaluation"],
    },

    "ground_truth_wrong_type": {
        "ground_truth": 12345,
        "candidate": "Some report.",
        "raw_evaluation": make_valid_entry()["raw_evaluation"],
    },

    # ── raw_evaluation problems ──
    "raw_evaluation_is_string": {
        "ground_truth": "Report A.",
        "candidate": "Report B.",
        "raw_evaluation": "not a dict",
    },

    "raw_evaluation_missing_keys": {
        "ground_truth": "Report A.",
        "candidate": "Report B.",
        "raw_evaluation": {
            "reference_findings": [],
            # missing predicted_findings, matched_findings, errors
        },
    },

    "reference_findings_not_list": {
        "ground_truth": "Report A.",
        "candidate": "Report B.",
        "raw_evaluation": {
            "reference_findings": "not a list",
            "predicted_findings": [],
            "matched_findings": [],
            "errors": {"false_findings": [], "missing_findings": [], "attribute_errors": []},
        },
    },

    "errors_not_dict": {
        "ground_truth": "Report A.",
        "candidate": "Report B.",
        "raw_evaluation": {
            "reference_findings": [],
            "predicted_findings": [],
            "matched_findings": [],
            "errors": "not a dict",
        },
    },

    "errors_missing_subkey": {
        "ground_truth": "Report A.",
        "candidate": "Report B.",
        "raw_evaluation": {
            "reference_findings": [],
            "predicted_findings": [],
            "matched_findings": [],
            "errors": {
                "false_findings": [],
                # missing missing_findings and attribute_errors
            },
        },
    },

    "attribute_errors_not_list": {
        "ground_truth": "Report A.",
        "candidate": "Report B.",
        "raw_evaluation": {
            "reference_findings": [],
            "predicted_findings": [],
            "matched_findings": [],
            "errors": {
                "false_findings": [],
                "missing_findings": [],
                "attribute_errors": "not a list",
            },
        },
    },

    "raw_evaluation_is_none": {
        "ground_truth": "Report A.",
        "candidate": "Report B.",
        "raw_evaluation": None,
    },
}


# ── Test runner ───────────────────────────────────────────────────
def main():
    # Add parent dir for CRIMSON imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dataset import CRIMSONDataset

    valid = make_valid_entry()
    # Also include a second valid entry to confirm >1 works
    valid2 = make_valid_entry()
    valid2["ground_truth"] = "No acute findings."
    valid2["candidate"] = "No acute abnormality detected."

    all_entries = [valid, valid2] + list(BAD_ENTRIES.values())
    n_bad = len(BAD_ENTRIES)
    n_valid = 2

    print(f"Testing CRIMSONDataset with {n_valid} valid + {n_bad} bad entries "
          f"({n_valid + n_bad} total)\n")
    print(f"Bad entry types:")
    for name in BAD_ENTRIES:
        print(f"  - {name}")
    print()

    # ── Construct dataset ──
    tokenizer = MockTokenizer()
    try:
        dataset = CRIMSONDataset(
            data=all_entries,
            tokenizer=tokenizer,
            max_length=512,
        )
    except Exception as e:
        print(f"\n✗ FAILED: Dataset construction crashed!\n  {type(e).__name__}: {e}")
        sys.exit(1)

    # ── Check results ──
    print(f"\nDataset size: {len(dataset)} (expected: {n_valid})")

    # Verify __getitem__ works on all surviving samples
    getitem_ok = True
    for i in range(len(dataset)):
        try:
            item = dataset[i]
            assert "input_ids" in item, "missing input_ids"
            assert "labels" in item, "missing labels"
            assert "attention_mask" in item, "missing attention_mask"
            assert "token_type_ids" in item, "missing token_type_ids"
        except Exception as e:
            print(f"  ✗ __getitem__({i}) failed: {type(e).__name__}: {e}")
            getitem_ok = False

    # ── Verify skip reasons were recorded ──
    print(f"\nSkip reasons recorded:")
    for reason, count in sorted(dataset._skip_reasons.items(), key=lambda x: -x[1]):
        print(f"  {count}x  {reason}")

    # ── Final verdict ──
    print(f"\n{'='*60}")
    all_ok = True

    if len(dataset) != n_valid:
        print(f"  ✗ FAIL: Expected {n_valid} samples, got {len(dataset)}")
        all_ok = False
    else:
        print(f"  ✓ PASS: Correct number of samples ({n_valid})")

    if not getitem_ok:
        print(f"  ✗ FAIL: __getitem__ crashed on some samples")
        all_ok = False
    else:
        print(f"  ✓ PASS: __getitem__ works on all {len(dataset)} samples")

    expected_skipped = n_bad
    actual_skipped = sum(dataset._skip_reasons.values())
    if actual_skipped != expected_skipped:
        print(f"  ✗ FAIL: Expected {expected_skipped} skipped, got {actual_skipped}")
        all_ok = False
    else:
        print(f"  ✓ PASS: All {expected_skipped} bad entries were skipped")

    if all_ok:
        print(f"\n  ✓ ALL TESTS PASSED")
    else:
        print(f"\n  ✗ SOME TESTS FAILED")

    print(f"{'='*60}\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
