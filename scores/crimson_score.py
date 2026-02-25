"""
MedGemma-backed CRIMSON scorer for RadGame report evaluation.

Loads a finetuned MedGemma-4B model (base + LoRA adapter) lazily on first
call, builds the CRIMSON evaluation prompt, generates a structured JSON
evaluation, computes the CRIMSON score, and translates the result into
the frontend-compatible format expected by the RadGame report UI.

Toggle via config.py:
    REPORT_SCORER = "medgemma"
"""

import json
import os
import re
import sys
import traceback

# ---------------------------------------------------------------------------
# Ensure the CRIMSON library is importable
# ---------------------------------------------------------------------------
try:
    from config import CRIMSON_ROOT as _CRIMSON_ROOT
except ImportError:
    _CRIMSON_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "CRIMSON")
    )
if _CRIMSON_ROOT not in sys.path:
    sys.path.insert(0, _CRIMSON_ROOT)

# Lazy-import heavy deps so the Flask app can still start even when torch
# / transformers / peft are not yet installed.
_model = None  # will hold the loaded PeftModel
_processor = None
_tokenizer = None
_model_loaded = False

# Re-export the pure scoring function from the CRIMSON library.
# generate_score.py has a top-level `from transformers import pipeline`
# which isn't needed for the scoring logic we use.  If transformers is
# missing we temporarily stub it so the import succeeds.
_transformers_stub_installed = False
try:
    import transformers as _tf_check  # noqa: F401
except ImportError:
    import types
    _stub = types.ModuleType("transformers")
    _stub.pipeline = None
    sys.modules["transformers"] = _stub
    _transformers_stub_installed = True

try:
    from CRIMSON.generate_score import CRIMSONScore
    from CRIMSON.prompt_parts import build_prompt as _build_crimson_prompt
except ImportError as _imp_err:
    CRIMSONScore = None
    _build_crimson_prompt = None
    print(f"[crimson_score] WARNING: Could not import CRIMSON library: {_imp_err}")
finally:
    # Remove the stub so it doesn't leak if transformers is later installed
    if _transformers_stub_installed:
        sys.modules.pop("transformers", None)


# ---------------------------------------------------------------------------
# Model loading (lazy, thread-safe enough for a single-worker Flask app)
# ---------------------------------------------------------------------------
def _load_model(base_model_id: str, lora_path: str, cache_dir: str):
    """Load the base MedGemma model + LoRA adapter.  Called once."""
    global _model, _processor, _tokenizer, _model_loaded

    if _model_loaded:
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    load_kw = {"cache_dir": cache_dir, "trust_remote_code": True}

    print(f"[crimson_score] Loading processor: {base_model_id}")
    _processor = AutoProcessor.from_pretrained(base_model_id, **load_kw)
    _tokenizer = _processor.tokenizer
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    print(f"[crimson_score] Loading base model: {base_model_id}")
    _model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        **load_kw,
    )

    print(f"[crimson_score] Loading LoRA adapter: {lora_path}")
    _model = PeftModel.from_pretrained(_model, lora_path)
    _model.eval()
    _model_loaded = True
    print("[crimson_score] Model loaded and ready for inference.")


def _generate(prompt: str, max_new_tokens: int = 4096) -> str:
    """Run a single forward pass through the loaded model."""
    import torch

    messages = [{"role": "user", "content": prompt}]

    enc = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    device = next(_model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = torch.zeros_like(input_ids)

    with torch.no_grad():
        out = _model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    return _tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


def _generate_vllm(prompt: str, max_new_tokens: int = 4096,
                   vllm_url: str = "", model_name: str = "") -> str:
    """Call a vLLM OpenAI-compatible server for generation."""
    import urllib.request

    url = (vllm_url or _CFG_VLLM_URL).rstrip("/") + "/chat/completions"
    model = model_name or _CFG_VLLM_MODEL or _CFG_MODEL_ID

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0,
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    return body["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Translation: CRIMSON structured output → RadGame frontend format
# ---------------------------------------------------------------------------

def _translate_to_frontend(crimson_result: dict) -> dict:
    """Convert the rich CRIMSON evaluation result into the format
    expected by the RadGame report popup.

    The new frontend format mirrors the 3 native CRIMSON error categories
    (false_findings, missing_findings, attribute_errors) with full
    severity / clinical-significance detail and the raw CRIMSON score
    on a -1 to +1 scale.
    """

    raw_eval = crimson_result.get("raw_evaluation", {})
    errors = raw_eval.get("errors", {})

    # Build lookup maps  ref_id → {text, significance}, pred_id → {text, significance}
    ref_list = raw_eval.get("reference_findings", [])
    pred_list = raw_eval.get("predicted_findings", [])
    ref_map = {r["id"]: r for r in ref_list}
    pred_map = {p["id"]: p for p in pred_list}

    # --- False findings ---
    false_findings = []
    for pid in errors.get("false_findings", []):
        p = pred_map.get(pid, {})
        false_findings.append({
            "id": pid,
            "finding": p.get("finding", pid),
            "clinical_significance": p.get("clinical_significance", "unknown"),
        })

    # --- Missing findings ---
    missing_findings = []
    for rid in errors.get("missing_findings", []):
        r = ref_map.get(rid, {})
        missing_findings.append({
            "id": rid,
            "finding": r.get("finding", rid),
            "clinical_significance": r.get("clinical_significance", "unknown"),
        })

    # --- Attribute errors (with full detail) ---
    attribute_errors = []
    for ae in errors.get("attribute_errors", []):
        ref_text = ref_map.get(ae.get("ref_id", ""), {}).get("finding", "")
        pred_text = pred_map.get(ae.get("pred_id", ""), {}).get("finding", "")
        attribute_errors.append({
            "ref_id": ae.get("ref_id", ""),
            "pred_id": ae.get("pred_id", ""),
            "ref_finding": ref_text,
            "pred_finding": pred_text,
            "severity": ae.get("severity", "unknown"),
            "error_types": ae.get("error_types", []),
            "explanation": ae.get("explanation", ""),
        })

    # --- Matched findings ---
    matched_findings = []
    for m in raw_eval.get("matched_findings", []):
        ref_id = m.get("ref_id", "")
        pred_id = m.get("pred_id", "")
        ref_text = ref_map.get(ref_id, {}).get("finding", ref_id)
        pred_text = pred_map.get(pred_id, {}).get("finding", pred_id)
        matched_findings.append({
            "ref_id": ref_id,
            "pred_id": pred_id,
            "ref_finding": ref_text,
            "pred_finding": pred_text,
        })

    # --- Reference / predicted finding lists for display ---
    reference_findings_display = [
        {"id": r.get("id", ""), "finding": r.get("finding", ""),
         "clinical_significance": r.get("clinical_significance", "unknown")}
        for r in ref_list
    ]
    predicted_findings_display = [
        {"id": p.get("id", ""), "finding": p.get("finding", ""),
         "clinical_significance": p.get("clinical_significance", "unknown")}
        for p in pred_list
    ]

    return {
        "crimson_score": crimson_result.get("crimson_score", 0),
        "error_counts": crimson_result.get("error_counts", {}),
        "weighted_error_counts": crimson_result.get("weighted_error_counts", {}),
        "metrics": crimson_result.get("metrics", {}),
        "false_findings": false_findings,
        "missing_findings": missing_findings,
        "attribute_errors": attribute_errors,
        "matched_findings": matched_findings,
        "reference_findings": reference_findings_display,
        "predicted_findings": predicted_findings_display,
        # Pass through full CRIMSON result for DB storage
        "_crimson_full": crimson_result,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Import central config so callers get the right defaults automatically
try:
    from config import MEDGEMMA_BASE_MODEL as _CFG_MODEL_ID
    from config import MEDGEMMA_LORA_PATH as _CFG_LORA_PATH
    from config import MEDGEMMA_CACHE_DIR as _CFG_CACHE_DIR
    from config import MEDGEMMA_BACKEND as _CFG_BACKEND
    from config import MEDGEMMA_VLLM_URL as _CFG_VLLM_URL
    from config import MEDGEMMA_VLLM_MODEL as _CFG_VLLM_MODEL
except ImportError:
    _CFG_MODEL_ID = "google/medgemma-4b-it"
    _CFG_LORA_PATH = ""
    _CFG_CACHE_DIR = ""
    _CFG_BACKEND = "transformers"
    _CFG_VLLM_URL = "http://localhost:8000/v1"
    _CFG_VLLM_MODEL = ""


def evaluate_report(
    reference_findings: str,
    predicted_findings: str,
    patient_context: dict | None = None,
    *,
    base_model_id: str = _CFG_MODEL_ID,
    lora_path: str = _CFG_LORA_PATH,
    cache_dir: str = _CFG_CACHE_DIR,
    max_new_tokens: int = 4096,
) -> dict:
    """Score a candidate report against a reference using finetuned MedGemma.

    Returns a dict with CRIMSON error categories (false_findings,
    missing_findings, attribute_errors, matched_findings) and the raw
    crimson_score value, plus a ``_crimson_full`` key for DB storage.

    Raises RuntimeError if the CRIMSON library is not available.
    """

    if CRIMSONScore is None or _build_crimson_prompt is None:
        raise RuntimeError(
            "CRIMSON library not available. Ensure "
            f"{_CRIMSON_ROOT} is importable (set CRIMSON_ROOT in config.py)."
        )

    # Determine inference backend
    backend = os.environ.get("MEDGEMMA_BACKEND", _CFG_BACKEND).lower()

    # Ensure model is loaded (lazy init) — only needed for transformers backend
    if backend != "vllm":
        _load_model(base_model_id, lora_path, cache_dir)

    # Build the CRIMSON prompt (matches training-time config: no extra
    # examples / guidelines for the finetuned model)
    prompt = _build_crimson_prompt(
        reference_findings,
        predicted_findings,
        patient_context=patient_context,
        include_significance_examples=False,
        include_attribute_guidelines=False,
        include_context_guidelines=False,
    )

    # Generate
    if backend == "vllm":
        response_text = _generate_vllm(prompt, max_new_tokens=max_new_tokens)
    else:
        response_text = _generate(prompt, max_new_tokens=max_new_tokens)
    print(f"[crimson_score] Raw MedGemma output ({backend}):\n{response_text}\n")

    # Parse JSON from response
    try:
        evaluation = json.loads(response_text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            try:
                evaluation = json.loads(json_match.group())
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON from MedGemma response: {exc}"
                    f"\nResponse:\n{response_text}"
                ) from exc
        else:
            raise ValueError(
                f"No JSON object found in MedGemma response:\n{response_text}"
            )

    # Compute CRIMSON score using the proven scoring logic
    crimson_result = CRIMSONScore._calculate_crimson(None, evaluation)

    # Translate to frontend format
    return _translate_to_frontend(crimson_result)
