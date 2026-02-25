import os

# show image filename in UI (debug)
SHOW_IMAGE_NAME = False

# path setup
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# data files
LOCALIZE_JSON = os.path.join(DATA_DIR, 'localize_small.json')
REPORT_METADATA_JSON = os.path.join(DATA_DIR, 'radgame_report.json')

# image directories
LOCALIZE_IMAGE_BASE = os.path.join(BASE_DIR, 'local_sampled')
REPORT_IMAGE_BASE = os.path.join(BASE_DIR, 'rex_sampled_additional_cases')

CRIMSON_ROOT = os.environ.get("CRIMSON_ROOT", os.path.join(BASE_DIR, "CRIMSON"))
# ---------------------------------------------------------------------------
# Report scoring backend
# ---------------------------------------------------------------------------
# Uses finetuned MedGemma-4B with LoRA adapter via the CRIMSON scoring framework.
REPORT_SCORER = "medgemma"

# MedGemma / CRIMSON model paths (only used when REPORT_SCORER == "medgemma")
MEDGEMMA_BASE_MODEL = os.environ.get(
    "MEDGEMMA_BASE_MODEL",
    "google/medgemma-4b-it",
)
MEDGEMMA_LORA_PATH = os.environ.get(
    "MEDGEMMA_LORA_PATH",
    None,  # set via env var; path to LoRA adapter weights
)
MEDGEMMA_CACHE_DIR = os.environ.get(
    "MEDGEMMA_CACHE_DIR",
    None,  # set via env var or leave None for HF default cache
)
MEDGEMMA_MAX_NEW_TOKENS = int(os.environ.get("MEDGEMMA_MAX_NEW_TOKENS", "4096"))

# Inference backend: "transformers" (HuggingFace, default) or "vllm" (faster,
# requires a separate vLLM server running on MEDGEMMA_VLLM_URL).
MEDGEMMA_BACKEND = os.environ.get("MEDGEMMA_BACKEND", "transformers")

# vLLM server URL (OpenAI-compatible endpoint).  The model name sent in the
# request defaults to the base model id; override with MEDGEMMA_VLLM_MODEL
# if you registered the LoRA under a custom name (e.g. "crimson").
MEDGEMMA_VLLM_URL = os.environ.get("MEDGEMMA_VLLM_URL", "http://localhost:8000/v1")
MEDGEMMA_VLLM_MODEL = os.environ.get("MEDGEMMA_VLLM_MODEL", "")

# ---------------------------------------------------------------------------
# Test mode — when True, localize and report always show these fixed cases
# first (in order) so you can anticipate ground truth during development.
# ---------------------------------------------------------------------------
TEST_MODE = True

TEST_LOCALIZE_CASES = [
    "1059090736492172890440690893294928964_qnqec4.png",       # 2 findings: interstitial pattern, pleural thickening
    "3337838038438312879412295722317051049_2_m1m86n.png",     # 8 findings: atelectasis, interstitial pattern, pleural thickening, pleural effusion, cardiomegaly
    "13224141948247255586026463437846237918_zt30no.png",       # 1 finding: hyperinflation
]

TEST_REPORT_CASES = [
    # 3 positive findings: cardiomegaly, tortuous aorta, T7 compression fracture
    "pGRDN2M6YB40JVKXT_aGRDNPXT6PESJ8PN9_s1.2.826.0.1.3680043.8.498.16966475801975550915919249203392254470",
    # 3 positive findings: left base atelectasis, right pleural effusion, thoracic degenerative changes
    "pGRDN0INSFKP87VBD_aGRDNNF6XJLWHO9E1_s1.2.826.0.1.3680043.8.498.36718788597288494251583097359735026737",
    # 3 positive findings: calcified mediastinal nodes, RUL scarring/volume loss, calcified granuloma
    "pGRDN0T0I2H9WNP7I_aGRDN6RC4UVXXAA7T_s1.2.826.0.1.3680043.8.498.27747643274283460905315749455107466194",
]