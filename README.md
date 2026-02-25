# RadGame

> **Submission for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)**

An interactive web application for training medical professionals in radiology report writing and localization of findings on chest X-rays — powered by **MedGemma**.

## Table of Contents

- [Overview](#radgame-an-ai-powered-platform-for-radiology-education)
- [Installation](#installation)
- [Dataset Generation](#dataset-generation)
- [Generating Localize Explanations](#generating-localize-explanations)
- [Running the Application](#running-the-application)
- [Configuration](#configuration)
- [Finding Classes](#finding-classes)
- [Development](#development)

## RadGame: An AI-Powered Platform for Radiology Education

**RadGame** is an AI-powered, gamified platform designed to teach two core radiology skills: **finding localization** and **report generation**.
The platform integrates large-scale public datasets and AI-driven feedback to deliver **interactive, scalable, and structured learning experiences** for medical trainees.

### Key Features

- **RadGame Localize:**
  Trainees identify and localize abnormalities on chest X-rays by drawing bounding boxes or selecting findings.
  Their annotations are automatically compared against expert radiologist labels from the **PadChest-GR** dataset (de Castro et al., 2025).
  Visual feedback is provided via **MedGemma 4B**, which generates concise explanations for missed or incorrect findings.

- **RadGame Report:**
  Trainees compose structured radiology reports based on chest X-rays, patient age, and indication.
  The system evaluates reports using **CRIMSON**, a context-aware metric adapted from GREEN (Ostmeier et al., 2024), implemented via a **finetuned MedGemma 4B** model with a LoRA adapter.
  The finetuned model (**MedGemmaCRIMSON**) is available on Hugging Face: [CRIMSONScore/medgemma-4b-it-crimson](https://huggingface.co/CRIMSONScore/medgemma-4b-it-crimson).
  Feedback includes a quantitative CRIMSON score and categorized error summaries (matched findings, attribute errors, false findings, and missing findings).

- **Performance Gains:**
  In a multi-institutional study, participants using RadGame achieved a **68% improvement in localization accuracy** and a **31% improvement in report-writing accuracy**, compared to 17% and 4% respectively for traditional passive methods.

### Datasets

RadGame builds upon publicly available datasets:
- **PadChest-GR** (de Castro et al., 2025) — chest radiographs with bounding box annotations.
- **ReXGradient-160K** (Zhang et al., 2025) — paired X-rays and radiologist-written reports.

## Installation

### Prerequisites

- Conda (Anaconda or Miniconda)
- Python 3.11+
- GPU with ≥16 GB VRAM (for MedGemma inference; not needed when using a vLLM server)

### Environment Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RadGame
   ```

2. **Create Conda environment**
   ```bash
   conda create -n radgame python=3.11
   conda activate radgame
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install MedGemma dependencies** (for report scoring or explanation generation)
   ```bash
   pip install torch transformers peft accelerate pillow
   ```

## Dataset Generation

RadGame uses two types of datasets that must be generated from source data before running the app.

### 1. Report Dataset

Generate the report writing dataset from ReXGradient metadata:

```bash
# Set paths to your ReXGradient data
export REX_METADATA="/path/to/rexgradient/metadata/train_metadata.csv"
export REX_TEST_METADATA="/path/to/rexgradient/metadata/test_metadata.json"

# Generate report dataset (default: 50 cases)
python generate_report_dataset.py

# Custom options
python generate_report_dataset.py --nrows 500 --skip-confirm
```

**What it does:**
- Extracts positive findings from radiology reports
- Filters out pediatric patients (<18 years) and reports referencing prior imaging
- Samples cases with target distribution (0–5 findings per report)

**Output:** `data/sample_rex.csv`

### 2. Localize Dataset

Generate the finding localization dataset from PadChest-GR grounded reports:

```bash
# Generate localize dataset (default: 250 images)
python generate_localize_dataset.py --src-dir /path/to/PadChest_GR/

# Custom options
python generate_localize_dataset.py --sample-size 300 --skip-copy
```

**What it does:**
- Filters out rare findings and empty bounding boxes
- Samples images with weighted distribution favoring clinically important findings
- Copies images to `local_sampled/`

**Output:**
- `data/localize_small.json` — sampled dataset manifest
- `local_sampled/` — copied image files

## Generating Localize Explanations

The `medgemma/generate_explanations.py` script uses MedGemma 4B (multimodal) to generate natural-language explanations for each finding in the localize dataset. These explanations are displayed to trainees when they review their annotations.

### Usage

```bash
python medgemma/generate_explanations.py \
  --image_dir local_sampled/ \
  --json_input_path data/localize_small.json \
  --json_output_path data/localize_small_with_explanations.json \
  --model_id google/medgemma-4b-it
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image_dir` | *(required)* | Directory containing the chest X-ray images |
| `--json_input_path` | `data/localize_small.json` | Input JSON with findings and bounding boxes |
| `--json_output_path` | `data/localize_small_with_explanations.json` | Output JSON with added `medgemma_explanation` per finding |
| `--model_id` | `google/medgemma-4b-it` | HuggingFace model ID or local path |
| `--num_samples` | all | Process only the first N images |
| `--save_debug_images` | off | Save images with bounding box overlays for debugging |
| `--debug_image_dir` | `medgemma/overlay_debug` | Where to save debug images |

### How It Works

For each finding with a bounding box:
1. Crops the region of interest from the chest X-ray
2. Draws a red bounding box on the full image for context
3. Sends both images to MedGemma with a prompt asking for a concise radiological explanation
4. Stores the generated explanation in the `medgemma_explanation` field of each finding

The output JSON has the same structure as the input, with an added `medgemma_explanation` key per finding.

## Running the Application

### Start the Flask Server

```bash
conda activate radgame
python app.py
```

The app runs on **port 5000** by default at `http://localhost:5000`.

### Configuration

All key settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `REPORT_SCORER` | `"medgemma"` | Report scoring backend |
| `MEDGEMMA_BASE_MODEL` | `"google/medgemma-4b-it"` | Base model for CRIMSON scoring |
| `MEDGEMMA_LORA_PATH` | *(env var)* | Path to finetuned LoRA adapter |
| `MEDGEMMA_CACHE_DIR` | *(env var)* | HuggingFace model cache directory |
| `MEDGEMMA_BACKEND` | `"transformers"` | `"transformers"` (HF) or `"vllm"` (faster, requires vLLM server) |
| `MEDGEMMA_VLLM_URL` | `"http://localhost:8000/v1"` | vLLM OpenAI-compatible API URL |
| `MEDGEMMA_VLLM_MODEL` | `""` | Model/LoRA name registered with vLLM |
| `TEST_MODE` | `True` | Pin specific cases to front for development |

All paths can be overridden via environment variables of the same name.

### Using vLLM for Faster Inference

For significantly faster report scoring, you can run MedGemma via a vLLM server:

```bash
# Terminal 1: start vLLM with LoRA
python -m vllm.entrypoints.openai.api_server \
  --model google/medgemma-4b-it \
  --enable-lora \
  --lora-modules crimson=/path/to/lora/adapter/ \
  --dtype bfloat16 \
  --port 8000

# Terminal 2: run the app with vLLM backend
export MEDGEMMA_BACKEND=vllm
export MEDGEMMA_VLLM_MODEL=crimson
python app.py
```

## Finding Classes

### Localizable Classes

| Finding | Description |
|---------|-------------|
| Atelectasis | Collapsed or airless lung tissue |
| Bronchiectasis | Permanent dilation of bronchi |
| Bullas | Air-filled spaces in lung parenchyma |
| Calcification | Calcium deposits in tissue |
| Catheter | Medical tube device |
| Consolidation | Dense lung tissue (infection/fluid) |
| Fibrotic band | Scar tissue in lungs |
| Fracture | Broken bone |
| Heart device | Cardiac pacemaker/ICD |
| Hiatal hernia | Stomach protrusion through diaphragm |
| Interstitial pattern | Abnormal lung interstitium texture |
| Infiltration | Abnormal substance in lung tissue |
| Nodule/Mass | Round opacity in lungs |
| Osteosynthesis/suture material | Surgical hardware |
| Pleural thickening | Thickened pleural lining |
| Postoperative change | Post-surgical alterations |
| Prosthesis/endoprosthesis | Artificial body part |
| Tube | Chest tube, ET tube, NG tube |

### Non-Localizable Classes

| Finding | Description |
|---------|-------------|
| Cardiomegaly | Enlarged heart (global finding) |
| Hilar enlargement | Enlarged lung hilum (diffuse) |
| Hyperinflation | Over-expanded lungs (global) |
| Pleural effusion | Fluid in pleural space (gravity-dependent) |
| Pulmonary fibrosis | Lung scarring (diffuse pattern) |
| Pneumothorax | Air in pleural space (can vary) |
| Scoliosis | Spinal curvature (structural) |

## Development

### Project Structure

```
RadGame/
├── app.py                      # Flask web application
├── config.py                   # Centralized configuration
├── models.py                   # SQLAlchemy database models
├── requirements.txt            # Python dependencies
├── scores/
│   └── crimson_score.py        # MedGemma + CRIMSON report scorer
├── medgemma/
│   ├── generate_explanations.py  # Generate localize explanations
│   └── inference.py              # Batch MedGemma inference
├── templates/                  # Jinja2 HTML templates
├── static/                     # CSS, JS, images
├── data/                       # Dataset JSON files
├── generate_report_dataset.py  # Report dataset pipeline
├── generate_localize_dataset.py # Localize dataset pipeline
├── rexgradient/                # ReXGradient data utilities
└── utils/                      # Analysis & processing scripts
```

### Application Routes

| Route | Description |
|-------|-------------|
| `/` | Landing page |
| `/main-menu` | Task selection menu |
| `/report` | Report writing interface |
| `/localize` | Finding localization interface |
| `/admin` | Admin dashboard (requires admin login) |

### Scoring System

- **Report Scoring:** Uses the CRIMSON metric via a finetuned MedGemma 4B model. The CRIMSON score ranges from -1 to +1, where positive scores indicate more correct findings than errors (weighted by clinical significance). See `scores/crimson_score.py`.
- **Localization Scoring:** IoU-based bounding box matching against ground truth. See `make_localize_test_scores.py`.

## Security Note

- Keep `secretcodes.py` and `CRIMSON/api_secrets.py` out of version control (both are in `.gitignore`)
- Use environment variables for sensitive data in production
- Review code for hardcoded credentials before sharing

## Acknowledgements

This work was conducted as part of the Machine Learning for Health (ML4H) 2025 proceedings. We thank the participating institutions and study volunteers for their contributions, and the creators of PadChest-GR and ReXGradient-160K datasets for enabling this research. We also acknowledge the developers of MedGemma 4B and the GREEN metric, whose tools and frameworks informed RadGame's design and evaluation.

**Note**: This application is for educational purposes only and should not be used for clinical decision-making.

## References

- de Castro, D.C., Bustos, A., Bannur, S., Hyland, S.L., Bouzid, K., Wetscherek, M.T., et al. *PadChest-GR: A bilingual chest X-ray dataset for grounded radiology report generation.* **NEJM AI**, 2(7):AIdbp2401120, 2025.
- Zhang, X., Acosta, J.N., Miller, J., Huang, O., Rajpurkar, P. *ReXGradient-160K: A large-scale publicly available dataset of chest radiographs with free-text reports.* arXiv:2505.00228, 2025.
- Ostmeier, S., Xu, J., Chen, Z., Varma, M., Blankemeier, L., et al. *GREEN: Generative Radiology Report Evaluation and Error Notation.* In *Findings of ACL: EMNLP 2024*, pp. 374–390, 2024.
- Sellergren, A., Kazemzadeh, S., Jaroensri, T., Kiraly, A., et al. *MedGemma Technical Report.* arXiv:2507.05201, 2025.

## Bibtex

Please cite **RadGame** whenever you use it.

```bibtex
@article{baharoon2025radgame,
  title={RadGame: An AI-Powered Platform for Radiology Education},
  author={Baharoon, Mohammed and Raissi, Siavash and Jun, John S and Heintz, Thibault and Alabbad, Mahmoud and Alburkani, Ali and Kim, Sung Eun and Kleinschmidt, Kent and Alhumaydhi, Abdulrahman O and Alghamdi, Mohannad Mohammed G and others},
  journal={arXiv preprint arXiv:2509.13270},
  year={2025}
}
```
