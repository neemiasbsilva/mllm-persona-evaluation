# Persona Effects in Multimodal LLMs: Urban Sentiment Annotation

> **Paper:** *Testing Persona Effects in Multimodal LLMs: Do Demographic Personas Produce Distinct Urban Sentiment Judgments?*
> Submitted to UrbCom 2026.

This repository contains all code, data, and pre-computed results for the empirical study described in the paper. The experiment deploys **1,200 synthetic annotators** — defined by a factorial grid of gender, economic status, political orientation, and personality — to evaluate **50 urban scene images** from the [PerceptSent](https://github.com/PerceptSent/PerceptSent) dataset, producing **59,708 structured sentiment annotations**.

---

## Overview

**Research question:** Do demographic persona prompts produce *distinct* and *stable* behavior in multimodal LLMs performing urban image sentiment annotation?

**Short answer:** Within-profile convergence is strong (mean modal ratio 0.871, median 0.980), confirming persona prompts stabilize behavior. Cross-profile differentiation is dimension-dependent: economic status and personality archetype drive variation, while political orientation and gender produce fully overlapping interquartile ranges under tag-based prompting. Agreement with human ground truth is strong for binary polarity (Macro F1 ≥ 0.797) but degrades on ordinal tasks due to systematic collapse of intermediate sentiment classes.

---

## Repository Structure

```
.
├── data/
│   ├── baseline_distribution.csv      # 1,200 seed demographic profiles (24 profiles × 50 agents)
│   ├── generate_demographics.py       # Script that produced baseline_distribution.csv
│   ├── unique_perceptions.json        # 593-label closed-vocabulary for perception annotation
│   ├── perceptsent-raw/
│   │   └── dataset.json               # PerceptSent dataset (5,000 images, human GT annotations)
│   ├── perceptsent-agreement/         # 12 agreement-filtered subsets (σ × problem type)
│   │   └── percept_dataset_sigma{3,4,5}_p{2neg,2plus,p3,p5}.csv
│   └── raw_images/                    # Place PerceptSent JPEG images here (see Data Setup)
│
├── src/
│   ├── persona_generator/             # Phase 1: LangGraph persona synthesis pipeline
│   │   ├── config.py                  # Settings (pydantic-settings, .env)
│   │   ├── state.py                   # PersonaState TypedDict
│   │   ├── graph.py                   # LangGraph graph definition
│   │   ├── pipeline.py                # Batch orchestrator (async, checkpoint-safe)
│   │   ├── nodes/
│   │   │   ├── seeder.py              # Node A: demographic seeder
│   │   │   ├── synthesizer.py         # Node B: backstory synthesizer
│   │   │   ├── reflector.py           # Node C: expert reflection generator
│   │   │   ├── evaluator.py           # Node D: coherence & guardrail evaluator
│   │   │   └── compiler.py            # Node E: final prompt compiler
│   │   └── prompts/
│   │       ├── backstory.py           # Node B prompt templates
│   │       ├── reflection.py          # Node C prompt templates
│   │       └── compiler.py            # Node E prompt templates
│   │
│   └── annotator/                     # Phase 2: LangGraph annotation pipeline
│       ├── config.py                  # Settings (extends persona_generator settings)
│       ├── state.py                   # AnnotationState TypedDict
│       ├── graph.py                   # LangGraph graph definition
│       ├── pipeline.py                # Async batch orchestrator (resume-safe)
│       ├── nodes/
│       │   ├── image_loader.py        # Node 1: load + base64-encode JPEG
│       │   ├── assembler.py           # Node 2: assemble demographic system prompt
│       │   └── annotator.py           # Node 3: vision model call + JSON parse + retry
│       └── prompts/
│           └── vision.py              # Annotation prompt template + perception vocabulary
│
├── scripts/
│   ├── generate_personas.py           # Phase 1 CLI entry point
│   └── run_annotation.py             # Phase 2 CLI entry point
│
├── notebooks/
│   └── 01_baseline_distribution_eda.ipynb   # Demographic profile EDA
│
├── figures/                           # Publication figures (PDF + PNG)
│   ├── fig_dataset_image_distribution.{pdf,png}
│   ├── parse_retry_distribution.{pdf,png}
│   ├── fig9a_sentiment_distribution_baseline.{pdf,png}
│   ├── fig_rq1a_profile_sentiment_heatmap.{pdf,png}
│   ├── rq1_boxplot_per_dimension.{pdf,png}
│   ├── rq1_boxplot_per_profile.{pdf,png}
│   ├── fig_pa03_metrics_baseline.{pdf,png}
│   ├── fig_pa05_confusion_baseline.{pdf,png}
│   └── table{1,2,3}_*.tex             # LaTeX tables
│
├── outputs/
│   ├── annotations_baseline.jsonl     # 59,708 annotation records (45 MB)
│   └── annotation_failures.jsonl      # 292 failed triples with error details
│
├── .env.example                       # Environment variable template
├── pyproject.toml                     # Project dependencies (uv)
└── README.md
```

---

## Experimental Design

### Factorial Persona Grid

| Dimension         | Categories                            | Levels |
|-------------------|---------------------------------------|--------|
| Gender            | Male, Female                          | 2      |
| Economic Status   | Low income, High income               | 2      |
| Political Spectrum| Progressive, Conservative             | 2      |
| Personality       | Analytical, Empathetic, Pragmatic     | 3      |

**24 unique profiles × 50 agents = 1,200 personas** (`random_seed = 42`).

Each persona is defined by four demographic tags injected directly into the system prompt, for example:
```
Gender: Male; Economic status: Low income;
Political leaning: Progressive; Personality archetype: Analytical
```

### Image Dataset

50 stratified images from PerceptSent (stratified by `sentiment × indoor/outdoor`), each evaluated by all 1,200 personas → **60,000 triples targeted, 59,708 completed** (0.49% failure rate).

### Annotation Model

**Qwen3-VL:8B** via Ollama, `temperature=0.1`, `think=True`, `seed=42`, `num_ctx=4096`, timeout 360 s.

Each annotation returns a structured JSON with:
- `sentiment` — 5-class ordinal (Negative → Positive)
- `perceptions` — 1–5 labels from the 593-term closed vocabulary
- `caption` — objective scene description
- `justification` — in-character one-sentence reasoning

---

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) running locally (only required to **re-run** experiments; pre-computed results are included)

### Install

```bash
git clone <repo-url>
cd mllm-persona-simulation-urbcom

# Install dependencies
uv sync

# Copy and configure environment variables
cp .env.example .env
# Edit .env as needed (model names, Ollama URL, concurrency)
```

### Data Setup

The pre-computed annotations in `outputs/annotations_baseline.jsonl` are sufficient for reproducing all figures and analysis. To **re-run** the annotation pipeline you also need the raw images:

1. Download the PerceptSent images from the [PerceptSent repository](https://github.com/PerceptSent/PerceptSent).
2. Place the JPEG files in `data/raw_images/` as `{image_id}.jpg`.
3. Download the agreement-filtered CSV subsets for `data/perceptsent-agreement/` from [Google Drive](https://drive.google.com/drive/folders/1LQAOGI2ojzE5ykjr5WbtDJFM1PQWF9On).
4. Update `IMAGE_DIR` in your `.env` if the path differs.

Note: the repository keeps the `data/perceptsent-agreement/` directory structure, but does not track the CSV files themselves.

---

## Reproducing the Results

### Pre-computed outputs (recommended)

All paper figures were generated from `outputs/annotations_baseline.jsonl`. Run the EDA notebook to reproduce the demographic distribution figures:

```bash
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=600 \
    notebooks/01_baseline_distribution_eda.ipynb \
    --output-dir /tmp/nb_exec/
```

### Re-running Phase 1 — Persona Generation

Requires an Ollama-served text model (paper used `llama3.3:70b`). Generating all 1,200 personas takes several hours depending on hardware.

```bash
# Pull model first
ollama pull llama3.3:70b

# Generate personas (produces outputs/personas_baseline.jsonl)
uv run python scripts/generate_personas.py \
    --csv data/baseline_distribution.csv

# Smoke-test with 5 personas
uv run python scripts/generate_personas.py \
    --csv data/baseline_distribution.csv \
    --limit 5 --max-concurrent 2
```

### Re-running Phase 2 — Annotation

Requires Ollama with `qwen3-vl:8b` and raw images in `data/raw_images/`. The full run took ~28 days on consumer-grade hardware (single-slot inference).

Set Ollama server variables **before** starting `ollama serve`:

```bash
export OLLAMA_NUM_PARALLEL=1        # match --max-concurrent
export OLLAMA_FLASH_ATTENTION=1     # reduces VRAM per slot
export OLLAMA_KV_CACHE_TYPE=q8_0   # halves K/V cache VRAM
ollama serve
```

Then in a separate terminal:

```bash
ollama pull qwen3-vl:8b

# Full paper replication: 1,200 personas × 50 images
uv run python scripts/run_annotation.py \
    --condition baseline \
    --n-personas 1200 --n-images 50

# Smoke-test: 3 personas × 5 images
uv run python scripts/run_annotation.py \
    --condition baseline \
    --n-personas 3 --n-images 5 \
    --limit 10
```

**GPU tuning guide:**

| VRAM  | `--max-concurrent` | `OLLAMA_NUM_PARALLEL` |
|-------|-------------------|----------------------|
| 20 GB | 1–2               | 1–2                  |
| 40 GB | 3–4               | 3–4                  |
| 95 GB | 6–8               | 6–8                  |

The pipeline is **crash-resilient**: on restart it skips `(persona_id, image_id)` pairs already written to the output file.

---

## Output Format

Each line in `annotations_baseline.jsonl` is a JSON object:

```json
{
  "annotation_id":         "p<prefix>_img_<image_id>_baseline",
  "persona_id":            "<uuid>",
  "image_id":              "<perceptsent_image_id>",
  "condition":             "baseline",
  "raw_demographics": {
    "gender":              "Female",
    "economic_status":     "High income",
    "political_spectrum":  "Progressive",
    "personality":         "Empathetic"
  },
  "predicted_sentiment":   "Positive",
  "predicted_perceptions": ["Green/Natural", "Lively"],
  "caption":               "A tree-lined street with pedestrians walking on a sunny day.",
  "justification":         "This kind of vibrant, green space is exactly what every neighborhood deserves.",
  "parse_retries":         0,
  "timestamp_utc":         "2025-01-15T14:23:07Z"
}
```

---

## Key Results

| Metric                          | Value                           |
|---------------------------------|---------------------------------|
| Successful annotations          | 59,708 / 60,000 (99.51%)        |
| Mean within-profile modal ratio | 0.871 ± 0.185                   |
| Median modal ratio              | 0.980                           |
| Binary polarity F1 (σ=4)        | 0.842–0.859                     |
| Ordinal-5 F1 (σ=3)              | 0.309 (granularity failure)     |
| Ordinal quadratic κ (σ=3)       | 0.791 (adjacent errors only)    |

Principal findings:
1. **Personas stabilize behavior** — identically prompted agents produce near-unanimous labels.
2. **Economic status is the primary differentiator** — Low income shifts both mean and median below zero.
3. **Political orientation and gender produce no IQR separation** under tag-based prompting.
4. **Dominant failure mode is granularity, not polarity** — intermediate sentiment classes are collapsed toward the extremes.

---

## Citation

```bibtex
@inproceedings{anonymous2026persona,
  title     = {Testing Persona Effects in Multimodal LLMs: Do Demographic
               Personas Produce Distinct Urban Sentiment Judgments?},
  booktitle = {Proceedings of UrbCom 2026},
  year      = {2026}
}
```

---

## Acknowledgments

This research was partially supported by the SocialNet project (FAPESP process 2023/00148-0), CNPq (processes 314603/2023-9, 441444/2023-7, 409669/2024-5, 444724/2024-9), and the INCT TILD-IAR funded by CNPq (proc. 408490/2024-1).
