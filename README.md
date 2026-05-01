# Stable Behavior, Limited Variation: Persona Validity in LLM Agents for Urban Sentiment Perception

Paper: [arXiv:2604.28048](https://arxiv.org/abs/2604.28048)

---

## Pipeline Overview

```mermaid
flowchart TD
    A["baseline_distribution.csv\n1,200 seed profiles\n(24 profiles × 50 agents)"] --> P1

    subgraph P1["Phase 1 — Persona Generation  (scripts/generate_personas.py)"]
        direction TB
        B["Node A · Seeder\ndemographic tag injection"] --> C["Node B · Synthesizer\nbackstory generation (llama3.3:70b)"]
        C --> D["Node C · Reflector\nexpert reflection"]
        D --> E["Node D · Evaluator\ncoherence & guardrail check"]
        E --> F["Node E · Compiler\nfinal prompt assembly"]
    end

    P1 --> G["personas_baseline.jsonl\n1,200 persona prompts"]

    H["PerceptSent dataset\n50 urban-scene images"] --> P2
    G --> P2

    subgraph P2["Phase 2 — Annotation  (scripts/run_annotation.py)"]
        direction TB
        I["Node 1 · Image Loader\nbase64-encode JPEG"] --> J["Node 2 · Assembler\ndemographic system prompt"]
        J --> K["Node 3 · Annotator\nQwen3-VL:8b  ·  JSON parse + retry"]
    end

    P2 --> L["annotations_baseline.jsonl\n59,708 records"]

    H --> NP

    subgraph NP["No-Persona Baseline  (colab/setup_colab_no_persona.ipynb)"]
        direction TB
        NP1["think=True\nannotations_no_persona_think.jsonl"] 
        NP2["think=False\nannotations_no_persona_no_think.jsonl"]
    end

    subgraph NB["Notebook Analysis"]
        direction TB
        N1["01 · Baseline EDA\ndemographic balance · Hamming distance"]
        N2["02 · Convergence Analysis\nmodal ratio · Cohen κ · Macro F1"]
        N3["03 · PerceptSent Agreement\n12 subsets · σ × p-type evaluation"]
        N4["05 · RQ1 Box Plots\nsentiment by profile & dimension"]
    end

    L --> N1
    L --> N2
    L --> N3
    L --> N4
    NP --> N3

    N1 --> R["figures/  +  docs/"]
    N2 --> R
    N3 --> R
    N4 --> R
```

---

## Setup

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com/) (only to re-run experiments — pre-computed results are included).

```bash
git clone https://github.com/neemiasbsilva/mllm-persona-evaluation.git
cd mllm-persona-evaluation
uv sync
cp .env.example .env   # edit model names, Ollama URL, concurrency
```

---

## Reproducing the Results

### Notebooks (pre-computed outputs — recommended)

All figures were generated from `outputs/annotations_baseline.jsonl`. Run any notebook directly:

```bash
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=600 \
    notebooks/01_baseline_distribution_eda.ipynb \
    --output-dir /tmp/nb_exec/
```

### Phase 1 — Persona Generation

```bash
ollama pull llama3.3:70b

uv run python scripts/generate_personas.py \
    --csv data/baseline_distribution.csv

# Smoke-test (5 personas)
uv run python scripts/generate_personas.py \
    --csv data/baseline_distribution.csv \
    --limit 5 --max-concurrent 2
```

### Phase 2 — Annotation (baseline: 1,200 personas × 50 images)

```bash
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
ollama serve
```

```bash
ollama pull qwen3-vl:8b

uv run python scripts/run_annotation.py \
    --condition baseline \
    --n-personas 1200 --n-images 50

# Smoke-test (3 personas × 5 images)
uv run python scripts/run_annotation.py \
    --condition baseline \
    --n-personas 3 --n-images 5 --limit 10
```

### No-Persona Baselines (think=True and think=False)

```bash
# think=True
uv run python scripts/run_annotation_no_persona.py \
    --think \
    --baseline-jsonl outputs/annotations_baseline.jsonl \
    --max-concurrent 1 \
    --output-dir outputs/

# think=False
uv run python scripts/run_annotation_no_persona.py \
    --no-think \
    --baseline-jsonl outputs/annotations_baseline.jsonl \
    --max-concurrent 1 \
    --output-dir outputs/
```

On Google Colab (T4 GPU), use `colab/setup_colab_no_persona.ipynb` — it handles Ollama installation, image extraction from Drive, and both conditions end-to-end.

**GPU tuning guide:**

| VRAM  | `--max-concurrent` | `OLLAMA_NUM_PARALLEL` |
|-------|-------------------|-----------------------|
| 20 GB | 1–2               | 1–2                   |
| 40 GB | 3–4               | 3–4                   |
| 95 GB | 6–8               | 6–8                   |

The pipeline is **crash-resilient**: on restart it skips `(persona_id, image_id)` pairs already written to the output file.

---

## Data

- **`data/baseline_distribution.csv`** — 1,200 seed demographic profiles (24 profiles × 50 agents, `random_seed=42`)
- **`data/perceptsent-raw/dataset.json`** — PerceptSent dataset metadata (5,000 images, human GT annotations)
- **`data/perceptsent-agreement/`** — 12 agreement-filtered subsets (σ ∈ {3,4,5} × problem type)
- **`data/raw_images/`** — place PerceptSent JPEG files here as `{image_id}.jpg`

Images: download from the [PerceptSent repository](https://github.com/PerceptSent/PerceptSent).

---

## Outputs

Pre-computed results are included. Each line in `outputs/annotations_baseline.jsonl` is a JSON record:

```json
{
  "annotation_id": "p<prefix>_img_<image_id>_baseline",
  "persona_id":    "<uuid>",
  "image_id":      "<perceptsent_image_id>",
  "condition":     "baseline",
  "raw_demographics": {
    "gender": "Female", "economic_status": "High income",
    "political_spectrum": "Progressive", "personality": "Empathetic"
  },
  "predicted_sentiment":   "Positive",
  "predicted_perceptions": ["Green/Natural", "Lively"],
  "caption":               "A tree-lined street with pedestrians walking on a sunny day.",
  "justification":         "This kind of vibrant, green space is exactly what every neighborhood deserves.",
  "parse_retries":         0,
  "timestamp_utc":         "2025-01-15T14:23:07Z"
}
```

Experiment documentation and per-notebook result summaries are in [`docs/`](docs/).

---

## Citation

```bibtex
@misc{dasilva2026stablebehaviorlimitedvariation,
      title={Stable Behavior, Limited Variation: Persona Validity in LLM Agents for Urban Sentiment Perception}, 
      author={Neemias B da Silva and Rodrigo Minetto and Daniel Silver and Thiago H Silva},
      year={2026},
      eprint={2604.28048},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.28048}, 
}
```

---

## Acknowledgments

Supported by the SocialNet project (FAPESP 2023/00148-0), CNPq (314603/2023-9, 441444/2023-7, 409669/2024-5, 444724/2024-9), and INCT TILD-IAR (CNPq 408490/2024-1).
