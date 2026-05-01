# No-Persona Annotation — Google Colab Setup

**Purpose:** Run the no-persona baseline annotation conditions on Google Colab (T4 GPU). Annotates the same 50 urban-scene images used in the main baseline experiment using `qwen3-vl:8b` *without* any demographic persona conditioning, in two variants: `think=True` (extended reasoning) and `think=False` (direct response). Outputs are used in notebook 03 to compare persona vs. no-persona agreement with human ground truth.

---

## Conditions

| Condition              | Think mode    | Output file                              |
|------------------------|---------------|------------------------------------------|
| `no_persona_think`     | `think=True`  | `annotations_no_persona_think.jsonl`     |
| `no_persona_no_think`  | `think=False` | `annotations_no_persona_no_think.jsonl`  |

Both conditions run on the exact 50 images used in the baseline persona experiment, enforced by passing `--baseline-jsonl` to the annotation script.

## Key Results

**Output pools after full run:**

| Condition              | Annotations | Images | Ann / image |
|------------------------|-------------|--------|-------------|
| No-persona think=True  | 50          | 50     | 1.0         |
| No-persona think=False | 50          | 50     | 1.0         |

Because these conditions run without a persona pool (single annotation per image), they serve as an ablation baseline — isolating the effect of demographic persona conditioning on the annotation result.

## Workflow Summary

1. Mount Google Drive and clone the repository
2. Install `uv` + sync project dependencies
3. Install Ollama and pull `qwen3-vl:8b` (requires T4 GPU runtime)
4. Extract the 50-image archive (`perceptsent_images_50.zip`) from Drive
5. Write `.env` and locate `annotations_baseline.jsonl` to constrain image selection
6. Smoke-test (3 images, think=True)
7. Run full annotation — think=True (cell 9a) → `annotations_no_persona_think.jsonl`
8. Run full annotation — think=False (cell 9b) → `annotations_no_persona_no_think.jsonl`
9. Inspect output row counts and preview last record

## Insights

- Both no-persona conditions produce exactly 1 annotation per image — there is no ensemble smoothing as in the 1,200-persona baseline. Metrics computed in notebook 03 use these raw single predictions directly.
- The `think=True` variant activates extended chain-of-thought reasoning before producing the structured JSON output; `think=False` produces the response directly. Comparing these isolates the effect of explicit reasoning on annotation quality.
- Results show that no-persona conditions achieve comparable binary polarity F1 to the baseline, suggesting that demographic persona tags influence sentiment *intensity* more than *direction*.
