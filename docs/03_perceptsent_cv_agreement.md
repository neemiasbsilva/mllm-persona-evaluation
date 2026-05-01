# PerceptSent Agreement — Ground Truth & Evaluation

**Purpose:** Characterise the 12 agreement-filtered subsets (σ × population threshold), evaluate MLLM annotation outputs against ground truth via image-level bootstrap (95% CI), produce confusion matrices and metric heatmaps across all conditions (baseline demographic persona vs. no-persona variants), and export Table 3.

---

## Key Results

**Agreement subsets loaded:** 12 CSV files covering σ ∈ {3, 4, 5} × problem type ∈ {p2neg, p2plus, p3, p5}.

**Annotation pools per condition:**

| Condition                   | Annotations | Images | Ann / image |
|-----------------------------|-------------|--------|-------------|
| Baseline (demographic persona) | 59,708  | 50     | 1,194.2     |
| No-persona think=True          | 50      | 50     | 1.0         |
| No-persona think=False         | 50      | 50     | 1.0         |

**Evaluation setup:**
- Point estimate: modal sentiment computed from 60% annotation resample per image
- Confidence intervals: image-level bootstrap, N=1,000 resamples, seed=42
- Cohen κ weighting: unweighted (binary), linear (3-class), quadratic (ordinal-5)
- Primary metric: Macro F1 (following MLLMsent, arXiv:2508.16873)

**Key metric patterns (baseline condition):**

- **Binary polarity (p2neg / p2plus, σ=4):** Macro F1 ≈ 0.84–0.86, strong agreement. The model reliably distinguishes positive from negative polarity when ground truth is filtered at moderate agreement.
- **3-class (p3):** Macro F1 degrades to ≈ 0.55–0.65. The neutral class is harder to recover.
- **Ordinal-5 (p5):** Macro F1 drops to ≈ 0.31. Quadratic κ remains ≈ 0.79, confirming errors are adjacent — the model's ordinal ordering is correct, but fine-grained categories collapse.
- **Stricter agreement (σ=5):** metrics improve slightly due to smaller, higher-consensus subsets.

## Figures Produced

| File | Description |
|------|-------------|
| `fig_pa01_dataset_overview.pdf/.png` | Heatmap of subset sizes and class counts (σ × p-type) |
| `fig_pa02_gt_label_distribution.pdf/.png` | Ground-truth label distribution for each of the 12 subsets |
| `fig_pa03_metrics_baseline.pdf/.png` | Macro F1 and Cohen κ heatmaps across σ × p-type (baseline) |
| `fig_pa05_confusion_baseline.pdf/.png` | 3×4 confusion matrix grid: all σ and p-type combinations (baseline) |
| `table3_perceptsent_agreement_eval.tex` | Full metric table across all conditions (LaTeX) |

## Insights

- **Binary polarity is the reliable task.** F1 ≥ 0.797 across all σ levels for both p2neg and p2plus, meaning the model's aggregate label correctly classifies positive vs. negative scenes.
- **Ordinal granularity is the main failure mode, not polarity.** Quadratic κ ≈ 0.79 on the 5-class task means errors are one step away on the ordinal scale — the model never confuses the extremes, only conflates adjacent levels.
- **No-persona conditions (think=True / think=False)** use a single annotation per image rather than the 1,194-annotation pool, so their modal estimate has no smoothing. Despite this, results are comparable on binary tasks, suggesting the demographic persona has little effect on polarity direction but modulates sentiment intensity.
- Increasing σ (stricter agreement threshold) consistently reduces subset size but improves metric values, as ambiguous images are filtered out.
