# Sentiment Convergence Analysis

**Purpose:** Answer RQ1 — do personas sharing the same demographic profile converge toward the same sentiment label when independently evaluating the same image? Computes within-group modal ratios and sentiment variance per (image, profile) pair, measures agreement with human ground truth via Cohen κ and Macro F1, and produces all convergence figures and Table 2.

---

## Key Results

**Dataset loaded:**
- 59,708 annotation records (`annotations_baseline.jsonl`)
- 5,000 unique images in ground-truth; 50 images used in experiment
- 1,200 personas × 24 unique profiles

**Predicted sentiment distribution (baseline):**

| Sentiment         | % of annotations |
|-------------------|-----------------|
| Negative          | 42.9%           |
| SlightlyNegative  | 6.1%            |
| Neutral           | 11.5%           |
| SlightlyPositive  | 5.1%            |
| Positive          | 34.5%           |

**Human ground-truth distribution (modal per image, 50 images):**

| Sentiment         | % of images |
|-------------------|------------|
| Negative          | 14.2%      |
| SlightlyNegative  | 22.7%      |
| Neutral           | 23.0%      |
| SlightlyPositive  | 19.1%      |
| Positive          | 21.1%      |

The agent distribution is strongly bimodal (Negative + Positive ≈ 77.4%), whereas human GT is approximately uniform. This is the primary source of metric degradation on ordinal tasks.

**Within-group convergence (1,200 image × profile pairs):**

| Metric         | Mean  | Std   | Min   | Median | Max   |
|----------------|-------|-------|-------|--------|-------|
| Modal ratio    | 0.871 | 0.185 | 0.260 | 0.980  | 1.000 |
| Sentiment var  | 0.211 | 0.405 | 0.000 | 0.020  | 3.517 |
| N personas/grp | 49.8  | 1.3   | 33    | 50     | 50    |

A median modal ratio of 0.980 means that in a typical (image, profile) group, 98% of the 50 personas chose the **same** sentiment label — confirming very strong within-profile convergence.

**Agreement with human ground truth (50 images, modal prediction vs. modal human):**

| Metric                   | Value  | 95% CI              |
|--------------------------|--------|---------------------|
| Cohen κ (linear-weighted)| 0.5431 | [0.3864, 0.6805]    |
| Macro F1                 | 0.3094 | [0.2237, 0.3959]    |

Per-class F1 breakdown:
- Negative: F1 = 0.67 (precision 0.50, recall 1.00)
- SlightlyNegative: F1 = 0.00 (precision 0.00, recall 0.00)
- Neutral: F1 = 0.24 (precision 0.33, recall 0.18)
- SlightlyPositive: F1 = 0.00 (precision 0.00, recall 0.00)
- Positive: F1 = 0.65 (precision 0.50, recall 0.91)
- Overall accuracy: 0.46

The dominant failure mode is **granularity collapse**: intermediate classes (SlightlyNegative, SlightlyPositive) are entirely missed. The model correctly identifies polarity direction but conflates fine-grained ordinal differences.

## Figures Produced

| File | Description |
|------|-------------|
| `fig_dataset_a_sentiment.pdf/.png` | Human modal sentiment distribution across 50 experiment images |
| `fig_dataset_b_scene_type.pdf/.png` | Indoor / Outdoor split of the 50 images |
| `fig_dataset_c_heatmap.pdf/.png` | Cross-tabulation: scene type × human sentiment |
| `fig9a_sentiment_agents.pdf/.png` | Predicted sentiment distribution — agents panel |
| `fig9a_sentiment_humans.pdf/.png` | Human GT sentiment distribution — humans panel |
| `fig9a_sentiment_distribution_baseline.pdf/.png` | Side-by-side agents vs. humans |
| `fig10a_confusion_matrix_baseline.pdf/.png` | Row-normalised confusion matrix (modal pred vs. human GT) |
| `fig12a_modal_ratio_heatmap_base.pdf/.png` | Mean modal ratio by economic status × human sentiment |
| `fig_rq1a_profile_sentiment_heatmap.pdf/.png` | Proportion heatmap: 24 profiles × 5 sentiment labels |
| `fig_rq1b_modal_ratio_by_profile.pdf/.png` | Modal ratio ranking with 95% Wilson CI, coloured by dominant sentiment |
| `fig_rq1c_convergence_by_dimension.pdf/.png` | Mean modal ratio per category across all 4 demographic dimensions |
| `table2_convergence_results.tex` | LaTeX convergence + GT-agreement summary table |

## Insights

- **Strong within-profile convergence:** personas with identical demographic tags reliably choose the same label. The design successfully stabilises individual behaviour.
- **Economic status is the primary differentiator** across profiles: Low income profiles skew negative; High income profiles skew positive.
- **Political orientation and gender produce no IQR separation** under tag-based prompting — their modal ratios are statistically indistinguishable from each other.
- **Intermediate sentiment classes are systematically suppressed.** SlightlyNegative and SlightlyPositive receive near-zero F1, indicating the model treats the scale as effectively binary at the extremes rather than as a 5-point ordinal.
- The high quadratic κ (0.791 from notebook 03) indicates errors are adjacent — when the model is wrong, it picks a neighbouring label, not the opposite pole.
