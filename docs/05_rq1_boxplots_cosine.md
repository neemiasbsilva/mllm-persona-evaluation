# RQ1 Box Plots — Sentiment Distribution by Profile & Dimension

**Purpose:** Produce per-profile and per-dimension sentiment box plots (paper Figures 6 and 7), and the parse/retry failure analysis (Figure 3). Each box shows the ordinal sentiment distribution (−2 to +2) for a demographic profile or dimension category, annotated with median, quartiles, mean, sample size, and modal ratio.

---

## Key Results

**Annotation pool:**
- 59,708 successful annotations (baseline)
- 292 failures
- 24 unique profiles, 1,200 unique personas, 50 unique images

**Ordinal sentiment distribution (numeric scale −2 to +2):**

| Score | Label            | Count  |
|-------|------------------|--------|
| −2    | Negative         | 25,604 |
| −1    | SlightlyNegative |  3,630 |
|  0    | Neutral          |  6,857 |
| +1    | SlightlyPositive |  3,041 |
| +2    | Positive         | 20,576 |

The distribution is strongly bimodal at the extremes (−2 and +2 account for 77.4% of annotations), with intermediate labels suppressed — consistent with the granularity-collapse finding in notebook 02.

**Parse/retry failure analysis:**
- Total annotation attempts: 60,000
- Successful: 59,708 (99.51%)
- Failed (retries exhausted): 292 (0.49%)
- The vast majority of annotations required zero retries; failures were uniformly distributed across images with no single image responsible for a disproportionate share.

## Figures Produced

| File | Description |
|------|-------------|
| `rq1_boxplot_per_profile.pdf/.png` | Box plots for all 24 demographic profiles, ordered by modal sentiment and coloured by dominant label |
| `rq1_boxplot_per_dimension.pdf/.png` | Box plots for each category within gender, economic status, political spectrum, and personality |
| `parse_retry_distribution.pdf/.png` | Stacked bar of retry counts (successful vs. failed) + top-10 failure images |

## Insights

- **Economic status drives the largest shift.** Low income profiles have consistently lower median sentiment scores and higher Negative modal ratios than High income profiles. This dimension produces clear IQR separation between the two categories.
- **Personality archetype creates a secondary split.** Analytical personas tend toward more neutral/negative scores, while Empathetic personas score slightly higher on average. Pragmatic personas fall in between.
- **Gender and political orientation produce overlapping IQRs.** The box plots confirm that neither Male vs. Female nor Progressive vs. Conservative produces a statistically distinguishable shift under tag-based prompting alone.
- **Modal ratios annotated on each box** show that profiles dominated by Negative sentiment converge most strongly (MR near 1.0), while profiles with mixed or Neutral dominance show more spread.
- **The 0.49% failure rate is negligible** and is not concentrated in any particular image, persona type, or demographic group — the annotation pipeline is robust.
