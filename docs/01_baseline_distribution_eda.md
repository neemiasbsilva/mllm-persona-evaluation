# Baseline Demographic Distribution — Exploratory Analysis

**Purpose:** Validate that the 1,200 synthetic persona seeds form a fully balanced factorial design (2 gender × 2 economic × 2 political × 3 personality = 24 profiles, 50 agents each). Produces demographic distribution figures and the LaTeX summary table used in the paper's Methods section.

---

## Key Results

**Dataset shape:** 1,200 rows × 6 columns (`persona_id`, `profile_id`, `gender`, `economic_status`, `political_spectrum`, `personality`).

**Profile completeness:**
- 24 unique profiles (all cells of the factorial grid populated)
- Exactly 50 agents per profile (min = max = 50)
- 0 duplicate `persona_id` values
- 0 missing values across all categorical columns

**Balance metrics (Shannon entropy):**

| Dimension          | N categories | Balance Ratio | Min agents | Max agents |
|--------------------|-------------|---------------|-----------|-----------|
| Gender             | 2           | 1.0000        | 600       | 600       |
| Economic Status    | 2           | 1.0000        | 600       | 600       |
| Political Spectrum | 2           | 1.0000        | 600       | 600       |
| Personality        | 3           | 1.0000        | 400       | 400       |

Balance Ratio = H_obs / H_max = 1.0 across all dimensions, confirming the design is perfectly balanced — no dimension over-represents any category.

## Figures Produced

| File | Description |
|------|-------------|
| `fig1_profile_completeness.pdf` | Heatmap of agent counts per (gender × economic / political × personality) cell |
| `fig2_univariate_distributions.pdf` | Bar charts of agent share per category for all 4 dimensions |
| `fig3_pairwise_heatmaps.pdf` | All 6 pairwise cross-tabulation heatmaps |
| `fig4_profile_grid.pdf` | Per-subgroup bar charts (gender × economic facets) |
| `fig5_stacked_by_political.pdf` | Stacked % bars: personality / economic / gender by political group |
| `fig6_balance_ratio.pdf` | Horizontal bar chart of Balance Ratio per dimension |
| `fig7_hamming_distance.pdf` | 24 × 24 pairwise Hamming distance matrix across all profiles |
| `fig8_personality_political.pdf` | Grouped bars: personality archetype counts by political orientation |
| `fig9_gender_economic.pdf` | Grouped bar + heatmap: gender × economic status interaction |
| `table1_demographic_distribution.tex` | LaTeX marginal distribution table (Methods section) |

## Insights

- The dataset is a **controlled synthetic experiment**, not a population sample. Equal representation per profile is the design goal, and it is met exactly.
- The Hamming distance matrix (fig7) shows profiles range from 1 to 4 attribute differences. Profiles at distance = 1 form *minimal pairs* — the primary targets for downstream response-variance analysis.
- Analytical, Empathetic, and Pragmatic archetypes are evenly split across both political orientations, ensuring that personality effects can be isolated from political effects in the annotation results.
