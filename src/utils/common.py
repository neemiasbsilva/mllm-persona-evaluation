"""Shared constants, paths, and small helpers used across analysis notebooks."""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"
AGREEMENT_DIR = DATA_DIR / "perceptsent-agreement"
DATASET_JSON = DATA_DIR / "perceptsent-raw" / "dataset.json"
DIST_CSV = DATA_DIR / "baseline_distribution.csv"

# ── Demographic dimension orders (must match baseline_distribution.csv) ──────
GEN_ORDER = ["Female", "Male"]
ECO_ORDER = ["Low income", "High income"]
POL_ORDER = ["Progressive", "Conservative"]
PERS_ORDER = ["Analytical", "Empathetic", "Pragmatic"]
CATEGORICAL_COLS = ["gender", "economic_status", "political_spectrum", "personality"]

# ── Sentiment encoding ───────────────────────────────────────────────────────
SENTIMENT_ORDER = ["Negative", "SlightlyNegative", "Neutral", "SlightlyPositive", "Positive"]
SENTIMENT_INT = {s: i + 1 for i, s in enumerate(SENTIMENT_ORDER)}  # 1..5
INT_SENTIMENT = {v: k for k, v in SENTIMENT_INT.items()}

SENT_COLORS_HEX = {
    "Negative":         "#d73027",
    "SlightlyNegative": "#fc8d59",
    "Neutral":          "#aaaaaa",
    "SlightlyPositive": "#91cf60",
    "Positive":         "#1a9850",
}

# Sentiment maps used by the agreement / cross-validation notebook (03)
SENTIMENT_MAPS = {
    "p2neg": {
        "Positive": 1, "SlightlyPositive": 1, "Neutral": 0,
        "SlightlyNegative": 0, "Negative": 0,
    },
    "p2plus": {
        "Positive": 0, "SlightlyPositive": 0, "Neutral": 0,
        "SlightlyNegative": 1, "Negative": 1,
    },
    "p3": {
        "Positive": 2, "SlightlyPositive": 2, "Neutral": 0,
        "SlightlyNegative": 1, "Negative": 1,
    },
    "p5": {
        "Positive": 4, "SlightlyPositive": 3, "Neutral": 2,
        "SlightlyNegative": 1, "Negative": 0,
    },
}
PROBLEM_TYPE = {
    "p2neg":  "binary",
    "p2plus": "binary",
    "p3":     "multiclass-3",
    "p5":     "ordinal-5",
}
P_ORDER = ["p2neg", "p2plus", "p3", "p5"]


# ── Theme setup ──────────────────────────────────────────────────────────────
def setup_theme(context: str = "paper", font_scale: float = 1.5) -> sns.palettes._ColorPalette:
    """Apply the shared seaborn theme used by the analysis notebooks."""
    sns.set_theme(style="whitegrid", context=context, font_scale=font_scale)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return sns.color_palette("muted")


# ── Figure saving ────────────────────────────────────────────────────────────
def save_fig(fig: plt.Figure, stem: str, figures_dir: Path | None = None) -> None:
    """Save a figure as both PDF and PNG at 300 dpi."""
    target_dir = figures_dir or FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(target_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    print(f"Saved → {stem}  [.pdf + .png]")


# ── Data loading ─────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> pd.DataFrame:
    """Load a JSONL file into a DataFrame; return empty DataFrame if missing."""
    if not Path(path).exists():
        print(f"WARNING: {path} not found.")
        return pd.DataFrame()
    records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return pd.DataFrame(records)


def load_human_gt_modal(dataset_json: Path | None = None) -> pd.DataFrame:
    """Load human ground-truth modal sentiment per image from dataset.json."""
    path = Path(dataset_json or DATASET_JSON)
    if not path.exists():
        print(f"WARNING: {path} not found — human GT not available.")
        return pd.DataFrame()
    with open(path) as fh:
        raw = json.load(fh)
    records = [img for task in raw["tasks"] for img in task["images"]]
    cols = [c for c in ("id", "sentiment", "in_out_door", "perceptions") if c in records[0]]
    gt = pd.DataFrame(records)[cols]
    modal = (
        gt.groupby("id")["sentiment"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"id": "image_id", "sentiment": "human_sentiment"})
    )
    return modal


def load_profile_map(dist_csv: Path | None = None) -> pd.DataFrame:
    """Load persona_id → profile_id mapping; return empty DataFrame if missing."""
    path = Path(dist_csv or DIST_CSV)
    if not path.exists():
        print(f"WARNING: {path} not found — profile_id will be unavailable.")
        return pd.DataFrame(columns=["persona_id", "profile_id"])
    return pd.read_csv(path, usecols=["persona_id", "profile_id"])


# ── Statistical helpers ──────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% binomial CI on the proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p_hat = k / n
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_mean_ci(values, n: int = 1000, seed: int = 42,
                      q_lo: float = 2.5, q_hi: float = 97.5) -> tuple[float, float]:
    """Bootstrap 95% CI for the mean of `values`."""
    arr = np.asarray(values)
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, q_lo)), float(np.percentile(means, q_hi))


def krippendorff_alpha_ordinal(ratings_matrix: np.ndarray) -> float:
    """Krippendorff's alpha for ordinal ratings on a (raters × items) matrix."""
    _, n_items = ratings_matrix.shape
    D_o = 0.0
    n_pairs = 0
    for u in range(n_items):
        col = ratings_matrix[:, u]
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            continue
        for i, j in combinations(range(len(valid)), 2):
            D_o += (valid[i] - valid[j]) ** 2
            n_pairs += 1
    if n_pairs == 0:
        return float("nan")
    D_o /= n_pairs

    all_vals = ratings_matrix[~np.isnan(ratings_matrix)]
    N = len(all_vals)
    if N < 2:
        return float("nan")
    D_e = sum(
        (all_vals[i] - all_vals[j]) ** 2
        for i, j in combinations(range(N), 2)
    ) / (N * (N - 1) / 2)
    return 1.0 if D_e == 0 else 1.0 - D_o / D_e


def modal_ratio(series: pd.Series) -> float:
    """Proportion of values equal to the modal value."""
    if series.empty:
        return float("nan")
    mode = series.value_counts().index[0]
    return (series == mode).mean()


def fmt_ci(lo: float, hi: float) -> str:
    """Format a 95% CI tuple as a human-readable string."""
    if np.isnan(lo) or np.isnan(hi):
        return "—"
    return f"[{lo:.3f}, {hi:.3f}]"
