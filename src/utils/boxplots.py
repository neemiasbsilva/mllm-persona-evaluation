"""Notebook 05 — RQ1 Box Plots & Parse-Retry Analysis."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.common import FIGURES_DIR, OUTPUT_DIR, ROOT, load_jsonl, save_fig

# Ordinal sentiment mapping (numeric scale used for box-plot heights)
SENT_MAP = {
    "Negative": -2,
    "SlightlyNegative": -1,
    "Neutral": 0,
    "SlightlyPositive": 1,
    "Positive": 2,
}
SENT_LABELS = list(SENT_MAP.keys())
SENT_COLORS = {
    -2: "#d62728",
    -1: "#ff7f0e",
     0: "#7f7f7f",
     1: "#2ca02c",
     2: "#1f77b4",
}

GENDER_SHORT = {"Male": "M", "Female": "F"}
ECON_SHORT   = {"Low income": "Low", "High income": "High"}
POL_SHORT    = {"Progressive": "Prog", "Conservative": "Cons"}
PERS_SHORT   = {"Analytical": "Ana", "Empathetic": "Emp", "Pragmatic": "Prag"}


# ── Loaders & preparation ────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_ann      = load_jsonl(OUTPUT_DIR / "annotations_baseline.jsonl")
    df_fail     = load_jsonl(OUTPUT_DIR / "annotation_failures.jsonl")
    df_personas = load_jsonl(OUTPUT_DIR / "personas_baseline.jsonl")
    print(f"Baseline annotations: {len(df_ann):,}")
    print(f"Annotation failures:  {len(df_fail):,}")
    print(f"Personas:             {len(df_personas):,}")
    return df_ann, df_fail, df_personas


def prepare_sentiment_data(df_ann: pd.DataFrame,
                           df_personas: pd.DataFrame) -> pd.DataFrame:
    """Attach numeric sentiment, demographics, profile label."""
    df_ann = df_ann.copy()
    df_ann["sentiment_num"] = df_ann["predicted_sentiment"].map(SENT_MAP)
    df_ann = df_ann.dropna(subset=["sentiment_num"])
    df_ann["sentiment_num"] = df_ann["sentiment_num"].astype(int)

    if isinstance(df_ann["raw_demographics"].iloc[0], str):
        df_ann["raw_demographics"] = df_ann["raw_demographics"].apply(json.loads)

    demo_df = pd.json_normalize(df_ann["raw_demographics"])
    for col in ["gender", "economic_status", "political_spectrum", "personality"]:
        df_ann[col] = demo_df[col].values

    if (not df_personas.empty and "persona_id" in df_personas.columns
            and "profile_id" in df_personas.columns):
        pid_to_profile = dict(zip(df_personas["persona_id"], df_personas["profile_id"]))
        df_ann["profile_id"] = df_ann["persona_id"].map(pid_to_profile)
    else:
        df_ann["profile_id"] = (
            df_ann["gender"] + "|" + df_ann["economic_status"]
            + "|" + df_ann["political_spectrum"] + "|" + df_ann["personality"]
        )

    df_ann["profile_label"] = (
        df_ann["gender"].map(GENDER_SHORT) + "·"
        + df_ann["economic_status"].map(ECON_SHORT) + "·"
        + df_ann["political_spectrum"].map(POL_SHORT) + "·"
        + df_ann["personality"].map(PERS_SHORT)
    )

    print(f"Unique profiles in annotations: {df_ann['profile_id'].nunique()}")
    print(f"Unique personas:                {df_ann['persona_id'].nunique()}")
    print(f"Unique images:                  {df_ann['image_id'].nunique()}")
    return df_ann


# ── Fig: box-plot per profile ────────────────────────────────────────────────
def plot_boxplot_per_profile(df_ann: pd.DataFrame) -> plt.Figure:
    profile_modal = (
        df_ann.groupby("profile_label")["sentiment_num"]
        .agg(lambda x: x.mode().iloc[0])
        .sort_values()
    )
    profile_order = profile_modal.index.tolist()
    modal_colors = [SENT_COLORS[profile_modal[p]] for p in profile_order]

    fig, ax = plt.subplots(figsize=(18, 7))
    sns.boxplot(
        data=df_ann, x="profile_label", y="sentiment_num",
        order=profile_order, palette=modal_colors,
        showfliers=False, width=0.6, ax=ax,
    )
    for i, profile in enumerate(profile_order):
        vals = df_ann.loc[df_ann["profile_label"] == profile, "sentiment_num"]
        q1, med, q3 = vals.quantile(0.25), vals.median(), vals.quantile(0.75)
        mean, n = vals.mean(), len(vals)
        mr = vals.value_counts().max() / n
        ax.text(i, med + 0.08, f"med={med:.0f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="black",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
        ax.text(i + 0.32, q1, f"Q1={q1:.1f}", ha="left", va="center",
                fontsize=8, color="#444444")
        ax.text(i + 0.32, q3, f"Q3={q3:.1f}", ha="left", va="center",
                fontsize=8, color="#444444")
        ax.plot(i, mean, marker="D", color="black", markersize=4, zorder=5)
        ax.text(i - 0.32, mean, f"μ={mean:.2f}", ha="right", va="center",
                fontsize=8, color="#222222", fontstyle="italic")
        ax.text(i, -2.45, f"n={n:,}\nMR={mr:.2f}", ha="center", va="top",
                fontsize=8, color="#555555")

    ax.set_xlabel("Demographic Profile", fontsize=14)
    ax.set_ylabel("Sentiment Score", fontsize=14)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels(["Negative\n(−2)", "Sl. Negative\n(−1)", "Neutral\n(0)",
                        "Sl. Positive\n(+1)", "Positive\n(+2)"])
    ax.set_ylim(-2.7, 2.5)
    plt.xticks(rotation=45, ha="center", fontsize=11)

    legend_patches = [
        mpatches.Patch(color=SENT_COLORS[v], label=k)
        for k, v in SENT_MAP.items()
    ]
    ax.legend(handles=legend_patches, title="Modal Sentiment", loc="upper left",
              fontsize=10, title_fontsize=11)
    plt.tight_layout()
    save_fig(fig, "rq1_boxplot_per_profile")
    return fig


# ── Fig: box-plot per dimension ──────────────────────────────────────────────
DIMENSIONS = [
    ("gender",             ["Male", "Female"]),
    ("economic_status",    ["Low income", "High income"]),
    ("political_spectrum", ["Progressive", "Conservative"]),
    ("personality",        ["Analytical", "Empathetic", "Pragmatic"]),
]


def plot_boxplot_per_dimension(df_ann: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(22, 7), sharey=True)
    for ax, (dim, cats) in zip(axes, DIMENSIONS):
        sns.boxplot(
            data=df_ann, x=dim, y="sentiment_num",
            order=cats, palette="Set2", showfliers=False, width=0.5, ax=ax,
        )
        for j, cat in enumerate(cats):
            vals = df_ann.loc[df_ann[dim] == cat, "sentiment_num"]
            q1, med, q3 = vals.quantile(0.25), vals.median(), vals.quantile(0.75)
            mean, n = vals.mean(), len(vals)
            mr = vals.value_counts().max() / n
            ax.text(j, med + 0.12, f"med={med:.0f}", ha="center", va="bottom",
                    fontsize=13, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
            ax.text(j + 0.28, q1, f"Q1={q1:.1f}", ha="left", va="center",
                    fontsize=12, color="#444444")
            ax.text(j + 0.28, q3, f"Q3={q3:.1f}", ha="left", va="center",
                    fontsize=12, color="#444444")
            ax.plot(j, mean, marker="D", color="black", markersize=5, zorder=5)
            ax.text(j - 0.28, mean, f"μ={mean:.2f}", ha="right", va="center",
                    fontsize=12, color="#222222", fontstyle="italic")
            y_label = (q1 + med) / 2
            ax.text(j, y_label, f"n={n:,}\nMR={mr:.2f}", ha="center", va="center",
                    fontsize=13, color="#333333",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc",
                              linewidth=0.6, pad=3))
        ax.set_xlabel(dim.replace("_", " ").title(), fontsize=15)
        ax.set_ylabel("" if ax != axes[0] else "Sentiment Score", fontsize=15)
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_ylim(-2.5, 2.8)
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=13)
        if ax == axes[0]:
            ax.set_yticklabels(["−2", "−1", "0", "+1", "+2"])
    plt.tight_layout()
    save_fig(fig, "rq1_boxplot_per_dimension")
    return fig


# ── Fig: parse-retry distribution ────────────────────────────────────────────
def plot_parse_retry_distribution(df_ann: pd.DataFrame,
                                  df_fail: pd.DataFrame) -> plt.Figure:
    retry_success = df_ann["parse_retries"].value_counts().sort_index()
    retry_fail = (df_fail["parse_retries"].value_counts().sort_index()
                  if len(df_fail) > 0 else pd.Series(dtype=int))
    retry_df = pd.DataFrame({
        "Successful": retry_success,
        "Failed (exhausted)": retry_fail,
    }).fillna(0).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    retry_df.plot(kind="bar", stacked=True, ax=axes[0],
                  color=["#2ca02c", "#d62728"], edgecolor="white")
    axes[0].set_xlabel("Parse Retries", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].tick_params(axis="x", rotation=0)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%d", fontsize=8)

    if len(df_fail) > 0:
        fail_by_img = df_fail["image_id"].value_counts().head(10)
        fail_by_img.plot(kind="barh", ax=axes[1], color="#d62728", edgecolor="white")
        axes[1].set_xlabel("Failure Count", fontsize=11)
        axes[1].set_ylabel("Image ID (truncated)", fontsize=11)
        axes[1].set_yticklabels(
            [t[:20] + "..." if len(t) > 20 else t for t in fail_by_img.index],
            fontsize=8,
        )
        axes[1].bar_label(axes[1].containers[0], fmt="%d", fontsize=9)
    else:
        axes[1].text(0.5, 0.5, "No failures recorded", ha="center", va="center",
                     fontsize=12, transform=axes[1].transAxes)

    plt.tight_layout()
    save_fig(fig, "parse_retry_distribution")

    total = len(df_ann) + len(df_fail)
    print(f"\nTotal annotation attempts: {total:,}")
    print(f"Successful: {len(df_ann):,} ({len(df_ann)/total*100:.1f}%)")
    print(f"Failed:     {len(df_fail):,} ({len(df_fail)/total*100:.2f}%)")
    print(f"Retries > 0: {(df_ann['parse_retries'] > 0).sum():,}")
    return fig


# ── Orchestrator ─────────────────────────────────────────────────────────────
def run_all() -> None:
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    sns.set_theme(style="whitegrid", font_scale=1.4)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df_ann, df_fail, df_personas = load_data()
    df_ann = prepare_sentiment_data(df_ann, df_personas)
    plot_boxplot_per_profile(df_ann)
    plot_boxplot_per_dimension(df_ann)
    plot_parse_retry_distribution(df_ann, df_fail)


if __name__ == "__main__":
    run_all()
