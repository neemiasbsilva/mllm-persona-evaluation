"""Notebook 01 — Baseline Demographic Distribution EDA.

Each public function generates one figure (or table) from the existing notebook,
preserving the original chart style, palettes, sizes, and saved file names.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy

from utils.common import (
    CATEGORICAL_COLS, ECO_ORDER, FIGURES_DIR, GEN_ORDER, PERS_ORDER, POL_ORDER,
    ROOT, save_fig, setup_theme,
)


def load_baseline(csv_path: Path | None = None) -> pd.DataFrame:
    """Load the baseline distribution CSV and apply categorical ordering."""
    path = csv_path or (ROOT / "data" / "baseline_distribution.csv")
    df = pd.read_csv(path)
    expected = {"persona_id", "profile_id", *CATEGORICAL_COLS}
    assert set(df.columns) >= expected, f"Missing columns: {expected - set(df.columns)}"
    df["gender"]             = pd.Categorical(df["gender"],             categories=GEN_ORDER,  ordered=False)
    df["economic_status"]    = pd.Categorical(df["economic_status"],    categories=ECO_ORDER,  ordered=True)
    df["political_spectrum"] = pd.Categorical(df["political_spectrum"], categories=POL_ORDER,  ordered=False)
    df["personality"]        = pd.Categorical(df["personality"],        categories=PERS_ORDER, ordered=False)
    return df


def describe_dataset(df: pd.DataFrame) -> None:
    """Print shape, missing-value counts, and profile inventory."""
    print(f"Shape: {df.shape}")
    print("Missing values per column:")
    print(df[CATEGORICAL_COLS].isna().sum())
    agents_per = df.groupby("profile_id").size()
    print(f"\nDuplicate persona_ids: {df['persona_id'].duplicated().sum()}")
    print(f"Unique profile_ids:    {df['profile_id'].nunique()}")
    print(f"Agents per profile:    min={agents_per.min()}, max={agents_per.max()}")


# ── Fig 1: Profile completeness ──────────────────────────────────────────────
def plot_profile_completeness(df: pd.DataFrame, palette=None) -> plt.Figure:
    pivot = (
        df.assign(
            row_key=df["gender"].astype(str) + "  /  " + df["economic_status"].astype(str),
            col_key=df["political_spectrum"].astype(str) + "  /  " + df["personality"].astype(str),
        )
        .groupby(["row_key", "col_key"])
        .size()
        .unstack(fill_value=0)
    )
    row_order = [f"{g}  /  {e}" for g in GEN_ORDER for e in ECO_ORDER]
    col_order = [f"{p}  /  {a}" for p in POL_ORDER for a in PERS_ORDER]
    pivot = pivot.reindex(index=row_order, columns=col_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(
        pivot, annot=True, fmt="d", cmap="Blues",
        linewidths=0.6, linecolor="white",
        cbar_kws={"label": "Agents"}, vmin=0, ax=ax,
    )
    ax.set_xlabel("Political Spectrum  /  Personality")
    ax.set_ylabel("Gender  /  Economic Status")
    ax.tick_params(axis="both", labelsize=9)
    plt.tight_layout()
    save_fig(fig, "fig1_profile_completeness")
    return fig


# ── Fig 2: Univariate distributions ──────────────────────────────────────────
def _plot_univariate(df: pd.DataFrame, col: str, order: list, ax: plt.Axes, color) -> None:
    counts = df[col].value_counts().reindex(order).fillna(0)
    pcts = counts / counts.sum() * 100
    bars = ax.bar(order, pcts, color=color, edgecolor="white", linewidth=0.5)
    uniform = 100 / len(order)
    ax.axhline(uniform, ls="--", color="crimson", lw=1.2, alpha=0.7,
               label=f"Uniform ({uniform:.1f}%)")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Share of agents (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, pcts.max() * 1.3)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)


def plot_univariate_distributions(df: pd.DataFrame, palette) -> plt.Figure:
    plots = [
        ("gender",             GEN_ORDER,  palette[0]),
        ("economic_status",    ECO_ORDER,  palette[1]),
        ("political_spectrum", POL_ORDER,  palette[2]),
        ("personality",        PERS_ORDER, palette[3]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, (col, order, color) in zip(axes, plots):
        _plot_univariate(df, col, order, ax, color)
    plt.tight_layout()
    save_fig(fig, "fig2_univariate_distributions")
    return fig


# ── Fig 3: Pairwise heatmaps ─────────────────────────────────────────────────
def plot_pairwise_heatmaps(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    pairs = [
        ("political_spectrum", POL_ORDER,  "personality",      PERS_ORDER),
        ("political_spectrum", POL_ORDER,  "economic_status",  ECO_ORDER),
        ("political_spectrum", POL_ORDER,  "gender",           GEN_ORDER),
        ("personality",        PERS_ORDER, "economic_status",  ECO_ORDER),
        ("personality",        PERS_ORDER, "gender",           GEN_ORDER),
        ("economic_status",    ECO_ORDER,  "gender",           GEN_ORDER),
    ]
    for ax, (row_col, row_ord, col_col, col_ord) in zip(axes, pairs):
        ct = pd.crosstab(df[row_col], df[col_col]).reindex(
            index=row_ord, columns=col_ord, fill_value=0
        )
        sns.heatmap(
            ct, annot=True, fmt="d", cmap="Blues",
            linewidths=0.4, linecolor="white",
            cbar_kws={"label": "Agents"}, vmin=0, ax=ax,
        )
        ax.set_xlabel(col_col.replace("_", " ").title())
        ax.set_ylabel(row_col.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    save_fig(fig, "fig3_pairwise_heatmaps")
    return fig


# ── Fig 4: Profile grid ──────────────────────────────────────────────────────
def plot_profile_grid(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, (gender, eco) in zip(
        axes.flatten(),
        [(g, e) for g in GEN_ORDER for e in ECO_ORDER],
    ):
        subset = df[(df["gender"] == gender) & (df["economic_status"] == eco)]
        ct = (
            pd.crosstab(subset["political_spectrum"], subset["personality"])
            .reindex(index=POL_ORDER, columns=PERS_ORDER, fill_value=0)
        )
        ct.plot(kind="bar", ax=ax, colormap="Set2",
                edgecolor="white", linewidth=0.4, width=0.7)
        ax.set_ylabel("Agents")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Personality", fontsize=8, loc="upper right")
        ax.set_ylim(0, ct.values.max() * 1.5)
    plt.tight_layout()
    save_fig(fig, "fig4_profile_grid")
    return fig


# ── Fig 5: Stacked-by-political ──────────────────────────────────────────────
def _stacked_bar(df: pd.DataFrame, row_col: str, row_ord: list,
                 col_col: str, col_ord: list, ax: plt.Axes) -> None:
    ct = pd.crosstab(df[row_col], df[col_col]).reindex(
        index=row_ord, columns=col_ord, fill_value=0
    )
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot(kind="bar", stacked=True, ax=ax,
                colormap="tab10", edgecolor="white", linewidth=0.4)
    ax.set_ylabel("% within group")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title=col_col.replace("_", " ").title(),
              bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)


def plot_stacked_by_political(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _stacked_bar(df, "political_spectrum", POL_ORDER, "personality",     PERS_ORDER, axes[0])
    _stacked_bar(df, "political_spectrum", POL_ORDER, "economic_status", ECO_ORDER,  axes[1])
    _stacked_bar(df, "political_spectrum", POL_ORDER, "gender",          GEN_ORDER,  axes[2])
    plt.tight_layout()
    save_fig(fig, "fig5_stacked_by_political")
    return fig


# ── Fig 6: Balance ratio ─────────────────────────────────────────────────────
def compute_balance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, order in {
        "gender": GEN_ORDER, "economic_status": ECO_ORDER,
        "political_spectrum": POL_ORDER, "personality": PERS_ORDER,
    }.items():
        counts = df[col].value_counts().reindex(order).fillna(0).values
        probs = counts / counts.sum()
        H_obs = entropy(probs, base=2)
        H_max = np.log2(len(order))
        rows.append({
            "Dimension":     col.replace("_", " ").title(),
            "N Categories":  len(order),
            "H_obs (bits)":  round(H_obs, 4),
            "H_max (bits)":  round(H_max, 4),
            "Balance Ratio": round(H_obs / H_max, 4),
            "Min agents":    int(counts.min()),
            "Max agents":    int(counts.max()),
        })
    return pd.DataFrame(rows)


def plot_balance_ratio(balance_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [
        "#2ecc71" if r >= 0.99 else "#f39c12" if r >= 0.90 else "#e74c3c"
        for r in balance_df["Balance Ratio"]
    ]
    bars = ax.barh(balance_df["Dimension"], balance_df["Balance Ratio"],
                   color=colors, edgecolor="white")
    ax.axvline(1.0, ls="--", color="grey",  lw=1, alpha=0.6, label="Perfect balance (1.0)")
    ax.axvline(0.9, ls=":",  color="green", lw=1, alpha=0.5, label="Good balance (0.9)")
    for bar, val in zip(bars, balance_df["Balance Ratio"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel(r"Balance Ratio  ($H_{obs} / H_{max}$)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_fig(fig, "fig6_balance_ratio")
    return fig


# ── Fig 7: Hamming distance ──────────────────────────────────────────────────
def _short_code(row: pd.Series) -> str:
    g = row["gender"][0]
    e = "Lo" if "Low" in row["economic_status"] else "Hi"
    p = "Pr" if row["political_spectrum"] == "Progressive" else "Co"
    a = row["personality"][:2]
    return f"{g}-{e}-{p}-{a}"


def plot_hamming_distance(df: pd.DataFrame) -> plt.Figure:
    profiles = (
        df.drop_duplicates("profile_id")
        .sort_values("profile_id")
        .reset_index(drop=True)
    )
    profiles["code"] = profiles.apply(_short_code, axis=1)
    n = len(profiles)
    ham = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            ham[i, j] = sum(
                profiles.iloc[i][col] != profiles.iloc[j][col]
                for col in CATEGORICAL_COLS
            )
    ham_df = pd.DataFrame(ham, index=profiles["code"], columns=profiles["code"])
    fig, ax = plt.subplots(figsize=(15, 13))
    sns.heatmap(
        ham_df, annot=True, fmt="d", cmap="RdYlGn_r",
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Hamming Distance (# differing attributes, 0–4)"},
        ax=ax,
    )
    ax.tick_params(axis="both", labelsize=7.5)
    plt.tight_layout()
    save_fig(fig, "fig7_hamming_distance")
    return fig


# ── Fig 8: Personality × political ───────────────────────────────────────────
def plot_personality_political(df: pd.DataFrame) -> plt.Figure:
    ct = pd.crosstab(df["political_spectrum"], df["personality"]).reindex(
        index=POL_ORDER, columns=PERS_ORDER, fill_value=0
    )
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ct.plot(kind="bar", ax=axes[0], colormap="Set2", edgecolor="white",
            linewidth=0.5, width=0.7)
    axes[0].set_ylabel("Agents")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(title="Personality", bbox_to_anchor=(1.01, 1),
                   loc="upper left", fontsize=9)

    ct_pct.plot(kind="bar", ax=axes[1], colormap="Set2", edgecolor="white",
                linewidth=0.5, width=0.7)
    axes[1].set_ylabel("% within political group")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="Personality", bbox_to_anchor=(1.01, 1),
                   loc="upper left", fontsize=9)
    plt.tight_layout()
    save_fig(fig, "fig8_personality_political")
    return fig


# ── Fig 9: Gender × economic ─────────────────────────────────────────────────
def plot_gender_economic(df: pd.DataFrame) -> plt.Figure:
    ct = pd.crosstab(df["gender"], df["economic_status"]).reindex(
        index=GEN_ORDER, columns=ECO_ORDER, fill_value=0
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ct.plot(kind="bar", ax=axes[0], colormap="Set1", edgecolor="white",
            linewidth=0.5, width=0.6)
    axes[0].set_ylabel("Agents")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(title="Economic Status", bbox_to_anchor=(1.01, 1),
                   loc="upper left", fontsize=9)
    sns.heatmap(
        ct, annot=True, fmt="d", cmap="Blues",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Agents"}, ax=axes[1],
    )
    axes[1].set_xlabel("Economic Status")
    axes[1].set_ylabel("Gender")
    axes[1].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    save_fig(fig, "fig9_gender_economic")
    return fig


# ── Summary table ────────────────────────────────────────────────────────────
def summary_table(df: pd.DataFrame, figures_dir: Path | None = None) -> pd.DataFrame:
    rows = []
    for col, order in [
        ("gender",             GEN_ORDER),
        ("economic_status",    ECO_ORDER),
        ("political_spectrum", POL_ORDER),
        ("personality",        PERS_ORDER),
    ]:
        counts = df[col].value_counts().reindex(order).fillna(0).astype(int)
        for cat, n in counts.items():
            rows.append({
                "Dimension":  col.replace("_", " ").title(),
                "Category":   cat,
                "n":          n,
                "% of total": f"{n / len(df) * 100:.1f}%",
            })
    table = pd.DataFrame(rows)
    latex = table.to_latex(
        index=False,
        caption=(
            "Marginal distribution of the 1,008 demographic agent personas "
            "(balanced factorial design: 24 profiles × 42 agents each)."
        ),
        label="tab:demographic_distribution",
        column_format="llrr",
        escape=True,
    )
    out_dir = figures_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "table1_demographic_distribution.tex").write_text(latex)
    print(f"LaTeX table saved → {out_dir / 'table1_demographic_distribution.tex'}")
    return table


# ── Orchestrator ─────────────────────────────────────────────────────────────
def run_all() -> None:
    """Generate every figure and table for notebook 01 in sequence."""
    palette = setup_theme(context="paper", font_scale=1.2)
    df = load_baseline()
    describe_dataset(df)
    plot_profile_completeness(df)
    plot_univariate_distributions(df, palette)
    plot_pairwise_heatmaps(df)
    plot_profile_grid(df)
    plot_stacked_by_political(df)
    balance_df = compute_balance_metrics(df)
    print(balance_df.to_string(index=False))
    plot_balance_ratio(balance_df)
    plot_hamming_distance(df)
    plot_personality_political(df)
    plot_gender_economic(df)
    summary_table(df)


if __name__ == "__main__":
    run_all()
