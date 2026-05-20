"""Notebook 02 — Sentiment Convergence Analysis."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (cohen_kappa_score, confusion_matrix, f1_score,
                             classification_report)

from utils.common import (DATASET_JSON, DIST_CSV, ECO_ORDER, FIGURES_DIR,
                          OUTPUT_DIR, PERS_ORDER, POL_ORDER, ROOT,
                          SENT_COLORS_HEX, SENTIMENT_INT, SENTIMENT_ORDER,
                          bootstrap_mean_ci, fmt_ci, load_human_gt_modal,
                          load_jsonl, load_profile_map, modal_ratio, save_fig,
                          setup_theme, wilson_ci)


# ── Loaders ──────────────────────────────────────────────────────────────────
def load_inputs():
    """Load baseline annotations, human GT, profile map."""
    df_base = load_jsonl(OUTPUT_DIR / "annotations_baseline.jsonl")
    if not df_base.empty:
        print(f"Loaded {len(df_base):,} records from annotations_baseline.jsonl")
    gt_modal = load_human_gt_modal()
    if not gt_modal.empty:
        print(f"Ground truth: {len(gt_modal):,} unique images")
    profile_map = load_profile_map()
    if not profile_map.empty:
        print(f"Profile map loaded: {len(profile_map):,} personas, "
              f"{profile_map['profile_id'].nunique()} unique profiles")
    return df_base, gt_modal, profile_map


def prepare(df: pd.DataFrame, condition: str, profile_map: pd.DataFrame,
            gt_modal: pd.DataFrame) -> pd.DataFrame:
    """Flatten raw_demographics, attach profile_id, merge GT."""
    if df.empty:
        return df
    df = df.copy()
    df["sentiment_int"] = df["predicted_sentiment"].map(SENTIMENT_INT)
    demo_df = pd.json_normalize(df["raw_demographics"])
    df = pd.concat([df.drop(columns=["raw_demographics"]), demo_df], axis=1)
    df = df.merge(profile_map, on="persona_id", how="left")
    n_missing = df["profile_id"].isna().sum()
    if n_missing:
        print(f"  WARNING: {n_missing} rows missing profile_id")
    if not gt_modal.empty:
        df = df.merge(gt_modal, on="image_id", how="left")
        df["human_sentiment_int"] = df["human_sentiment"].map(SENTIMENT_INT)
    else:
        df["human_sentiment"] = pd.NA
        df["human_sentiment_int"] = pd.NA
    df["condition"] = condition
    return df


# ── Convergence ──────────────────────────────────────────────────────────────
_CONVERGENCE_COLS = [
    "image_id", "profile_id", "n_personas", "modal_ratio",
    "sentiment_var", "human_sentiment",
]


def within_group_convergence(df: pd.DataFrame) -> pd.DataFrame:
    """Per (image, profile) modal ratio + variance over personas."""
    empty = pd.DataFrame(columns=_CONVERGENCE_COLS)
    if df.empty or "profile_id" not in df.columns:
        return empty
    rows = []
    for (image_id, profile_id), grp in df.groupby(["image_id", "profile_id"]):
        if len(grp) < 2:
            continue
        rows.append({
            "image_id":        image_id,
            "profile_id":      profile_id,
            "n_personas":      len(grp),
            "modal_ratio":     round(modal_ratio(grp["predicted_sentiment"]), 4),
            "sentiment_var":   round(grp["sentiment_int"].var(), 4),
            "human_sentiment": grp["human_sentiment"].iloc[0],
        })
    return pd.DataFrame(rows, columns=_CONVERGENCE_COLS) if rows else empty


# ── Ground-truth agreement ───────────────────────────────────────────────────
def _image_level_modal(df: pd.DataFrame, gt_modal: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("image_id")["predicted_sentiment"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"predicted_sentiment": "modal_pred"})
        .merge(gt_modal, on="image_id", how="inner")
    )


def _bootstrap_ci_metric(y_true, y_pred, metric_fn, n_boot: int = 1000,
                         seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 5:
        return float("nan"), float("nan")
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            v = metric_fn(y_true[idx], y_pred[idx])
            if not np.isnan(v):
                boot.append(v)
        except Exception:
            pass
    if len(boot) < 50:
        return float("nan"), float("nan")
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def compute_gt_metrics(df_cond: pd.DataFrame, gt_modal: pd.DataFrame,
                      label: str) -> dict:
    if df_cond.empty:
        return {}
    modal = _image_level_modal(df_cond, gt_modal)
    valid = modal.dropna(subset=["modal_pred", "human_sentiment"])
    if valid.empty:
        return {}
    y_pred = valid["modal_pred"].values
    y_true = valid["human_sentiment"].values

    kappa = cohen_kappa_score(y_true, y_pred, weights="linear", labels=SENTIMENT_ORDER)
    f1    = f1_score(y_true, y_pred, average="macro", labels=SENTIMENT_ORDER, zero_division=0)
    kappa_ci = _bootstrap_ci_metric(
        y_true, y_pred,
        lambda yt, yp: cohen_kappa_score(yt, yp, weights="linear", labels=SENTIMENT_ORDER),
    )
    f1_ci = _bootstrap_ci_metric(
        y_true, y_pred,
        lambda yt, yp: f1_score(yt, yp, average="macro", labels=SENTIMENT_ORDER, zero_division=0),
    )
    print(f"\n{'=' * 60}")
    print(f"Condition: {label}  (n_images={len(valid)})")
    print(f"  Cohen κ (linear-weighted) : {kappa:.4f}  95% CI [{kappa_ci[0]:.4f}, {kappa_ci[1]:.4f}]")
    print(f"  Macro F1                  : {f1:.4f}  95% CI [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")
    print(classification_report(y_true, y_pred, labels=SENTIMENT_ORDER, zero_division=0))
    return {
        "condition": label, "cohen_kappa": kappa, "macro_f1": f1,
        "n_images": len(valid),
        "kappa_ci_lo": kappa_ci[0], "kappa_ci_hi": kappa_ci[1],
        "f1_ci_lo": f1_ci[0], "f1_ci_hi": f1_ci[1],
    }


# ── Section 1.5: image dataset distribution panels ───────────────────────────
def plot_image_dataset_distribution(gt_modal: pd.DataFrame,
                                    df_base: pd.DataFrame,
                                    palette) -> None:
    if gt_modal.empty:
        print("Ground-truth not loaded — skipping image dataset distribution figure.")
        return

    try:
        with open(DATASET_JSON) as fh:
            raw = json.load(fh)
        img_records = [img for task in raw["tasks"] for img in task["images"]]
        gt_full = pd.DataFrame(img_records)[["id", "sentiment", "in_out_door"]]
        gt_full = gt_full.rename(columns={"id": "image_id"})
    except Exception as e:
        print(f"WARNING: could not reload dataset.json — {e}")
        gt_full = pd.DataFrame()

    exp_images = (df_base["image_id"].unique() if not df_base.empty
                  else gt_modal["image_id"].unique())
    gt_exp = gt_modal[gt_modal["image_id"].isin(exp_images)].copy()
    if not gt_full.empty:
        gt_io = (
            gt_full[gt_full["image_id"].isin(exp_images)]
            .groupby("image_id")["in_out_door"]
            .agg(lambda x: x.value_counts().index[0])
            .reset_index()
            .rename(columns={"in_out_door": "scene_type"})
        )
        gt_exp = gt_exp.merge(gt_io, on="image_id", how="left")
    else:
        gt_exp["scene_type"] = "Unknown"

    # Panel A — sentiment distribution
    sent_counts = gt_exp["human_sentiment"].value_counts().reindex(SENTIMENT_ORDER).fillna(0)
    colors = sns.color_palette("coolwarm", n_colors=len(SENTIMENT_ORDER))
    fig_a, ax_a = plt.subplots(figsize=(6, 4))
    ax_a.bar(SENTIMENT_ORDER, sent_counts, color=colors, edgecolor="white")
    ax_a.set_ylabel("Number of images")
    ax_a.set_xticklabels([s.replace("Slightly", "Slightly\n") for s in SENTIMENT_ORDER],
                         rotation=0, ha="center")
    for bar, val in zip(ax_a.patches, sent_counts):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                  f"{int(val)}", ha="center", va="bottom", fontsize=9)
    fig_a.tight_layout()
    save_fig(fig_a, "fig_dataset_a_sentiment")

    # Panel B — indoor / outdoor
    fig_b, ax_b = plt.subplots(figsize=(4, 4))
    if "scene_type" in gt_exp.columns and gt_exp["scene_type"].notna().any():
        io_counts = gt_exp["scene_type"].value_counts()
        io_colors = [palette[0], palette[2], palette[4]]
        bars = ax_b.bar(io_counts.index, io_counts.values,
                        color=io_colors[: len(io_counts)], edgecolor="white")
        ax_b.set_ylabel("Number of images")
        for bar, val in zip(bars, io_counts.values):
            ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                      f"{int(val)}", ha="center", va="bottom", fontsize=9)
    else:
        ax_b.text(0.5, 0.5, "Indoor/outdoor\nnot available",
                  ha="center", va="center", transform=ax_b.transAxes)
    fig_b.tight_layout()
    save_fig(fig_b, "fig_dataset_b_scene_type")

    # Panel C — heatmap
    fig_c, ax_c = plt.subplots(figsize=(7, 3))
    if "scene_type" in gt_exp.columns and gt_exp["scene_type"].nunique() > 1:
        ct = pd.crosstab(gt_exp["scene_type"], gt_exp["human_sentiment"]).reindex(
            columns=SENTIMENT_ORDER, fill_value=0
        )
        sns.heatmap(ct, annot=True, fmt="d", cmap="Blues",
                    linewidths=0.4, cbar_kws={"label": "# images"}, ax=ax_c)
        ax_c.set_ylabel("Scene type")
        ax_c.set_xticklabels([s.replace("Slightly", "Slightly\n") for s in SENTIMENT_ORDER],
                             rotation=0, ha="center")
    else:
        ax_c.set_visible(False)
    fig_c.tight_layout()
    save_fig(fig_c, "fig_dataset_c_heatmap")


# ── Section 6.1a: sentiment distribution baseline ────────────────────────────
def plot_sentiment_distribution_baseline(df_base: pd.DataFrame,
                                        gt_modal: pd.DataFrame, palette) -> None:
    if df_base.empty:
        return
    counts_base = (df_base["predicted_sentiment"].value_counts(normalize=True)
                   .mul(100).reindex(SENTIMENT_ORDER).fillna(0))
    counts_gt = (gt_modal["human_sentiment"].value_counts(normalize=True)
                 .mul(100).reindex(SENTIMENT_ORDER).fillna(0))
    ymax = max(counts_base.max(), counts_gt.max()) * 1.15
    disp = [s.replace("Slightly", "Slightly\n") for s in SENTIMENT_ORDER]

    fig_ag, ax_ag = plt.subplots(figsize=(6, 4))
    ax_ag.bar(SENTIMENT_ORDER, counts_base, color=palette[0], edgecolor="white")
    ax_ag.set_title("Agents", fontsize=14, fontweight="bold")
    ax_ag.set_ylabel("% of responses")
    ax_ag.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_ag.set_xticklabels(disp, rotation=0, ha="center")
    ax_ag.set_ylim(0, ymax)
    fig_ag.tight_layout()
    save_fig(fig_ag, "fig9a_sentiment_agents")

    fig_hu, ax_hu = plt.subplots(figsize=(6, 4))
    ax_hu.bar(SENTIMENT_ORDER, counts_gt, color=palette[3], edgecolor="white")
    ax_hu.set_title("Humans", fontsize=14, fontweight="bold")
    ax_hu.set_ylabel("% of responses")
    ax_hu.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_hu.set_xticklabels(disp, rotation=0, ha="center")
    ax_hu.set_ylim(0, ymax)
    fig_hu.tight_layout()
    save_fig(fig_hu, "fig9a_sentiment_humans")

    fig_comb, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ax_left.bar(range(len(SENTIMENT_ORDER)), counts_base, color=palette[0], edgecolor="white")
    ax_left.set_title("Agents", fontsize=14, fontweight="bold")
    ax_left.set_ylabel("% of responses")
    ax_left.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_left.set_xticks(range(len(SENTIMENT_ORDER)))
    ax_left.set_xticklabels(disp, rotation=0, ha="center")
    ax_left.set_ylim(0, ymax)
    ax_right.bar(range(len(SENTIMENT_ORDER)), counts_gt, color=palette[3], edgecolor="white")
    ax_right.set_title("Humans", fontsize=14, fontweight="bold")
    ax_right.set_ylabel("% of responses")
    ax_right.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_right.set_xticks(range(len(SENTIMENT_ORDER)))
    ax_right.set_xticklabels(disp, rotation=0, ha="center")
    ax_right.set_ylim(0, ymax)
    fig_comb.tight_layout()
    save_fig(fig_comb, "fig9a_sentiment_distribution_baseline")


# ── Section 6.2a: confusion matrix baseline ──────────────────────────────────
def plot_confusion_matrix_baseline(df_base: pd.DataFrame, gt_modal: pd.DataFrame) -> None:
    if df_base.empty:
        return
    modal_cm = _image_level_modal(df_base, gt_modal).dropna()
    if modal_cm.empty:
        return
    cm = confusion_matrix(
        modal_cm["human_sentiment"], modal_cm["modal_pred"],
        labels=SENTIMENT_ORDER, normalize="true",
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm * 100, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=SENTIMENT_ORDER, yticklabels=SENTIMENT_ORDER,
        linewidths=0.4, cbar_kws={"label": "Row %"}, ax=ax,
    )
    ax.set_xlabel("Predicted (modal)")
    ax.set_ylabel("Human (modal)")
    ax.set_xticklabels([s.replace("Slightly", "Slightly\n") for s in SENTIMENT_ORDER],
                       rotation=0, ha="center")
    plt.tight_layout()
    save_fig(fig, "fig10a_confusion_matrix_baseline")


# ── Section 6.4: modal_ratio heatmap by economic_status ──────────────────────
def plot_modal_ratio_heatmap_economic(df_base: pd.DataFrame,
                                      conv_base: pd.DataFrame) -> None:
    if conv_base.empty or df_base.empty:
        print("Skipped: Condition A — Baseline")
        return

    eco_lookup = df_base[["profile_id", "economic_status"]].drop_duplicates("profile_id")
    pivot_df = conv_base.merge(eco_lookup, on="profile_id", how="left")
    agg = (
        pivot_df.groupby(["human_sentiment", "economic_status"])["modal_ratio"]
        .agg(["mean", "std", "count"])
        .unstack(level="economic_status")
    )
    heat_mean = agg["mean"].reindex(index=SENTIMENT_ORDER, columns=ECO_ORDER, fill_value=np.nan)
    heat_std  = agg["std"].reindex(index=SENTIMENT_ORDER, columns=ECO_ORDER, fill_value=np.nan)
    if heat_mean.isna().all().all():
        print("Skipped (no data)")
        return

    annot = pd.DataFrame(index=heat_mean.index, columns=heat_mean.columns, dtype=object)
    for row in heat_mean.index:
        for col in heat_mean.columns:
            m, s = heat_mean.loc[row, col], heat_std.loc[row, col]
            if pd.isna(m):
                annot.loc[row, col] = "—"
            elif pd.isna(s) or s == 0:
                annot.loc[row, col] = f"{m:.2f}"
            else:
                annot.loc[row, col] = f"{m:.2f}\n±{s:.2f}"
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        heat_mean.fillna(0), annot=annot, fmt="",
        cmap="YlGn", vmin=0, vmax=1, linewidths=0.4,
        cbar_kws={"label": "Mean Modal Ratio"}, ax=ax,
    )
    ax.set_xlabel("Economic Status")
    ax.set_ylabel("Human Ground-Truth Sentiment")
    plt.tight_layout()
    save_fig(fig, "fig12a_modal_ratio_heatmap_base")


# ── Profile metadata helper ──────────────────────────────────────────────────
def load_profile_meta() -> pd.DataFrame:
    dist_full = pd.read_csv(DIST_CSV)
    meta = (
        dist_full.groupby("profile_id")
        .first()[["gender", "economic_status", "political_spectrum", "personality"]]
        .reset_index()
    )
    meta["label"] = (
        meta["gender"] + " | " + meta["economic_status"]
        + " | " + meta["political_spectrum"] + " | " + meta["personality"]
    )
    return meta


# ── Section 6.5a: Persona × sentiment heatmap ───────────────────────────────
def plot_profile_sentiment_heatmap(df_base: pd.DataFrame,
                                   profile_meta: pd.DataFrame) -> None:
    if df_base.empty or "profile_id" not in df_base.columns:
        print("⚠  No data — skipping profile-sentiment heatmap.")
        return

    agg = (df_base.groupby(["profile_id", "predicted_sentiment"])
           .size().reset_index(name="n"))
    pivot = (
        agg.pivot(index="profile_id", columns="predicted_sentiment", values="n")
        .reindex(index=profile_meta["profile_id"], columns=SENTIMENT_ORDER, fill_value=0)
        .fillna(0).astype(int)
    )
    row_sums = pivot.sum(axis=1)
    pct = pivot.div(row_sums.replace(0, np.nan), axis=0).fillna(0)
    label_map = profile_meta.set_index("profile_id")["label"]
    pct.index = pct.index.map(label_map)
    row_sums.index = row_sums.index.map(label_map)
    pivot.index = pivot.index.map(label_map)

    neg_score = pct["Negative"] + pct.get("SlightlyNegative", pd.Series(0, index=pct.index))
    sort_order = neg_score.sort_values(ascending=False).index
    pct = pct.loc[sort_order]
    pivot = pivot.loc[sort_order]

    annot = pd.DataFrame(index=pct.index, columns=pct.columns, dtype=object)
    for lbl in pct.index:
        n_row = int(row_sums.loc[lbl])
        for sent in SENTIMENT_ORDER:
            p = pct.loc[lbl, sent]
            if n_row == 0 or p == 0:
                annot.loc[lbl, sent] = "—"
            else:
                k = int(pivot.loc[lbl, sent])
                ci_lo, ci_hi = wilson_ci(k, n_row)
                annot.loc[lbl, sent] = f"{p:.2f}\n±{(ci_hi - ci_lo) / 2:.2f}"

    fig, ax = plt.subplots(figsize=(10, 14))
    sns.heatmap(
        pct, annot=annot, fmt="", cmap="RdYlGn",
        vmin=0, vmax=1, linewidths=0.5,
        annot_kws={"fontsize": 11},
        cbar_kws={"label": "Proportion of profile's annotations", "shrink": 0.55},
        ax=ax,
    )
    ax.set_xlabel("Predicted Sentiment", labelpad=10, fontsize=13)
    ax.set_ylabel("Persona  (gender | economic | political | personality)",
                  labelpad=10, fontsize=13)
    ax.set_xticklabels([s.replace("Slightly", "Slightly\n") for s in SENTIMENT_ORDER],
                       rotation=0, ha="center", fontsize=12)
    ax.tick_params(axis="y", rotation=0, labelsize=12)
    plt.tight_layout()
    save_fig(fig, "fig_rq1a_profile_sentiment_heatmap")


# ── Section 6.5b: modal ratio by profile ─────────────────────────────────────
def compute_modal_df(df_base: pd.DataFrame, profile_meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, grp in df_base.groupby("profile_id"):
        vc = grp["predicted_sentiment"].value_counts()
        mode = vc.index[0]
        k = int(vc.iloc[0])
        n = int(vc.sum())
        ratio = k / n
        ent_p = grp["predicted_sentiment"].value_counts(normalize=True)
        ent = -(ent_p * np.log2(ent_p + 1e-12)).sum()
        ci_lo, ci_hi = wilson_ci(k, n)
        rows.append({
            "profile_id":      pid,
            "modal_sentiment": mode,
            "modal_ratio":     ratio,
            "ci_lo":           ci_lo,
            "ci_hi":           ci_hi,
            "entropy_bits":    round(ent, 3),
            "n_annotations":   n,
        })
    return (pd.DataFrame(rows)
            .merge(profile_meta[["profile_id", "label"]], on="profile_id", how="left")
            .sort_values("modal_ratio", ascending=True)
            .reset_index(drop=True))


def plot_modal_ratio_by_profile(modal_df: pd.DataFrame) -> None:
    if modal_df.empty:
        print("⚠  modal_df is empty — skipping modal-ratio chart.")
        return
    bar_colors = [SENT_COLORS_HEX.get(s, "#888888") for s in modal_df["modal_sentiment"]]
    err_lo = (modal_df["modal_ratio"] - modal_df["ci_lo"]).clip(lower=0) * 100
    err_hi = (modal_df["ci_hi"] - modal_df["modal_ratio"]).clip(lower=0) * 100

    fig, ax = plt.subplots(figsize=(10, max(5, len(modal_df) * 0.52)))
    bars = ax.barh(
        modal_df["label"], modal_df["modal_ratio"] * 100,
        color=bar_colors, edgecolor="white", height=0.72,
        xerr=np.array([err_lo, err_hi]),
        error_kw={"linewidth": 1.2, "ecolor": "#444444", "capsize": 4},
    )
    ax.axvline(20, color="black", linestyle="--", linewidth=1.3,
               label="Chance level  (5 labels → 20 %)")
    ax.set_xlabel(
        "Modal Ratio (%)  ·  error bars = 95% Wilson CI\n"
        "(Wilson CI: exact binomial CI on the proportion of annotations\n"
        " that fell on the most-chosen label for each profile)", labelpad=10,
    )
    ax.set_xlim(0, 135)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    for bar, (_, row) in zip(bars, modal_df.iterrows()):
        ax.text(
            row["ci_hi"] * 100 + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{row['modal_sentiment']}  (n={row['n_annotations']})",
            va="center", fontsize=8.5, color="#333333",
        )
    legend_handles = [mpatches.Patch(color=c, label=s) for s, c in SENT_COLORS_HEX.items()]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5,
              title="Modal sentiment", title_fontsize=8.5, framealpha=0.9)
    plt.tight_layout()
    save_fig(fig, "fig_rq1b_modal_ratio_by_profile")
    print(
        modal_df[["label", "modal_sentiment", "modal_ratio", "ci_lo", "ci_hi",
                  "entropy_bits", "n_annotations"]]
        .sort_values("modal_ratio", ascending=False).to_string(index=False)
    )


# ── Section 6.5c: convergence by dimension ───────────────────────────────────
def plot_convergence_by_dimension(modal_df: pd.DataFrame,
                                  profile_meta: pd.DataFrame, palette) -> None:
    if modal_df.empty:
        print("⚠  modal_df is empty — skipping dimension chart.")
        return
    merged = modal_df.merge(
        profile_meta[["profile_id", "gender", "economic_status",
                      "political_spectrum", "personality"]],
        on="profile_id", how="left",
    )
    dim_specs = [
        ("gender",             sorted(merged["gender"].dropna().unique()), "Gender"),
        ("economic_status",    ECO_ORDER,  "Economic Status"),
        ("political_spectrum", POL_ORDER,  "Political Spectrum"),
        ("personality",        PERS_ORDER, "Personality"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.5))
    for ax, (dim, order, title) in zip(axes, dim_specs):
        avail = [v for v in order if v in merged[dim].values]
        agg = (merged.groupby(dim)["modal_ratio"]
               .agg(["mean", "std", "count"])
               .reindex(avail).dropna(subset=["mean"]))
        if agg.empty:
            ax.set_visible(False)
            continue
        x = range(len(agg))
        ax.bar(x, agg["mean"] * 100, color=palette[0], edgecolor="white",
               yerr=agg["std"].fillna(0) * 100, capsize=6,
               error_kw={"linewidth": 1.5, "ecolor": "#444444"})
        for xi, (_, row) in zip(x, agg.iterrows()):
            y_top = (row["mean"] + row.get("std", 0)) * 100 + 3
            ax.text(xi, y_top, f"n={int(row['count'])}", ha="center", fontsize=9)
        ax.axhline(20, color="crimson", linestyle="--", linewidth=1.3, label="Chance (20 %)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(list(agg.index), rotation=0, ha="center")
        if ax is axes[0]:
            ax.set_ylabel("Mean Modal Ratio (%)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylim(0, 130)
        ax.legend(fontsize=8.5, loc="upper right")
    plt.tight_layout()
    save_fig(fig, "fig_rq1c_convergence_by_dimension")


# ── Summary table ────────────────────────────────────────────────────────────
def summary_table(conv_base: pd.DataFrame, df_base: pd.DataFrame,
                  metrics_base: dict, figures_dir: Path | None = None) -> pd.DataFrame:
    row = {"Condition": "Baseline (A)"}
    if not conv_base.empty:
        mr_lo, mr_hi = bootstrap_mean_ci(conv_base["modal_ratio"].dropna())
        sv_lo, sv_hi = bootstrap_mean_ci(conv_base["sentiment_var"].dropna())
        row.update({
            "Mean modal ratio":     round(conv_base["modal_ratio"].mean(), 3),
            "Modal ratio 95% CI":   fmt_ci(mr_lo, mr_hi),
            "Mean sentiment var":   round(conv_base["sentiment_var"].mean(), 3),
            "Sentiment var 95% CI": fmt_ci(sv_lo, sv_hi),
        })
    else:
        row.update({
            "Mean modal ratio": "—", "Modal ratio 95% CI": "—",
            "Mean sentiment var": "—", "Sentiment var 95% CI": "—",
        })
    if not df_base.empty and metrics_base:
        row.update({
            "N annotations":    len(df_base),
            "N images":         df_base["image_id"].nunique(),
            "N personas":       df_base["persona_id"].nunique(),
            "Cohen κ (linear)": round(metrics_base.get("cohen_kappa", float("nan")), 3),
            "Cohen κ 95% CI":   fmt_ci(metrics_base.get("kappa_ci_lo", float("nan")),
                                       metrics_base.get("kappa_ci_hi", float("nan"))),
            "Macro F1":         round(metrics_base.get("macro_f1", float("nan")), 3),
            "Macro F1 95% CI":  fmt_ci(metrics_base.get("f1_ci_lo", float("nan")),
                                       metrics_base.get("f1_ci_hi", float("nan"))),
        })
    table = pd.DataFrame([row])
    latex = table.to_latex(
        index=False,
        caption=(
            "Convergence and ground-truth agreement metrics per experimental condition. "
            "95\\% CI computed via non-parametric bootstrap over within-profile groups "
            "(N=1\\,000 resamples, seed 42)."
        ),
        label="tab:convergence_results",
        escape=True,
    )
    out_dir = figures_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "table2_convergence_results.tex").write_text(latex)
    print(f"LaTeX table saved → {out_dir / 'table2_convergence_results.tex'}")
    return table


# ── Orchestrator ─────────────────────────────────────────────────────────────
def run_all() -> None:
    palette = setup_theme(font_scale=1.5)
    df_base, gt_modal, profile_map = load_inputs()
    df_base = prepare(df_base, "baseline", profile_map, gt_modal)
    conv_base = within_group_convergence(df_base)
    metrics_base = compute_gt_metrics(df_base, gt_modal, "baseline")

    plot_image_dataset_distribution(gt_modal, df_base, palette)
    plot_sentiment_distribution_baseline(df_base, gt_modal, palette)
    plot_confusion_matrix_baseline(df_base, gt_modal)
    plot_modal_ratio_heatmap_economic(df_base, conv_base)

    profile_meta = load_profile_meta()
    plot_profile_sentiment_heatmap(df_base, profile_meta)
    modal_df = compute_modal_df(df_base, profile_meta)
    plot_modal_ratio_by_profile(modal_df)
    plot_convergence_by_dimension(modal_df, profile_meta, palette)

    summary_table(conv_base, df_base, metrics_base)


if __name__ == "__main__":
    run_all()
