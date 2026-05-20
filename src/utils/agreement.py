"""Notebook 03 — PerceptSent Agreement & Cross-Validation Evaluation."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick  # noqa: F401  (used for consistent imports)
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, mean_absolute_error,
                             roc_auc_score)

from utils.common import (AGREEMENT_DIR, FIGURES_DIR, OUTPUT_DIR, P_ORDER,
                          PROBLEM_TYPE, SENTIMENT_MAPS, SENTIMENT_ORDER,
                          load_human_gt_modal, load_jsonl, save_fig,
                          setup_theme)

SAMPLE_FRAC = 0.6     # fraction of annotations resampled per image for modal
N_BOOTSTRAP = 1000    # bootstrap iterations for 95% CI
BOOTSTRAP_SEED = 42


# ── Dataset CSV loader ───────────────────────────────────────────────────────
_PAT = re.compile(r"percept_dataset_sigma(\d+)_(p\w+)\.csv")


def load_datasets(agreement_dir: Path | None = None
                  ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load all agreement CSVs and return (datasets dict, meta DataFrame)."""
    target = Path(agreement_dir or AGREEMENT_DIR)
    csv_files = sorted(target.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files:\n")
    datasets: dict[str, pd.DataFrame] = {}
    meta_rows = []
    for fpath in csv_files:
        m = _PAT.match(fpath.name)
        if not m:
            continue
        sigma, p_type = int(m.group(1)), m.group(2)
        key = f"sigma{sigma}_{p_type}"
        df = pd.read_csv(fpath)
        label_counts = df["ground_truth"].value_counts().sort_index().to_dict()
        datasets[key] = df
        meta_rows.append({
            "key": key, "sigma": sigma, "p_type": p_type,
            "problem": PROBLEM_TYPE.get(p_type, "unknown"),
            "filename": fpath.name,
            "n_total": len(df),
            "n_classes": df["ground_truth"].nunique(),
            "label_dist": label_counts,
        })
        dist_str = "  ".join(f"label {k}={v}" for k, v in label_counts.items())
        print(f"  {key:28s}  n={len(df):5,}  classes={df['ground_truth'].nunique()}  [{dist_str}]")
    return datasets, pd.DataFrame(meta_rows)


def load_all_annotations() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and filter the three condition JSONL files."""
    ann_base = _filter_valid(load_jsonl(OUTPUT_DIR / "annotations_baseline.jsonl"))
    ann_np_think = _filter_valid(load_jsonl(OUTPUT_DIR / "annotations_no_persona_think.jsonl"))
    ann_np_no_think = _filter_valid(load_jsonl(OUTPUT_DIR / "annotations_no_persona_no_think.jsonl"))
    return ann_base, ann_np_think, ann_np_no_think


def _filter_valid(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["predicted_sentiment"].notna() & (df["predicted_sentiment"] != "")].copy()


# ── Fig pa01: dataset overview ───────────────────────────────────────────────
def plot_dataset_overview(meta_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    pivot_size = meta_df.pivot(index="sigma", columns="p_type", values="n_total").reindex(columns=P_ORDER)
    pivot_cls  = meta_df.pivot(index="sigma", columns="p_type", values="n_classes").reindex(columns=P_ORDER)
    sns.heatmap(pivot_size, annot=True, fmt=",.0f", cmap="Blues", linewidths=0.4,
                cbar_kws={"label": "# images"}, ax=axes[0])
    axes[0].set_xlabel("Population threshold (p-type)")
    axes[0].set_ylabel("Sigma (σ)")
    sns.heatmap(pivot_cls, annot=True, fmt="d", cmap="Purples", linewidths=0.4,
                cbar_kws={"label": "# ground-truth classes"}, vmin=0, vmax=5, ax=axes[1])
    axes[1].set_xlabel("Population threshold (p-type)")
    axes[1].set_ylabel("Sigma (σ)")
    for ax in axes:
        ax.set_xticklabels(
            [f"{p}\n({PROBLEM_TYPE.get(p, '?')})" for p in P_ORDER],
            rotation=0, fontsize=9,
        )
    plt.tight_layout()
    save_fig(fig, "fig_pa01_dataset_overview")


# ── Fig pa02: GT label distribution ──────────────────────────────────────────
def plot_gt_label_distribution(datasets: dict[str, pd.DataFrame], palette) -> None:
    sigma_vals = sorted({int(k.split("_")[0].replace("sigma", "")) for k in datasets})
    fig, axes = plt.subplots(len(sigma_vals), len(P_ORDER), figsize=(14, 9), sharey=False)
    for row_i, sigma in enumerate(sigma_vals):
        for col_j, p_type in enumerate(P_ORDER):
            ax = axes[row_i][col_j]
            key = f"sigma{sigma}_{p_type}"
            if key not in datasets:
                ax.set_visible(False)
                continue
            counts = datasets[key]["ground_truth"].value_counts().sort_index()
            ax.bar([str(c) for c in counts.index], counts.values,
                   color=palette[0], alpha=0.75, edgecolor="white")
            ax.set_xlabel("ground_truth label", fontsize=7)
            ax.set_ylabel("images", fontsize=7)
            ax.tick_params(labelsize=7)
    plt.tight_layout()
    save_fig(fig, "fig_pa02_gt_label_distribution")


# ── Per-image modal helpers ──────────────────────────────────────────────────
def modal_per_image(ann: pd.DataFrame) -> pd.DataFrame:
    """Full-pool modal sentiment per image."""
    if ann.empty:
        return pd.DataFrame(columns=["image_id", "modal_sentiment"])
    return (ann.groupby("image_id")["predicted_sentiment"]
            .agg(lambda x: x.value_counts().index[0])
            .reset_index(name="modal_sentiment"))


def sampled_modal_per_image(ann: pd.DataFrame, frac: float, rng) -> pd.DataFrame:
    """Resample `frac` of annotations per image, then take the modal."""
    if ann.empty:
        return pd.DataFrame(columns=["image_id", "modal_sentiment"])
    rows = []
    for img_id, grp in ann.groupby("image_id"):
        n = max(1, int(len(grp) * frac))
        sample = grp.sample(n=n, random_state=int(rng.integers(0, 2 ** 31)))
        rows.append({
            "image_id": img_id,
            "modal_sentiment": sample["predicted_sentiment"].value_counts().index[0],
        })
    return pd.DataFrame(rows)


def describe_conditions(ann_base: pd.DataFrame, ann_np_think: pd.DataFrame,
                        ann_np_no_think: pd.DataFrame) -> None:
    for label, df in [
        ("Baseline (persona)",     ann_base),
        ("No-persona think=True",  ann_np_think),
        ("No-persona think=False", ann_np_no_think),
    ]:
        if df.empty:
            print(f"{label:<30s}: not available")
            continue
        n_img = df["image_id"].nunique()
        ann_per = len(df) / n_img if n_img else 0
        print(f"{label:<30s}: {len(df):>7,} annotations  |  {n_img:>3} images  |  {ann_per:.1f} ann/image")


# ── Evaluation pipeline ──────────────────────────────────────────────────────
def _compute_metrics(y_true, y_pred, problem: str) -> dict:
    m = {"accuracy": accuracy_score(y_true, y_pred)}
    present_labels = sorted(set(y_true.tolist()))
    if problem == "binary":
        m["f1_macro"]    = f1_score(y_true, y_pred, average="macro",
                                    labels=present_labels, zero_division=0)
        m["f1_pos"]      = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        m["cohen_kappa"] = cohen_kappa_score(y_true, y_pred, weights=None)
        try:
            m["roc_auc"] = roc_auc_score(y_true, y_pred)
        except ValueError:
            pass
    elif problem == "multiclass-3":
        m["f1_macro"]    = f1_score(y_true, y_pred, average="macro",
                                    labels=present_labels, zero_division=0)
        m["cohen_kappa"] = cohen_kappa_score(y_true, y_pred, weights="linear")
    elif problem == "ordinal-5":
        m["f1_macro"]    = f1_score(y_true, y_pred, average="macro",
                                    labels=present_labels, zero_division=0)
        m["cohen_kappa"] = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        m["mae"]         = mean_absolute_error(y_true, y_pred)
    return m


def evaluate_cv(key: str, p_type: str, df: pd.DataFrame,
                ann: pd.DataFrame, condition: str,
                sample_frac: float = SAMPLE_FRAC,
                n_bootstrap: int = N_BOOTSTRAP,
                seed: int = BOOTSTRAP_SEED) -> dict:
    base = {"key": key, "condition": condition,
            "p_type": p_type, "problem": PROBLEM_TYPE.get(p_type, "?")}
    if ann.empty:
        return {**base, "note": "no annotations"}
    smap = SENTIMENT_MAPS[p_type]
    problem = PROBLEM_TYPE[p_type]

    rng_modal = np.random.default_rng(seed)
    modal_df = sampled_modal_per_image(ann, sample_frac, rng_modal)
    merged = df.merge(modal_df, on="image_id", how="inner")
    if merged.empty or merged["ground_truth"].nunique() < 2:
        return {**base, "note": "insufficient data"}

    y_true_all = merged["ground_truth"].values
    y_pred_raw = merged["modal_sentiment"].map(smap)
    valid_mask = y_pred_raw.notna()
    y_true_all = y_true_all[valid_mask]
    y_pred_all = y_pred_raw[valid_mask].values.astype(int)
    n_images = len(y_true_all)
    if n_images < 2 or len(set(y_true_all.tolist())) < 2:
        return {**base, "note": "insufficient data after mapping"}

    try:
        point = _compute_metrics(y_true_all, y_pred_all, problem)
    except Exception as e:
        return {**base, "note": f"metric error: {e}"}

    rng_boot = np.random.default_rng(seed + 1)
    boot_results = []
    for _ in range(n_bootstrap):
        idx = rng_boot.integers(0, n_images, size=n_images)
        yt, yp = y_true_all[idx], y_pred_all[idx]
        if len(set(yt.tolist())) < 2:
            continue
        try:
            boot_results.append(_compute_metrics(yt, yp, problem))
        except Exception:
            pass

    n_img_in_ann = ann["image_id"].nunique()
    ann_per_img = round(len(ann) / n_img_in_ann * sample_frac) if n_img_in_ann else 0
    result = {**base, "n_images": n_images, "ann_per_image": int(ann_per_img),
              "n_annotations": int(n_images * ann_per_img)}
    for metric in ["accuracy", "f1_macro", "cohen_kappa", "f1_pos", "roc_auc", "mae"]:
        if metric in point:
            vals = [r[metric] for r in boot_results if metric in r]
            result[f"{metric}_mean"]    = round(point[metric], 4)
            result[f"{metric}_ci_low"]  = round(float(np.percentile(vals,  2.5)), 4) if vals else float("nan")
            result[f"{metric}_ci_high"] = round(float(np.percentile(vals, 97.5)), 4) if vals else float("nan")
    return result


def evaluate_all_conditions(datasets: dict[str, pd.DataFrame],
                            conditions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cond_label, ann in conditions.items():
        for key, df in datasets.items():
            p_type = key.split("_", 1)[1]
            rows.append(evaluate_cv(key, p_type, df, ann, cond_label))
    eval_df = pd.DataFrame(rows)
    eval_df["sigma"] = eval_df["key"].str.extract(r"sigma(\d+)").astype(int)
    print(f"Evaluation complete — {len(eval_df)} rows\n")
    for cond_label in conditions:
        sub = eval_df[eval_df["condition"] == cond_label]
        avail_cols = [c for c in ["key", "n_images", "ann_per_image",
                                  "f1_macro_mean", "cohen_kappa_mean"]
                      if c in sub.columns]
        print(f"  [{cond_label}]")
        print(sub[avail_cols].to_string(index=False) if avail_cols else "  (no data)")
        print()
    return eval_df


# ── Fig pa03: metrics heatmaps ───────────────────────────────────────────────
def _make_annot(sub: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_piv = sub.pivot(index="sigma", columns="p_type", values=f"{metric}_mean").reindex(columns=P_ORDER)
    lo_piv  = sub.pivot(index="sigma", columns="p_type", values=f"{metric}_ci_low").reindex(columns=P_ORDER)
    hi_piv  = sub.pivot(index="sigma", columns="p_type", values=f"{metric}_ci_high").reindex(columns=P_ORDER)
    annot = val_piv.copy().astype(object)
    for idx in val_piv.index:
        for col in val_piv.columns:
            v = val_piv.loc[idx, col]
            lo = lo_piv.loc[idx, col]
            hi = hi_piv.loc[idx, col]
            if pd.isna(v):
                annot.loc[idx, col] = "—"
            else:
                annot.loc[idx, col] = f"{v:.2f}\n±{(hi - lo) / 2:.2f}"
    return val_piv, annot


def plot_metrics_heatmap_baseline(eval_df: pd.DataFrame) -> None:
    sub = eval_df[eval_df["condition"] == "baseline"].copy()
    if sub.empty or "f1_macro_mean" not in sub.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (metric, label) in zip(axes, [("f1_macro", "Macro F1"),
                                          ("cohen_kappa", "Cohen's κ")]):
        if f"{metric}_mean" not in sub.columns:
            continue
        val_piv, annot = _make_annot(sub, metric)
        sns.heatmap(val_piv, annot=annot, fmt="", cmap="RdYlGn", vmin=0, vmax=1,
                    linewidths=0.4, cbar_kws={"label": label}, ax=ax)
        ax.set_ylabel("σ  (agreement threshold)")
        ax.set_xticklabels(
            [f"{p}\n({PROBLEM_TYPE.get(p, '?')})" for p in P_ORDER],
            rotation=0, fontsize=10,
        )
    plt.tight_layout()
    save_fig(fig, "fig_pa03_metrics_baseline")


# ── Fig sentiment dist all conditions ────────────────────────────────────────
def plot_sentiment_dist_all_conditions(ann_base: pd.DataFrame,
                                       ann_np_think: pd.DataFrame,
                                       ann_np_no_think: pd.DataFrame,
                                       gt_modal: pd.DataFrame, palette) -> None:
    # Order: Human GT, Persona, No-persona (think), No-persona (no-think)
    entries: list[tuple[str, pd.DataFrame | None]] = []
    if not gt_modal.empty:
        entries.append(("Human GT", None))
    entries += [
        ("Persona",               ann_base),
        ("No-persona (think)",    ann_np_think),
        ("No-persona (no-think)", ann_np_no_think),
    ]
    n_bars = len(entries)
    x = np.arange(len(SENTIMENT_ORDER))
    bar_w = 0.18
    colors = palette[:n_bars]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (label, ann) in enumerate(entries):
        if ann is None:
            counts = (gt_modal["human_sentiment"].value_counts(normalize=True)
                      .mul(100).reindex(SENTIMENT_ORDER).fillna(0))
        elif ann.empty:
            continue
        else:
            counts = (ann["predicted_sentiment"].value_counts(normalize=True)
                      .mul(100).reindex(SENTIMENT_ORDER).fillna(0))
        offset = bar_w * (i - (n_bars - 1) / 2)
        ax.bar(x + offset, counts, bar_w, label=label,
               color=colors[i], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace("Slightly", "Slightly\n") for s in SENTIMENT_ORDER],
        rotation=0, ha="center",
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel("% of annotations")
    ax.legend(loc="upper right", fontsize=10, ncol=2)
    fig.tight_layout()
    save_fig(fig, "fig_sentiment_dist_all_conditions")


# ── Fig pa05: confusion matrices ─────────────────────────────────────────────
SIGMA_ORDER = ["sigma3", "sigma4", "sigma5"]

SIGMA_LABEL = {
    "sigma3": r"$\sigma_3$",
    "sigma4": r"$\sigma_4$",
    "sigma5": r"$\sigma_5$",
}
P_LABEL = {
    "p2neg":  r"$P_2^{-}$",
    "p2plus": r"$P_2^{+}$",
    "p3":     r"$P_3$",
    "p5":     r"$P_5$",
}
TICK_MAP = {
    "binary":       ["Neg", "Pos"],
    "multiclass-3": ["Neg", "Neu", "Pos"],
    "ordinal-5":    ["Neg", "SlNeg", "Neu", "SlPos", "Pos"],
}
ANNOT_FS = {2: 26, 3: 20, 5: 14}


def plot_confusion_matrices(datasets: dict[str, pd.DataFrame],
                            modal_base_full: pd.DataFrame,
                            eval_df: pd.DataFrame) -> None:
    valid_keys = set(
        k for k in datasets
        if k in eval_df[eval_df["condition"] == "baseline"]["key"].values
    )
    p_order = ["p2neg", "p2plus", "p3", "p5"]
    fig, axes = plt.subplots(len(SIGMA_ORDER), len(p_order),
                             figsize=(len(p_order) * 5, len(SIGMA_ORDER) * 5))
    for row_i, sigma in enumerate(SIGMA_ORDER):
        for col_j, p_key in enumerate(p_order):
            ax = axes[row_i][col_j]
            key = f"{sigma}_{p_key}"
            if key not in valid_keys:
                ax.set_visible(False)
                continue
            p_type = key.split("_", 1)[1]
            smap = SENTIMENT_MAPS[p_type]
            labels = sorted(set(smap.values()))
            if PROBLEM_TYPE[p_type] == "multiclass-3":
                labels = [1, 0, 2]
            merged = datasets[key].merge(modal_base_full, on="image_id", how="inner")
            if merged.empty:
                ax.set_visible(False)
                continue
            y_true = merged["ground_truth"].values
            y_pred = merged["modal_sentiment"].map(smap).values
            cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
            n_cls = len(labels)
            annot_fs = ANNOT_FS.get(n_cls, max(10, 26 - (n_cls - 2) * 4))
            sns.heatmap(cm * 100, annot=True, fmt=".1f", cmap="Blues",
                        annot_kws={"fontsize": annot_fs, "fontweight": "bold"},
                        xticklabels=TICK_MAP[PROBLEM_TYPE[p_type]],
                        yticklabels=TICK_MAP[PROBLEM_TYPE[p_type]],
                        linewidths=0.5, cbar=False, ax=ax)
            ax.set_xlabel("Predicted", fontsize=22, labelpad=10)
            ax.set_ylabel("True", fontsize=22, labelpad=10)
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=20, rotation=45, ha="right")
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, rotation=0)
            ax.set_title(f"{SIGMA_LABEL[sigma]},  {P_LABEL[p_key]}",
                         fontsize=20, pad=8)
    plt.tight_layout(pad=2.0)
    save_fig(fig, "fig_pa05_confusion_baseline")


# ── Summary table ────────────────────────────────────────────────────────────
def summary_table(eval_df: pd.DataFrame,
                  figures_dir: Path | None = None) -> pd.DataFrame:
    shared_cols = ["key", "p_type", "problem", "condition", "n_images", "ann_per_image"]
    metric_cols = ["accuracy_mean", "f1_macro_mean", "cohen_kappa_mean",
                   "f1_pos_mean", "roc_auc_mean", "mae_mean"]
    avail_cols = shared_cols + [c for c in metric_cols if c in eval_df.columns]
    summary = (eval_df[avail_cols]
               .sort_values(["p_type", "key", "condition"])
               .reset_index(drop=True))
    cond_display = {
        "baseline":            "Baseline (persona)",
        "no_persona_think":    "No-persona think=True",
        "no_persona_no_think": "No-persona think=False",
    }
    summary["condition"] = summary["condition"].map(cond_display).fillna(summary["condition"])
    latex = summary.to_latex(
        index=False,
        caption=(
            "Evaluation metrics per agreement subset $(\\sigma, p)$ and condition. "
            "Baseline: 1,200 personas $\\times$ 50 images; "
            "no-persona conditions: \\texttt{qwen3-vl:8b} without persona conditioning, "
            "same 50 images, \\texttt{think=True} and \\texttt{think=False} variants. "
            f"Point estimate from {SAMPLE_FRAC:.0%} annotation resampling per image; "
            f"95\\% CI via image bootstrap ($N={N_BOOTSTRAP}$, seed {BOOTSTRAP_SEED}). "
            "Cohen's $\\kappa$ weighting: unweighted (binary), linear (3-class), "
            "quadratic (ordinal-5). "
            "Macro F1 is the primary metric."
        ),
        label="tab:perceptsent_agreement_eval",
        escape=True,
        float_format="{:.4f}".format,
        na_rep="---",
    )
    out_dir = figures_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "table3_perceptsent_agreement_eval.tex").write_text(latex)
    print(f"LaTeX table saved → {out_dir / 'table3_perceptsent_agreement_eval.tex'}")
    return summary


# ── Orchestrator ─────────────────────────────────────────────────────────────
def run_all() -> None:
    palette = setup_theme(font_scale=1.5)
    datasets, meta_df = load_datasets()
    plot_dataset_overview(meta_df)
    plot_gt_label_distribution(datasets, palette)

    ann_base, ann_np_think, ann_np_no_think = load_all_annotations()
    describe_conditions(ann_base, ann_np_think, ann_np_no_think)

    eval_df = evaluate_all_conditions(
        datasets,
        {
            "baseline":            ann_base,
            "no_persona_think":    ann_np_think,
            "no_persona_no_think": ann_np_no_think,
        },
    )
    plot_metrics_heatmap_baseline(eval_df)

    gt_modal = load_human_gt_modal()
    plot_sentiment_dist_all_conditions(
        ann_base, ann_np_think, ann_np_no_think, gt_modal, palette,
    )

    modal_base_full = modal_per_image(ann_base)
    plot_confusion_matrices(datasets, modal_base_full, eval_df)
    summary_table(eval_df)


if __name__ == "__main__":
    run_all()
