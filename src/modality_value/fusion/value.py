"""The table the whole repo exists to produce.

compute_value_table is deliberately metric-agnostic (see eval/metrics.py's
YRef convention): the PTB-XL arm calls it with macro_auroc_metric and a
classification y_ref, the TCGA arm calls it with c_index_metric and a
survival y_ref. Same code path, different metric -- that is the point.

`compute_value_table` / `compute_leave_one_out_table` are generic (any
metric, any fuse_fn, any modality-set ordering). `main()` below is the
PTB-XL arm's own orchestration -- fits tabular/waveform/text, late-fuses
via a val-fold-fit stacking classifier, evaluates on the test fold -- and
is what `make fusion` runs. The TCGA arm's orchestration lives in
`modalities/molecular.py::main()` instead, calling the same two generic
functions with `c_index_metric` and its own KFold-OOF scores; do not
change `compute_value_table`'s/`compute_leave_one_out_table`'s signatures
without checking both call sites.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from modality_value.eval.metrics import MetricFn, YRef, paired_bootstrap_ci_delta

# Combines a dict of modality score arrays (already OOF/cross-fitted) into
# one fused score array of the shape metric_fn expects.
FuseFn = Callable[[dict[str, np.ndarray]], np.ndarray]


@dataclass
class ModalitySet:
    """One row's worth of modalities included, for both the incremental and
    leave-one-out views."""

    included: list[str]
    label: str  # e.g. "tabular" or "tabular+waveform" or "all - waveform"


def compute_value_table(
    y_ref: YRef,
    oof_scores: dict[str, np.ndarray],
    costs: dict[str, float],
    metric_fn: MetricFn,
    fuse_fn: FuseFn,
    incremental_order: list[str],
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the incremental (cheapest-first, greedy) value table.

    Columns: modality_set, metric, delta_vs_previous, ci_low, ci_high,
    cost_gbp, delta_per_100_gbp.

    `oof_scores` values MUST already be out-of-fold predictions (cross-fit
    across the training folds) -- this function does not do the
    cross-fitting itself, callers are responsible for the leakage guard.
    """
    rows = []
    included: list[str] = []
    prev_score = None
    prev_metric = None

    for modality in incremental_order:
        included = included + [modality]
        fused = fuse_fn({m: oof_scores[m] for m in included})
        metric_value = metric_fn(y_ref, fused)

        if prev_score is None:
            delta, lo, hi = metric_value, np.nan, np.nan
        else:
            delta, lo, hi = paired_bootstrap_ci_delta(
                metric_fn, y_ref, prev_score, fused, n_boot=n_boot, seed=seed
            )

        cost = costs[modality]
        rows.append(
            {
                "modality_set": "+".join(included),
                "metric": metric_value,
                "delta_vs_previous": delta,
                "ci_low": lo,
                "ci_high": hi,
                "acquisition_cost_gbp": cost,
                "delta_per_100_gbp": (delta / cost * 100) if cost and not np.isnan(delta) else np.nan,
            }
        )
        prev_score = fused
        prev_metric = metric_value

    return pd.DataFrame(rows)


def compute_leave_one_out_table(
    y_ref: YRef,
    oof_scores: dict[str, np.ndarray],
    costs: dict[str, float],
    metric_fn: MetricFn,
    fuse_fn: FuseFn,
    all_modalities: list[str],
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """For each modality, the marginal drop in metric from removing it out of
    the full fused set -- answers 'what could I stop paying for'."""
    full_fused = fuse_fn({m: oof_scores[m] for m in all_modalities})
    full_metric = metric_fn(y_ref, full_fused)

    rows = []
    for modality in all_modalities:
        remaining = [m for m in all_modalities if m != modality]
        reduced_fused = fuse_fn({m: oof_scores[m] for m in remaining})
        delta, lo, hi = paired_bootstrap_ci_delta(
            metric_fn, y_ref, reduced_fused, full_fused, n_boot=n_boot, seed=seed
        )
        rows.append(
            {
                "modality_removed": modality,
                "full_metric": full_metric,
                "metric_without": metric_fn(y_ref, reduced_fused),
                "delta_vs_full": delta,
                "ci_low": lo,
                "ci_high": hi,
                "acquisition_cost_gbp": costs[modality],
                "delta_per_100_gbp": (delta / costs[modality] * 100) if costs[modality] else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 5: PTB-XL arm (tabular + waveform + text -> late-fusion value table)
#
# Unlike the TCGA arm (molecular.py), which has no dataset-provided held-out
# split and so cross-fits OOF scores via KFold over the whole cohort, PTB-XL
# already has a fixed, previously-reported train(fold 1-8)/val(fold 9)/
# test(fold 10) split that Phase 1/2/3's individual numbers are all reported
# against. The fusion arm reuses that split rather than inventing a second,
# inconsistent evaluation frame:
#   - each base ModalityModel is fit on the train folds only (identical to
#     its own phase script);
#   - the second-stage stacking classifier is fit on fold 9 (val) scores --
#     out-of-sample for every base model, so no leakage into the stacker;
#   - the value/LOO tables are evaluated on fold 10 (test), matching the
#     individual per-modality numbers already in phase1_metrics.json.
def _ptbxl_make_fuse_fn(val_scores: dict[str, np.ndarray], val_y: np.ndarray) -> FuseFn:
    """Returns a fuse_fn that, for >1 modality, fits a fresh one-vs-rest
    logistic regression stacker on the *captured* fold-9 val scores/labels
    for whichever subset of modalities it's called with, then applies it to
    the scores actually passed in (fold-10 test, from compute_value_table's
    perspective). Refitting per call (rather than once for the full set)
    is what lets one fuse_fn serve every subset in both the incremental and
    leave-one-out tables, mirroring molecular.py's make_fuse_fn."""

    def fuse(scores: dict[str, np.ndarray]) -> np.ndarray:
        if len(scores) == 1:
            return next(iter(scores.values()))
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier

        names = sorted(scores)
        X_val = np.concatenate([val_scores[m] for m in names], axis=1)
        clf = OneVsRestClassifier(LogisticRegression(max_iter=2000))
        clf.fit(X_val, val_y)
        X_query = np.concatenate([scores[m] for m in names], axis=1)
        return clf.predict_proba(X_query)

    return fuse


def _ptbxl_plot_value(value_table: pd.DataFrame, out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = value_table["acquisition_cost_gbp"].to_numpy(dtype=float)
    y = value_table["delta_vs_previous"].to_numpy(dtype=float)
    lo = value_table["ci_low"].to_numpy(dtype=float)
    hi = value_table["ci_high"].to_numpy(dtype=float)
    is_baseline = value_table["ci_low"].isna().to_numpy()

    if is_baseline.any():
        ax.scatter(x[is_baseline], y[is_baseline], color="#a0aec0", s=90, zorder=2,
                   label="baseline (absolute macro AUROC)")
    if (~is_baseline).any():
        xi, yi = x[~is_baseline], y[~is_baseline]
        yerr = np.vstack([yi - lo[~is_baseline], hi[~is_baseline] - yi])
        ax.errorbar(xi, yi, yerr=yerr, fmt="o", capsize=4, color="#2b6cb0", ecolor="#718096",
                    markersize=9, zorder=3, label="Δ vs. previous step (95% CI)")

    for xi, yi, label in zip(x, y, value_table["modality_set"]):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.axhline(0, color="#a0aec0", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Acquisition cost of added modality (£/patient)")
    ax.set_ylabel("Macro AUROC (baseline)  /  Δ macro AUROC (increments)")
    ax.set_title("PTB-XL arm: discrimination gain per £ spent")
    ax.legend(loc="center right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    from modality_value.config import MODALITY_COSTS, RESULTS_DIR, SEED
    from modality_value.eval.metrics import macro_auroc, macro_auroc_metric
    from modality_value.io.ptbxl import build_or_load_signal_cache, labels_matrix, load_metadata, split_indices
    from modality_value.modalities.tabular import TabularModel
    from modality_value.modalities.text import TextModel, pd_index_subsample
    from modality_value.modalities.waveform import WaveformModel

    df = load_metadata()
    splits = split_indices(df)
    y = labels_matrix(df)
    pos = {name: df.index.get_indexer(idx) for name, idx in splits.items()}
    y_train, y_val, y_test = y[pos["train"]], y[pos["val"]], y[pos["test"]]

    print("=== Fitting tabular ===")
    tabular = TabularModel().fit(df.loc[splits["train"]], y_train)
    tabular_val = tabular.score(df.loc[splits["val"]])
    tabular_test = tabular.score(df.loc[splits["test"]])
    print(f"  tabular: val={macro_auroc(y_val, tabular_val):.4f} test={macro_auroc(y_test, tabular_test):.4f}")

    print("\n=== Fitting waveform ===")
    signals = build_or_load_signal_cache(df)
    Xw_train = np.array(signals[pos["train"]])
    Xw_val = np.array(signals[pos["val"]])
    Xw_test = np.array(signals[pos["test"]])
    waveform = WaveformModel().fit(Xw_train, y_train, X_val=Xw_val, y_val=y_val)
    waveform_val = waveform.score(Xw_val)
    waveform_test = waveform.score(Xw_test)
    print(f"  waveform: val={macro_auroc(y_val, waveform_val):.4f} test={macro_auroc(y_test, waveform_test):.4f}")

    print("\n=== Fitting text (LoRA, same 4000-example train subsample as Phase 2/3) ===")
    rng = np.random.default_rng(SEED)
    train_idx = splits["train"]
    max_train_examples = 4000
    if len(train_idx) > max_train_examples:
        train_idx = pd_index_subsample(train_idx, max_train_examples, rng)
    reports_train = df.loc[train_idx, "report"].fillna("").astype(str).tolist()
    y_train_sub = y[df.index.get_indexer(train_idx)]
    reports_val = df.loc[splits["val"], "report"].fillna("").astype(str).tolist()
    reports_test = df.loc[splits["test"], "report"].fillna("").astype(str).tolist()

    text = TextModel(backend="lora", num_epochs=2)
    text.fit(reports_train, y_train_sub)
    text_val = text.score(reports_val)
    text_test = text.score(reports_test)
    text._free_heavy()
    print(f"  text: val={macro_auroc(y_val, text_val):.4f} test={macro_auroc(y_test, text_test):.4f}")

    y_ref_val = {"y": y_val}
    y_ref_test = {"y": y_test}
    val_scores = {"tabular": tabular_val, "waveform": waveform_val, "text": text_val}
    test_scores = {"tabular": tabular_test, "waveform": waveform_test, "text": text_test}
    fuse_fn = _ptbxl_make_fuse_fn(val_scores, y_val)
    costs = {name: MODALITY_COSTS[name].cost_gbp for name in ("tabular", "waveform", "text")}

    # Cheapest-first: tabular (£1) -> text (£15) -> waveform (£25).
    incremental_order = sorted(costs, key=lambda m: costs[m])

    value_table = compute_value_table(
        y_ref=y_ref_test,
        oof_scores=test_scores,
        costs=costs,
        metric_fn=macro_auroc_metric,
        fuse_fn=fuse_fn,
        incremental_order=incremental_order,
        n_boot=1000,
        seed=SEED,
    )
    loo_table = compute_leave_one_out_table(
        y_ref=y_ref_test,
        oof_scores=test_scores,
        costs=costs,
        metric_fn=macro_auroc_metric,
        fuse_fn=fuse_fn,
        all_modalities=list(costs),
        n_boot=1000,
        seed=SEED,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    value_csv = RESULTS_DIR / "ptbxl_value.csv"
    loo_csv = RESULTS_DIR / "ptbxl_value_loo.csv"
    value_table.to_csv(value_csv, index=False)
    loo_table.to_csv(loo_csv, index=False)

    print("\nIncremental value table (fold-10 test):")
    print(value_table.to_string(index=False))
    print("\nLeave-one-out table (fold-10 test):")
    print(loo_table.to_string(index=False))
    print(f"\nWrote {value_csv}")
    print(f"Wrote {loo_csv}")

    _ptbxl_plot_value(value_table, RESULTS_DIR / "fig_ptbxl_value.png")


if __name__ == "__main__":
    main()
