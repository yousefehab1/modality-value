"""The table the whole repo exists to produce.

compute_value_table is deliberately metric-agnostic (see eval/metrics.py's
YRef convention): the PTB-XL arm calls it with macro_auroc_metric and a
classification y_ref, the TCGA arm calls it with c_index_metric and a
survival y_ref. Same code path, different metric. Do not change
compute_value_table's/compute_leave_one_out_table's signatures without
checking both call sites (this file's main() and
modalities/molecular.py::main()).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from modality_value.eval.metrics import MetricFn, YRef, paired_bootstrap_ci_delta

# Combines a dict of modality score arrays (already OOF/cross-fitted) into
# one fused score array of the shape metric_fn expects.
FuseFn = Callable[[dict[str, np.ndarray]], np.ndarray]


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


def make_stacking_fuse_fn(
    fit_and_predict: Callable[[list[str], dict[str, np.ndarray]], np.ndarray],
) -> FuseFn:
    """Wrap a model-specific `fit_and_predict(sorted_names, scores) -> np.ndarray`
    closure with the single-modality-passthrough and sorted-name-ordering
    boilerplate shared by every stacking fuse_fn in this repo (this module's
    PTB-XL stacker and modalities/molecular.py's Cox stacker)."""

    def fuse(scores: dict[str, np.ndarray]) -> np.ndarray:
        if len(scores) == 1:
            return next(iter(scores.values()))
        return fit_and_predict(sorted(scores), scores)

    return fuse


# ---------------------------------------------------------------------------
# PTB-XL arm (tabular + waveform + text -> late-fusion value table)
#
# PTB-XL has a fixed, previously-reported train(fold 1-8)/val(fold 9)/
# test(fold 10) split that every individual modality's own numbers are
# already reported against, so the fusion arm reuses it rather than
# cross-fitting a second, inconsistent evaluation frame: each base
# ModalityModel is fit on the train folds only, the second-stage stacking
# classifier is fit on fold 9 (val) scores (out-of-sample for every base
# model), and the value/LOO tables are evaluated on fold 10 (test).
def _ptbxl_make_fuse_fn(val_scores: dict[str, np.ndarray], val_y: np.ndarray) -> FuseFn:
    """For >1 modality, fits a fresh one-vs-rest logistic regression stacker
    on the captured fold-9 val scores/labels for whichever subset it's
    called with, then applies it to the scores actually passed in (fold-10
    test, from compute_value_table's perspective)."""

    def fit_and_predict(names: list[str], scores: dict[str, np.ndarray]) -> np.ndarray:
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier

        X_val = np.concatenate([val_scores[m] for m in names], axis=1)
        clf = OneVsRestClassifier(LogisticRegression(max_iter=2000))
        clf.fit(X_val, val_y)
        X_query = np.concatenate([scores[m] for m in names], axis=1)
        return clf.predict_proba(X_query)

    return make_stacking_fuse_fn(fit_and_predict)


def main() -> None:
    from modality_value.config import MODALITY_COSTS, RESULTS_DIR, SEED
    from modality_value.eval.metrics import macro_auroc, macro_auroc_metric
    from modality_value.eval.report import plot_value_table
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

    print("\n=== Fitting text (LoRA, same 4000-example train subsample as the standalone text arm) ===")
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

    plot_value_table(
        value_table,
        RESULTS_DIR / "fig_ptbxl_value.png",
        metric_label="macro AUROC",
        title="PTB-XL arm: discrimination gain per £ spent",
    )


if __name__ == "__main__":
    main()
