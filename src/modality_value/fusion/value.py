"""The table the whole repo exists to produce.

compute_value_table is deliberately metric-agnostic (see eval/metrics.py's
YRef convention): the PTB-XL arm calls it with macro_auroc_metric and a
classification y_ref, the TCGA arm calls it with c_index_metric and a
survival y_ref. Same code path, different metric -- that is the point.

STATUS: Phase 3 owns the real implementation (OOF cross-fitted late fusion
+ incremental/leave-one-out value analysis + the CSV/figure emission).
Phase 4 (molecular arm) depends on this module's public signature to run
its own value analysis once Phase 3 lands -- do not change this signature
without updating both call sites.
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
