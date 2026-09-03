"""Shared metrics contract used by every modality and by fusion/value.py.

fusion/value.py must run identically for the PTB-XL arm (metric = macro
AUROC, targets = a multi-hot label matrix) and the TCGA arm (metric =
C-index, targets = (time, event) pairs). Those two target shapes are
incompatible as plain arrays, so every metric here takes a
`y_ref: dict[str, np.ndarray]`, an opaque, task-specific bundle of targets,
rather than a positional array. Do not add a second calling convention;
wrap new metrics the same way (see `macro_auroc_metric` / `c_index_metric`).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

# y_ref is a dict of numpy arrays, all the same length n, e.g.
#   classification: {"y": (n, k) multi-hot labels}
#   survival:       {"time": (n,), "event": (n,)}
YRef = dict[str, np.ndarray]
MetricFn = Callable[[YRef, np.ndarray], float]


def index_y_ref(y_ref: YRef, idx: np.ndarray) -> YRef:
    return {k: v[idx] for k, v in y_ref.items()}


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """y_true, y_score: (n, k) multi-label. One-vs-rest AUROC averaged over columns
    that have both classes present in y_true (skips degenerate columns)."""
    aurocs = []
    for k in range(y_true.shape[1]):
        col = y_true[:, k]
        if len(np.unique(col)) < 2:
            continue
        aurocs.append(roc_auc_score(col, y_score[:, k]))
    if not aurocs:
        raise ValueError("no column had both classes present")
    return float(np.mean(aurocs))


def macro_auroc_metric(y_ref: YRef, score: np.ndarray) -> float:
    return macro_auroc(y_ref["y"], score)


def c_index(event_time: np.ndarray, predicted_risk: np.ndarray, event_observed: np.ndarray) -> float:
    """Harrell's C-index. predicted_risk should be higher = higher risk
    (e.g. the linear predictor / log partial hazard from a Cox model)."""
    return float(concordance_index(event_time, -predicted_risk, event_observed))


def c_index_metric(y_ref: YRef, score: np.ndarray) -> float:
    """score is the (n,) or (n, 1) linear predictor / log partial hazard."""
    risk = score[:, 0] if score.ndim == 2 else score
    return c_index(y_ref["time"], risk, y_ref["event"])


def paired_bootstrap_ci_delta(
    metric_fn: MetricFn,
    y_ref: YRef,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap the *difference* metric_fn(y_ref, score_b) - metric_fn(y_ref, score_a)
    over paired resamples of rows (same resampled indices applied to y_ref and both
    score sets). Works for any metric_fn following the YRef convention above --
    this is the one function fusion/value.py calls for both the PTB-XL (AUROC) and
    TCGA (C-index) arms.

    Returns (point_estimate_delta, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    n = score_a.shape[0]
    point = metric_fn(y_ref, score_b) - metric_fn(y_ref, score_a)

    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_ref_i = index_y_ref(y_ref, idx)
        try:
            deltas.append(metric_fn(y_ref_i, score_b[idx]) - metric_fn(y_ref_i, score_a[idx]))
        except (ValueError, ZeroDivisionError):
            continue  # degenerate resample (e.g. a class/only-censored draw); skip it

    deltas = np.asarray(deltas)
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)
