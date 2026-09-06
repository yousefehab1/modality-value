"""Shared results-writing helpers: the phase1_metrics.json read-merge-write,
and the value-table plot shared by both cohorts' fusion arms.

`make tabular`, `make waveform`, and `make text` each run independently and
must not clobber each other's entry in phase1_metrics.json -- this merges by
top-level key, atomically, with retries, instead of overwriting the whole
file.
"""
from __future__ import annotations

import json
import time

from modality_value.config import RESULTS_DIR

PHASE1_METRICS_PATH = RESULTS_DIR / "phase1_metrics.json"


def merge_json_atomic(path, updates: dict, retries: int = 5) -> None:
    """Read-merge-write `updates` (top-level keys) into the JSON file at `path`.

    Writes via a temp file + atomic rename, and retries on a concurrent
    writer's read/write race, so parallel `make` targets don't clobber each
    other's entry.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            existing = {}
            if path.exists():
                with open(path) as f:
                    existing = json.load(f)
            existing.update(updates)
            tmp = path.with_suffix(path.suffix + f".tmp{attempt}")
            with open(tmp, "w") as f:
                json.dump(existing, f, indent=2, sort_keys=True)
            tmp.replace(path)
            return
        except (json.JSONDecodeError, OSError):
            time.sleep(0.5 + attempt)
    raise RuntimeError(f"Could not safely update {path} after {retries} retries (concurrent writer?)")


def update_phase1_metrics(key: str, metrics: dict, retries: int = 5) -> None:
    """Read-merge-write `metrics` under `key` into phase1_metrics.json."""
    merge_json_atomic(PHASE1_METRICS_PATH, {key: metrics}, retries=retries)
    print(f"Updated {PHASE1_METRICS_PATH} [{key}]")


def plot_value_table(value_table, out_path, *, metric_label: str, title: str) -> None:
    """Plot a compute_value_table() result: baseline point plus Δ-vs-previous
    points with 95% CI error bars, against acquisition cost. Shared by both
    cohorts' fusion arms (fusion/value.py's PTB-XL arm, modalities/molecular.py's
    TCGA arm) -- only the metric name and title differ between them."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = value_table["acquisition_cost_gbp"].to_numpy(dtype=float)
    y = value_table["delta_vs_previous"].to_numpy(dtype=float)
    lo = value_table["ci_low"].to_numpy(dtype=float)
    hi = value_table["ci_high"].to_numpy(dtype=float)
    is_baseline = value_table["ci_low"].isna().to_numpy()  # first row: no "previous" to gain over

    if is_baseline.any():
        ax.scatter(x[is_baseline], y[is_baseline], color="#a0aec0", s=90, zorder=2,
                   label=f"baseline (absolute {metric_label})")
    if (~is_baseline).any():
        xi, yi = x[~is_baseline], y[~is_baseline]
        yerr = np.vstack([yi - lo[~is_baseline], hi[~is_baseline] - yi])
        ax.errorbar(xi, yi, yerr=yerr, fmt="o", capsize=4, color="#2b6cb0", ecolor="#718096",
                    markersize=9, zorder=3, label="Δ vs. previous step (95% CI)")

    for xi, yi, label in zip(x, y, value_table["modality_set"]):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.axhline(0, color="#a0aec0", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Acquisition cost of added modality (£/patient)")
    ax.set_ylabel(f"{metric_label} (baseline)  /  Δ {metric_label} (increments)")
    ax.set_title(title)
    ax.legend(loc="center right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")
