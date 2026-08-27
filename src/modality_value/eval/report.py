"""Shared results/phase1_metrics.json read-merge-write helper.

`make tabular` and `make waveform` (or `make waveform`'s fallback to
waveform_features) each run independently and must not clobber each other's
entry in the same JSON file -- this merges by top-level key instead of
overwriting the whole file.
"""
from __future__ import annotations

import json
from pathlib import Path

from modality_value.config import RESULTS_DIR

PHASE1_METRICS_PATH = RESULTS_DIR / "phase1_metrics.json"


def update_phase1_metrics(key: str, metrics: dict) -> None:
    PHASE1_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if PHASE1_METRICS_PATH.exists():
        with open(PHASE1_METRICS_PATH) as f:
            existing = json.load(f)
    existing[key] = metrics
    with open(PHASE1_METRICS_PATH, "w") as f:
        json.dump(existing, f, indent=2, sort_keys=True)
    print(f"Updated {PHASE1_METRICS_PATH} [{key}]")
