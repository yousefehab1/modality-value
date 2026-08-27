"""Loader for the molecular arm's exported CSV (see Phase 4 / R export step).

The CSV is produced once, read-only, from the existing thesis R pipeline:
sample_id, <one column per signature score>, age, sex, stage, os_time, os_event.
This module only reads it -- no R code lives here.
"""
from __future__ import annotations

import pandas as pd

from modality_value.config import TCGA_MOLECULAR_SCORES_CSV

NON_SIGNATURE_COLUMNS = {"sample_id", "age", "sex", "stage", "os_time", "os_event"}


def load_molecular_scores() -> pd.DataFrame:
    df = pd.read_csv(TCGA_MOLECULAR_SCORES_CSV)
    required = {"sample_id", "age", "sex", "stage", "os_time", "os_event"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"molecular_scores.csv missing required columns: {missing}")
    return df


def signature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_SIGNATURE_COLUMNS]
