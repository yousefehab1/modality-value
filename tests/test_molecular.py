"""Molecular arm (Phase 4): registration + a data-dependent sanity check.

Interface conformance (fit/embed/score exist) is already covered generically
by tests/test_interface.py's registry-based parametrization once this module
is imported (see conftest.py). This file adds:

1. An explicit check that both TCGA modalities are registered under the
   names fusion/value.py expects ("clinical", "molecular").
2. A data-dependent test, skipped if the exported CSV doesn't exist yet
   (mirrors the skip pattern in tests/test_splits.py for PTB-XL), that the
   CSV has the required columns and that a simple holdout Cox fit beats
   chance (C-index > 0.5).
"""
from __future__ import annotations

import numpy as np
import pytest

from modality_value.config import TCGA_MOLECULAR_SCORES_CSV
from modality_value.modalities import registered_modalities
from modality_value.modalities.molecular import ClinicalCoxModality, MolecularCoxModality


def test_molecular_modalities_registered():
    reg = registered_modalities()
    assert "ClinicalCoxModality" in reg
    assert "MolecularCoxModality" in reg
    assert reg["ClinicalCoxModality"] is ClinicalCoxModality
    assert reg["MolecularCoxModality"] is MolecularCoxModality
    assert ClinicalCoxModality.name == "clinical"
    assert MolecularCoxModality.name == "molecular"


pytestmark_data = pytest.mark.skipif(
    not TCGA_MOLECULAR_SCORES_CSV.exists(),
    reason="TCGA molecular_scores.csv not exported yet "
    "(run `Rscript scripts/export_tcga_molecular.R`)",
)


@pytestmark_data
def test_molecular_scores_csv_schema():
    from modality_value.io.tcga import load_molecular_scores, signature_columns

    df = load_molecular_scores()
    required = {"sample_id", "age", "sex", "stage", "os_time", "os_event"}
    assert required.issubset(df.columns)
    assert len(signature_columns(df)) > 0
    assert len(df) > 0
    assert df["os_event"].isin([0, 1]).all()
    assert (df["os_time"] >= 0).all()


@pytestmark_data
def test_molecular_cox_beats_chance():
    from sklearn.model_selection import train_test_split

    from modality_value.eval.metrics import c_index_metric
    from modality_value.io.tcga import load_molecular_scores, signature_columns

    df = load_molecular_scores().reset_index(drop=True)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.3, random_state=42
    )
    y = np.stack([df["os_time"].to_numpy(dtype=float), df["os_event"].to_numpy(dtype=float)], axis=1)

    X = df[signature_columns(df)]
    model = MolecularCoxModality()
    model.fit(X.iloc[train_idx], y[train_idx])

    score = model.score(X.iloc[test_idx])
    assert score.shape == (len(test_idx), 1)

    embedded = model.embed(X.iloc[test_idx])
    assert embedded.shape == (len(test_idx), len(signature_columns(df)))

    y_ref_test = {"time": y[test_idx, 0], "event": y[test_idx, 1]}
    c = c_index_metric(y_ref_test, score)
    assert c > 0.5, f"molecular C-index {c} not better than chance"


@pytestmark_data
def test_clinical_cox_beats_chance():
    from sklearn.model_selection import train_test_split

    from modality_value.eval.metrics import c_index_metric
    from modality_value.io.tcga import load_molecular_scores

    df = load_molecular_scores().reset_index(drop=True)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.3, random_state=42
    )
    y = np.stack([df["os_time"].to_numpy(dtype=float), df["os_event"].to_numpy(dtype=float)], axis=1)

    X = df[["age", "sex", "stage"]]
    model = ClinicalCoxModality()
    model.fit(X.iloc[train_idx], y[train_idx])
    score = model.score(X.iloc[test_idx])

    y_ref_test = {"time": y[test_idx, 0], "event": y[test_idx, 1]}
    c = c_index_metric(y_ref_test, score)
    assert c > 0.5, f"clinical C-index {c} not better than chance"
