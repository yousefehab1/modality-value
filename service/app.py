"""Local demo API for the PTB-XL tabular+waveform fusion arm.

Scope cut (see README "What was cut"): the text/LoRA modality is excluded
from this service. Fine-tuning it takes ~30+ minutes, which is fine for an
offline `make fusion` run but not for something you'd want to refit on every
`make serve` restart. Serving it would require persisting the fine-tuned
adapter to disk and loading it, which is real but out of scope for a local
demo -- adding it back is the first line item in "Next steps".

On startup this either loads persisted service/artifacts/ (see
scripts/export_service_artifacts.py: a fitted TabularModel + WaveformModel,
the small val-fold score arrays the stacker needs, and the held-out
test-fold's tabular fields + signals), or, if that directory is absent,
fits TabularModel + WaveformModel on the real PTB-XL train fold and a
stacking fuse_fn on the val fold directly from data/, exactly like
`fusion/value.py::main()`'s PTB-XL arm. Either way it serves scores for any
ecg_id in the held-out test fold (fold 10) -- the fold nothing above was fit
on. The artifacts path is what makes this deployable without the full
~1.7GB PTB-XL dataset present in the container.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from modality_value.config import SUPERCLASSES
from modality_value.fusion.value import _ptbxl_make_fuse_fn
from modality_value.io.ptbxl import build_or_load_signal_cache, labels_matrix, load_metadata, split_indices
from modality_value.modalities.tabular import TABULAR_COLUMNS, TabularModel
from modality_value.modalities.waveform import WaveformModel

logger = logging.getLogger(__name__)

app = FastAPI(title="modality-value demo: PTB-XL tabular+waveform fusion")

_state: dict = {}

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


@app.on_event("startup")
def _fit_models() -> None:
    if (ARTIFACTS_DIR / "tabular_model.pkl").exists():
        _load_from_artifacts()
    else:
        _fit_from_raw_data()


def _load_from_artifacts() -> None:
    logger.info("Loading persisted service artifacts from %s...", ARTIFACTS_DIR)
    with open(ARTIFACTS_DIR / "tabular_model.pkl", "rb") as f:
        tabular = pickle.load(f)
    with open(ARTIFACTS_DIR / "waveform_model.pkl", "rb") as f:
        waveform = pickle.load(f)
    val_scores = dict(np.load(ARTIFACTS_DIR / "val_scores.npz"))
    y_val = np.load(ARTIFACTS_DIR / "y_val.npy")
    fuse_fn = _ptbxl_make_fuse_fn(val_scores, y_val)

    test_df = pd.read_csv(ARTIFACTS_DIR / "test_fold.csv", index_col="ecg_id")
    test_signals = np.load(ARTIFACTS_DIR / "test_signals.npy").astype(np.float32)

    _state["df"] = test_df
    _state["signals"] = test_signals
    _state["tabular"] = tabular
    _state["waveform"] = waveform
    _state["fuse_fn"] = fuse_fn
    _state["test_ids"] = list(test_df.index)
    logger.info("Startup complete (from artifacts): %d test-fold patients available.", len(_state["test_ids"]))


def _fit_from_raw_data() -> None:
    logger.info("No service/artifacts/ found; loading PTB-XL metadata and signal cache from data/...")
    df = load_metadata()
    splits = split_indices(df)
    y = labels_matrix(df)
    pos = {name: df.index.get_indexer(idx) for name, idx in splits.items()}
    signals = build_or_load_signal_cache(df)

    X_train, y_train = df.loc[splits["train"]], y[pos["train"]]
    X_val, y_val = df.loc[splits["val"]], y[pos["val"]]
    sig_train, sig_val = np.array(signals[pos["train"]]), np.array(signals[pos["val"]])

    logger.info("Fitting tabular model...")
    tabular = TabularModel().fit(X_train, y_train)

    logger.info("Fitting waveform model (this takes ~30-60s)...")
    waveform = WaveformModel().fit(sig_train, y_train, X_val=sig_val, y_val=y_val)

    val_scores = {"tabular": tabular.score(X_val), "waveform": waveform.score(sig_val)}
    fuse_fn = _ptbxl_make_fuse_fn(val_scores, y_val)

    _state["df"] = df.loc[splits["test"], TABULAR_COLUMNS + SUPERCLASSES]
    _state["signals"] = np.array(signals[pos["test"]])
    _state["tabular"] = tabular
    _state["waveform"] = waveform
    _state["fuse_fn"] = fuse_fn
    _state["test_ids"] = list(splits["test"])
    logger.info("Startup complete (fit from raw data): %d test-fold patients available.", len(_state["test_ids"]))


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if "fuse_fn" in _state else "starting", "modalities": ["tabular", "waveform"]}


@app.get("/patients")
def list_patients(n: int = 10) -> dict:
    """Sample ecg_ids from the held-out test fold, for trying /score/{ecg_id}."""
    if "test_ids" not in _state:
        raise HTTPException(503, "Models still loading, try again shortly.")
    return {"ecg_ids": _state["test_ids"][:n], "total_test_fold_size": len(_state["test_ids"])}


@app.get("/score/{ecg_id}")
def score(ecg_id: int) -> dict:
    if "fuse_fn" not in _state:
        raise HTTPException(503, "Models still loading, try again shortly.")
    df = _state["df"]
    if ecg_id not in df.index or ecg_id not in _state["test_ids"]:
        raise HTTPException(404, f"ecg_id {ecg_id} not found in the held-out test fold. See /patients for valid ids.")

    row_pos = df.index.get_indexer([ecg_id])
    X_row = df.loc[[ecg_id]]
    sig_row = np.array(_state["signals"][row_pos])

    tabular_score = _state["tabular"].score(X_row)
    waveform_score = _state["waveform"].score(sig_row)
    fused_score = _state["fuse_fn"]({"tabular": tabular_score, "waveform": waveform_score})

    true_labels = {sc: bool(df.loc[ecg_id, sc]) for sc in SUPERCLASSES}

    return {
        "ecg_id": ecg_id,
        "fused": dict(zip(SUPERCLASSES, fused_score[0].tolist())),
        "tabular_only": dict(zip(SUPERCLASSES, tabular_score[0].tolist())),
        "waveform_only": dict(zip(SUPERCLASSES, waveform_score[0].tolist())),
        "true_labels": true_labels,
    }
