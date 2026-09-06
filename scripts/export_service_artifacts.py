#!/usr/bin/env python
"""Export a self-contained set of service artifacts to service/artifacts/, so
service/app.py can start without the full PTB-XL dataset (data/) present.

Fits the exact same TabularModel + WaveformModel + stacking fuse_fn as
service/app.py's own startup path (mirroring fusion/value.py's PTB-XL arm),
then persists: the two fitted models, the small val-fold score arrays the
fuse_fn stacker needs (not the val fold's raw inputs), and the held-out
test-fold's tabular fields + ECG signals (not the full 21,799-record
dataset) so /score can answer requests for any test-fold ecg_id.

Run once, locally, where PTB-XL is already cached:
    python scripts/export_service_artifacts.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from modality_value.config import SUPERCLASSES
from modality_value.io.ptbxl import build_or_load_signal_cache, labels_matrix, load_metadata, split_indices
from modality_value.modalities.tabular import TABULAR_COLUMNS, TabularModel
from modality_value.modalities.waveform import WaveformModel

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "service" / "artifacts"


def main() -> None:
    print("Loading PTB-XL metadata and signal cache...")
    df = load_metadata()
    splits = split_indices(df)
    y = labels_matrix(df)
    pos = {name: df.index.get_indexer(idx) for name, idx in splits.items()}
    signals = build_or_load_signal_cache(df)

    X_train, y_train = df.loc[splits["train"]], y[pos["train"]]
    X_val, y_val = df.loc[splits["val"]], y[pos["val"]]
    sig_train, sig_val = np.array(signals[pos["train"]]), np.array(signals[pos["val"]])

    print("Fitting tabular model...")
    tabular = TabularModel().fit(X_train, y_train)

    print("Fitting waveform model (this takes ~30-60s)...")
    waveform = WaveformModel().fit(sig_train, y_train, X_val=sig_val, y_val=y_val)

    # The fuse_fn stacker only ever needs these two small score arrays (not
    # the val fold's raw tabular fields / signals) to refit at request time.
    val_scores = {"tabular": tabular.score(X_val), "waveform": waveform.score(sig_val)}

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(ARTIFACTS_DIR / "tabular_model.pkl", "wb") as f:
        pickle.dump(tabular, f)
    with open(ARTIFACTS_DIR / "waveform_model.pkl", "wb") as f:
        pickle.dump(waveform, f)
    np.savez(ARTIFACTS_DIR / "val_scores.npz", tabular=val_scores["tabular"], waveform=val_scores["waveform"])
    np.save(ARTIFACTS_DIR / "y_val.npy", y_val)

    test_idx = splits["test"]
    test_df = df.loc[test_idx, TABULAR_COLUMNS + SUPERCLASSES].copy()
    test_df.index.name = "ecg_id"
    test_df.to_csv(ARTIFACTS_DIR / "test_fold.csv")
    # float16 halves this file's size (~101MB -> ~51MB, comfortably under
    # GitHub's 100MB hard cap); service/app.py upcasts back to float32 on
    # load, matching what WaveformModel expects.
    sig_test = np.array(signals[pos["test"]]).astype(np.float16)
    np.save(ARTIFACTS_DIR / "test_signals.npy", sig_test)

    print(f"Wrote artifacts to {ARTIFACTS_DIR} ({len(test_idx)} test-fold patients).")


if __name__ == "__main__":
    main()
