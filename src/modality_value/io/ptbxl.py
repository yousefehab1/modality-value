"""PTB-XL loader: signals, metadata, reports, labels, folds.

21,799 records / 18,869 patients, 10s each, sampled at 100Hz and 500Hz;
we use only the 100Hz set (see scripts/download_ptbxl.sh).

Splits are NOT random: strat_fold 1-8 train, 9 val, 10 test, per the
dataset's own published recommendation. Never re-split -- patients do not
cross folds.
"""
from __future__ import annotations

import argparse
import ast
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from modality_value.config import (
    NUM_LEADS,
    PTBXL_DATABASE_CSV,
    PTBXL_DIR,
    PTBXL_SCP_STATEMENTS_CSV,
    PTBXL_SIGNALS_MEMMAP,
    SIGNAL_LENGTH,
    SUPERCLASSES,
    TEST_FOLD,
    TRAIN_FOLDS,
    VAL_FOLD,
)


def _parse_scp_codes(raw: str) -> dict:
    return ast.literal_eval(raw)


@lru_cache(maxsize=1)
def load_scp_to_superclass() -> dict[str, str]:
    """Map SCP code -> one of the 5 diagnostic superclasses (or drop if none)."""
    scp_df = pd.read_csv(PTBXL_SCP_STATEMENTS_CSV, index_col=0)
    scp_df = scp_df[scp_df["diagnostic"] == 1]
    return scp_df["diagnostic_class"].dropna().to_dict()


def load_metadata() -> pd.DataFrame:
    """Return the ptbxl_database with scp_codes parsed and superclass multi-hot labels attached."""
    df = pd.read_csv(PTBXL_DATABASE_CSV, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(_parse_scp_codes)

    scp_to_super = load_scp_to_superclass()

    def codes_to_superclasses(codes: dict) -> set[str]:
        supers = set()
        for code in codes:
            sc = scp_to_super.get(code)
            if sc in SUPERCLASSES:
                supers.add(sc)
        return supers

    df["superclasses"] = df["scp_codes"].apply(codes_to_superclasses)
    for sc in SUPERCLASSES:
        df[sc] = df["superclasses"].apply(lambda s, sc=sc: int(sc in s))

    return df


def labels_matrix(df: pd.DataFrame) -> np.ndarray:
    """(n, 5) multi-hot label matrix in SUPERCLASSES order."""
    return df[SUPERCLASSES].to_numpy(dtype=np.float32)


def split_indices(df: pd.DataFrame) -> dict[str, pd.Index]:
    train = df.index[df["strat_fold"].isin(TRAIN_FOLDS)]
    val = df.index[df["strat_fold"] == VAL_FOLD]
    test = df.index[df["strat_fold"] == TEST_FOLD]
    return {"train": train, "val": val, "test": test}


def load_raw_signal(record_path_100: str) -> np.ndarray:
    """Load one record's 100Hz signal as (NUM_LEADS, SIGNAL_LENGTH) float32."""
    signal, _ = wfdb.rdsamp(str(PTBXL_DIR / record_path_100))
    signal = signal.T.astype(np.float32)  # (leads, time)
    assert signal.shape == (NUM_LEADS, SIGNAL_LENGTH), signal.shape
    return signal


def build_or_load_signal_cache(
    df: pd.DataFrame, path: Path = PTBXL_SIGNALS_MEMMAP, min_free_ratio: float = 1.5
) -> np.memmap:
    """Build (once) or load a (n, NUM_LEADS, SIGNAL_LENGTH) float32 memmap of every
    record's 100Hz signal, in the row order of `df` (row i <-> df.iloc[i], i.e.
    NOT df.index -- callers needing a specific ecg_id's row must translate via
    `df.index.get_indexer([ecg_id])` or equivalent positional lookup, same as
    labels_matrix(df) which is aligned to df's row order).

    Avoids re-reading WFDB per-batch during training -- the whole point of this
    cache is that after the one-time build, every epoch reads only this memmap.

    Refuses to build if there isn't at least `min_free_ratio`x the required
    bytes free on disk (a failed/partial write here would corrupt the cache).
    """
    path = Path(path)
    ids_path = path.with_name(path.stem + "_ids.npy")
    n = len(df)
    shape = (n, NUM_LEADS, SIGNAL_LENGTH)

    if path.exists() and ids_path.exists():
        cached_ids = np.load(ids_path)
        if np.array_equal(cached_ids, df.index.to_numpy()) and path.stat().st_size == n * NUM_LEADS * SIGNAL_LENGTH * 4:
            return np.memmap(path, dtype=np.float32, mode="r", shape=shape)
        print(f"Signal cache at {path} is stale for the given df; rebuilding.")

    required_bytes = n * NUM_LEADS * SIGNAL_LENGTH * 4
    path.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(path.parent).free
    if free_bytes < required_bytes * min_free_ratio:
        raise RuntimeError(
            f"Refusing to build signal cache at {path}: need ~{required_bytes / 1e9:.2f}GB "
            f"(x{min_free_ratio} safety margin -> {required_bytes * min_free_ratio / 1e9:.2f}GB), "
            f"only {free_bytes / 1e9:.2f}GB free on disk."
        )

    mm = np.memmap(path, dtype=np.float32, mode="w+", shape=shape)
    for i, (_, row) in enumerate(df.iterrows()):
        mm[i] = load_raw_signal(row["filename_lr"])
        if i % 2000 == 0:
            print(f"  signal cache: {i}/{n} records...")
    mm.flush()
    np.save(ids_path, df.index.to_numpy())
    print(f"Signal cache built at {path} ({required_bytes / 1e9:.2f}GB, {n} records).")
    return np.memmap(path, dtype=np.float32, mode="r", shape=shape)


def _summary() -> None:
    df = load_metadata()
    total = len(df)
    print(f"Total records: {total}")

    fold_counts = df["strat_fold"].value_counts().sort_index()
    print("\nRecords per fold:")
    for fold, count in fold_counts.items():
        print(f"  fold {fold}: {count}")
    print(f"  sum: {fold_counts.sum()}")

    print("\nRecords per superclass (multi-label, not mutually exclusive):")
    for sc in SUPERCLASSES:
        print(f"  {sc}: {int(df[sc].sum())}")

    n_patients = df["patient_id"].nunique()
    print(f"\nUnique patients: {n_patients}")

    assert fold_counts.sum() == 21799, f"expected 21799 records, got {fold_counts.sum()}"
    print("\nOK: counts sum to 21,799.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        _summary()
    else:
        parser.print_help()
