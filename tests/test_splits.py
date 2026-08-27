"""Split integrity: no patient crosses train/val/test, fold 10 is held out."""
import pytest

from modality_value.config import PTBXL_DATABASE_CSV
from modality_value.io.ptbxl import load_metadata, split_indices

pytestmark = pytest.mark.skipif(
    not PTBXL_DATABASE_CSV.exists(), reason="PTB-XL not downloaded (run `make data`)"
)


def test_no_patient_crosses_splits():
    df = load_metadata()
    splits = split_indices(df)
    patients = {name: set(df.loc[idx, "patient_id"]) for name, idx in splits.items()}
    assert patients["train"].isdisjoint(patients["val"])
    assert patients["train"].isdisjoint(patients["test"])
    assert patients["val"].isdisjoint(patients["test"])


def test_fold_counts_sum_to_total():
    df = load_metadata()
    splits = split_indices(df)
    total = sum(len(idx) for idx in splits.values())
    assert total == len(df) == 21799
