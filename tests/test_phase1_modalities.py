"""Phase 1 (tabular + waveform) unit tests that don't require PTB-XL to be
downloaded -- they use small synthetic arrays so they always run in CI.

Covers:
- the normalization leakage trap named in the build plan: LeadNormalizer
  (and WaveformModel.fit) must compute mean/std from the rows it's given,
  not from any wider set -- demonstrated by showing train-only stats differ
  materially from train+val stats.
- tabular imputation leakage: TabularModel's median/missingness handling
  must come from the training rows passed to fit(), not from val/test.
- basic shape contracts for score()/embed() on both tabular and waveform
  (features) arms, using tiny fixtures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modality_value.config import NUM_LEADS, SIGNAL_LENGTH
from modality_value.modalities.tabular import TabularModel
from modality_value.modalities.waveform import LeadNormalizer, WaveformModel
from modality_value.modalities.waveform_features import FEATURE_NAMES, extract_features


def test_lead_normalizer_uses_only_the_rows_it_is_given():
    rng = np.random.default_rng(0)
    train = rng.normal(loc=0.0, scale=1.0, size=(50, NUM_LEADS, SIGNAL_LENGTH)).astype(np.float32)
    val = rng.normal(loc=100.0, scale=50.0, size=(10, NUM_LEADS, SIGNAL_LENGTH)).astype(np.float32)

    normalizer_train_only = LeadNormalizer.fit(train)
    normalizer_train_plus_val = LeadNormalizer.fit(np.concatenate([train, val], axis=0))

    # Fitting on train-only vs train+val must give materially different stats --
    # this is the concrete demonstration of the leakage trap: if a caller
    # accidentally normalizes before splitting, this is the number that moves.
    assert not np.allclose(normalizer_train_only.mean, normalizer_train_plus_val.mean, atol=1.0)

    # And train-only stats must exactly match directly-computed train stats.
    assert np.allclose(normalizer_train_only.mean, train.mean(axis=(0, 2)), atol=1e-5)
    assert np.allclose(normalizer_train_only.std, train.std(axis=(0, 2)), atol=1e-5)


def test_waveform_model_fit_normalizer_ignores_val_rows():
    rng = np.random.default_rng(1)
    n_train, n_val = 8, 4
    X_train = rng.normal(0.0, 1.0, size=(n_train, NUM_LEADS, SIGNAL_LENGTH)).astype(np.float32)
    y_train = rng.integers(0, 2, size=(n_train, 5)).astype(np.float32)
    X_val = rng.normal(50.0, 10.0, size=(n_val, NUM_LEADS, SIGNAL_LENGTH)).astype(np.float32)
    y_val = rng.integers(0, 2, size=(n_val, 5)).astype(np.float32)

    model = WaveformModel(epochs=1, batch_size=4, patience=1)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)

    assert model.normalizer is not None
    assert np.allclose(model.normalizer.mean, X_train.mean(axis=(0, 2)), atol=1e-4)
    # The val distribution is shifted by +50; if it had leaked into the
    # normalizer's stats, the mean would be nowhere near the train-only mean.
    assert not np.allclose(model.normalizer.mean, X_val.mean(axis=(0, 2)), atol=1.0)


def test_waveform_model_score_and_embed_shapes():
    rng = np.random.default_rng(2)
    n_train = 8
    X_train = rng.normal(0.0, 1.0, size=(n_train, NUM_LEADS, SIGNAL_LENGTH)).astype(np.float32)
    y_train = rng.integers(0, 2, size=(n_train, 5)).astype(np.float32)

    model = WaveformModel(epochs=1, batch_size=4, patience=1)
    model.fit(X_train, y_train, verbose=False)

    n_test = 3
    X_test = rng.normal(0.0, 1.0, size=(n_test, NUM_LEADS, SIGNAL_LENGTH)).astype(np.float32)
    scores = model.score(X_test)
    emb = model.embed(X_test)
    assert scores.shape == (n_test, 5)
    assert emb.shape == (n_test, 256)


def test_tabular_imputation_uses_only_training_rows():
    X_train = pd.DataFrame(
        {
            "age": [10.0, 20.0, 30.0, 40.0],
            "sex": [0, 1, 0, 1],
            "height": [150.0, np.nan, 170.0, np.nan],
            "weight": [50.0, 60.0, np.nan, 80.0],
        }
    )
    y_train = np.array(
        [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0]], dtype=np.float32
    )
    model = TabularModel().fit(X_train, y_train)

    assert model._height_median == pytest.approx(160.0)  # median of [150, 170]
    assert model._weight_median == pytest.approx(60.0)  # median of [50, 60, 80]

    # A val row with wildly different height/weight must NOT change the
    # imputation stats used to fill its own missing values.
    X_val = pd.DataFrame({"age": [99.0], "sex": [1], "height": [np.nan], "weight": [np.nan]})
    feats = model.embed(X_val)
    # columns: age, sex, height, height_missing, weight, weight_missing
    assert feats[0, 2] == pytest.approx(160.0)
    assert feats[0, 3] == 1.0
    assert feats[0, 4] == pytest.approx(60.0)
    assert feats[0, 5] == 1.0


def test_tabular_score_shape():
    X_train = pd.DataFrame(
        {
            "age": np.linspace(20, 80, 20),
            "sex": [i % 2 for i in range(20)],
            "height": [170.0] * 10 + [np.nan] * 10,
            "weight": [70.0] * 20,
        }
    )
    rng = np.random.default_rng(3)
    y_train = rng.integers(0, 2, size=(20, 5)).astype(np.float32)
    model = TabularModel().fit(X_train, y_train)
    scores = model.score(X_train)
    assert scores.shape == (20, 5)
    assert np.all((scores >= 0) & (scores <= 1))


def test_waveform_features_extraction_shape():
    rng = np.random.default_rng(4)
    signals = rng.normal(0.0, 1.0, size=(3, NUM_LEADS, SIGNAL_LENGTH)).astype(np.float32)
    feats = extract_features(signals)
    assert feats.shape == (3, len(FEATURE_NAMES))
