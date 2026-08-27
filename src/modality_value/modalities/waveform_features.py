"""FALLBACK waveform arm: heart-rate variability + QRS + per-lead
amplitude/energy features fed into LightGBM.

Only used if the 1D CNN in waveform.py does not clear the ~0.80 macro AUROC
stop-line named in the build plan, after a reasonable training attempt (a
handful of honest hyperparameter/debug iterations, not weeks of tuning). If
that fallback is invoked, `make waveform` is repointed at this module's
_main() instead of waveform.py's, and the README says so explicitly -- this
is a sanctioned fallback, not a hidden failure.
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from scipy.signal import butter, filtfilt, find_peaks, peak_widths

from modality_value.config import MODALITY_COSTS, NUM_LEADS, SAMPLING_RATE_HZ, SEED
from modality_value.modalities import register
from modality_value.modalities.base import ModalityModel

# PTB-XL's 12-lead order is I, II, III, aVR, aVL, aVF, V1..V6 -- lead II
# (index 1) is the conventional rhythm-strip lead for QRS/HRV detection.
LEAD_II_INDEX = 1


def _bandpass(sig: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 2) -> np.ndarray:
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, sig)


def _detect_r_peaks(lead_ii: np.ndarray, fs: int = SAMPLING_RATE_HZ) -> np.ndarray:
    """Crude Pan-Tompkins-style QRS detector: bandpass -> square -> moving-average
    integrate -> peak find. Adequate for HRV summary statistics; not a
    clinical-grade detector."""
    if np.std(lead_ii) < 1e-9:
        return np.array([], dtype=int)
    filtered = _bandpass(lead_ii, lowcut=5, highcut=15, fs=fs)
    squared = filtered**2
    window = max(1, int(0.08 * fs))
    integrated = np.convolve(squared, np.ones(window) / window, mode="same")
    if integrated.std() < 1e-9:
        return np.array([], dtype=int)
    min_distance = max(1, int(0.3 * fs))  # refractory period: max plausible HR ~200bpm
    height = integrated.mean() + 0.5 * integrated.std()
    peaks, _ = find_peaks(integrated, distance=min_distance, height=height)
    return peaks


def _record_features(record: np.ndarray) -> np.ndarray:
    """record: (NUM_LEADS, SIGNAL_LENGTH) -> 1D feature vector (see FEATURE_NAMES)."""
    feats: list[float] = []

    for lead in range(record.shape[0]):
        sig = record[lead]
        feats.extend(
            [
                float(sig.mean()),
                float(sig.std()),
                float(sig.min()),
                float(sig.max()),
                float(np.mean(sig**2)),  # energy
                float(np.mean(np.abs(np.diff(np.sign(sig)))) / 2),  # zero-crossing rate
            ]
        )

    peaks = _detect_r_peaks(record[LEAD_II_INDEX])
    if len(peaks) >= 2:
        rr = np.diff(peaks) / SAMPLING_RATE_HZ  # seconds
        mean_rr = float(rr.mean())
        sdnn = float(rr.std())
        rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) >= 2 else float("nan")
        mean_hr = 60.0 / mean_rr if mean_rr > 0 else float("nan")
        try:
            widths = peak_widths(record[LEAD_II_INDEX], peaks, rel_height=0.5)[0]
            qrs_width_ms = float(np.mean(widths)) / SAMPLING_RATE_HZ * 1000
        except Exception:
            qrs_width_ms = float("nan")
    else:
        mean_rr = sdnn = rmssd = mean_hr = qrs_width_ms = float("nan")

    feats.extend([mean_rr, sdnn, rmssd, mean_hr, qrs_width_ms, float(len(peaks))])
    return np.array(feats, dtype=np.float32)


FEATURE_NAMES = [
    f"lead{i}_{stat}" for i in range(NUM_LEADS) for stat in ["mean", "std", "min", "max", "energy", "zcr"]
] + ["mean_rr_s", "sdnn_s", "rmssd_s", "mean_hr_bpm", "qrs_width_ms", "n_beats"]


def extract_features(signals: np.ndarray) -> np.ndarray:
    """signals: (n, NUM_LEADS, SIGNAL_LENGTH) -> (n, len(FEATURE_NAMES)).
    LightGBM handles the NaNs left by failed peak detection natively (treated
    as a learned split direction), so we don't impute here."""
    signals = np.asarray(signals)
    return np.stack([_record_features(signals[i]) for i in range(signals.shape[0])])


@register
class WaveformFeaturesModel(ModalityModel):
    """5x LGBMClassifier one-vs-rest over HRV/QRS/amplitude features, mirroring
    TabularModel's approach. Registered under its own name so both this and
    WaveformModel can coexist in the registry/tests; the Makefile decides
    which one's __main__ actually populates results/phase1_metrics.json's
    "waveform" key."""

    name = "waveform_features"
    acquisition_cost_gbp = MODALITY_COSTS["waveform"].cost_gbp  # same acquisition (the ECG itself)

    def __init__(self, seed: int = SEED, n_estimators: int = 300):
        self.seed = seed
        self.n_estimators = n_estimators
        self._models: list[LGBMClassifier | None] = []

    def _as_features(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 3:  # raw (n, leads, time) signal -- extract on the fly
            return extract_features(X)
        return X.astype(np.float32)  # already-extracted (n, d) features

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WaveformFeaturesModel":
        feats = self._as_features(X)
        self._models = []
        for k in range(y.shape[1]):
            col = y[:, k]
            if len(np.unique(col)) < 2:
                self._models.append(None)
                continue
            model = LGBMClassifier(random_state=self.seed, n_estimators=self.n_estimators, verbosity=-1)
            model.fit(feats, col)
            self._models.append(model)
        return self

    def embed(self, X: np.ndarray) -> np.ndarray:
        return self._as_features(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        feats = self._as_features(X)
        n = feats.shape[0]
        out = np.zeros((n, len(self._models)), dtype=np.float32)
        for k, model in enumerate(self._models):
            out[:, k] = 0.0 if model is None else model.predict_proba(feats)[:, 1]
        return out


def _main() -> None:
    from modality_value.eval.metrics import macro_auroc
    from modality_value.eval.report import update_phase1_metrics
    from modality_value.io.ptbxl import build_or_load_signal_cache, labels_matrix, load_metadata, split_indices

    df = load_metadata()
    splits = split_indices(df)
    y = labels_matrix(df)
    pos = {name: df.index.get_indexer(idx) for name, idx in splits.items()}

    print("Loading/building signal cache...")
    signals = build_or_load_signal_cache(df)

    print("Extracting HRV/QRS/amplitude features (train)...")
    X_train = extract_features(np.array(signals[pos["train"]]))
    y_train = y[pos["train"]]
    print("Extracting features (val)...")
    X_val = extract_features(np.array(signals[pos["val"]]))
    y_val = y[pos["val"]]
    print("Extracting features (test)...")
    X_test = extract_features(np.array(signals[pos["test"]]))
    y_test = y[pos["test"]]

    model = WaveformFeaturesModel().fit(X_train, y_train)

    val_auroc = macro_auroc(y_val, model.score(X_val))
    test_auroc = macro_auroc(y_test, model.score(X_test))
    print(f"waveform_features: fold9 (val) macro AUROC = {val_auroc:.4f}")
    print(f"waveform_features: fold10 (test) macro AUROC = {test_auroc:.4f}")

    update_phase1_metrics(
        "waveform",
        {
            "fold9_val_macro_auroc": val_auroc,
            "fold10_test_macro_auroc": test_auroc,
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "model": "HRV/QRS/amplitude features -> 5x LGBMClassifier one-vs-rest",
            "note": "FALLBACK arm: the 1D CNN (waveform.py) did not clear the ~0.80 "
            "macro AUROC stop-line after a reasonable training attempt; see README.",
        },
    )


if __name__ == "__main__":
    _main()
