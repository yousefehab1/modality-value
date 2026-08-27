"""TCGA molecular arm: Cox-model-backed ModalityModel implementations.

Proves the ModalityModel interface is genuinely modality-agnostic: the same
fit/embed/score surface that wraps a 1D CNN over ECG waveforms elsewhere in
this repo here wraps `lifelines.CoxPHFitter` over TCGA-COAD gene-set (ssGSEA)
scores and clinical covariates, scored by Harrell's C-index instead of AUROC.

Two registered modalities, mirroring `config.MODALITY_COSTS["clinical"]` and
`["molecular"]` so `fusion/value.py`'s incremental order
`["clinical", "molecular"]` has one ModalityModel per cost row:

- `ClinicalCoxModality` ("clinical"): age, sex, tumour stage.
- `MolecularCoxModality` ("molecular"): ssGSEA signature scores exported by
  `scripts/export_tcga_molecular.R` (see `io/tcga.py`).

`y` convention (documented once, used by both classes):
    y is an (n, 2) float ndarray, y[:, 0] = event_time, y[:, 1] = event
    (1 = event/death observed, 0 = censored). This mirrors the `YRef`
    {"time", "event"} convention in eval/metrics.py without forcing a dict
    through the positional `fit(X, y)` signature every other ModalityModel
    uses.

`score()` returns the linear predictor / log partial hazard as an (n, 1)
array -- higher means higher risk, matching `eval.metrics.c_index_metric`'s
expected convention directly (lifelines' `predict_log_partial_hazard` already
follows it, no sign flip needed).

`embed()` returns the standardized (z-scored) design matrix actually handed
to the Cox model, as an (n, d) array -- the "latent representation" for a
linear model is just its own (standardized) inputs.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from modality_value.config import MODALITY_COSTS
from modality_value.modalities import register
from modality_value.modalities.base import ModalityModel

_STAGE_ORDER = {"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0}


def _y_to_frame(y: np.ndarray) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError(f"y must be (n, 2) [time, event], got shape {y.shape}")
    return pd.DataFrame({"time": y[:, 0], "event": y[:, 1]})


class _CoxModality(ModalityModel):
    """Shared plumbing for the two Cox-backed modalities below.

    Subclasses only need to implement `_raw_features(X) -> pd.DataFrame`
    (unstandardized numeric covariates); this base class handles
    standardization (fit on training data only, reused at predict time),
    the lifelines CoxPHFitter itself, and the ModalityModel surface.
    """

    feature_names: Sequence[str]

    def __init__(self, penalizer: float = 0.1) -> None:
        self._model = CoxPHFitter(penalizer=penalizer)
        self._mean: pd.Series | None = None
        self._std: pd.Series | None = None
        self._fitted = False

    # -- subclasses implement this -----------------------------------------
    def _raw_features(self, X) -> pd.DataFrame:
        raise NotImplementedError

    # -- shared standardization ----------------------------------------------
    def _standardize(self, raw: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if fit:
            self._mean = raw.mean()
            self._std = raw.std(ddof=0).replace(0.0, 1.0)
        if self._mean is None or self._std is None:
            raise RuntimeError("model not fitted yet")
        return (raw - self._mean) / self._std

    def _design_matrix(self, X, fit: bool) -> pd.DataFrame:
        raw = self._raw_features(X)
        if fit:
            # Impute any missing covariate values with the training median,
            # computed once at fit time and reused at predict time (no
            # leakage from val/test rows into the imputation statistic).
            self._impute_median = raw.median()
        raw = raw.fillna(self._impute_median)
        return self._standardize(raw, fit=fit)

    # -- ModalityModel surface ------------------------------------------------
    def fit(self, X, y) -> "_CoxModality":
        design = self._design_matrix(X, fit=True)
        df = pd.concat([design.reset_index(drop=True), _y_to_frame(y).reset_index(drop=True)], axis=1)
        self._model.fit(df, duration_col="time", event_col="event")
        self._fitted = True
        return self

    def embed(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() before embed()")
        return self._design_matrix(X, fit=False).to_numpy(dtype=np.float64)

    def score(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() before score()")
        design = self._design_matrix(X, fit=False)
        risk = self._model.predict_log_partial_hazard(design).to_numpy(dtype=np.float64)
        return risk.reshape(-1, 1)


@register
class ClinicalCoxModality(_CoxModality):
    """Clinical covariates only: age, sex, tumour stage (ordinal I-IV)."""

    name = "clinical"
    acquisition_cost_gbp = MODALITY_COSTS["clinical"].cost_gbp
    feature_names = ("age", "sex_male", "stage_ordinal")

    def _raw_features(self, X) -> pd.DataFrame:
        df = pd.DataFrame(X)
        out = pd.DataFrame(index=df.index)
        out["age"] = pd.to_numeric(df["age"], errors="coerce")
        out["sex_male"] = (df["sex"].astype(str).str.lower() == "male").astype(float)
        out["stage_ordinal"] = df["stage"].astype(str).str.upper().map(_STAGE_ORDER)
        return out


@register
class MolecularCoxModality(_CoxModality):
    """ssGSEA signature-score covariates only (no clinical fields)."""

    name = "molecular"
    acquisition_cost_gbp = MODALITY_COSTS["molecular"].cost_gbp

    def __init__(self, penalizer: float = 0.1) -> None:
        super().__init__(penalizer=penalizer)
        self.feature_names: Sequence[str] = ()

    def _raw_features(self, X) -> pd.DataFrame:
        df = pd.DataFrame(X)
        if not self.feature_names:
            # First call (fit): infer signature columns from whatever was
            # passed (io.tcga.signature_columns already filters these out
            # upstream, so X here is expected to be signature columns only).
            self.feature_names = tuple(df.columns)
        return df[list(self.feature_names)].apply(pd.to_numeric, errors="coerce")


_MODALITY_CLASSES = {"clinical": ClinicalCoxModality, "molecular": MolecularCoxModality}


def _modality_X(df: pd.DataFrame, modality: str):
    if modality == "clinical":
        return df[["age", "sex", "stage"]]
    if modality == "molecular":
        from modality_value.io.tcga import signature_columns

        return df[signature_columns(df)]
    raise ValueError(modality)


def _build_y_ref(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "time": df["os_time"].to_numpy(dtype=float),
        "event": df["os_event"].to_numpy(dtype=float),
    }


def compute_oof_scores(df: pd.DataFrame, seed: int = 42, n_splits: int = 5) -> dict[str, np.ndarray]:
    """5-fold out-of-fold cross-fitting for both TCGA modalities.

    Plain `KFold` (unlike PTB-XL's `strat_fold`): TCGA has no published,
    dataset-provided fold assignment to reuse, and no multi-label
    stratification target to preserve (a single binary event column is not
    worth a bespoke stratified splitter here) -- so a fixed-seed KFold is the
    honest, simplest choice, documented rather than silently substituted.
    """
    from sklearn.model_selection import KFold

    n = len(df)
    y = np.stack([df["os_time"].to_numpy(dtype=float), df["os_event"].to_numpy(dtype=float)], axis=1)
    oof: dict[str, np.ndarray] = {name: np.full((n, 1), np.nan) for name in _MODALITY_CLASSES}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(df):
        for name, cls in _MODALITY_CLASSES.items():
            X = _modality_X(df, name)
            model = cls()
            model.fit(X.iloc[train_idx], y[train_idx])
            oof[name][test_idx] = model.score(X.iloc[test_idx])

    for name, arr in oof.items():
        if np.isnan(arr).any():
            raise RuntimeError(f"OOF cross-fitting left NaNs for modality {name!r}")
    return oof


def make_fuse_fn(y_ref: dict[str, np.ndarray]):
    """Late-fusion for survival: refit a small second-stage CoxPHFitter using
    the concatenated per-modality OOF risk scores as covariates -- the
    survival analogue of the classification arm's late-fusion logistic
    regression. With a single modality, fusion is the identity (nothing to
    combine).
    """

    def fuse(scores: dict[str, np.ndarray]) -> np.ndarray:
        if len(scores) == 1:
            return next(iter(scores.values()))
        names = sorted(scores)
        df = pd.DataFrame({name: scores[name].reshape(-1) for name in names})
        df["time"] = y_ref["time"]
        df["event"] = y_ref["event"]
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(df, duration_col="time", event_col="event")
        risk = cph.predict_log_partial_hazard(df[names]).to_numpy(dtype=float)
        return risk.reshape(-1, 1)

    return fuse


def main() -> None:
    import sys

    from modality_value.config import MODALITY_COSTS, RESULTS_DIR, SEED, TCGA_MOLECULAR_SCORES_CSV
    from modality_value.eval.metrics import c_index_metric
    from modality_value.fusion.value import compute_leave_one_out_table, compute_value_table
    from modality_value.io.tcga import load_molecular_scores

    if not TCGA_MOLECULAR_SCORES_CSV.exists():
        print(
            f"Missing {TCGA_MOLECULAR_SCORES_CSV}. Run "
            "`Rscript scripts/export_tcga_molecular.R` first (see that script's "
            "header for what it requires from the thesis repo).",
            file=sys.stderr,
        )
        sys.exit(1)

    df = load_molecular_scores().reset_index(drop=True)
    y_ref = _build_y_ref(df)

    print(f"Loaded {len(df)} TCGA-COAD patients; event rate {y_ref['event'].mean():.3f}.")

    oof_scores = compute_oof_scores(df, seed=SEED, n_splits=5)
    fuse_fn = make_fuse_fn(y_ref)
    costs = {name: MODALITY_COSTS[name].cost_gbp for name in _MODALITY_CLASSES}

    value_table = compute_value_table(
        y_ref=y_ref,
        oof_scores=oof_scores,
        costs=costs,
        metric_fn=c_index_metric,
        fuse_fn=fuse_fn,
        incremental_order=["clinical", "molecular"],
        n_boot=1000,
        seed=SEED,
    )
    loo_table = compute_leave_one_out_table(
        y_ref=y_ref,
        oof_scores=oof_scores,
        costs=costs,
        metric_fn=c_index_metric,
        fuse_fn=fuse_fn,
        all_modalities=["clinical", "molecular"],
        n_boot=1000,
        seed=SEED,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    value_csv = RESULTS_DIR / "molecular_value.csv"
    loo_csv = RESULTS_DIR / "molecular_value_loo.csv"
    value_table.to_csv(value_csv, index=False)
    loo_table.to_csv(loo_csv, index=False)

    print("\nIncremental value table:")
    print(value_table.to_string(index=False))
    print("\nLeave-one-out table:")
    print(loo_table.to_string(index=False))
    print(f"\nWrote {value_csv}")
    print(f"Wrote {loo_csv}")

    _plot_value(value_table, RESULTS_DIR / "fig_molecular_value.png")


def _plot_value(value_table, out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = value_table["acquisition_cost_gbp"].to_numpy(dtype=float)
    y = value_table["delta_vs_previous"].to_numpy(dtype=float)
    lo = value_table["ci_low"].to_numpy(dtype=float)
    hi = value_table["ci_high"].to_numpy(dtype=float)
    is_baseline = value_table["ci_low"].isna().to_numpy()  # first row: no "previous" to gain over

    # Baseline row's "delta" is its own absolute metric (nothing preceded it),
    # not a marginal gain -- plot it distinctly (grey, no CI) so it isn't read
    # as a gain on the same footing as the incremental points.
    if is_baseline.any():
        ax.scatter(x[is_baseline], y[is_baseline], color="#a0aec0", s=90, zorder=2,
                   label="baseline (absolute C-index)")
    if (~is_baseline).any():
        xi, yi = x[~is_baseline], y[~is_baseline]
        yerr = np.vstack([yi - lo[~is_baseline], hi[~is_baseline] - yi])
        ax.errorbar(xi, yi, yerr=yerr, fmt="o", capsize=4, color="#2b6cb0", ecolor="#718096",
                    markersize=9, zorder=3, label="Δ vs. previous step (95% CI)")

    for xi, yi, label in zip(x, y, value_table["modality_set"]):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.axhline(0, color="#a0aec0", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Acquisition cost of added modality (£/patient)")
    ax.set_ylabel("C-index (baseline)  /  Δ C-index (increments)")
    ax.set_title("TCGA molecular arm: discrimination gain per £ spent")
    ax.legend(loc="center right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
