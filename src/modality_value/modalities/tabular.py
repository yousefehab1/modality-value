"""LightGBM over PTB-XL demographics: age, sex, height, weight.

height/weight are substantially missing in PTB-XL (many records only have
age/sex populated). We median-impute them AND keep a per-column missingness
indicator, since missingness itself is informative here (it correlates with
which site/era a record came from) and costs nothing to compute.

Design note: rather than one multi-output model, we fit 5 independent
LGBMClassifier one-vs-rest binary models (one per SUPERCLASS). LightGBM's
native multi-output support is limited/awkward for probabilistic multi-label
output, and 5 independent models keep each class trivially inspectable and
map directly onto the "score() returns 5 one-vs-rest probabilities" spec.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from modality_value.config import MODALITY_COSTS, SEED
from modality_value.modalities import register
from modality_value.modalities.base import ModalityModel

TABULAR_COLUMNS = ["age", "sex", "height", "weight"]


@register
class TabularModel(ModalityModel):
    name = "tabular"
    acquisition_cost_gbp = MODALITY_COSTS["tabular"].cost_gbp

    def __init__(self, seed: int = SEED, n_estimators: int = 200):
        self.seed = seed
        self.n_estimators = n_estimators
        self._height_median: float | None = None
        self._weight_median: float | None = None
        self._models: list[LGBMClassifier | None] = []

    def _preprocess(self, X: pd.DataFrame) -> np.ndarray:
        """(n, 6) feature matrix: age, sex, height, height_missing, weight, weight_missing.

        Imputation stats (_height_median / _weight_median) must already be set by
        fit() -- this is what keeps val/test rows from leaking their own
        height/weight distribution into the imputed value.
        """
        if self._height_median is None or self._weight_median is None:
            raise RuntimeError("TabularModel.fit() must be called before embed()/score().")

        df = X[TABULAR_COLUMNS].copy()
        height_missing = df["height"].isna().to_numpy(dtype=np.float32)
        weight_missing = df["weight"].isna().to_numpy(dtype=np.float32)
        df["height"] = df["height"].fillna(self._height_median)
        df["weight"] = df["weight"].fillna(self._weight_median)

        return np.column_stack(
            [
                df["age"].to_numpy(dtype=np.float32),
                df["sex"].to_numpy(dtype=np.float32),
                df["height"].to_numpy(dtype=np.float32),
                height_missing,
                df["weight"].to_numpy(dtype=np.float32),
                weight_missing,
            ]
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TabularModel":
        """X: DataFrame with TABULAR_COLUMNS, TRAINING ROWS ONLY.
        y: (n, len(SUPERCLASSES)) multi-hot label matrix, same row order as X."""
        # Imputation statistics come from these (training) rows only.
        self._height_median = float(X["height"].median())
        self._weight_median = float(X["weight"].median())

        feats = self._preprocess(X)
        self._models = []
        for k in range(y.shape[1]):
            col = y[:, k]
            if len(np.unique(col)) < 2:
                # Degenerate column (shouldn't happen on PTB-XL train fold, but
                # guard against it rather than let LightGBM raise).
                self._models.append(None)
                continue
            model = LGBMClassifier(
                random_state=self.seed,
                n_estimators=self.n_estimators,
                verbosity=-1,
            )
            model.fit(feats, col)
            self._models.append(model)
        return self

    def embed(self, X: pd.DataFrame) -> np.ndarray:
        """(n, 6) preprocessed feature matrix (post-impute, with missingness flags)."""
        return self._preprocess(X)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """(n, len(SUPERCLASSES)) one-vs-rest probabilities."""
        feats = self._preprocess(X)
        n = feats.shape[0]
        out = np.zeros((n, len(self._models)), dtype=np.float32)
        for k, model in enumerate(self._models):
            if model is None:
                out[:, k] = 0.0
            else:
                out[:, k] = model.predict_proba(feats)[:, 1]
        return out


def _main() -> None:
    from modality_value.eval.metrics import macro_auroc
    from modality_value.eval.report import update_phase1_metrics
    from modality_value.io.ptbxl import labels_matrix, load_metadata, split_indices

    df = load_metadata()
    splits = split_indices(df)

    y = labels_matrix(df)
    pos = {name: df.index.get_indexer(idx) for name, idx in splits.items()}

    X_train, y_train = df.loc[splits["train"]], y[pos["train"]]
    X_val, y_val = df.loc[splits["val"]], y[pos["val"]]
    X_test, y_test = df.loc[splits["test"]], y[pos["test"]]

    model = TabularModel().fit(X_train, y_train)

    val_auroc = macro_auroc(y_val, model.score(X_val))
    test_auroc = macro_auroc(y_test, model.score(X_test))

    print(f"tabular: fold9 (val) macro AUROC = {val_auroc:.4f}")
    print(f"tabular: fold10 (test) macro AUROC = {test_auroc:.4f}")

    update_phase1_metrics(
        "tabular",
        {
            "fold9_val_macro_auroc": val_auroc,
            "fold10_test_macro_auroc": test_auroc,
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "features": ["age", "sex", "height", "height_missing", "weight", "weight_missing"],
            "model": "5x LGBMClassifier one-vs-rest",
        },
    )


if __name__ == "__main__":
    _main()
