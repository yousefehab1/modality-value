"""The single interface every modality implements.

A 1D CNN over ECG waveforms and a Cox model over gene-set scores are wildly
different models. They are NOT allowed to be wildly different objects. Every
modality -- tabular, waveform, text, molecular -- exposes exactly this
surface, and nothing upstream (fusion, value analysis, the service) is
allowed to know which concrete class it is holding. This is what lets the
molecular arm and the waveform arm plug into one engine.

Do not let any modality leak its own signature upward.
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class ModalityModel(ABC):
    name: str
    acquisition_cost_gbp: float

    @abstractmethod
    def fit(self, X, y) -> "ModalityModel":
        """Fit in place; return self."""
        ...

    @abstractmethod
    def embed(self, X) -> np.ndarray:
        """Return an (n, d) latent representation."""
        ...

    @abstractmethod
    def score(self, X) -> np.ndarray:
        """Return an (n, k) risk/class-probability array."""
        ...

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "ModalityModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return obj
