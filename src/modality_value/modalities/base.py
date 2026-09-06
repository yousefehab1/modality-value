"""The single interface every modality implements.

A 1D CNN over ECG waveforms and a Cox model over gene-set scores are
different models, but they expose exactly this surface so that fusion, value
analysis, and the service never need to know which concrete class they're
holding.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

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


def get_device():
    """The torch device shared by every GPU-capable modality (waveform, text)."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
