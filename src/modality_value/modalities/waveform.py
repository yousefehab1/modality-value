"""1D residual CNN over the 12-lead x 1000-sample (10s @ 100Hz) ECG waveform.

Normalization: per-lead z-score, with mean/std computed on TRAINING rows only
and then applied identically to every record (train/val/test/inference).
This is the leakage trap named in the build plan: computing those stats over
val/test rows (even by accident, e.g. by normalizing before splitting) would
let test-set amplitude/offset statistics leak into every model input.

Architecture (as specified in the build plan): stem conv (kernel 7, 64ch,
stride 2) -> 4 residual blocks (64/128/128/256 channels, stride 2 each) ->
global average pool -> dropout -> linear to 5 logits. BCEWithLogitsLoss,
AdamW, cosine LR schedule, batch 64, <=30 epochs, early stop on fold-9 macro
AUROC.

embed() returns the pre-logit 256-d pooled vector (post GAP, pre-dropout,
pre-fc) -- this is the "latent representation" the ModalityModel interface
promises, and what a downstream fusion layer could use instead of / in
addition to the 5-way score.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from modality_value.config import MODALITY_COSTS, NUM_LEADS, SEED, SIGNAL_LENGTH, SUPERCLASSES
from modality_value.modalities import register
from modality_value.modalities.base import ModalityModel


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class LeadNormalizer:
    """Per-lead z-score stats, fit on training rows only."""

    mean: np.ndarray  # (NUM_LEADS,)
    std: np.ndarray  # (NUM_LEADS,)

    @classmethod
    def fit(cls, signals: np.ndarray) -> "LeadNormalizer":
        # signals: (n, leads, time) -- MUST be training rows only.
        mean = signals.mean(axis=(0, 2)).astype(np.float32)
        std = signals.std(axis=(0, 2)).astype(np.float32)
        std = np.where(std < 1e-6, np.float32(1.0), std)
        return cls(mean=mean, std=std)

    def transform(self, signals: np.ndarray) -> np.ndarray:
        return (signals - self.mean[None, :, None]) / self.std[None, :, None]


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(out + identity)


class ECGResNet1D(nn.Module):
    def __init__(self, num_leads: int = NUM_LEADS, num_classes: int = len(SUPERCLASSES), dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(num_leads, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResBlock1D(64, 64, stride=2),
            ResBlock1D(64, 128, stride=2),
            ResBlock1D(128, 128, stride=2),
            ResBlock1D(128, 256, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.blocks(x)
        emb = self.pool(x).squeeze(-1)  # (n, 256)
        logits = self.fc(self.dropout(emb))
        return logits, emb


@register
class WaveformModel(ModalityModel):
    name = "waveform"
    acquisition_cost_gbp = MODALITY_COSTS["waveform"].cost_gbp

    def __init__(
        self,
        seed: int = SEED,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 6,
        device: torch.device | None = None,
    ):
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device or get_device()
        self.normalizer: LeadNormalizer | None = None
        self.model: ECGResNet1D | None = None
        self.history: list[dict] = []

    def _forward_all(self, X: np.ndarray, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """Run the whole array through the model, batched. Returns (logits, emb)."""
        assert self.model is not None and self.normalizer is not None
        self.model.eval()
        Xn = self.normalizer.transform(np.asarray(X, dtype=np.float32))
        logits_out, emb_out = [], []
        with torch.no_grad():
            for start in range(0, len(Xn), batch_size):
                batch = torch.from_numpy(Xn[start : start + batch_size]).to(self.device)
                logits, emb = self.model(batch)
                logits_out.append(logits.cpu().numpy())
                emb_out.append(emb.cpu().numpy())
        return np.concatenate(logits_out, axis=0), np.concatenate(emb_out, axis=0)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = True,
    ) -> "WaveformModel":
        """X: (n, NUM_LEADS, SIGNAL_LENGTH) raw signal, TRAINING rows only.
        y: (n, len(SUPERCLASSES)) multi-hot labels, same row order as X.

        X_val/y_val (fold 9), if given, drive early stopping on macro AUROC --
        NOT used to fit the normalizer, only to pick when to stop / which
        epoch's weights to keep.
        """
        from modality_value.eval.metrics import macro_auroc

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.normalizer = LeadNormalizer.fit(X)  # train rows only, by contract
        Xn = self.normalizer.transform(X)

        self.model = ECGResNet1D().to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.BCEWithLogitsLoss()

        n = len(Xn)
        best_val_auroc = -np.inf
        best_state = None
        epochs_without_improvement = 0
        self.history = []

        for epoch in range(self.epochs):
            self.model.train()
            perm = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            t0 = time.time()
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = torch.from_numpy(Xn[idx]).to(self.device)
                yb = torch.from_numpy(y[idx]).to(self.device)

                optimizer.zero_grad()
                logits, _ = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()

            record = {"epoch": epoch, "train_loss": epoch_loss / max(n_batches, 1), "time_s": time.time() - t0}

            if X_val is not None and y_val is not None:
                val_logits, _ = self._forward_all(X_val)
                val_scores = 1 / (1 + np.exp(-val_logits))
                val_auroc = macro_auroc(np.asarray(y_val), val_scores)
                record["val_macro_auroc"] = val_auroc

                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc
                    best_state = copy.deepcopy(self.model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            self.history.append(record)
            if verbose:
                msg = f"  epoch {epoch+1}/{self.epochs}  loss={record['train_loss']:.4f}  time={record['time_s']:.1f}s"
                if "val_macro_auroc" in record:
                    msg += f"  val_macro_auroc={record['val_macro_auroc']:.4f} (best={best_val_auroc:.4f})"
                print(msg)

            if X_val is not None and epochs_without_improvement >= self.patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch+1} (no val improvement for {self.patience} epochs)")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def embed(self, X: np.ndarray) -> np.ndarray:
        """(n, 256) pre-logit pooled embedding."""
        _, emb = self._forward_all(X)
        return emb

    def score(self, X: np.ndarray) -> np.ndarray:
        """(n, len(SUPERCLASSES)) sigmoid probabilities."""
        logits, _ = self._forward_all(X)
        return 1 / (1 + np.exp(-logits))

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model is not None:
            state["model"] = None
            state["_model_state_dict"] = {k: v.cpu() for k, v in self.model.state_dict().items()}
        return state

    def __setstate__(self, state):
        model_state_dict = state.pop("_model_state_dict", None)
        self.__dict__.update(state)
        if model_state_dict is not None:
            self.model = ECGResNet1D().to(self.device)
            self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)


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

    print("Materializing train/val/test subsets into memory...")
    X_train, y_train = np.array(signals[pos["train"]]), y[pos["train"]]
    X_val, y_val = np.array(signals[pos["val"]]), y[pos["val"]]
    X_test, y_test = np.array(signals[pos["test"]]), y[pos["test"]]

    print(f"device: {get_device()}")
    model = WaveformModel().fit(X_train, y_train, X_val=X_val, y_val=y_val)

    val_auroc = macro_auroc(y_val, model.score(X_val))
    test_auroc = macro_auroc(y_test, model.score(X_test))
    print(f"waveform: fold9 (val) macro AUROC = {val_auroc:.4f}")
    print(f"waveform: fold10 (test) macro AUROC = {test_auroc:.4f}")

    update_phase1_metrics(
        "waveform",
        {
            "fold9_val_macro_auroc": val_auroc,
            "fold10_test_macro_auroc": test_auroc,
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "model": "1D residual CNN (stem + 4 res blocks 64/128/128/256) -> 256d GAP -> 5 logits",
            "epochs_run": len(model.history),
            "device": str(model.device),
        },
    )


if __name__ == "__main__":
    _main()
