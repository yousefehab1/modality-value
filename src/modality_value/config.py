"""Paths, seeds, split definition, and the acquisition-cost table.

Every number in results/modality_value.csv must trace back to something
reproducible from here plus a Makefile target. Cost figures are estimates
where noted -- see README for sources.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = DATA_DIR / "cache"

PTBXL_DIR = DATA_DIR / "ptbxl"
PTBXL_DATABASE_CSV = PTBXL_DIR / "ptbxl_database.csv"
PTBXL_SCP_STATEMENTS_CSV = PTBXL_DIR / "scp_statements.csv"
PTBXL_SIGNALS_MEMMAP = CACHE_DIR / "signals_100.npy"

TCGA_DIR = DATA_DIR / "tcga"
TCGA_MOLECULAR_SCORES_CSV = TCGA_DIR / "molecular_scores.csv"

# PTB-XL's own recommended split: never re-split, never mix folds.
TRAIN_FOLDS = tuple(range(1, 9))  # 1-8
VAL_FOLD = 9
TEST_FOLD = 10

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

SAMPLING_RATE_HZ = 100
SIGNAL_LENGTH = 1000  # 10s at 100Hz
NUM_LEADS = 12


@dataclass(frozen=True)
class ModalityCost:
    """Per-patient acquisition cost in GBP. Flag every estimate as an estimate."""

    name: str
    cost_gbp: float
    is_estimate: bool
    source: str


# Populated in Phase 3 (PTB-XL arm) and Phase 4 (TCGA arm). Kept centrally so
# fusion/value.py never hardcodes a number the README doesn't also cite.
MODALITY_COSTS: dict[str, ModalityCost] = {
    "tabular": ModalityCost(
        name="tabular",
        cost_gbp=1.0,
        is_estimate=True,
        source="Nominal cost of recording routine demographics (age/sex/height/weight) "
        "already captured at intake; effectively the marginal cost of data entry.",
    ),
    "waveform": ModalityCost(
        name="waveform",
        cost_gbp=25.0,
        is_estimate=True,
        source="Approximate UK cost of a standard 12-lead resting ECG "
        "(NHS reference-cost-style estimate for outpatient ECG acquisition + technician time).",
    ),
    "text": ModalityCost(
        name="text",
        cost_gbp=15.0,
        is_estimate=True,
        source="Estimated marginal cost of cardiologist reporting time for a single ECG report, "
        "prorated (reports are typically produced alongside the waveform, but reporting time "
        "is a distinct line cost from acquisition).",
    ),
    "molecular": ModalityCost(
        name="molecular",
        cost_gbp=300.0,
        is_estimate=True,
        source="Approximate list price for a bulk RNA-seq assay per sample "
        "(commercial sequencing-core estimate, order of magnitude only).",
    ),
    "clinical": ModalityCost(
        name="clinical",
        cost_gbp=1.0,
        is_estimate=True,
        source="Nominal cost of recording routine clinical covariates (age/sex/stage) "
        "already captured at intake.",
    ),
}
