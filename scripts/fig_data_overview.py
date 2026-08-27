#!/usr/bin/env python
"""Generate results/fig_data_overview.png: label prevalence, age/sex
distribution, and report-length histogram, straight off ptbxl_database.csv.

Phase 0 acceptance also lives here in spirit (though the actual --summary
assertion is in io/ptbxl.py): this script is the "one figure of data shape"
requirement, wired to `make data`.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modality_value.config import RESULTS_DIR, SUPERCLASSES
from modality_value.io.ptbxl import load_metadata


def main() -> None:
    df = load_metadata()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Label prevalence (multi-label, not mutually exclusive).
    prevalence = [df[sc].sum() for sc in SUPERCLASSES]
    axes[0].bar(SUPERCLASSES, prevalence, color="#4C72B0")
    axes[0].set_title("Label prevalence (5 diagnostic superclasses)")
    axes[0].set_ylabel("n records")
    for i, v in enumerate(prevalence):
        axes[0].text(i, v, str(int(v)), ha="center", va="bottom", fontsize=8)

    # 2. Age distribution split by sex (0/1 in ptbxl_database.csv).
    ages_by_sex = [df.loc[df["sex"] == s, "age"].dropna() for s in sorted(df["sex"].dropna().unique())]
    labels = [f"sex={int(s)}" for s in sorted(df["sex"].dropna().unique())]
    axes[1].hist(ages_by_sex, bins=20, stacked=True, label=labels, color=["#4C72B0", "#DD8452"][: len(labels)])
    axes[1].set_title("Age distribution by sex")
    axes[1].set_xlabel("age (years)")
    axes[1].set_ylabel("n records")
    axes[1].legend()

    # 3. Report-length histogram (the free-text arm's raw material; German-
    # dominant per the build plan -- length is measured in characters).
    report_lengths = df["report"].dropna().astype(str).str.len()
    axes[2].hist(report_lengths, bins=40, color="#55A868")
    axes[2].set_title("Report length (characters)")
    axes[2].set_xlabel("len(report)")
    axes[2].set_ylabel("n records")
    axes[2].axvline(report_lengths.median(), color="black", linestyle="--", linewidth=1, label=f"median={report_lengths.median():.0f}")
    axes[2].legend()

    fig.suptitle(f"PTB-XL data overview (n={len(df)} records, {df['patient_id'].nunique()} patients)")
    fig.tight_layout()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fig_data_overview.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
