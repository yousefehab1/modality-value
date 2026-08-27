# modality-value

> A deployed multimodal risk engine that quantifies what each additional data modality is worth, in discrimination gain per unit of acquisition cost.

Every additional data source a health system buys — an ECG, a free-text report, a genomic panel — has both a discrimination gain and an acquisition cost. This prototype builds a modality-agnostic risk engine, spanning waveform, text, tabular, and molecular data, and asks: what is each modality actually worth, in AUROC (or C-index) gained per £100 spent? Both directions matter commercially: **what should I buy next**, and **what could I stop paying for**.

## Honest limitation, stated up front

No openly available cohort pairs raw biosignals with multi-omics **in the same patients**. This prototype therefore demonstrates the *architecture* — one interface, `fit / embed / score`, that any modality can plug into — across **two separate cohorts** that share that interface, rather than faking a joint dataset:

1. **PTB-XL** (ECG waveforms, free-text reports, tabular demographics) — 12-lead ECG diagnostic classification.
2. **TCGA** (molecular signature scores, clinical covariates) — survival analysis via the existing thesis pipeline.

This is exactly the data-access barrier the field runs into in practice. Naming it here is deliberate.

## Architecture

Every modality — a 1D CNN over waveforms, a LoRA-tuned language model over reports, LightGBM over tabular fields, a Cox model over gene-set scores — implements the same interface (`src/modality_value/modalities/base.py`):

```python
class ModalityModel(ABC):
    def fit(self, X, y) -> "ModalityModel": ...
    def embed(self, X) -> np.ndarray: ...   # (n, d) latent representation
    def score(self, X) -> np.ndarray: ...   # (n, k) risk / class probabilities
```

This is what lets a late-fusion layer and a value-analysis function treat wildly different models identically, and it's the thing to point at first when explaining this repo.

(Architecture diagram: TODO once Phase 5.)

## Results — Cohort 1: PTB-XL (ECG)

Macro AUROC across 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP), fold-10 test set. Bootstrap 95% CIs are paired, 1,000 resamples.

| Modality set | Macro AUROC | Δ vs. previous | 95% CI on Δ | Acquisition cost (£/patient) | Δ AUROC per £100 |
|---|---|---|---|---|---|
| _(not yet run)_ | | | | | |

Leave-one-modality-out marginal gains:

| Modality removed | Macro AUROC (full − modality) | Δ | 95% CI on Δ |
|---|---|---|---|
| _(not yet run)_ | | | |

## Results — Cohort 2: TCGA (molecular / survival)

Harrell's C-index, TCGA-COAD, n=455 patients with overall-survival follow-up (event rate 22.4%). Same `fusion/value.py` code path as Cohort 1, parametrised on metric (`c_index_metric` instead of `macro_auroc_metric`) — not a separate implementation. 5-fold `KFold` out-of-fold cross-fitting (fixed seed 42; TCGA has no published `strat_fold`-style split to reuse, unlike PTB-XL, so a plain KFold is the honest choice here — see `modalities/molecular.py::compute_oof_scores`). Bootstrap 95% CIs are paired, 1,000 resamples. See [honest limitation](#honest-limitation-stated-up-front): this is a **different cohort** from PTB-XL, not the same patients with an added modality.

Reproduce with `make molecular` (requires `data/tcga/molecular_scores.csv`, produced once by `Rscript scripts/export_tcga_molecular.R` from the existing thesis R pipeline — see that script's header).

| Modality set | C-index | Δ vs. previous | 95% CI on Δ | Acquisition cost (£/patient) | ΔC-index per £100 |
|---|---|---|---|---|---|
| clinical | 0.702 | 0.702 | — | 1 | 70.19 |
| clinical+molecular | 0.711 | +0.0087 | [-0.0004, 0.0185] | 300 | 0.0029 |

Leave-one-modality-out marginal gains (both modalities included = C-index 0.711):

| Modality removed | C-index (full − modality) | Δ | 95% CI on Δ |
|---|---|---|---|
| clinical | 0.558 | -0.153 | [-0.232, -0.077] |
| molecular | 0.702 | -0.0087 | [-0.0185, 0.0004] |

**Reading this honestly:** clinical covariates (age, sex, stage) carry almost all of the discrimination in this cohort — dropping clinical costs 0.15 of C-index, a large and clearly significant loss. Adding the £300/patient ssGSEA molecular panel on top of clinical covariates buys a small, borderline gain (its 95% CI on Δ includes numbers at/near zero), i.e. **not yet a clearly worthwhile marginal purchase at this sample size** — the kind of answer this table exists to surface rather than obscure. This says nothing about whether molecular data adds value in general; it reflects this specific signature panel, this specific cohort size, and stage/age already being strong prognostic covariates in colorectal cancer.

## Acquisition cost sources

All cost figures are estimates unless stated otherwise; see `src/modality_value/config.py::MODALITY_COSTS` for the single source of truth and per-modality citation/rationale. An interviewer will ask — a defensible order of magnitude with a stated source beats a precise fabricated figure.

## Data

- **PTB-XL**: PhysioNet, [CC BY 4.0](https://physionet.org/content/ptb-xl/1.0.3/), open access, no credentialing required. 21,799 records / 18,869 patients. Citation: Wagner et al., "PTB-XL, a large publicly available electrocardiography dataset," Scientific Data, 2020.
- **TCGA**: molecular scores exported once from the existing thesis R pipeline (`core/scoring.R`, `core/signatures.R`, `modules/crc_survival.R`, `core/clinical.R`) into `data/tcga/molecular_scores.csv`. No thesis code was modified.

## Reproducing

```bash
uv venv --python 3.11
uv pip install -e .
make test        # interface conformance, split integrity, no-leakage checks
make data        # download + cache PTB-XL; asserts 21,799 records
make tabular waveform text
make fusion
make molecular
make serve       # local demo on :8080
```

## Deployment

TODO (Phase 5): GCP Cloud Run, `service/Dockerfile`, `gcloud run deploy --source`.

## What was cut

TODO: updated honestly as phases land or are cut per the stop-lines below. Nothing here should claim a capability that doesn't exist in the code.

## Next steps

TODO (Phase 5).
