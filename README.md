# modality-value

> A deployed multimodal risk engine that quantifies what each additional data modality is worth, in discrimination gain per unit of acquisition cost.

Every additional data source a health system buys (an ECG, a free-text report, a genomic panel) has both a discrimination gain and an acquisition cost. This prototype builds a modality-agnostic risk engine, spanning waveform, text, tabular, and molecular data, and asks: what is each modality actually worth, in AUROC (or C-index) gained per £100 spent? Both directions matter commercially: **what should I buy next**, and **what could I stop paying for**.

## Honest limitation, stated up front

No openly available cohort pairs raw biosignals with multi-omics **in the same patients**. This prototype therefore demonstrates the *architecture*: one interface, `fit / embed / score`, that any modality can plug into, across **two separate cohorts** that share that interface, rather than faking a joint dataset:

1. **PTB-XL** (ECG waveforms, free-text reports, tabular demographics): 12-lead ECG diagnostic classification.
2. **TCGA** (molecular signature scores, clinical covariates): survival analysis via the existing thesis pipeline.

This is exactly the data-access barrier the field runs into in practice. Naming it here is deliberate.

## Architecture

Every modality (a 1D CNN over waveforms, a LoRA-tuned language model over reports, LightGBM over tabular fields, a Cox model over gene-set scores) implements the same interface (`src/modality_value/modalities/base.py`):

```python
class ModalityModel(ABC):
    def fit(self, X, y) -> "ModalityModel": ...
    def embed(self, X) -> np.ndarray: ...   # (n, d) latent representation
    def score(self, X) -> np.ndarray: ...   # (n, k) risk / class probabilities
```

This is what lets a late-fusion layer and a value-analysis function treat wildly different models identically, and it's the thing to point at first when explaining this repo.

```
                    ┌──────────────┐
   tabular (£1)  →  │ TabularModel │ ──┐
                    └──────────────┘   │
                    ┌──────────────┐   │      ┌────────────────────┐      ┌─────────────────────┐
   waveform (£25) → │ WaveformModel│ ──┼──→   │ fuse_fn (stacking   │ →    │ compute_value_table /│
                    └──────────────┘   │      │ classifier, fit on  │      │ compute_leave_one_   │
                    ┌──────────────┐   │      │ val-fold scores)    │      │ out_table            │
   text (£15)    →  │  TextModel   │ ──┘      └────────────────────┘      └─────────────────────┘
                    │ (LoRA/Qwen)  │                    ↓                            ↓
                    └──────────────┘          fused test-fold scores      value.csv / loo.csv / fig.png
```

The TCGA arm (`modalities/molecular.py`) reuses the exact same two functions shown above, with `c_index_metric` and `CoxPHFitter` in place of `macro_auroc_metric` and logistic regression. `service/app.py` runs the same `TabularModel` + `WaveformModel` + `fuse_fn` path live behind a FastAPI endpoint (text is excluded there, see "What was cut").

## Cohort 1 results: PTB-XL (ECG)

Macro AUROC across 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP), fold-10 test set. Bootstrap 95% CIs are paired, 1,000 resamples.

Reproduce with `make tabular waveform text fusion` (tabular and waveform train on the full fold; the text/LoRA sample-size caveat is in "What was cut" below).

| Modality set | Macro AUROC | Δ vs. previous | 95% CI on Δ | Acquisition cost (£/patient) | Δ AUROC per £100 |
|---|---|---|---|---|---|
| tabular | 0.676 | 0.676 | n/a | 1 | 67.60 |
| tabular+text | 0.894 | +0.2178 | [0.2044, 0.2319] | 15 | 1.45 |
| tabular+text+waveform | 0.945 | +0.0511 | [0.0454, 0.0578] | 25 | 0.20 |

Leave-one-modality-out marginal gains (all three modalities included = macro AUROC 0.945):

| Modality removed | Macro AUROC (full − modality) | Δ | 95% CI on Δ |
|---|---|---|---|
| tabular | 0.944 | -0.0010 | [-0.0019, -0.0003] |
| waveform | 0.894 | -0.0511 | [-0.0578, -0.0454] |
| text | 0.908 | -0.0370 | [-0.0433, -0.0309] |

**Reading this honestly:**
- Free-text cardiologist reports carry most of the diagnostic signal here: adding text to tabular demographics is a huge, clearly significant jump (+0.218 AUROC), because the reports frequently paraphrase the diagnosis itself.
- The waveform CNN then adds a further significant gain on top (+0.051), at a higher acquisition cost (£25) than text (£15), so text is the better marginal buy of the two by a wide margin (1.45 vs. 0.20 AUROC per £100).
- The leave-one-out table shows an asymmetry worth flagging: **the cost-ordered buying sequence and the "what hurts most to lose" ranking disagree.** Tabular is bought first because it's cheapest, but once text and waveform are present it is nearly redundant (dropping it costs 0.001 AUROC); waveform, added last, turns out to be the single modality most costly to remove from the full set (-0.051), ahead of text (-0.037).
- A "what could I stop paying for" query and a "what should I buy next" query are genuinely different questions with different answers here.
- Every number above that involves text carries the 4,000-example training-subsample caveat; see "What was cut".

## Cohort 2 results: TCGA (molecular / survival)

Harrell's C-index, TCGA-COAD, n=455 patients with overall-survival follow-up (event rate 22.4%). Same `fusion/value.py` code path as Cohort 1, parametrised on metric (`c_index_metric` instead of `macro_auroc_metric`), not a separate implementation. 5-fold `KFold` out-of-fold cross-fitting (fixed seed 42; TCGA has no published `strat_fold`-style split to reuse, unlike PTB-XL, so a plain KFold is the honest choice here, see `modalities/molecular.py::compute_oof_scores`). Bootstrap 95% CIs are paired, 1,000 resamples. See [honest limitation](#honest-limitation-stated-up-front): this is a **different cohort** from PTB-XL, not the same patients with an added modality.

Reproduce with `make molecular` (requires `data/tcga/molecular_scores.csv`, produced once by `Rscript scripts/export_tcga_molecular.R` from the existing thesis R pipeline, see that script's header).

| Modality set | C-index | Δ vs. previous | 95% CI on Δ | Acquisition cost (£/patient) | ΔC-index per £100 |
|---|---|---|---|---|---|
| clinical | 0.702 | 0.702 | n/a | 1 | 70.19 |
| clinical+molecular | 0.711 | +0.0087 | [-0.0004, 0.0185] | 300 | 0.0029 |

Leave-one-modality-out marginal gains (both modalities included = C-index 0.711):

| Modality removed | C-index (full − modality) | Δ | 95% CI on Δ |
|---|---|---|---|
| clinical | 0.558 | -0.153 | [-0.232, -0.077] |
| molecular | 0.702 | -0.0087 | [-0.0185, 0.0004] |

**Reading this honestly:**
- Clinical covariates (age, sex, stage) carry almost all of the discrimination in this cohort: dropping clinical costs 0.15 of C-index, a large and clearly significant loss.
- Adding the £300/patient ssGSEA molecular panel on top of clinical covariates buys a small, borderline gain (its 95% CI on Δ includes numbers at/near zero), i.e. **not yet a clearly worthwhile marginal purchase at this sample size**. That is the kind of answer this table exists to surface, not obscure.
- This says nothing about whether molecular data adds value in general; it reflects this specific signature panel, this specific cohort size, and stage/age already being strong prognostic covariates in colorectal cancer.

## Acquisition cost sources

All cost figures are estimates unless stated otherwise; see `src/modality_value/config.py::MODALITY_COSTS` for the single source of truth and per-modality citation/rationale. An interviewer will ask: a defensible order of magnitude with a stated source beats a precise fabricated figure.

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

`service/app.py` is a real FastAPI app: on startup it fits `TabularModel` + `WaveformModel` on the PTB-XL train fold and a stacking `fuse_fn` on the val fold (same code path as `fusion/value.py`'s PTB-XL arm), then serves `GET /health`, `GET /patients` (sample held-out test-fold ids), and `GET /score/{ecg_id}` (fused + per-modality risk scores, plus the true label, for that test-fold patient).

Locally:

```bash
make serve   # fits on startup (~30-60s), then serves on :8080
curl localhost:8080/patients
curl localhost:8080/score/<ecg_id from the list above>
```

Containerized, for Cloud Run:

```bash
docker build -t modality-value-service -f service/Dockerfile .
docker run -p 8080:8080 -v "$(pwd)/data:/app/data" modality-value-service   # local sanity check first

gcloud run deploy modality-value-service \
  --source . \
  --region <your-region> \
  --allow-unauthenticated \
  --memory 4Gi
```

`--source .` builds `service/Dockerfile` via Cloud Build. PTB-XL isn't baked into the image (see the Dockerfile's header comment); a real deploy needs the dataset reachable at `/app/data` inside the container, e.g. a GCS FUSE volume mount or an init step that runs `make data`, neither of which is wired up here. This command has not been run against a live GCP project as part of this repo; it's documented, not executed, since that requires the deployer's own GCP credentials and billing.

## What was cut

- **Text modality is excluded from the live service.** LoRA fine-tuning takes ~30 minutes (see `modalities/text.py`); fine for an offline `make fusion` run, not for something refit on every `make serve` restart. Serving it for real needs the fine-tuned adapter persisted to disk and loaded at startup instead of retrained; see Next steps.
- **Text arm trains on a 4,000-example subsample**, not the full 17,418-record train fold, purely for LoRA compute budget on a single Apple Silicon machine. Every PTB-XL number above that includes text should be read as a lower bound on what more training data could buy.
- **No true joint multimodal cohort.** PTB-XL and TCGA are separate cohorts sharing an interface, not the same patients with an added modality; stated up front rather than glossed over.
- **TCGA uses a plain 5-fold `KFold`**, not a published stratified split, because none exists for this cohort (unlike PTB-XL's `strat_fold`), the honest option given in `modalities/molecular.py::compute_oof_scores`.
- **No model persistence layer.** Every `make <target>` and the service both refit from scratch; nothing is pickled/checkpointed to disk. Fine for a research prototype at this scale, but the first thing a real deployment would need (see Next steps).
- **No auth, rate limiting, or PHI handling** in `service/app.py`: it returns ecg_ids and labels straight from a public research dataset for demo purposes; treat it as a local/portfolio demo, not a template for handling real patient data.

## Next steps

1. **Persist trained models** (tabular: `joblib`; waveform: `torch.save` state dict; text: `peft`'s `save_pretrained`/`from_pretrained` for the LoRA adapter) so the service loads instantly instead of refitting, and so text can be added back to the live demo without a 30-minute cold start.
2. **Accept raw input** at `/score` (tabular fields + a 12-lead signal array) instead of only looking up existing test-fold `ecg_id`s, so the demo can score a genuinely new record.
3. **Wire up `make data` (or a GCS FUSE mount) into the Cloud Run deploy** so the container has PTB-XL available without a manual volume mount, and actually run `gcloud run deploy` against a real project to confirm the documented command above works end to end.
4. **Re-run the TCGA value table at a larger n** (or with a second molecular signature panel): the molecular Δ's 95% CI currently straddles zero (see "Reading this honestly" under Cohort 2 results), not yet a confident answer to whether the £300/patient panel is worth it.
