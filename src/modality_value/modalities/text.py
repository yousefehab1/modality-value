"""Text arm: PTB-XL free-text ECG report -> structured superclass probabilities.

PTB-XL reports are predominantly German, mixing device-generated template
interpretations with cardiologist-written prose, so the base model needs
genuine multilingual coverage rather than an English-only prompting trick.

Three backends behind the same class (selectable via `backend=`): "lora"
(Qwen2.5-0.5B-Instruct + a LoRA adapter fine-tuned to emit strict JSON with
the five superclass booleans), "zero_shot" (same base model and prompt, no
adapter, the required un-tuned baseline), and "tfidf" (TF-IDF + one-vs-rest
LogisticRegression, a fallback used only if LoRA fails to converge).

`embed()` returns the mean-pooled last-hidden-state of the report from the
base model (no LoRA, no generation).
"""
from __future__ import annotations

import gc
import json
import logging
import re
from pathlib import Path

import numpy as np

from modality_value.config import MODALITY_COSTS, SEED, SUPERCLASSES
from modality_value.modalities import register
from modality_value.modalities.base import ModalityModel, get_device

logger = logging.getLogger(__name__)

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a cardiology assistant extracting structured labels from ECG "
    "reports. Reports may be written in German or English, and may be a "
    "short device-generated phrase rather than full prose. Always respond "
    "with STRICT JSON ONLY (no markdown, no extra commentary) with exactly "
    "these five boolean keys: NORM, MI, STTC, CD, HYP. These are PTB-XL "
    "diagnostic superclasses: NORM=normal ECG, MI=myocardial infarction, "
    "STTC=ST/T change, CD=conduction disturbance, HYP=hypertrophy. More "
    "than one key may be true."
)

JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

BOOL_PROB_TRUE = 0.95
BOOL_PROB_FALSE = 0.05

REPORT_CHAR_CAP = 800  # reports are short; this is generous, not a real truncation risk


def _as_report_list(X) -> list[str]:
    if hasattr(X, "tolist"):
        X = X.tolist()
    return ["" if r is None else str(r) for r in X]


def _build_messages(report: str) -> list[dict]:
    report = "" if report is None else str(report).strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Report: {report[:REPORT_CHAR_CAP]}"},
    ]


def _target_json(labels_row) -> str:
    obj = {sc: bool(labels_row[i]) for i, sc in enumerate(SUPERCLASSES)}
    return json.dumps(obj)


def parse_json_labels(text: str) -> dict[str, bool] | None:
    """Parse one model generation into {SUPERCLASS: bool}. Returns None on any failure.

    Deliberately strict: missing keys, non-JSON text, or non-boolean-ish
    values all count as a parse failure so the failure-rate number is honest
    rather than papering over partial/garbled generations.
    """
    if not text:
        return None
    match = JSON_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    out: dict[str, bool] = {}
    for sc in SUPERCLASSES:
        if sc not in obj:
            return None
        v = obj[sc]
        if isinstance(v, bool):
            out[sc] = v
        elif isinstance(v, (int, float)):
            out[sc] = bool(v)
        elif isinstance(v, str):
            vs = v.strip().lower()
            if vs not in ("true", "false", "1", "0", "yes", "no"):
                return None
            out[sc] = vs in ("true", "1", "yes")
        else:
            return None
    return out


def scores_from_generations(texts: list[str], prior: np.ndarray) -> tuple[np.ndarray, float]:
    """Parse a batch of raw model generations into (n, k) probability scores.

    Any generation that fails to parse falls back to `prior` (the training-set
    class prevalence, computed in `fit()`) row-wise. Pulled out as a standalone
    function so the fallback behaviour is testable without a loaded model --
    see tests/test_text.py.

    Returns (scores, parse_failure_rate).
    """
    n = len(texts)
    k = len(SUPERCLASSES)
    prior = np.asarray(prior, dtype=np.float32).reshape(k)
    scores = np.zeros((n, k), dtype=np.float32)
    n_failed = 0
    for i, text in enumerate(texts):
        parsed = parse_json_labels(text)
        if parsed is None:
            n_failed += 1
            scores[i] = prior
        else:
            scores[i] = [BOOL_PROB_TRUE if parsed[sc] else BOOL_PROB_FALSE for sc in SUPERCLASSES]
    failure_rate = (n_failed / n) if n else 0.0
    return scores, failure_rate


@register
class TextModel(ModalityModel):
    """LoRA-tuned (or zero-shot, or TF-IDF-fallback) text arm over PTB-XL reports."""

    name = "text"
    acquisition_cost_gbp = MODALITY_COSTS["text"].cost_gbp

    def __init__(
        self,
        backend: str = "lora",
        base_model_name: str = BASE_MODEL_NAME,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        num_epochs: int = 2,
        lr: float = 1e-4,
        batch_size: int = 8,
        micro_batch_size: int = 2,
        max_new_tokens: int = 48,
        gen_batch_size: int = 16,
        seed: int = SEED,
    ):
        if backend not in ("lora", "zero_shot", "tfidf"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        self.base_model_name = base_model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.num_epochs = num_epochs
        self.lr = lr
        # `batch_size` is the *effective* batch size the LR/convergence
        # behavior is tuned for. LoRA training runs it as `micro_batch_size`
        # micro-batches with gradient accumulation instead -- passing the
        # full batch_size straight into one forward pass spikes MPS memory
        # linearly with batch size (measured ~1.85GB/example at seq_len~300
        # for this vocab size), which OOMs a ~20GB unified-memory ceiling at
        # batch_size=8. Accumulation keeps the same effective batch size and
        # gradient statistics at a fraction of the peak memory.
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.max_new_tokens = max_new_tokens
        self.gen_batch_size = gen_batch_size
        self.seed = seed

        self._class_prior: np.ndarray | None = None
        self.last_parse_failure_rate_: float | None = None
        self.train_loss_history_: list[float] = []

        # lazy-loaded heavy objects -- never pickled directly (see save/load)
        self._tokenizer = None
        self._base_model = None
        self._peft_model = None
        self._tfidf = None  # (TfidfVectorizer, OneVsRestClassifier)

    # ---- device / model loading -----------------------------------------
    def _ensure_base_loaded(self):
        if self._base_model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.base_model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self._tokenizer = tok
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name, torch_dtype=torch.bfloat16
        )
        model.to(get_device())
        model.eval()
        self._base_model = model

    def _ensure_lora_attached(self):
        if self._peft_model is not None:
            return
        from peft import LoraConfig, get_peft_model

        self._ensure_base_loaded()
        cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            task_type="CAUSAL_LM",
        )
        self._peft_model = get_peft_model(self._base_model, cfg)

    def _free_heavy(self):
        """Drop model refs and reclaim MPS/CPU memory -- useful between backends."""
        self._peft_model = None
        self._base_model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    # ---- fit --------------------------------------------------------------
    def fit(self, X, y) -> "TextModel":
        reports = _as_report_list(X)
        y = np.asarray(y, dtype=np.float32)
        # Always compute the class prior -- it is the parse-failure fallback
        # for every backend, including zero-shot (which is never "fit" in
        # the gradient sense but still needs a prior from labelled data).
        self._class_prior = y.mean(axis=0)

        if self.backend == "zero_shot":
            self._ensure_base_loaded()
        elif self.backend == "tfidf":
            self._fit_tfidf(reports, y)
        elif self.backend == "lora":
            self._fit_lora(reports, y)
        else:
            raise ValueError(self.backend)
        return self

    def _fit_tfidf(self, reports: list[str], y: np.ndarray) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier

        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
        Xv = vec.fit_transform(reports)
        clf = OneVsRestClassifier(
            LogisticRegression(max_iter=2000, class_weight="balanced")
        )
        clf.fit(Xv, y)
        self._tfidf = (vec, clf)

    def _fit_lora(self, reports: list[str], y: np.ndarray) -> None:
        import torch
        from torch.utils.data import DataLoader, Dataset

        self._ensure_lora_attached()
        tok = self._tokenizer
        model = self._peft_model
        device = get_device()

        class ReportDataset(Dataset):
            def __init__(self, reports, y):
                self.reports = reports
                self.y = y

            def __len__(self):
                return len(self.reports)

            def __getitem__(self, i):
                messages = _build_messages(self.reports[i])
                prompt = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                target = _target_json(self.y[i]) + tok.eos_token
                return prompt, target

        def collate(batch):
            input_ids_list, labels_list = [], []
            for prompt, target in batch:
                prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
                target_ids = tok(target, add_special_tokens=False)["input_ids"]
                ids = prompt_ids + target_ids
                labels = [-100] * len(prompt_ids) + target_ids
                input_ids_list.append(ids)
                labels_list.append(labels)
            max_len = max(len(ids) for ids in input_ids_list)
            pad_id = tok.pad_token_id
            input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), max_len), dtype=torch.long)
            labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
            for i, (ids, lab) in enumerate(zip(input_ids_list, labels_list)):
                input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attn[i, : len(ids)] = 1
                labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
            return input_ids, attn, labels

        ds = ReportDataset(reports, y)
        gen = torch.Generator().manual_seed(self.seed)
        micro_bs = min(self.micro_batch_size, self.batch_size)
        accum_steps = max(1, self.batch_size // micro_bs)
        loader = DataLoader(
            ds, batch_size=micro_bs, shuffle=True, collate_fn=collate, generator=gen
        )

        model.train()
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=self.lr
        )

        self.train_loss_history_ = []
        for epoch in range(self.num_epochs):
            epoch_losses = []
            opt.zero_grad()
            for step, (input_ids, attn, labels) in enumerate(loader):
                input_ids = input_ids.to(device)
                attn = attn.to(device)
                labels = labels.to(device)
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                # Un-scaled loss for logging; scaled-down before backward so
                # `accum_steps` micro-batches sum to one effective-batch-size
                # gradient (same training dynamics as one big batch, at
                # micro_bs's peak memory instead of batch_size's).
                loss = out.loss
                (loss / accum_steps).backward()
                epoch_losses.append(loss.item())
                if (step + 1) % accum_steps == 0:
                    opt.step()
                    opt.zero_grad()
                if step % 50 == 0:
                    logger.info(
                        "LoRA epoch %d step %d loss %.4f", epoch + 1, step, loss.item()
                    )
            if (step + 1) % accum_steps != 0:
                opt.step()
                opt.zero_grad()
            mean_loss = float(np.mean(epoch_losses))
            self.train_loss_history_.append(mean_loss)
            logger.info("LoRA epoch %d/%d mean loss %.4f", epoch + 1, self.num_epochs, mean_loss)
        model.eval()

    # ---- embed --------------------------------------------------------------
    def embed(self, X) -> np.ndarray:
        """Mean-pooled last-hidden-state of the report from the *base* model
        (no LoRA, no generation) -- an encoder-style pooling pass."""
        import torch

        reports = _as_report_list(X)
        self._ensure_base_loaded()
        tok = self._tokenizer
        model = self._base_model
        device = get_device()

        old_padding_side = tok.padding_side
        tok.padding_side = "right"  # mean-pool with a mask, side doesn't matter, but be explicit
        embeddings = []
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(reports), batch_size):
                batch = [r[:REPORT_CHAR_CAP] for r in reports[i : i + batch_size]]
                enc = tok(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=256
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc, output_hidden_states=True)
                hidden = out.hidden_states[-1]  # (b, t, d)
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                pooled = summed / counts
                embeddings.append(pooled.float().cpu().numpy())
        tok.padding_side = old_padding_side
        return np.concatenate(embeddings, axis=0)

    # ---- score --------------------------------------------------------------
    def score(self, X) -> np.ndarray:
        reports = _as_report_list(X)
        if self.backend == "tfidf":
            return self._score_tfidf(reports)
        return self._score_generative(reports)

    def _score_tfidf(self, reports: list[str]) -> np.ndarray:
        vec, clf = self._tfidf
        Xv = vec.transform(reports)
        probs = clf.predict_proba(Xv)
        probs = np.asarray(probs, dtype=np.float32)
        self.last_parse_failure_rate_ = 0.0  # no generation step, nothing to parse
        return probs

    def _score_generative(self, reports: list[str]) -> np.ndarray:
        import torch

        model = self._peft_model if self.backend == "lora" else None
        if model is None:
            self._ensure_base_loaded()
            model = self._base_model
        tok = self._tokenizer
        device = get_device()
        model.eval()

        old_padding_side = tok.padding_side
        tok.padding_side = "left"  # required for correct batched causal-LM generation

        prior = (
            self._class_prior
            if self._class_prior is not None
            else np.full(len(SUPERCLASSES), 0.5, dtype=np.float32)
        )

        all_texts: list[str] = []
        with torch.no_grad():
            for start in range(0, len(reports), self.gen_batch_size):
                batch_reports = reports[start : start + self.gen_batch_size]
                prompts = [
                    tok.apply_chat_template(
                        _build_messages(r), tokenize=False, add_generation_prompt=True
                    )
                    for r in batch_reports
                ]
                enc = tok(
                    prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(device)
                gen = model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
                gen_only = gen[:, enc["input_ids"].shape[1] :]
                texts = tok.batch_decode(gen_only, skip_special_tokens=True)
                all_texts.extend(texts)

        tok.padding_side = old_padding_side
        scores, failure_rate = scores_from_generations(all_texts, prior)
        self.last_parse_failure_rate_ = failure_rate
        return scores

    # ---- save / load --------------------------------------------------------
    # Overridden vs. the ABC's default pickle-the-whole-object: torch models
    # and tokenizers are large and have their own (de)serialisation; we save
    # only the lightweight config/metadata plus, per backend, the LoRA
    # adapter (via peft) or the TF-IDF+LogisticRegression pickle.
    def save(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "backend": self.backend,
            "base_model_name": self.base_model_name,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "max_new_tokens": self.max_new_tokens,
            "gen_batch_size": self.gen_batch_size,
            "seed": self.seed,
            "class_prior": self._class_prior.tolist() if self._class_prior is not None else None,
            "last_parse_failure_rate": self.last_parse_failure_rate_,
            "train_loss_history": self.train_loss_history_,
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f)
        if self.backend == "lora" and self._peft_model is not None:
            self._peft_model.save_pretrained(str(path / "adapter"))
        if self.backend == "tfidf" and self._tfidf is not None:
            import pickle

            with open(path / "tfidf.pkl", "wb") as f:
                pickle.dump(self._tfidf, f)

    @classmethod
    def load(cls, path) -> "TextModel":
        path = Path(path)
        with open(path / "meta.json") as f:
            meta = json.load(f)
        obj = cls(
            backend=meta["backend"],
            base_model_name=meta["base_model_name"],
            lora_r=meta["lora_r"],
            lora_alpha=meta["lora_alpha"],
            lora_dropout=meta["lora_dropout"],
            num_epochs=meta["num_epochs"],
            lr=meta["lr"],
            batch_size=meta["batch_size"],
            max_new_tokens=meta["max_new_tokens"],
            gen_batch_size=meta["gen_batch_size"],
            seed=meta["seed"],
        )
        obj._class_prior = (
            np.array(meta["class_prior"], dtype=np.float32)
            if meta["class_prior"] is not None
            else None
        )
        obj.last_parse_failure_rate_ = meta["last_parse_failure_rate"]
        obj.train_loss_history_ = meta["train_loss_history"]
        if obj.backend == "lora" and (path / "adapter").exists():
            from peft import PeftModel

            obj._ensure_base_loaded()
            obj._peft_model = PeftModel.from_pretrained(obj._base_model, str(path / "adapter"))
        if obj.backend == "tfidf" and (path / "tfidf.pkl").exists():
            import pickle

            with open(path / "tfidf.pkl", "rb") as f:
                obj._tfidf = pickle.load(f)
        return obj


# ---- CLI entrypoint: `make text` -> python -m modality_value.modalities.text ---


def _inspect_reports(df, n: int = 20) -> None:
    print(f"\n--- Sample of {n} raw report strings (language/structure check) ---")
    sample = df["report"].dropna().sample(n=min(n, len(df)), random_state=SEED)
    for i, r in sample.items():
        print(f"  [{i}] {r!r}"[:220])
    print()


def main() -> None:
    import argparse
    import time

    from modality_value.eval.metrics import macro_auroc
    from modality_value.eval.report import PHASE1_METRICS_PATH, merge_json_atomic
    from modality_value.io.ptbxl import labels_matrix, load_metadata, split_indices

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-examples", type=int, default=4000,
                         help="Subsample of train folds used for LoRA fine-tuning "
                              "(compute/time budget choice, documented here and in the README; "
                              "evaluation is always on the full requested test set).")
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--skip-lora", action="store_true",
                         help="Skip LoRA fine-tuning and go straight to the TF-IDF fallback "
                              "(for fast iteration / debugging).")
    args = parser.parse_args()

    df = load_metadata()
    _inspect_reports(df)

    splits = split_indices(df)
    y_all = labels_matrix(df)
    pos = {idx: i for i, idx in enumerate(df.index)}

    rng = np.random.default_rng(SEED)

    train_idx = splits["train"]
    test_idx = splits["test"]

    if args.max_train_examples and len(train_idx) > args.max_train_examples:
        train_idx = pd_index_subsample(train_idx, args.max_train_examples, rng)
    if args.max_test_examples and len(test_idx) > args.max_test_examples:
        test_idx = pd_index_subsample(test_idx, args.max_test_examples, rng)

    reports_train = df.loc[train_idx, "report"].fillna("").astype(str).tolist()
    y_train = y_all[[pos[i] for i in train_idx]]
    reports_test = df.loc[test_idx, "report"].fillna("").astype(str).tolist()
    y_test = y_all[[pos[i] for i in test_idx]]

    print(f"Train (fit) examples: {len(reports_train)} (subsampled from "
          f"{len(splits['train'])} available train-fold records)")
    print(f"Test (fold-10) examples: {len(reports_test)}")

    metrics: dict = {}
    stop_line_hit = False
    stop_line_reason = ""

    # ---- Zero-shot baseline ----
    print("\n=== Zero-shot baseline (Qwen2.5-0.5B-Instruct, no LoRA) ===")
    t0 = time.time()
    zero_shot = TextModel(backend="zero_shot")
    zero_shot.fit(reports_train, y_train)
    zs_scores = zero_shot.score(reports_test)
    zs_auroc = macro_auroc(y_test, zs_scores)
    zs_parse_fail = zero_shot.last_parse_failure_rate_
    print(f"zero-shot macro AUROC={zs_auroc:.4f} parse_failure_rate={zs_parse_fail:.4f} "
          f"({time.time() - t0:.0f}s)")
    metrics["text_zero_shot"] = {
        "backend": "zero_shot",
        "macro_auroc_fold10": zs_auroc,
        "json_parse_failure_rate": zs_parse_fail,
        "n_test": len(reports_test),
        "n_train_fit": len(reports_train),
    }
    zero_shot._free_heavy()
    del zero_shot

    # ---- LoRA fine-tuned ----
    if not args.skip_lora:
        print("\n=== LoRA fine-tuning (r=16, alpha=32, dropout=0.05) ===")
        t0 = time.time()
        lora_model = TextModel(backend="lora", num_epochs=args.epochs)
        lora_model.fit(reports_train, y_train)
        losses = lora_model.train_loss_history_
        print(f"LoRA training loss per epoch: {losses} ({time.time() - t0:.0f}s)")

        converged = len(losses) >= 2 and losses[-1] < losses[0] * 0.9
        if not converged:
            stop_line_hit = True
            stop_line_reason = (
                f"LoRA training loss did not clearly decrease across epochs "
                f"(loss history: {losses}); falling back to TF-IDF per the stop-line."
            )
            print(f"STOP-LINE HIT: {stop_line_reason}")
        else:
            t0 = time.time()
            lora_scores = lora_model.score(reports_test)
            lora_auroc = macro_auroc(y_test, lora_scores)
            lora_parse_fail = lora_model.last_parse_failure_rate_
            print(f"LoRA fine-tuned macro AUROC={lora_auroc:.4f} "
                  f"parse_failure_rate={lora_parse_fail:.4f} ({time.time() - t0:.0f}s)")
            metrics["text_lora"] = {
                "backend": "lora",
                "macro_auroc_fold10": lora_auroc,
                "json_parse_failure_rate": lora_parse_fail,
                "n_test": len(reports_test),
                "n_train_fit": len(reports_train),
                "train_loss_history": losses,
                "beats_zero_shot": bool(lora_auroc > zs_auroc),
            }
        lora_model._free_heavy()
        del lora_model
    else:
        stop_line_hit = True
        stop_line_reason = "--skip-lora passed explicitly."

    # ---- TF-IDF fallback (only if stop-line hit) ----
    if stop_line_hit:
        print(f"\n=== TF-IDF + LogisticRegression fallback ({stop_line_reason}) ===")
        tfidf_model = TextModel(backend="tfidf")
        tfidf_model.fit(reports_train, y_train)
        tfidf_scores = tfidf_model.score(reports_test)
        tfidf_auroc = macro_auroc(y_test, tfidf_scores)
        print(f"TF-IDF fallback macro AUROC={tfidf_auroc:.4f}")
        metrics["text_tfidf_fallback"] = {
            "backend": "tfidf",
            "macro_auroc_fold10": tfidf_auroc,
            "json_parse_failure_rate": 0.0,
            "n_test": len(reports_test),
            "n_train_fit": len(reports_train),
            "stop_line_reason": stop_line_reason,
            "beats_zero_shot": bool(tfidf_auroc > zs_auroc),
        }

    metrics["text_stop_line_hit"] = stop_line_hit
    if stop_line_hit:
        metrics["text_stop_line_reason"] = stop_line_reason

    # ---- write results: merge into phase1_metrics.json ----
    merge_json_atomic(PHASE1_METRICS_PATH, metrics)
    print(f"\nMerged text-arm metrics into {PHASE1_METRICS_PATH}")


def pd_index_subsample(idx, n, rng):
    import pandas as pd

    chosen = rng.choice(idx.to_numpy(), size=n, replace=False)
    return pd.Index(chosen)


if __name__ == "__main__":
    main()
