"""Shared paths, Telugu text normalisation and scoring helpers."""
import os
import re
import unicodedata
from pathlib import Path

import jiwer
import numpy as np

REPO = Path(__file__).resolve().parent
# Heavy artefacts (audio, checkpoints, exports) live under WT_WORK, which should be
# on the Linux filesystem in WSL2; results/ always lives in the repo.
WORK = Path(os.environ.get("WT_WORK", REPO)).expanduser()
DATA = WORK / "data"
CKPT = WORK / "ckpt"
EXPORT = WORK / "export"
RESULTS = REPO / "results"

BASE = "openai/whisper-small"
SR = 16_000
# Telugu costs several byte-level tokens per word: allow the full 448-token window
# minus the 4 prompt tokens, or long transcripts are cut off
MAX_NEW = 444

ZW = re.compile("[\u200c\u200d]")  # zero-width non-joiner / joiner


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s.replace("\\n", " "))  # OpenSLR 66 has literal "\n"
    s = ZW.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_te(s: str) -> str:
    """Strip punctuation (P) and symbols (S) only; keep vowel signs and virama (M)."""
    s = unicodedata.normalize("NFC", s)
    s = ZW.sub("", s)
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    return re.sub(r"\s+", " ", s).strip()


def norm_whisper(s: str) -> str:
    """Whisper's BasicTextNormalizer, which turns Telugu vowel signs into spaces."""
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    return BasicTextNormalizer()(s)


def utt_errors(refs, preds):
    """Per-utterance word errors and reference word counts."""
    errs, words = [], []
    for r, p in zip(refs, preds):
        if not r or not p:  # jiwer rejects empty strings: all insertions or all deletions
            errs.append(len((p or r).split()))
        else:
            o = jiwer.process_words(r, p)
            errs.append(o.substitutions + o.deletions + o.insertions)
        words.append(len(r.split()))
    return np.array(errs), np.array(words)


def wer_ci(refs, preds, n=1000, seed=0):
    """Corpus WER with a 95% bootstrap CI from resampling utterances."""
    errs, words = utt_errors(refs, preds)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(refs), (n, len(refs)))
    boots = errs[idx].sum(1) / words[idx].sum(1)
    return errs.sum() / words.sum(), np.percentile(boots, [2.5, 97.5])


def cer(refs, preds):
    pairs = [(r, p if p else " ") for r, p in zip(refs, preds) if r]
    return jiwer.cer([r for r, _ in pairs], [p for _, p in pairs])


def score(refs, preds, norm=norm_te):
    R, P = [norm(r) for r in refs], [norm(p) for p in preds]
    w, (lo, hi) = wer_ci(R, P)
    return dict(wer=w, wer_lo=lo, wer_hi=hi, cer=cer(R, P), n=len(R))
