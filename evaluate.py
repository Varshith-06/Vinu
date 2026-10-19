"""Transcribe a test split with Transformers and score WER/CER with bootstrap CIs.

Writes results/preds/<name>_<split>.csv (one row per clip) and appends to results/wer.csv.
"""
import argparse
import csv
import time

import pandas as pd
import torch
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from common import BASE, DATA, MAX_NEW, RESULTS, SR, norm_te, norm_whisper, score


def load_model(adapter=None, base=BASE):
    model = WhisperForConditionalGeneration.from_pretrained(base)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model.half().cuda().eval()


def transcribe(model, proc, ds, batch=16, num_beams=1):
    preds, t0 = [], time.perf_counter()
    for i in range(0, len(ds), batch):
        b = ds[i:i + batch]
        feats = proc.feature_extractor([a["array"] for a in b["audio"]], sampling_rate=SR,
                                       return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(input_features=feats.cuda().half(), language="te",
                                 task="transcribe", max_new_tokens=MAX_NEW, num_beams=num_beams)
        preds += [p.strip() for p in proc.batch_decode(ids, skip_special_tokens=True)]
        print(f"\r{len(preds)}/{len(ds)}", end="", flush=True)
    print(f"  {time.perf_counter() - t0:.0f}s")
    return preds


def record(name, split, df):
    """Score a predictions frame both ways and append the rows to results/wer.csv."""
    path = RESULTS / "wer.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "split", "normaliser", "wer", "wer_lo", "wer_hi", "cer", "n"])
        for nname, norm in [("norm_te", norm_te), ("whisper_basic", norm_whisper)]:
            s = score(df["ref"].fillna("").tolist(), df["pred"].fillna("").tolist(), norm)
            w.writerow([name, split, nname] + [round(s[k], 4) for k in
                                                ("wer", "wer_lo", "wer_hi", "cer")] + [s["n"]])
            print(f"{name} {split} [{nname}] WER {s['wer']:.2%} "
                  f"({s['wer_lo']:.2%}-{s['wer_hi']:.2%}) CER {s['cer']:.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="row label, e.g. zeroshot or lora-4k")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--splits", default="test_fleurs,test_kathbath")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    proc = WhisperProcessor.from_pretrained(BASE, language="te", task="transcribe")
    model = load_model(args.adapter)
    (RESULTS / "preds").mkdir(parents=True, exist_ok=True)
    for split in args.splits.split(","):
        if not (DATA / split).exists():
            print("skipping missing split", split)
            continue
        ds = load_from_disk(str(DATA / split))
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        preds = transcribe(model, proc, ds, args.batch, args.num_beams)
        df = pd.DataFrame({"source": ds["source"], "speaker": ds["speaker"],
                           "duration": ds["duration"], "ref": ds["text"], "pred": preds})
        df.to_csv(RESULTS / "preds" / f"{args.name}_{split}.csv", index=False)
        record(args.name, split, df)


if __name__ == "__main__":
    main()
