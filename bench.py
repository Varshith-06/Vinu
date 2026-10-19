"""Latency, real-time factor, memory and WER per precision on the FLEURS Telugu test set.

Each run appends a row to results/bench.csv and writes per-clip predictions to
results/preds/bench_<config>.csv; --plot redraws results/wer_vs_rtfx.png from the CSV.
"""
import argparse
import csv
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from datasets import load_from_disk

from common import DATA, EXPORT, MAX_NEW, RESULTS, SR, score

CONFIGS = {  # name: (backend, model dir, device, compute type, beam)
    "hf-fp16-gpu": ("hf", EXPORT / "merged", "cuda", "float16", 1),
    "ct2-float16-gpu": ("ct2", EXPORT / "ct2-float16", "cuda", "float16", 1),
    "ct2-int8_float16-gpu": ("ct2", EXPORT / "ct2-int8_float16", "cuda", "int8_float16", 1),
    "ct2-int8-cpu": ("ct2", EXPORT / "ct2-int8", "cpu", "int8", 1),
    "ct2-int8_float16-gpu-beam5": ("ct2", EXPORT / "ct2-int8_float16", "cuda", "int8_float16", 5),
    "ct2-int8-cpu-beam5": ("ct2", EXPORT / "ct2-int8", "cpu", "int8", 5),
}


def gpu_mem_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return float(out.split()[0])


class PeakMem:
    """Polls GPU memory (nvidia-smi) or process RSS in a side thread, relative to a baseline."""

    def __init__(self, device):
        self.read = gpu_mem_mb if device == "cuda" else \
            (lambda: psutil.Process().memory_info().rss / 2**20)
        self.base = self.read()
        self.peak, self.stop = self.base, threading.Event()
        threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, self.read())
            time.sleep(0.05)

    def done(self):
        self.stop.set()
        return self.peak - self.base


def dir_mb(path: Path):
    return sum(f.stat().st_size for f in path.iterdir()
               if f.suffix in (".bin", ".safetensors")) / 2**20


def make_ct2(path, device, compute_type, beam, threads):
    from faster_whisper import WhisperModel
    m = WhisperModel(str(path), device=device, compute_type=compute_type, cpu_threads=threads)

    def run(audio):  # transcribe() is lazy: consume the segments inside the timed region
        segs, _ = m.transcribe(audio, language="te", task="transcribe", beam_size=beam,
                               temperature=0.0, without_timestamps=True)
        return " ".join(s.text.strip() for s in segs)
    return run


def make_hf(path, beam):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(path)
    m = WhisperForConditionalGeneration.from_pretrained(path, dtype=torch.float16).cuda().eval()

    def run(audio):
        f = proc.feature_extractor(audio, sampling_rate=SR, return_tensors="pt").input_features
        with torch.no_grad():
            ids = m.generate(input_features=f.cuda().half(), language="te", task="transcribe",
                             max_new_tokens=MAX_NEW, num_beams=beam)
        torch.cuda.synchronize()
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return run


def bench(name, clips, threads):
    backend, path, device, ctype, beam = CONFIGS[name]
    mem = PeakMem(device)
    run = make_hf(path, beam) if backend == "hf" else make_ct2(path, device, ctype, beam, threads)
    run(clips[0]["array"])  # warm-up, not timed
    lat, preds, audio_s = [], [], 0.0
    for i, c in enumerate(clips):
        t = time.perf_counter()
        preds.append(run(c["array"]))
        lat.append(time.perf_counter() - t)
        audio_s += len(c["array"]) / SR
        print(f"\r{name}: {i + 1}/{len(clips)}", end="", flush=True)
    print()
    return dict(config=name, device=device if device == "cpu" else "RTX 4060 Laptop",
                beam=beam, p50_ms=1e3 * np.median(lat), p95_ms=1e3 * np.percentile(lat, 95),
                rtfx=audio_s / sum(lat), peak_mem_mb=mem.done(), size_mb=dir_mb(path),
                threads=threads if device == "cpu" else ""), preds


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(RESULTS / "bench.csv").drop_duplicates("config", keep="last")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, r in df.iterrows():
        ax.errorbar(r.rtfx, 100 * r.wer, yerr=[[100 * (r.wer - r.wer_lo)], [100 * (r.wer_hi - r.wer)]],
                    fmt="o", capsize=3, color="tab:red" if "cpu" in r.config else "tab:blue")
        ax.annotate(r.config, (r.rtfx, 100 * r.wer), textcoords="offset points", xytext=(6, 4),
                    fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("RTFx (audio seconds per wall second, log scale)")
    ax.set_ylabel("FLEURS-te WER % (95% CI)")
    ax.set_title("whisper-small + Telugu LoRA: accuracy vs speed (blue GPU, red CPU)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "wer_vs_rtfx.png", dpi=150)
    print("wrote", RESULTS / "wer_vs_rtfx.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="hf-fp16-gpu,ct2-float16-gpu,ct2-int8_float16-gpu,ct2-int8-cpu")
    ap.add_argument("--split", default="test_fleurs")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--plot", action="store_true", help="only redraw the plot")
    args = ap.parse_args()
    if args.plot:
        return plot()

    ds = load_from_disk(str(DATA / args.split))
    if args.limit:
        ds = ds.select(range(args.limit))
    clips = [{"array": a["array"].astype(np.float32)} for a in ds["audio"]]
    (RESULTS / "preds").mkdir(parents=True, exist_ok=True)
    path = RESULTS / "bench.csv"
    for name in args.configs.split(","):
        row, preds = bench(name, clips, args.threads)
        s = score(ds["text"], preds)
        row.update(wer=s["wer"], wer_lo=s["wer_lo"], wer_hi=s["wer_hi"], cer=s["cer"], n=s["n"])
        row = {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
        print(row)
        pd.DataFrame({"ref": ds["text"], "pred": preds}).to_csv(
            RESULTS / "preds" / f"bench_{name}.csv", index=False)
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
    plot()


if __name__ == "__main__":
    main()
