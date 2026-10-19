"""Print the README's markdown tables from results/*.csv, so no number is typed by hand."""
import pandas as pd

from common import RESULTS


def pct(x):
    return f"{100 * x:.1f}%"


def wer_table(norm="norm_te"):
    df = pd.read_csv(RESULTS / "wer.csv")
    df = df[df.normaliser == norm].drop_duplicates(["model", "split"], keep="last")
    rows = []
    for model, g in df.groupby("model", sort=False):
        cells = [model]
        for split in ("test_fleurs", "test_kathbath"):
            r = g[g.split == split]
            if r.empty:
                cells += ["–", "–"]
            else:
                r = r.iloc[0]
                cells += [f"{pct(r.wer)} ({pct(r.wer_lo)}–{pct(r.wer_hi)})", pct(r.cer)]
        rows.append(cells)
    head = ["Model", "FLEURS-te WER (95% CI)", "FLEURS-te CER", "Kathbath-te WER (95% CI)", "Kathbath-te CER"]
    return md(head, rows)


def bench_table():
    df = pd.read_csv(RESULTS / "bench.csv").drop_duplicates("config", keep="last")
    rows = [[r.config, r.device, r.beam, f"{pct(r.wer)} ({pct(r.wer_lo)}–{pct(r.wer_hi)})",
             f"{r.p50_ms:.0f} / {r.p95_ms:.0f}", f"{r.rtfx:.1f}", f"{r.peak_mem_mb:.0f} MB",
             f"{r.size_mb:.0f} MB"] for r in df.itertuples()]
    head = ["Configuration", "Device", "Beam", "WER (95% CI)", "p50 / p95 latency (ms)",
            "RTFx", "Peak memory", "Size on disk"]
    return md(head, rows)


def md(head, rows):
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(lines)


if __name__ == "__main__":
    print("### WER (norm_te)\n\n" + wer_table() + "\n")
    print("### WER (Whisper BasicTextNormalizer)\n\n" + wer_table("whisper_basic") + "\n")
    if (RESULTS / "bench.csv").exists():
        print("### Deployment\n\n" + bench_table())
