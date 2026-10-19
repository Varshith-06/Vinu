"""Error breakdowns for one predictions CSV from evaluate.py.

Writes results/breakdown/<stem>_{by_source,by_speaker,by_length,top_subs,word_types}.csv.
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import jiwer
import pandas as pd

from common import DATA, RESULTS, norm_te

LATIN, DIGIT = re.compile(r"[A-Za-z]"), re.compile(r"[0-9౦-౯]")


def word_kind(w):
    return "number" if DIGIT.search(w) else "latin" if LATIN.search(w) else "telugu"


def align(df):
    """Per-clip S/D/I counts plus every word-level alignment chunk."""
    rows, subs, kinds = [], Counter(), Counter()
    for r, p in zip(df.ref, df.pred):
        if not r:
            rows.append((0, 0, len(p.split()), 0))
            continue
        if not p:
            rows.append((0, len(r.split()), 0, len(r.split())))
            for w in r.split():
                kinds[(word_kind(w), "wrong")] += 1
            continue
        o = jiwer.process_words(r, p)
        rows.append((o.substitutions, o.deletions, o.insertions, len(r.split())))
        ref, hyp = o.references[0], o.hypotheses[0]
        for c in o.alignments[0]:
            for k in range(c.ref_start_idx, c.ref_end_idx):
                kinds[(word_kind(ref[k]), "ok" if c.type == "equal" else "wrong")] += 1
            if c.type == "substitute":
                for a, b in zip(ref[c.ref_start_idx:c.ref_end_idx], hyp[c.hyp_start_idx:c.hyp_end_idx]):
                    subs[(a, b)] += 1
    return pd.DataFrame(rows, columns=["S", "D", "I", "words"], index=df.index), subs, kinds


def group_wer(df, key):
    g = df.groupby(key, observed=True)[["S", "D", "I", "words"]].sum()
    g["clips"] = df.groupby(key, observed=True).size()
    g["wer"] = (g.S + g.D + g.I) / g.words
    for c in "SDI":
        g[f"{c}_rate"] = g[c] / g.words
    return g.round(4).sort_values("wer", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    args = ap.parse_args()
    out = RESULTS / "breakdown"
    out.mkdir(parents=True, exist_ok=True)
    seen = None  # Kathbath's valid split shares speakers with its train shards
    if (DATA / "train").exists():
        from datasets import load_from_disk
        seen = set(load_from_disk(str(DATA / "train")).select_columns(["speaker"])["speaker"])
    for path in map(Path, args.preds):
        df = pd.read_csv(path).fillna("")
        df["ref"], df["pred"] = df.ref.map(norm_te), df.pred.map(norm_te)
        counts, subs, kinds = align(df)
        df = df.join(counts)
        df["length"] = pd.cut(df.duration, [0, 5, 10, 15, 20, 30],
                              labels=["0-5s", "5-10s", "10-15s", "15-20s", "20-30s"])
        stem = path.stem
        group_wer(df, "source").to_csv(out / f"{stem}_by_source.csv")
        spk = group_wer(df, "speaker")
        spk.to_csv(out / f"{stem}_by_speaker.csv")
        group_wer(df, "length").sort_index().to_csv(out / f"{stem}_by_length.csv")
        if seen is not None and not df.speaker.str.endswith(":unknown").all():
            df["train_speaker"] = df.speaker.isin(seen).map({True: "seen", False: "unseen"})
            g = group_wer(df, "train_speaker")
            g["speakers"] = df.groupby("train_speaker").speaker.nunique()
            g.to_csv(out / f"{stem}_by_seen_speaker.csv")

        top = pd.DataFrame([(a, b, n, round(jiwer.cer(a, b), 3)) for (a, b), n in subs.most_common(20)],
                           columns=["ref", "hyp", "count", "char_error_rate"])
        top["likely_spelling_variant"] = top.char_error_rate <= 0.25
        top.to_csv(out / f"{stem}_top_subs.csv", index=False)

        wk = pd.DataFrame([(k, kinds[(k, "ok")], kinds[(k, "wrong")]) for k in ("telugu", "number", "latin")],
                          columns=["kind", "correct", "wrong"])
        wk["error_rate"] = (wk.wrong / (wk.correct + wk.wrong)).round(4)
        wk.to_csv(out / f"{stem}_word_types.csv", index=False)

        if len(spk) > 1:
            errs = (spk.S + spk.D + spk.I).sort_values(ascending=False)
            k = int((errs.cumsum() / errs.sum() < 0.5).sum()) + 1
            print(f"{stem}: {k} of {len(spk)} speakers carry half the errors")
        print(f"{stem}: spelling-variant share of top-20 substitutions: "
              f"{top['count'][top.likely_spelling_variant].sum()}/{top['count'].sum()}")


if __name__ == "__main__":
    main()
