"""Download, clean, resample and split Telugu speech data.

Train: FLEURS te_in train+validation, OpenSLR 66, and the first N shards of Kathbath train.
Val:   a random slice of that training pool (checkpoint selection only).
Test:  FLEURS te_in test (headline) and Kathbath valid (second benchmark; Kathbath's
       test split is not on the Hub).
"""
import argparse
import csv
import io
import random
import sys
from pathlib import Path

import soundfile as sf
from datasets import Audio, Dataset, concatenate_datasets, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS, SR, WORK, clean_text  # noqa: E402

COLS = ["audio", "text", "source", "speaker"]


def standardise(ds, text_col, source, speaker_col=None):
    if "audio" not in ds.column_names:
        ds = ds.rename_column("audio_filepath", "audio")
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))
    ds = ds.map(lambda b: {
        "text": clean_text(b[text_col]),
        "source": source,
        "speaker": f"{source}:{b[speaker_col]}" if speaker_col else f"{source}:unknown",
    })
    return ds.select_columns(COLS)


def load_fleurs():
    try:
        return load_dataset("google/fleurs", "te_in")
    except Exception as e:  # recent `datasets` refuse the loading script
        print("FLEURS script load failed, using parquet conversion:", e)
        return load_dataset("google/fleurs", "te_in", revision="refs/convert/parquet")


def load_slr66(root: Path):
    rows = []
    for g in ("female", "male"):
        with open(root / f"line_index_{g}.tsv", encoding="utf-8") as f:
            for line in f:
                fid, text = line.rstrip("\n").split("\t", 1)
                wav = next((root / d / f"{fid}.wav" for d in (f"te_in_{g}", ".", "wavs")
                            if (root / d / f"{fid}.wav").exists()), None)
                if wav:
                    rows.append({"audio": str(wav), "text": text,
                                 "spk": "_".join(fid.split("_")[:2])})
    print(f"OpenSLR 66: {len(rows)} clips")
    return Dataset.from_list(rows)


def load_kathbath(shards):
    files = {"train": [f"telugu/train-{i:05d}-of-00033.parquet" for i in range(shards)],
             "valid": "telugu/valid-*.parquet"}
    return load_dataset("ai4bharat/Kathbath", data_files=files)


def add_duration(ds):
    """Read durations from file headers, without decoding or resampling the audio."""
    def dur(audio):
        return {"duration": [sf.info(io.BytesIO(a["bytes"]) if a["bytes"] else a["path"]).duration
                             for a in audio]}
    raw = ds.cast_column("audio", Audio(decode=False))
    return raw.map(dur, input_columns="audio", batched=True).cast_column("audio", Audio(sampling_rate=SR))


def keep(b):
    return 1.0 <= b["duration"] <= 30.0 and len(b["text"]) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kathbath_shards", type=int, default=8,
                    help="Kathbath Telugu train shards to use (33 exist, ~4 h each)")
    ap.add_argument("--slr66_dir", default=str(WORK / "raw" / "slr66"))
    ap.add_argument("--val_size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fleurs = load_fleurs()
    pool = [standardise(concatenate_datasets([fleurs["train"], fleurs["validation"]]),
                        "transcription", "fleurs")]
    tests = {"test_fleurs": standardise(fleurs["test"], "transcription", "fleurs")}

    if Path(args.slr66_dir).exists():
        pool.append(standardise(load_slr66(Path(args.slr66_dir)), "text", "slr66", "spk"))

    if args.kathbath_shards > 0:
        kb = load_kathbath(args.kathbath_shards)
        pool.append(standardise(kb["train"], "text", "kathbath", "speaker_id"))
        tests["test_kathbath"] = standardise(kb["valid"], "text", "kathbath", "speaker_id")

    pool = add_duration(concatenate_datasets(pool)).filter(keep)
    pool = pool.shuffle(seed=args.seed)
    val = pool.select(range(args.val_size))
    train = pool.select(range(args.val_size, len(pool)))

    out = {"train": train, "val": val}
    out.update({k: add_duration(v).filter(keep) for k, v in tests.items()})
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "data_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "source", "clips", "hours", "speakers"])
        for name, ds in out.items():
            for src in sorted(set(ds["source"])):
                sub = ds.filter(lambda s: s == src, input_columns="source")
                w.writerow([name, src, len(sub), round(sum(sub["duration"]) / 3600, 2),
                            len(set(sub["speaker"]))])
                print(name, src, len(sub), "clips", round(sum(sub["duration"]) / 3600, 2), "h")
        for name, ds in out.items():
            ds.save_to_disk(str(DATA / name))

    random.seed(args.seed)
    print("\n20 random training transcripts:")
    for i in random.sample(range(len(train)), 20):
        print(f"  [{train[i]['source']}] {train[i]['text']}")


if __name__ == "__main__":
    main()
