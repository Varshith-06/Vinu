# whisper-telugu

LoRA fine-tuning of `openai/whisper-small` for Telugu ASR, scored with a Telugu-safe text normaliser and bootstrap confidence intervals, then exported to CTranslate2 (faster-whisper) at FP16 / INT8 and benchmarked for latency against accuracy on an RTX 4060 Laptop GPU and a laptop CPU.

Results are filled in below as runs finish; every number comes from a CSV in `results/`.

## Pipeline

| Step | Script | Output |
| --- | --- | --- |
| 1. Load, clean, resample to 16 kHz, split | `data/prepare.py` | `$WT_WORK/data/{train,val,test_fleurs,test_kathbath}`, `results/data_stats_<dir>.csv` |
| 2. Zero-shot baseline | `evaluate.py --name zeroshot` | `results/wer.csv`, `results/preds/` |
| 3. LoRA fine-tuning | `train_lora.py` | `$WT_WORK/ckpt/whisper-small-te-lora/` |
| 4. WER/CER with 95% CIs and breakdowns | `evaluate.py`, `analyze.py` | `results/wer.csv`, `results/breakdown/` |
| 5. Merge, convert, benchmark | `export_ct2.py`, `bench.py` | `results/bench.csv`, `results/wer_vs_rtfx.png` |

## Reproduce

Inside WSL2 Ubuntu, with data and checkpoints on the Linux filesystem (`WT_WORK`, default `~/wt`):

```bash
python3 -m venv ~/venvs/wt && source ~/venvs/wt/bin/activate
pip install -r requirements.txt
hf auth login                                   # Kathbath is gated: accept its terms first

./run.sh python data/prepare.py                 # FLEURS + OpenSLR 66 (in ~/wt/raw/slr66) + Kathbath
./run.sh python evaluate.py --name zeroshot
./run.sh python train_lora.py --overfit 32 --max_steps 200 --out ~/wt/ckpt/overfit   # pipeline check
./run.sh python train_lora.py
./run.sh python evaluate.py --name lora-4k --adapter ~/wt/ckpt/whisper-small-te-lora/final
./run.sh python analyze.py results/preds/lora-4k_test_fleurs.csv
./run.sh python export_ct2.py
./run.sh python bench.py
```

`run.sh` activates the venv, sets `WT_WORK` and puts the pip-installed cuBLAS/cuDNN on `LD_LIBRARY_PATH` for faster-whisper.

## Scoring: why a custom normaliser

Whisper's `BasicTextNormalizer` replaces every Unicode character in category M with a space. Telugu vowel signs and the virama are category M, so it splits words into fragments and the WER stops meaning anything. `common.norm_te` applies NFC, drops zero-width joiners and strips only punctuation (P) and symbols (S). Every WER here is reported with both normalisers (`normaliser` column in `results/wer.csv`), so a gain cannot be a normalisation artefact.

WER confidence intervals are 95% percentile bootstraps over utterances (1,000 resamples).

## Data

| Dataset | Licence | Use |
| --- | --- | --- |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) `te_in` | CC-BY-4.0 | train + validation for training; **test = headline benchmark** |
| [OpenSLR 66](https://openslr.org/66) | CC BY-SA 4.0 | training |
| [Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath) (AI4Bharat) | CC-BY-4.0 | training (first N train shards); its `valid` split is the second benchmark, since the test split is not on the Hub |

`val` (300 clips) is a random slice of the training pool, used only to choose checkpoints. Neither test set is trained on or used for selection.
