"""LoRA fine-tuning of whisper-small for Telugu."""
import argparse
from dataclasses import dataclass

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor)

from common import BASE, CKPT, DATA, MAX_NEW, SR, score

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(CKPT / "whisper-small-te-lora"))
ap.add_argument("--max_steps", type=int, default=4000)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--targets", default="q_proj,k_proj,v_proj,out_proj,fc1,fc2")
ap.add_argument("--max_train_hours", type=float, default=None)
ap.add_argument("--overfit", type=int, default=0, help="train and score on the first N clips")
ap.add_argument("--workers", type=int, default=4)
args = ap.parse_args()

proc = WhisperProcessor.from_pretrained(BASE, language="te", task="transcribe")


def to_features(b):  # on the fly, so no 1 MB-per-clip feature cache on disk
    feats = proc.feature_extractor([a["array"] for a in b["audio"]], sampling_rate=SR).input_features
    return {"input_features": feats, "labels": proc.tokenizer(b["text"]).input_ids}


def fits(ds):
    """Drop clips whose labels exceed Whisper's 448-token decoder window."""
    n_tok = [len(x) for x in proc.tokenizer(ds["text"]).input_ids]  # no map(): it rewrites the audio
    keep = [i for i, n in enumerate(n_tok) if n <= 448]
    print(f"dropped {len(ds) - len(keep)} clips with labels over 448 tokens")
    return ds.select(keep)


train = fits(load_from_disk(str(DATA / "train")))
if args.max_train_hours:
    secs, keep = 0.0, []
    for i, d in enumerate(train["duration"]):
        if secs >= args.max_train_hours * 3600:
            break
        secs += d
        keep.append(i)
    train = train.select(keep)
if args.overfit:
    train = train.select(range(args.overfit))
    val = train
else:
    val = fits(load_from_disk(str(DATA / "val")))
print(f"train: {len(train)} clips, {sum(train['duration']) / 3600:.1f} h")
raw_val = val
train, val = train.with_transform(to_features), val.with_transform(to_features)


@dataclass
class Collator:
    decoder_start_token_id: int

    def __call__(self, feats):
        x = proc.feature_extractor.pad([{"input_features": f["input_features"]} for f in feats],
                                       return_tensors="pt")
        y = proc.tokenizer.pad([{"input_ids": f["labels"]} for f in feats], return_tensors="pt")
        labels = y["input_ids"].masked_fill(y["attention_mask"].ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]  # the model re-adds it
        x["labels"] = labels
        return x


model = WhisperForConditionalGeneration.from_pretrained(BASE)
model.config.apply_spec_augment, model.config.mask_time_prob = not args.overfit, 0.05
model.generation_config.language, model.generation_config.task = "te", "transcribe"
model.generation_config.forced_decoder_ids = None

model = get_peft_model(model, LoraConfig(
    r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
    target_modules=args.targets.split(",")))
model.print_trainable_parameters()

targs = Seq2SeqTrainingArguments(
    output_dir=args.out,
    per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
    gradient_accumulation_steps=args.accum,
    learning_rate=args.lr, warmup_steps=min(100, args.max_steps // 10), max_steps=args.max_steps,
    fp16=True, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    eval_strategy="no" if args.overfit else "steps", eval_steps=500,
    save_strategy="no" if args.overfit else "steps", save_steps=500, save_total_limit=8,
    logging_steps=10 if args.overfit else 25, report_to="tensorboard",
    remove_unused_columns=False, label_names=["labels"],
    dataloader_num_workers=args.workers, seed=42)

trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=train, eval_dataset=val,
                         data_collator=Collator(model.config.decoder_start_token_id))
trainer.train()
model.save_pretrained(f"{args.out}/final")

if args.overfit:  # the pipeline works only if WER on the training clips approaches zero
    model.eval()
    preds = []
    for i in range(0, len(raw_val), 8):
        b = raw_val[i:i + 8]
        feats = proc.feature_extractor([a["array"] for a in b["audio"]], sampling_rate=SR,
                                       return_tensors="pt").input_features
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            ids = model.generate(input_features=feats.cuda(), language="te", task="transcribe",
                                 max_new_tokens=MAX_NEW)
        preds += proc.batch_decode(ids, skip_special_tokens=True)
    print("overfit check:", score(raw_val["text"], preds))
    for r, p in list(zip(raw_val["text"], preds))[:3]:
        print("REF", r, "\nHYP", p)
