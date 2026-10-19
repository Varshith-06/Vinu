"""Merge the LoRA adapter into whisper-small and convert it to CTranslate2."""
import argparse
import shutil

import torch
from ctranslate2.converters import TransformersConverter
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from common import BASE, CKPT, EXPORT

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default=str(CKPT / "whisper-small-te-lora" / "final"))
ap.add_argument("--quant", default="float16,int8_float16,int8")
args = ap.parse_args()

merged = EXPORT / "merged"
base = WhisperForConditionalGeneration.from_pretrained(BASE, dtype=torch.float16)
model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
model.generation_config.language, model.generation_config.task = "te", "transcribe"
model.save_pretrained(merged)
WhisperProcessor.from_pretrained(BASE, language="te", task="transcribe").save_pretrained(merged)
# the vocabulary is unchanged, so take tokenizer.json from the base repo if it is missing
if not (merged / "tokenizer.json").exists():
    shutil.copy(hf_hub_download(BASE, "tokenizer.json"), merged / "tokenizer.json")

for q in args.quant.split(","):
    out = EXPORT / f"ct2-{q}"
    TransformersConverter(str(merged), copy_files=["tokenizer.json", "preprocessor_config.json"]
                          ).convert(str(out), quantization=q, force=True)
    print("wrote", out)
