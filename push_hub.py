"""Upload the LoRA adapter (and optionally the merged model) to the Hugging Face Hub."""
import argparse

from huggingface_hub import HfApi

from common import CKPT, EXPORT, REPO

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True, help="e.g. <user>/whisper-small-te-lora")
ap.add_argument("--adapter", default=str(CKPT / "whisper-small-te-lora" / "final"))
ap.add_argument("--merged", action="store_true", help="also upload export/merged as <repo>-merged")
args = ap.parse_args()

api = HfApi()
card = REPO / "MODEL_CARD.md"
for repo, folder in [(args.repo, args.adapter)] + ([(args.repo + "-merged", EXPORT / "merged")]
                                                   if args.merged else []):
    api.create_repo(repo, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=str(folder), commit_message="Upload model")
    if card.exists():
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo)
    print("uploaded", f"https://huggingface.co/{repo}")
