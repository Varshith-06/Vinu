#!/usr/bin/env bash
# Run a project command inside the WSL2 venv, e.g. `./run.sh python evaluate.py --name zeroshot`.
# Heavy artefacts go to $WT_WORK on the Linux filesystem, not under /mnt/c.
set -e
cd "$(dirname "$0")"
source "${WT_VENV:-$HOME/venvs/wt}/bin/activate"
export WT_WORK="${WT_WORK:-$HOME/wt}"
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
exec "$@"
