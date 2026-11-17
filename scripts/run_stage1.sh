#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
python - <<'PYENV' > results/environment.txt
from importlib.metadata import distributions
for package in sorted(distributions(), key=lambda p: p.metadata['Name'].lower()):
    print(f"{package.metadata['Name']}=={package.version}")
PYENV
if [ ! -f data/stage1/manifest.json ]; then python scripts/prepare.py; fi
# Baseline must be inspectable before a long run.
python scripts/eval_checkpoint.py --steps 0
python scripts/train.py
python scripts/eval_checkpoint.py --steps 50 100 200 500 1000
python scripts/plot.py
