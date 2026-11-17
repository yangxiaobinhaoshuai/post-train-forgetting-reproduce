#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/stage2.py "$@"
