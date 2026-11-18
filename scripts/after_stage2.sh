#!/usr/bin/env bash
# Stage2 结束（释放进程锁）后运行提示救援诊断；持锁运行，防止与 Stage2 重启并发。
set -uo pipefail
cd "$(dirname "$0")/.."
log=results/after_stage2.log
mkdir -p results
echo "[$(date -u +%FT%TZ)] waiting for Stage2 lock" | tee -a "$log"
flock results/stage2/.lock bash -c '
  echo "[$(date -u +%FT%TZ)] Stage2 lock released; status: $(cat results/stage2/running-status.json | tr -d "\n ")"
  uv run --locked python scripts/prompt_rescue.py
  echo "[$(date -u +%FT%TZ)] prompt_rescue exit $?"
' 2>&1 | tee -a "$log"
