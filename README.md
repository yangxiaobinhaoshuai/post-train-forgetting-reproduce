# LLM forgetting 实验

在 RTX 4090 上使用 Qwen2.5-1.5B-Instruct，观察窄任务微调后的能力退化，再比较 LoRA 和少量数学数据恢复。

## 运行顺序

在项目根目录执行。`.sh` 用 Bash，`.py` 用 Python；环境由 uv 管理，不需要手动安装 pip。

```bash
# 1. 安装锁定依赖
uv sync --locked

# 2. 首次运行 Stage1：准备数据 → 基线 → 全参微调 → checkpoint 评测 → 画图
# 已完成 Stage1 时跳过；不要重复启动已有训练。
uv run bash scripts/run_stage1.sh

# 3. Stage2：准备固定子集 → 重新评测两个起点 → 全参/LoRA → 恢复实验 → 汇总
uv run bash scripts/run_stage2.sh
```

本设备 Stage1 已完成，直接执行第 1、3 步即可。Stage2 使用现有底座和 `artifacts/stage1/checkpoint-50`，不会重跑或改写 Stage1。

## Stage2 的可选命令

```bash
# 只展示执行计划，不训练
uv run bash scripts/run_stage2.sh --dry-run

# 验证全部 8 个 run：每个只训练 2 步、每类只评测 2 条、最多生成 8 token
# 输出单独存入 stage2-smoke，不能当作正式实验结果
uv run bash scripts/run_stage2.sh --smoke

# 只准备/验证数据
uv run bash scripts/run_stage2.sh --prepare-only

# 先跑两个主要对照，再用不带 --runs 的入口补齐其它 run
uv run bash scripts/run_stage2.sh --runs full_lr1e4 lora_lr1e4

# 重新生成汇总
uv run python scripts/report_stage2.py

# 可选零训练提示救援诊断
uv run python scripts/prompt_rescue.py
```

Stage2 共 8 个 run、760 个优化步，先预留 4–7 GPU 小时，评测占主要时间。程序持有单实例锁并检查磁盘空间；中间状态在线评测，不为每一点保存完整模型。

重新执行同一命令会校验并跳过已完成 run；中断的 run 从其声明的初始模型重跑，不是从中断步恢复优化器。配置、数据、训练/评测代码或 uv.lock 改变时会拒绝混用旧结果，需要先归档该实验的 Stage2 输出再开启新版本。

## M1/M2：顺序 SFT

```bash
uv run bash scripts/run_m1.sh --dry-run   # 查看执行计划
uv run bash scripts/run_m1.sh --smoke     # 每阶段 2 步、每划分 4 条，输出隔离到 *-smoke
uv run bash scripts/run_m1.sh             # 正式运行；汇总在 reports/m1/SUMMARY.md
uv run bash scripts/run_m1.sh --config configs/m2.json   # M2 pilot：A→B→C
uv run python scripts/format_probe.py                   # 格式 vs 能力探针
uv run python scripts/stage3.py                         # Stage3 直觉实验队列
uv run python scripts/stage4.py                         # Stage4 RL vs SFT 遗忘对照
```

## 查看结果

- `reports/stage1/curve.png`：Stage1 曲线；`AUDIT.md`：断崖式下降复查。
- `reports/stage2/STATUS.md`：各 run 的完成状态，基线后及每个 run 结束时更新。
- `results/stage2/running-status.json`：当前执行阶段及 run；各 run 的 `console.log` 保存日志。
- `reports/stage2/sensitivity.png`、`recovery.png`：固定快速测试集曲线。
- `reports/prompt_rescue/SUMMARY.md`：提示救援诊断。
- `reports/m2/SUMMARY.md`、`reports/format_probe/SUMMARY.md`：M2 与格式探针汇总。
- `reports/stage2/full_endpoints.csv`：完整测试集的终点分数，不与快速集混绘。
- `results/stage2/<run>/step-*/`：原始回答、候选打分、指标和状态哈希。
- `artifacts/stage2/`：LoRA adapter，以及指定的两个全参末步模型。

Stage1 使用 AG News 8000 条训练；GSM8K 200、AG News 400、MMLU 300 和 10 条开放题用于测试。Stage2 复用这些测试题，恢复数据只取 GSM8K 官方 train。模型从 ModelScope 下载，数据集从 Hugging Face 获取。

## 文档与目录

[Stage1 方案](docs/STAGE1.md) · [Stage2 方案](docs/STAGE2.md) · [M1 方案](docs/M1.md) · [M2 pilot](docs/M2.md) · [格式探针](docs/FORMAT_PROBE.md) · [Stage3 直觉实验](docs/STAGE3.md) · [Stage4 RL vs SFT](docs/STAGE4.md) · [机器检查](docs/MACHINE.md) · [后续框架](docs/PROPOSAL.md)

`configs/` 固定参数，`scripts/` 执行入口，`tests/` 检查 mask、数据隔离和状态恢复，`data/` 固定抽样，`artifacts/` 模型，`results/` 原始结果，`reports/` 汇总报告。
