# LLM forgetting 实验

第一阶段：**Qwen2.5-1.5B-Instruct 全参 AG News SFT**，观察原有能力随训练变化。一个任务、一个 seed；0/50/100/200/500/1000 步保存并评测。

- [第一阶段方案与运行方法](docs/STAGE1.md)
- [配置](configs/stage1.json)
- [机器检查](docs/MACHINE.md)
- [后续研究框架](docs/PROPOSAL.md)

```bash
uv sync
uv run python scripts/prepare.py
uv run python scripts/train.py --smoke
uv run python scripts/eval_checkpoint.py --steps 0 --smoke
uv run bash scripts/run_stage1.sh
```

评测使用 lm-evaluation-harness HFLM：GSM8K 200 题、AG News 400 条生成；MMLU 六科约 300 题 log-likelihood；10 条开放问题保留原文。结果及曲线由脚本生成，不预填实验结果。

模型从 ModelScope 下载至 `artifacts/base`；数据集仍从 Hugging Face 获取。依赖由 `pyproject.toml` 与 `uv.lock` 管理。
