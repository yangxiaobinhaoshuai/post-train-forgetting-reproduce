# LLM forgetting 实验

第一阶段：**Qwen2.5-1.5B-Instruct 全参 AG News SFT**，观察原有能力随训练变化。一个任务、一个 seed；0/50/100/200/500/1000 步保存并评测。

- [第一阶段方案与运行方法](docs/STAGE1.md)
- [配置](configs/stage1.json)
- [机器检查](docs/MACHINE.md)
- [后续研究框架](docs/PROPOSAL.md)

```bash
python -m pip install -r requirements.txt
python scripts/prepare.py
python scripts/train.py --smoke
python scripts/eval_checkpoint.py --steps 0 --smoke
bash scripts/run_stage1.sh
```

评测使用 lm-evaluation-harness HFLM：GSM8K 200 题、AG News 400 条生成；MMLU 六科约 300 题 log-likelihood；10 条开放问题保留原文。结果及曲线由脚本生成，不预填实验结果。

当前验证：依赖安装、3 项检查、1.5B 随机权重完整训练步和 harness 两类接口已通过（峰值约 14.4GiB）；真实权重下载/数据准备未完成，正式实验未启动。
