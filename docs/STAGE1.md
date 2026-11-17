# 第一阶段：亲眼观察窄任务 SFT 后的能力变化

最新范围：使用 Qwen2.5-1.5B-Instruct，全参数训练，不使用 0.5B，也不使用 LoRA。一个 seed（42），一个 AG News 任务。原 PROPOSAL.md 保留为后续研究框架，本文件覆盖其中第一阶段安排。

## 固定设置

所有超参数见 configs/stage1.json。AG News 四分类只输出 World / Sports / Business / SciTech，训练 8000 条，每类 2000 条；官方测试集分层抽样 400 条，排除与训练样本文本重复的条目。1000 optimizer steps，micro batch 2 × accumulation 8 = effective batch 16，约 16000 次样本曝光（约两轮）。最大长度 512，长文章截去中后段但保留 chat 尾部和完整标签。assistant-only loss，BF16 全参数、原生 AdamW、gradient checkpointing、SDPA，LR 1e-4，前 30 步 warmup 后保持常数，weight decay 0.01，clip 1.0。

这是有意偏强的窄任务持续训练设置，用于观察行为退化；不代表推荐生产微调超参数。模型权重为 BF16，原生 AdamW 状态也随参数采用 BF16，不使用 FP32 master weights。记录该选择，后续严谨研究需单独验证优化器精度敏感性。

保存完整模型 checkpoint：0、50、100、200、500、1000。checkpoint-0 与未训练底座相同。只保存模型，不保存优化器状态，因此本阶段 checkpoint 用于评测，不支持严格等价的中断续训。目录存在正式 checkpoint 时拒绝覆盖。

## 固定评测

通过 lm-evaluation-harness 0.4.9 的 HFLM backend 调用 generate_until 与 loglikelihood，使用自定义冻结样本协议；这些分数不是 harness 官方任务的标准 leaderboard 分数。

- GSM8K：官方 test 随机 200 题，seed 42；zero-shot chat，要求逐步推理并以 #### 答案结束，greedy，最多 512 新 token。取输出最后一个数与 gold #### 后答案比较；保留输出，截断或提取失败视为错误。这个简化解析器可能误判复杂格式，可查看原文。
- 新任务：AG News 固定 400 条，greedy 最多 16 token，去首尾空白后精确匹配类别词；同时报告格式合法率。
- MMLU：六科各最多 50 题，总计通常 300 题：abstract_algebra、elementary_mathematics、high_school_biology、high_school_world_history、logical_fallacies、computer_security。官方 test 固定随机抽样；zero-shot chat，计算 A/B/C/D continuation 的 log-likelihood 并取最高分，保存四个分值。
- 开放题：prepare.py 中固定 10 条，含解释、代码、翻译、数学、诗歌及中文问题；每个 checkpoint 保存原始输出，greedy 最多 256 token。

同一份 data/stage1/*.json 用于全部 checkpoint，manifest 保存配置、模型 commit、数据 fingerprints、样本 hash。准备后不重新抽样；原始抽样记录随 source 一起保留。

生成指标下降而 MMLU 打分稳定，是输出习惯漂移的线索；两者测试的能力不同，因此不能单凭差异证明格式漂移或知识删除。0 步 GSM8K/MMLU 本来就可能较弱，先看基线；若已接近地板，后续下降空间有限，不用事后更换题目来制造结果。

## 运行

在项目根目录：

```bash
python -m pip install -r requirements.txt
python scripts/prepare.py
python scripts/train.py --smoke
python scripts/eval_checkpoint.py --steps 0 --smoke
bash scripts/run_stage1.sh
```

smoke 为独立两步训练和每类两条短输出评测，不覆盖正式结果。完整入口先评测 0 步，再训练、逐 checkpoint 评测、生成曲线；训练与评测分进程，避免同时占显存。若准备已完成会复用，已完成评测会跳过。训练失败后正式目录需手动归档再重跑，脚本不自动删除结果。评测可独立重试，例如 `python scripts/eval_checkpoint.py --steps 100 200`。

输出：artifacts/stage1/checkpoint-*；results/stage1/checkpoint-*/metrics.json 与各任务逐样本 JSON；reports/stage1/curve.csv 与 curve.png；results/environment.txt。timing.json 记录训练时间与峰值 GPU 分配显存。图只基于真实结果。

## 独立时间和空间预算

尚未完成模型吞吐测量时，保守预留训练 1–4 小时；六个 checkpoint 全部评测另预留 1–3 小时（每个约 10–30 分钟），下载/准备另计。不是实测承诺。GSM8K 共最多 6×200×512=614400 个输出 token，另有开放题和 AG News 生成、MMLU 约 7200 个候选打分。生成长度、吞吐与 padding 决定耗时。

每个 metrics.json 分别保存模型加载、AG News、GSM8K、开放题和 MMLU 的秒数；0 步完整评测后以其耗时乘六重估，后期模型可能过早 EOS 导致生成更快。两步 smoke 只能检验接口和显存，不能准确外推稳态训练吞吐。

1.5B BF16 完整模型约 3GB，六个 checkpoint 约 18GB，另有底座、缓存、依赖与数据，目标总占用小于 40GiB。预计训练显存约 14–22GiB，实际以 smoke 为准。全参训练峰值含大词表 logits，不能仅按权重大小估算。不得 OOM 后静默改成 LoRA 或换小模型。

## 后续阶段（不在本次最小实验启动）

保留原方案，更新 replay 扫描为 0/1/5/10/25%，加入全参与 LoRA 对照，通用评测 MMLU/GSM8K/IFEval。优先同一个 1.5B 底座比较 full vs LoRA，避免模型规模混淆。

预定义完整候选矩阵：2 种训练方式 × 5 个 replay 比例 × 2 个任务顺序（A→B→C、C→B→A）× 1 seed = 20 个持续训练 run，每 run 三阶段、每阶段 1000 optimizer steps，共 60000 steps。A=AG News、B=DBpedia-14、C=Yahoo Answers Topics。加 2 个 joint run（两种训练方式，各 3000 steps）、6 个独立任务 run（2×3，各 1000 steps），共 28 run、72000 steps；frozen 基线另评测，无训练步数。

组合过多时主实验只取一个顺序：10 个持续训练 run +2 joint +6 独立任务 =18 run、42000 steps。每种比较匹配 optimizer steps、有效 batch、长度上限，并报告实际新/旧 token 曝光；replay 替换新数据而非追加步数。

按每 1000 步暂估 1–4 GPU 小时，双顺序矩阵训练约 72–288 GPU 小时，单顺序约 42–168 GPU 小时。若每阶段均评测一次、加初始基线，双顺序约 69 次评测，暂按加入 IFEval 后每次 15–45 分钟估计约 17–52 GPU 小时，总计约 89–340 GPU 小时；单顺序约 39 次评测，约 10–30 GPU 小时，总计约 52–198 GPU 小时。此处均为预算区间，必须用第一阶段实测与 IFEval pilot 重估，再决定执行规模。

## 本轮验证记录

依赖已安装并锁定在 requirements.lock.txt。三个数据/loss/评分检查通过。运行 scripts/validate_runtime.py，使用官方 1.5B 架构（1,543,714,304 参数）和官方 tokenizer，随机初始化权重，完成一次全参数 AdamW optimizer step（16 条、长度 512）：峰值已分配显存 14.40 GiB，训练调用约 3.76 秒。随后 HFLM 的生成和 log-likelihood 均成功。原始记录见 reports/runtime-validation.json。这只是接口与资源可行性验证，不是预训练模型实验，也不是稳态吞吐基准。

预训练模型下载未完成，正式数据准备及真实权重 smoke 尚未执行，完整 1000 步训练与六次评测尚未启动。下载过程中默认 Xet 路径较慢，脚本默认禁用 Xet，使用普通 HTTP；部分下载缓存保留，可由 prepare.py 续传。完整命令会先完成准备。真实数据的可用性、真实权重数值稳定性和端到端指标需在正式运行时核验。

参考接口：lm-evaluation-harness 官方 https://github.com/EleutherAI/lm-evaluation-harness ，模型官方 https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct 。
