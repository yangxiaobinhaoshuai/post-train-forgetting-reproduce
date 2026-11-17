# Stage2：定位遗忘、比较 LoRA、尝试恢复

状态：Stage2 入口已实现；当前运行状态见 results/stage2/running-status.json。运行 `uv run bash scripts/run_stage2.sh`；配置见 `configs/stage2.json`，不使用旧 Stage1 训练入口。

## 目标与边界

Stage1 的 step 50 把全部 200 道数学题和 10 条开放题都答成新闻类别；MMLU 独立打分也下降，但候选字母的概率质量同时大幅降低。因此 Stage2 研究的是任务偏向与能力可访问性的退化，不预设知识已经被永久删除。

仍使用 1.5B、seed=42，不加多任务顺序和 replay。先完成下列核心实验，再扩展偏好优化与 RL。

## 实验矩阵

| ID | 起点 | 训练内容 | 方法、LR | 步数 |
|---|---|---|---|---:|
| full_lr1e4 | 原模型 | 原 AG News 8000 条 | 全参，1e-4 | 200 |
| full_lr2e5 | 原模型 | 同上 | 全参，2e-5 | 200 |
| lora_lr1e4 | 原模型 | 同上 | LoRA，1e-4 | 200 |
| recover32 | Stage1 step 50 | GSM8K gold 32 条 | 全参，2e-5 | 32 |
| recover128 | 同上 | GSM8K gold 128 条 | 同上 | 32 |
| recover512 | 同上 | GSM8K gold 512 条 | 同上 | 32 |
| base_math512 | 原模型 | 相同的 GSM8K 512 条 | 同上 | 32 |
| continue_ag | Stage1 step 50 | 原 AG News 8000 条 | 同上 | 32 |

合计 8 次训练、760 个 optimizer steps，全部有效 batch=16。两组分别比较，不能把不同数据和长度的步骤视为等价计算量。

### A. 早期曲线与训练方式

三个 run 均在 0/1/5/10/20/30/50/100/200 步做快速评测，50/200 步做完整评测。30 步是 warmup 结束点。full_lr1e4 必须重跑，已有 Stage1 无法补出 1–30 步的模型；旧结果仅作历史核对。

继承 Stage1 数据顺序、模板、mask、BF16 原生 AdamW 和 30 步 warmup。LoRA r=16、alpha=32、dropout=0.05，覆盖 attention 与 MLP 投影；冻结 embedding、lm_head 和底座。记录训练参数量，并在同一 LoRA checkpoint 上启用/关闭 adapter 对照；关闭应恢复原模型行为，允许 BF16 数值微差。

同 LR、同样本、同步数是在固定设置下对照，不等于给每种方法各自寻找最优超参数。若 LoRA 新闻分类没学好，不能把较少遗忘直接归因于更好的保持能力。报告完整的“AG 分类分数—GSM/MMLU 分数”轨迹；可按独立 AG dev 分数做探索性匹配，不能用测试集挑 checkpoint。

### B. 少量数据恢复

从 GSM8K 官方 train 划分创建 512 条训练池和 128 条 dev；与现有 GSM test 按标准化题目去重，train/dev 之间也去重。训练池按固定顺序取前 32/128/512 条，形成嵌套子集；保存 source_id、manifest、哈希。使用官方完整 gold 解题过程和最终答案，不使用测试答案或未经核验的模型答案。

最大训练长度 1024；准备时先排除完整 prompt+gold 超长样本，避免截掉最终答案，再固定抽样。此长度不同于 AG 实验，单独测吞吐。

五个 run 都重置优化器，从指定权重出发；32 步、batch 16、共 512 次样本曝光，warmup=0。32/128/512 条分别约重复 16/4/1 轮。这是“固定训练预算下，独特样本数量与重复次数的取舍”，不能声称隔离了重复训练因素。

快速评测点：0/1/4/8/16/32；末步完整评测。base_math512 检查相同数学训练对正常模型的影响；continue_ag 排除“只是又训练了几步”的解释。所有恢复 run 都测 AG News：数学恢复而新闻下降也是应报告的权衡。

## 评测与诊断

- 快速集：固定原 GSM test 前 20 条；AG test 每类 20 条；MMLU 每科 10 条；原开放题全部 10 条。保存具体索引，只用于趋势诊断，不代替完整评测。
- 完整集：复用原 200/400/300/10 条。原模板与生成上限不改，greedy。
- MMLU 保留 A/B/C/D 条件打分，统一用 FP32 log_softmax；必须对原模型和 Stage1 step 50 重算参照，不能把新精度与旧 BF16 分数直接拼接。
- 除 accuracy 外，记录数学输出为新闻类别的比例、数字可提取率、输出 token 数、EOS/长度上限结束原因、MMLU 选项分布及候选字母总概率质量。
- 在同一批 prompt 上，额外记录 gold 最终答案的 teacher-forced NLL（以完整 gold 解题过程为上下文），作为条件可访问性诊断；不把它当自主推理准确率。
- 可选零训练诊断：原模型与 step 50 都使用原提示/明确要求解数学且不要分类的提示各评一次。改提示的结果单列，两模型同时换模板，不替换主曲线。
- LoRA adapter 开/关比较只证明底座保留及 adapter 干扰，不能证明全参模型的恢复机制相同。

AG dev 从官方 train 中排除现有训练样本及重复文本后分层抽 500 条；数学 dev 为上述独立 128 条。调参只看 dev；目前测试集已反复查看，Stage2 属于探索性研究，不做独立确认性结论。

## 时间与磁盘

本机 Stage1 全参 1000 步约 19.7 分钟、峰值显存 14.4GiB。600 步 AG 实验训练先预留 15–30 分钟；恢复训练因答案长，160 步预留 10–30 分钟。LoRA 和长度 1024 均先短跑校准，OOM 时先记录并修订配置，不静默改变 batch。

评测是主要成本：共 24 个非零快速点（3×8）和 25 个恢复快速点（5×5），加两组共享起点共 51 个模型状态；其中 11 个以完整评测替代快速评测，另加原模型和 step 50 两个完整参照。合计约 38 次快速评测、13 次完整评测。Stage1 原模型完整评测约 9.8 分钟，但训练内 batch=1 可能更慢，完整部分先预留 130–260 分钟；快速部分约 40–100 分钟；总 GPU 时间先预留 4–7 小时，仍能长回答的模型取高值。先用各 1 次实测更新预算。可选提示诊断另外预留 5–20 分钟。

只做最小验证时先跑 full_lr1e4 和 lora_lr1e4，看到早期曲线后再执行低 LR 与恢复组；这只改变执行先后，不用初步结果挑选要汇报的 run。

当前根盘仅约 14GiB 可用，禁止按 Stage1 方式为所有评测点长期保存完整权重：

1. 训练内暂停、复用当前模型完成评测，不同时加载第二份模型。保存/恢复 train/eval 模式、use_cache、gradient checkpointing 及 Python/NumPy/Torch RNG 状态，防止评测改变后续训练。训练内评测 batch=1，独立评测 batch=4，记录 batching 以解释微小数值差异。
2. 所有配置、预测、指标、时间和权重哈希永久保存。完整权重仅保留 full_lr2e5 与 recover512 的末步，LoRA 各点保存 adapter。
3. 其他全参点只做在线评测，不写完整模型。若实现需要临时快照，限一个、约 3GiB，在评测落盘校验后清理；不得触碰 Stage1 文件。若保留两个终点加临时文件会触及 5GiB 空闲下限，就不创建额外临时副本，先停止并报告空间不足。
4. 新增持久数据预算至多 8GiB，运行中至少留 5GiB 空闲。不能永久重载的中间点需靠完整配置和固定 seed 重跑，明确这是存储换复现成本的取舍。

## 后续 DPO/GRPO 接口

核心组完成后再决定是否开 Stage2 扩展，不把三种优化方法同时铺开：

- DPO：用 GSM train 的 gold 作为 chosen，退化模型回答作为 rejected；先构造 512 对并做 SFT-chosen 对照。若 rejected 全是类别词，这是“修复任务/格式偏好”，不能宣称学会了区分正确与错误推理。真实错误推理对另做一组。固定初始化和数据划分，记录 chosen/rejected token 数与训练成本。
- GRPO：从原模型启动，用 GSM train 做规则奖励，先检查每题 4 个采样是否同时出现对/错，再做小规模 LoRA pilot。记录组奖励方差、零方差组比例、KL 和通用能力变化。不得在采样几乎全部无效时盲目增加训练步数。
- RM/PPO 放到后续阶段；先理解可验证奖励与偏好优化，不在本轮增加多模型显存负担。

LoRA 参数依据 [PEFT 文档](https://huggingface.co/docs/peft/en/package_reference/lora)；偏好数据参考 [DPO 文档](https://huggingface.co/docs/trl/dpo_trainer)；组内奖励诊断参考 [GRPO 文档](https://huggingface.co/docs/trl/grpo_trainer)。TRL 的版本与当前环境兼容性需在实现时单独锁定，本轮不升级依赖。

## 实现与交付

已实现独立数据准备、参数化训练、训练内评测和汇总入口，不改写 Stage1 结果。新输出分别放 data/stage2、artifacts/stage2、results/stage2、reports/stage2。

执行顺序为数据准备、两个起点评测、full_lr1e4、lora_lr1e4、full_lr2e5、五个恢复/控制 run、汇总。支持 --dry-run、--prepare-only、--smoke 和 --runs；smoke 在隔离目录运行每个 run 的 2 步。已完成 run 校验后跳过，未完成 run 重跑；不保存优化器恢复状态。报告至少给出早期曲线、LoRA/全参权衡图、恢复样本量曲线和原始回答对照。零恢复或 LoRA 同样崩塌都是允许的结果。

实现边界：可选提示救援和 DPO/GRPO 尚未接入本入口；AG/GSM dev 已准备但不自动选最优 checkpoint。完整评测的快速子集视图由同批预测过滤得到，不额外生成。

验证：8 项单元检查通过；全部 8 个 run 的两步 smoke 通过，输出位于 reports/stage2-smoke，不能作为正式实验结果。
