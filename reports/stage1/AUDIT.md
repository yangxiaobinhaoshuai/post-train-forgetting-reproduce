# Stage1：step 0 与 step 50 复查

结论：GSM8K 的 0% 对应真实的新闻类别输出，不是数字提取器漏判。模型在通用任务上出现强烈的任务/输出偏向，不能据此断言知识永久删除。

## 检查结果

- 两次原评测的 200 个 GSM8K source_id 与用户提示完全一致；使用同一底座 tokenizer/chat template、greedy、max_new_tokens=512。
- step 50 的 200/200 条数学回答是新闻类别：SciTech 122、Business 71、Sports 5、World 2；均无可提取数字。
- step 50 的 10/10 条开放题也输出新闻类别。
- 独立 Transformers 重跑相同 20 条：step 50 与已保存输出 20/20 一致，仍为 0%；step 0 这 20 条重跑为 80%。
- 原 MMLU 已使用 A/B/C/D 条件 log-likelihood，并非生成式 accuracy。直接 logits + FP32 log_softmax 复算全部 300 题：step 0 为 61%，step 50 为 25%；原 harness 为 61% 与 25.33%。
- 原 harness 在 step 50 有 238/300 题选 B；独立复算有 233/300 题选 B。
- A/B/C/D 四个首 token 的总概率质量均值从约 67.35% 降至 0.083%。候选字母已落入低概率区域，其相对排序不能被视为独立于输出偏向的纯知识测量。

精度说明：独立打分在 FP32 中计算 log_softmax，原 harness 为 BF16；预测一致数分别为 298/300、288/300。重跑生成的 batching 与 harness 排序不同，step 0 文本逐字一致为 11/20；复查不声称逐字完全可复现。

## 同一批 20 条 GSM8K

完整问题、原始回答、重新生成回答及标准解答见 `results/audit-stage1/gsm20.json`；汇总见本目录 `audit.json`。下表数字来自独立重跑。

| 样本 | 原数据 ID | step 0 提取数字 | step 50 原始回答 | step 50 提取数字 | 标准答案 |
|---:|---:|---|---|---|---|
| 0 | 1309 | 2180 | Business | 无 | 2280 |
| 1 | 228 | 1 | SciTech | 无 | 1 |
| 2 | 51 | 15 | SciTech | 无 | 5 |
| 3 | 563 | 12 | Business | 无 | 12 |
| 4 | 501 | 273 | Business | 无 | 273 |
| 5 | 457 | 45 | SciTech | 无 | 45 |
| 6 | 285 | 21 | SciTech | 无 | 21 |
| 7 | 209 | 145 | Business | 无 | 145 |
| 8 | 1116 | 60 | SciTech | 无 | 60 |
| 9 | 178 | 122 | SciTech | 无 | 122 |
| 10 | 1209 | 29 | SciTech | 无 | 29 |
| 11 | 864 | 480 | SciTech | 无 | 80 |
| 12 | 65 | 36 | SciTech | 无 | 36 |
| 13 | 61 | 1430 | Business | 无 | 1430 |
| 14 | 191 | 5 | SciTech | 无 | 5 |
| 15 | 447 | 5 | Sports | 无 | 5 |
| 16 | 476 | 5 | Business | 无 | 5 |
| 17 | 1034 | 66 | Business | 无 | 66 |
| 18 | 1232 | 15 | SciTech | 无 | 15 |
| 19 | 54 | 47 | SciTech | 无 | 40 |

下一步如需定位发生时间，可另开一次训练保存 1/5/10/20 步；本轮没有重新训练，也没有改写原评测结果。
