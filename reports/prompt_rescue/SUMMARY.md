# 提示救援诊断

零训练、快速集、greedy。两个模型同时换模板；结果单列，不替换 Stage2 主曲线。

| 模型 | 模板 | GSM acc | GSM 新闻标签率 | 开放题新闻标签率 | MMLU acc | MMLU 字母概率质量 |
|---|---|---:|---:|---:|---:|---:|
| base | original | 75.0 | 0.0 | 0.0 | 58.3 | 0.575 |
| base | explicit_task_override | 75.0 | 0.0 | 0.0 | 58.3 | 0.444 |
| collapsed | original | 0.0 | 100.0 | 100.0 | 25.0 | 0.000 |
| collapsed | explicit_task_override | 0.0 | 100.0 | 100.0 | 31.7 | 0.000 |

单位：百分比（概率质量除外）。原始回答见 results 同名目录。
