# 格式 vs 能力探针

单 seed、GSM 200 题、贪心解码、batch 16。主指标为宽松准确率（取输出最后一个数）；格式遵循 = 能按该格式要求解析出答案。

## 宽松准确率 %（格式遵循 %）

| 模型 | hash | boxed | answer_is | plain | direct |
|---|---:|---:|---:|---:|---:|
| base | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (75) |
| collapsed | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) |
| recover512 | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (100) | 0.0 (0) |
| base_math512 | 25.0 (0) | 0.0 (0) | 0.0 (25) | 0.0 (25) | 25.0 (100) |

## GSM 训练风格（输出含 `<<算式>>`）%

| 模型 | hash | boxed | answer_is | plain | direct |
|---|---:|---:|---:|---:|---:|
| base | 0 | 0 | 0 | 0 | 0 |
| collapsed | 0 | 0 | 0 | 0 | 0 |
| recover512 | 75 | 50 | 50 | 50 | 25 |
| base_math512 | 25 | 0 | 25 | 25 | 0 |

## 逐题重合（同一格式内；phi 为对错相关系数）

- hash：base vs recover512：都对 0（独立期望 0.0），P(y对|x对) None，P(y对|x错) 0.0，phi None
- hash：base vs base_math512：都对 0（独立期望 0.0），P(y对|x对) None，P(y对|x错) 0.25，phi None
- hash：base_math512 vs recover512：都对 0（独立期望 0.0），P(y对|x对) 0.0，P(y对|x错) 0.0，phi None
- boxed：base vs recover512：都对 0（独立期望 0.0），P(y对|x对) None，P(y对|x错) 0.0，phi None
- boxed：base vs base_math512：都对 0（独立期望 0.0），P(y对|x对) None，P(y对|x错) 0.0，phi None
- boxed：base_math512 vs recover512：都对 0（独立期望 0.0），P(y对|x对) None，P(y对|x错) 0.0，phi None
- answer_is：base vs recover512：都对 0（独立期望 0.0），P(y对|x对) None，P(y对|x错) 0.0，phi None
- answer_is：base vs base_math512：都对 0（独立期望 0.0），P(y对|x对) None，P(y对|x错) 0.0，phi None
- answer_is：base_math512 vs recover512：都对 0（独立期望 0.0），P(y对|x对) None，P(y对|x错) 0.0，phi None
- plain：base vs recover512：都对 0（独立期望 0.0），P(y对|x对) None，P(y对|x错) 0.0，phi None
- plain：base vs base_math512：都对 0（独立期望 0.0），P(y对|x对) None，P(y对|x错) 0.0，phi None
- plain：base_math512 vs recover512：都对 0（独立期望 0.0），P(y对|x对) None，P(y对|x错) 0.0，phi None
- direct：base vs recover512：都对 0（独立期望 0.0），P(y对|x对) None，P(y对|x错) 0.0，phi None
- direct：base vs base_math512：都对 0（独立期望 0.0），P(y对|x对) None，P(y对|x错) 0.25，phi None
- direct：base_math512 vs recover512：都对 0（独立期望 0.0），P(y对|x对) 0.0，P(y对|x错) 0.0，phi None
