# 格式 vs 能力探针

单 seed、GSM 200 题、贪心解码、batch 16。主指标为宽松准确率（取输出最后一个数）；格式遵循 = 能按该格式要求解析出答案。

## 宽松准确率 %（格式遵循 %）

| 模型 | hash | boxed | answer_is | plain | direct |
|---|---:|---:|---:|---:|---:|
| base | 71.0 (0) | 76.0 (97) | 68.5 (92) | 73.0 (100) | 8.5 (96) |
| collapsed | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) |
| recover512 | 48.0 (100) | 49.5 (41) | 49.5 (100) | 47.5 (99) | 39.5 (2) |
| base_math512 | 51.5 (100) | 52.0 (24) | 49.0 (99) | 53.0 (100) | 18.0 (60) |

## GSM 训练风格（输出含 `<<算式>>`）%

| 模型 | hash | boxed | answer_is | plain | direct |
|---|---:|---:|---:|---:|---:|
| base | 0 | 0 | 0 | 0 | 0 |
| collapsed | 0 | 0 | 0 | 0 | 0 |
| recover512 | 98 | 95 | 98 | 95 | 91 |
| base_math512 | 100 | 94 | 100 | 98 | 38 |

## 逐题重合（同一格式内；phi 为对错相关系数）

- hash：base vs recover512：都对 81（独立期望 68.2），P(y对|x对) 0.57，P(y对|x错) 0.259，phi 0.283
- hash：base vs base_math512：都对 85（独立期望 73.1），P(y对|x对) 0.599，P(y对|x错) 0.31，phi 0.262
- hash：base_math512 vs recover512：都对 75（独立期望 49.4），P(y对|x对) 0.728，P(y对|x错) 0.216，phi 0.512
- boxed：base vs recover512：都对 89（独立期望 75.2），P(y对|x对) 0.586，P(y对|x错) 0.208，phi 0.322
- boxed：base vs base_math512：都对 88（独立期望 79.0），P(y对|x对) 0.579，P(y对|x错) 0.333，phi 0.21
- boxed：base_math512 vs recover512：都对 76（独立期望 51.5），P(y对|x对) 0.731，P(y对|x错) 0.24，phi 0.491
- answer_is：base vs recover512：都对 80（独立期望 67.8），P(y对|x对) 0.584，P(y对|x错) 0.302，phi 0.262
- answer_is：base vs base_math512：都对 79（独立期望 67.1），P(y对|x对) 0.577，P(y对|x错) 0.302，phi 0.256
- answer_is：base_math512 vs recover512：都对 74（独立期望 48.5），P(y对|x对) 0.755，P(y对|x错) 0.245，phi 0.51
- plain：base vs recover512：都对 85（独立期望 69.3），P(y对|x对) 0.582，P(y对|x错) 0.185，phi 0.353
- plain：base vs base_math512：都对 95（独立期望 77.4），P(y对|x对) 0.651，P(y对|x错) 0.204，phi 0.398
- plain：base_math512 vs recover512：都对 73（独立期望 50.4），P(y对|x对) 0.689，P(y对|x错) 0.234，phi 0.454
- direct：base vs recover512：都对 10（独立期望 6.7），P(y对|x对) 0.588，P(y对|x错) 0.377，phi 0.12
- direct：base vs base_math512：都对 6（独立期望 3.1），P(y对|x对) 0.353，P(y对|x错) 0.164，phi 0.137
- direct：base_math512 vs recover512：都对 27（独立期望 14.2），P(y对|x对) 0.75，P(y对|x错) 0.317，phi 0.34
