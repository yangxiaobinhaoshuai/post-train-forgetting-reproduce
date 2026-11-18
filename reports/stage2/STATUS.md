# Stage2 运行状态

快速集曲线和完整集终点分别汇总；partial 表示尚未完成。

| Run | 状态 |
|---|---|
| full_lr1e4 | complete |
| lora_lr1e4 | complete |
| full_lr2e5 | complete |
| recover32 | complete |
| recover128 | complete |
| recover512 | complete |
| base_math512 | complete |
| continue_ag | complete |

输出：quick_curve.csv、full_endpoints.csv、sensitivity.png、recovery.png（有对应数据时生成）。
完整原始输出在同名 results 目录。未完成的 run 只作进度展示，不能当作最终结果。
