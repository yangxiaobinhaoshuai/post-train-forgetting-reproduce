# 机器检查

检查日期：2026-11-17 UTC，目录 /root/projects/forgetting；初始目录为空。

| 项目 | 实测 |
|---|---|
| GPU | NVIDIA RTX 4090，24564 MiB，检查时无训练进程 |
| 驱动 | 610.57.04；nvidia-smi 标示 CUDA 能力 13.3 |
| PyTorch | 2.13.0+cu132，CUDA build 13.2 |
| GPU 验证 | CUDA 可用、BF16 支持；512×512 BF16 GPU 矩阵乘法成功 |
| CPU | AMD EPYC 7542；在线逻辑 CPU 0–13，共 14 个；不要按主机报告的 128 个计预算 |
| CPU cgroup | cpu.max = max 100000，未发现该层 CPU 配额上限 |
| 内存 | 64 GiB，约 63 GiB available；cgroup 上限 64 GiB，无 swap |
| 磁盘 | 根 overlay 50 GiB，约 48 GiB 可用；项目位于此盘 |
| /dev/shm | 32 GiB |
| Python | 3.12.11 |
| 依赖 | torch 已装；transformers / datasets / peft / trl / accelerate / bitsandbytes 未装 |

这是可行性检查，不是 LLM 吞吐测试。未验证完整模型加载、反向传播、优化器状态显存或模型/数据下载速度。正式实验前需补齐这些检查，并保存 pip freeze、模型 revision、数据 revision 和 GPU 峰值显存。
