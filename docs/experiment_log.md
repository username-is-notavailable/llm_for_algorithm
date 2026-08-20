# Experiment Log

每次云端实验记录：experiment ID、Git commit、配置路径、GPU/CUDA/依赖版本、命令、结果与异常。

## Milestone 0

- 2026-08-20，本地 WSL + Docker Desktop 验证（旧环境路径，现已退役）：
  - Git commit：`f1aa0c2`（`dtype` 参数修复后）；
  - GPU：NVIDIA GeForce RTX 4060 Ti，16 GiB；
  - NVIDIA KMD：610.74；CUDA UMD：13.3；PyTorch CUDA runtime：13.0；
  - Python 3.12.3；PyTorch 2.11.0+cu130；Transformers 5.5.3；
  - verl 0.10.0.dev0，commit `b256ebf83b304d83be5c1207fdf6867c04a0d077`；
  - vLLM 0.24.0；FlashAttention 2.8.3；
  - 3 个 CPU 单元测试通过；依赖 import、CUDA tensor 和 Qwen3-0.6B-Base 生成测试通过；
  - 峰值 CUDA allocated memory：1,214,435,328 bytes；
  - 产物：`outputs/experiments/m0-qwen3-smoke-20260820-072035/`（本地忽略，不提交）；
  - 已知问题：该次容器只挂载 outputs，导致 `environment.json` 的 `git_commit` 为 `null`。
- 实施调整：目标云 GPU 平台不提供 Docker，M0 改为 Conda bootstrap + verl uv frozen lock；Docker 文件已移除。
- 云端原生 Linux GPU：等待重新执行 `bash scripts/cloud_setup.sh` 与 `bash scripts/cloud_smoke_test.sh` 后补充结果。
