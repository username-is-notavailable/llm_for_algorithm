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
- 2026-08-21，云端原生 Linux GPU 最终验收：
  - Experiment ID：`m0-qwen3-smoke-20260821-122539`；
  - Git commit：`e1218fa6afd71f926abc7d02c54424646b46cfbf`；
  - 配置：`configs/environment/smoke.yaml`；
  - GPU：NVIDIA A100-PCIE-40GB，42,405,855,232 bytes；
  - NVIDIA driver：590.48.01；PyTorch CUDA runtime：13.0；
  - Python 3.12.13；PyTorch 2.11.0+cu130；Transformers 5.5.3；
  - verl 0.10.0.dev0，commit `b256ebf83b304d83be5c1207fdf6867c04a0d077`；
  - vLLM 0.24.0；FlashAttention 2.8.3；socksio 1.0.0；
  - `scripts/cloud_verify_environment.py`、3 个 CPU 单元测试和 Qwen3-0.6B-Base 生成测试全部通过；
  - CUDA allocated memory：1,200,636,928 bytes；峰值：1,214,435,328 bytes；
  - 产物：`outputs/experiments/m0-qwen3-smoke-20260821-122539/`；
  - 结论：M0 验收通过，无阻塞问题。

## Milestone 1

- 2026-08-21，Code Verifier 云端验收：
  - Git commit：`4b74008`；
  - 环境：Linux，Python 3.12.13，g++ C++17；
  - 定向测试：`tests/test_code_extraction.py` 与 `tests/test_verifier.py`，14 passed in 1.55s；
  - 项目全量测试：17 passed in 3.78s；
  - 覆盖：Output Protocol v1 提取、正确代码、Compile Error、Wrong Answer、Runtime Error、TLE、输出上限、多 testcase 和临时目录清理；
  - 结论：M1 验收通过，无阻塞问题。

## Milestone 2

- 2026-08-21，Evaluation Pipeline 云端验收：
  - Experiment ID：`m2-eval-toy-20260821-161401`；
  - Git commit：`d416643`；
  - 配置：`configs/eval/default.yaml`；
  - 项目全量测试：24 passed in 3.66s；
  - 数据：7 个手工 C++ toy problems，每题生成 1 个响应；
  - code extraction success rate：1.0；compile rate：1.0；
  - test pass rate：0.5882352941176471（10/17 testcase）；
  - pass@1：0.5714285714285714（4/7 problems）；
  - average response length：590.2857142857143 characters；
  - 产物：`outputs/eval/m2-eval-toy-20260821-161401/`，包含配置、环境、逐样本结果和汇总指标；
  - 结论：M2 验收通过，生成、提取、编译、判题、指标和实验产物链路完整，无阻塞问题。
