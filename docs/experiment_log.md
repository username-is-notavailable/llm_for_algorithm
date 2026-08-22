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

## Milestone 3

- 2026-08-21，Fixed Eval Set 云端 smoke 验收：
  - Experiment ID：`m3-livecodebench-smoke-v1-20260821-193449`；
  - Git commit：`a4adce6`；
  - 数据源：`livecodebench/code_generation_lite`，官方 `release_v6`，revision `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`；
  - 固定 split：399 eval、101 dev；smoke 从 dev 固定选择 10 题（easy 3、medium 3、hard 4）；
  - 项目全量测试：31 passed in 4.94s；
  - smoke：10 generations，code extraction success rate 1.0，compile rate 0.5，test pass rate 0.1794871794871795，pass@1 0.1；
  - 分难度 pass@1：easy 0.3333333333333333，medium 0.0，hard 0.0；
  - average response length：2,214.2 characters；
  - 错误分布：5 compile errors、4 wrong answers、1 pass；编译失败均为模型生成被截断、缺少 `main` 或错误 C++ API，不是 verifier/环境故障；
  - 产物：`outputs/eval/m3-livecodebench-smoke-v1-20260821-193449/`；
  - 结论：M3 验收通过，固定 split、难度分层、manifest 校验、数据隔离和 10 题完整 Eval 链路均正常。

## Milestone 4

- 2026-08-22，Qwen3-0.6B Base baseline 云端正式评测：
  - Experiment ID：`m4-base-eval-v1-20260822-105643`；Git commit：`074615d`；
  - 模型：`Qwen/Qwen3-0.6B-Base` revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`；
  - 协议：原生 32,768-token context、16,384-token generation cap、greedy decoding、pass@1；
  - 数据：固定 `eval_v1` 399 题；399 generations 和 399 unique problems 均完整；
  - 后端：vLLM 0.24 continuous batching；PyTorch 2.11.0+cu130；NVIDIA A100-PCIE-40GB；
  - overall：extraction rate 0.8521303258、compile rate 0.4937343358、test pass rate 0.1344578783、pass@1 0.0576441103（23/399）；
  - 分难度 pass@1：easy 0.1885245902（23/122）、medium 0.0（0/115）、hard 0.0（0/162）；
  - 错误分布：23 pass、146 wrong answer、143 compile error、59 extraction failure、16 runtime error、12 timeout；
  - 输出：平均 5,317.08 tokens；共 2,121,513 output tokens；273 stop、126 length；31.58% 的生成触及 16K cap；
  - 耗时：109m47.242s，平均聚合吞吐约 322.06 output tokens/s；
  - 产物：`outputs/eval/m4-base-eval-v1-20260822-105643/`；
  - 结论：M4 验收通过并冻结为后续 SFT/GRPO 的 Base baseline。16K 截断主要反映 Base 模型在失败题上的长生成或退化，不继续扩大评测上限；后续模型仍使用完全相同协议比较。

## Milestone 5

- 2026-08-22，OpenCodeReasoning-2 C++ SFT Data v1 准备与 audit 完成：
  - 固定 OCR2 revision：`eadf535931451525f3e5621d0f960c240bc62fd9`；完整扫描 1,174,475 rows；
  - 10K：10,000 unique problems，全部 verified，严格嵌套生成 1K/5K/10K；
  - 质量过滤新增 25 条总长超过 16,384 tokens、14 条 interactive task、1 条缺失题面；禁止截断末尾代码；
  - 长度：total p50 5,333、p90 12,093、p95 13,319、p99 14,953、max 16,369；
  - 数据隔离：拒绝 11 条 Eval near-duplicate，最终 retained Eval matches 为 0；
  - 固定 100 条人工 audit 完成：无剩余阻塞项；详细记录见 `docs/sft_v1_audit.md`；
  - 最终 SFT-10K SHA-256：`16d25b5ad5780b4b5925a6a504210c11c7d39f35b535e7c783e6f3e9398a3581`；
  - 结论：M5 验收通过，数据协议冻结，可以进入 M6 SFT smoke test。

## Milestone 6

- 2026-08-22，SFT smoke、checkpoint 与 DDP 吞吐云端验收：
  - Git commit：`47bb4f6`；GPU：NVIDIA A100-PCIE-40GB；训练方式：full-parameter bf16、FlashAttention 2、response-only loss、gradient checkpointing；
  - overfit：从 Base 在 64 条最短 SFT-1K 样本上训练 100 steps（global batch 8），loss 从约 1.25--1.40 降至 0.57 左右，整体 train loss 0.730369；无 NaN、OOM 或梯度爆炸；
  - checkpoint：`checkpoint-75` 和 `checkpoint-100` 均包含 model、optimizer、scheduler、RNG 与 Trainer state；final checkpoint 可重新加载并生成 reasoning + C++ code block；
  - resume：成功从 checkpoint-75 恢复并完成 step 76--100，续跑 loss 与原始轨迹一致；
  - 固定 global batch 16 的 30-step 吞吐测试：1 GPU 326.017s / 8,725 tokens/s，2 GPU 201.753s / 约 14,100 tokens/s，4 GPU 127.858s / 约 22,250 tokens/s；对应相对单卡加速 1.00x / 1.62x / 2.55x；
  - 峰值训练显存约 29GB/卡，A100 40GB 满足 16,384-token 上限；
  - 正式训练冻结为 2-GPU DDP：相对单卡约 1.62x 加速、约 81% 并行效率，在墙钟时间与 GPU 总成本间更均衡；
  - 产物：`outputs/training/m6-sft-overfit-smoke-20260822-153152/` 及三组 `m6-sft-throughput-*`；
  - 结论：M6 验收通过，可以从原始 Base 独立开始 M7 SFT-1K。

## Milestone 7

- 2026-08-22，M7-v1 SFT-1K 首次正式训练判定失败：
  - Git commit：`b719d45`（训练准备），后续使用 `baacbe5` 启用 expandable CUDA allocator segments 后完成训练；
  - 配方：固定 Base → 原始 SFT-1K，full-parameter bf16，2-GPU DDP，global batch 16，3 epochs，peak LR 2e-5；
  - 原始 SFT-1K response 长度：P50 4,469、P90 10,940、max 15,670、mean 5,339.9 tokens；
  - final smoke：10 题中 9 条触及 16,384-token generation cap，平均 14,939.7 tokens，extraction rate 0.1、compile rate 0.1、pass@1 0；
  - checkpoint-63（首 epoch）smoke：10/10 全部触及 16,384-token cap，extraction/compile/pass@1 均为 0；说明退化从首 epoch 已发生，不能通过 early stopping 修复；
  - 诊断：训练/评测 prompt 一致，response target 包含 EOS，tokenizer/model/generation EOS 均为 151643；问题不是停止符遗漏或评测后端，而是超长 reasoning token 主导 0.6B 模型的 SFT，导致 `<think>` 重复循环且无法稳定闭合到最终代码；
  - 对照：M6 使用最短 64 条样本时能正常停止并输出 C++，进一步支持长度配方是主要变量；
  - 结论：M7-v1 checkpoint 全部拒绝进入正式 399 题评测，产物保留为失败实验。下一步从 Base 独立运行仅改变 response 长度上限的 M7-v2 pilot。
