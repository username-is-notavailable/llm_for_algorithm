# Execution-Guided Agentic Code Post-Training

研究 execution feedback、Agentic SFT 与 Agentic GRPO 能否提升小模型利用环境反馈进行
代码自我修复的能力。Qwen3-1.7B 用于低成本 pipeline 和训练验证，Qwen3-4B 是正式训练主模型，
Qwen3-8B 官方后训练模型只作为 inference upper-bound/reference。

M0–M7 保留为环境、verifier、eval、SFT pipeline 和模型容量诊断历史。当前从 M8 开始构建有限
horizon 的 execution-guided Code Agent。本地 WSL 负责开发和 CPU 测试；模型 rollout、训练和
正式评测运行在云端 Linux GPU。

## Code Agent v1

Agent 每轮生成一个动作标签和一份完整 C++17 程序：

```text
<action>execute_code</action> + complete C++
<action>final</action>        + complete C++
```

`execute_code` 最多调用三次，只运行 visible tests 并返回受限的执行反馈；`final` 不返回反馈，
直接用 hidden tests 判定最终结果。三次执行额度用完后再次请求 `execute_code`，该候选会被明确
记录为 auto-final。缺失或非法 action 使用预算感知的兼容回退，同时保留 parse status 供分析。

M8 的内部架构位于 `src/agent/`：模型动作协议、trajectory schema、feedback formatter、可替换
`ExecutionBackend`、同步 controller 和 Agent metrics。`LocalVerifierBackend` 只用于可信代码的
开发测试，不是强安全沙箱。

## M9 Agent baseline

M9 从冻结 LiveCodeBench dev 中确定性选择 60 题，其中原固定 10 题是 smoke 子集。每题按固定
seed 将 tests 拆为 visible/hidden；one-shot 和 Agent final 使用完全相同的 hidden tests。

云端先生成派生数据，再运行 1.7B smoke 对照：

```bash
.third_party/verl/.venv/bin/python scripts/prepare_agent_eval.py
bash scripts/cloud_eval_agent.sh 1.7b smoke both \
  2>&1 | tee /tmp/qwen3-m9-1.7b-smoke.log
```

smoke 通过后可运行 60 题 dev。Agent rollout 是逐题多轮的，可用两张 GPU 做 problem sharding：

```bash
bash scripts/cloud_eval_agent.sh 1.7b dev oneshot \
  2>&1 | tee /tmp/qwen3-m9-1.7b-oneshot-dev.log

CUDA_VISIBLE_DEVICES=0,1 bash scripts/cloud_eval_agent.sh 1.7b dev agent-sharded \
  2>&1 | tee /tmp/qwen3-m9-1.7b-agent-dev.log
```

将 `1.7b` 替换为 `4b` 可使用固定 revision 运行主模型 baseline。单 GPU `agent`/`oneshot`
支持 `--resume OUTPUT_DIR`；sharded 模式保留每个 shard，可分别恢复后再合并。

官方 post-trained 1.7B 作为 Agent protocol reference，使用单独的模型变体名，避免和 Base
baseline 混淆：

```bash
bash scripts/cloud_eval_agent.sh 1.7b-post smoke both \
  2>&1 | tee /tmp/qwen3-m9-1.7b-post-smoke.log
```

官方 post-trained 4B 使用独立的 `4b-post` 变体：

```bash
bash scripts/cloud_eval_agent.sh 4b-post smoke both \
  2>&1 | tee /tmp/qwen3-m9-4b-post-smoke.log
```

诊断模型是否只是被默认 6K 输出上限截断时，可临时覆盖为单轮 8K；Agent 的累计生成预算同时
提高到 32K，后续轮次仍会按模型 context 动态收紧：

```bash
M9_MAX_NEW_TOKENS=8192 \
M9_MAX_TOTAL_GENERATION_TOKENS=32768 \
bash scripts/cloud_eval_agent.sh 1.7b-post smoke both \
  2>&1 | tee /tmp/qwen3-m9-1.7b-post-8k-smoke.log
```

该覆盖只用于诊断，不修改冻结 baseline 配置；产物 experiment ID 带 `long8k` 后缀。

Agent artifacts 位于 `outputs/agent_eval/`，包括 config、environment、逐题完整
`trajectories.jsonl` 和 `metrics.json`。指标包含 first-attempt/Agent/repair success、hidden testcase
pass rate、action validity/fallback、主动 final、execution/token efficiency、termination 和难度分层。
即使某轮被截断或没有可提取代码，原始 response、token 数与 finish reason 也会写入 trajectory，
并在 hidden testcase 汇总中按失败计入分母。

当前 `LocalVerifierBackend` 延续项目原 verifier 的 resource limit，只能用于固定 benchmark 和受控
实验。它不提供文件系统或网络强隔离；正式扩大 rollout 前仍需在云平台接入 isolate/nsjail。

## M10 API repair data pilot

M10 保留 1.7B/4B Base GPU rollout 作为真实 failure producer，不再设置本地 teacher 修复层。失败
样本由阿里云百炼 OpenAI-compatible API 的 `qwen3-8b` 并发修复；每轮候选必须经过本地 verifier，
模型只看到 visible feedback，hidden tests 仅用于接收/拒绝数据。

先将 GPU generation artifacts 和对应的、与 eval 隔离的训练题合并为 failure pool：

```bash
.third_party/verl/.venv/bin/python scripts/build_repair_failure_pool.py \
  --problems data/processed/repair_sft_v1/train_agent.jsonl \
  --generations outputs/failure_rollout/SHARD_*/generations.jsonl \
  --student-model Qwen/Qwen3-4B-Base \
  --exclude-manifest data/splits/agent_eval_v1_problem_ids.json \
  --output data/processed/repair_sft_v1/failure_pool_pilot.jsonl
```

设置密钥并启动 50 条 pilot。密钥只从环境读取，不得写入配置或日志：

```bash
export DASHSCOPE_API_KEY='...'
bash scripts/cloud_generate_repair_api.sh \
  2>&1 | tee /tmp/qwen3-m10-api-repair-pilot.log
```

任务状态保存在 run 目录的 `tasks.sqlite3`，支持中断恢复：

```bash
bash scripts/cloud_generate_repair_api.sh \
  configs/data/m10_repair_api_pilot.yaml \
  --resume outputs/data_generation/RUN_DIR
```

产物分别写入 `accepted.jsonl` 与 `rejected.jsonl`。API 的 reasoning、最终 content、usage 和 request
ID 分开记录；接收要求 full tests 通过、每轮 action 显式有效且最后主动 `final`。

## 本地开发

项目使用两个职责不同、互不混用的环境：

- `.venv`：轻量 CPU 开发环境，只运行配置、数据处理和 verifier 单元测试；
- `.third_party/verl/.venv`：由 `cloud_setup.sh` 管理的完整 GPU/M0/训练环境。

日常本地开发只需：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

不要向 `.venv` 安装 PyTorch、vLLM、FlashAttention 或 verl，也不要手工修改 `.third_party/verl/.venv` 的锁定依赖。

## 云端 smoke test

云端平台无需提供 Docker，但必须提供 Linux、NVIDIA 驱动、`nvidia-smi`、Conda、Git 和支持 C++17 的 `g++`。建议至少预留 50 GB 磁盘空间。

Conda 只提供 bootstrap Python；PyTorch、vLLM、FlashAttention、Transformers 和 verl 由 verl 自己的 uv lock 管理，避免 Conda/pip 混装训练依赖：

```bash
conda create -n qwen3-bootstrap python=3.12 pip -y
conda activate qwen3-bootstrap
bash scripts/cloud_setup.sh
bash scripts/cloud_smoke_test.sh
```

默认固定：

- uv `0.11.16`；
- verl commit `b256ebf83b304d83be5c1207fdf6867c04a0d077`；
- verl 的 `vllm + fsdp` frozen lock。
- socksio `1.0.0`，用于云平台的 SOCKS HTTP 代理。

依赖下载优先使用阿里云 PyPI，官方 PyPI 作为新版本 PyTorch/CUDA 包缺失时的回退。可在运行 setup 前通过 `UV_INDEX` 和 `UV_DEFAULT_INDEX` 覆盖。

可将上传的缓存放在项目目录：

```text
cache/
├── uv/
└── huggingface/
```

`cloud_setup.sh` 和 `cloud_smoke_test.sh` 会分别检测这两个目录。存在时优先使用项目缓存；不存在时回退到 uv 和 Hugging Face 的默认用户缓存。`cache/` 已整体加入 Git 忽略。

从其他机器或容器导出的 uv 缓存可能带有指向旧 `$HOME/.cache/uv` 的绝对符号链接；脚本会在目标文件确实存在时自动将这些链接改写为项目缓存内的相对链接。

正式环境位于 `.third_party/verl/.venv`。重复运行 setup 会复用 uv 下载缓存并将 verl 恢复到固定 commit。需要诊断时可分别运行：

```bash
.third_party/verl/.venv/bin/python scripts/cloud_verify_environment.py
.third_party/verl/.venv/bin/python -m pytest -q
.third_party/verl/.venv/bin/python scripts/smoke_test_model.py \
  --config configs/environment/smoke.yaml
```

如需 Hugging Face 更高下载限额，在执行 smoke test 前设置：

```bash
export HF_TOKEN=<token>
```

运行记录写入 `outputs/experiments/<experiment_id>/`，包括解析后的配置、环境元数据和日志。

## Code Verifier

Milestone 1 提供 C++17 代码提取、编译、受限执行和多测试用例判题：

```python
from src.verifier import extract_code, judge

code = extract_code(model_response)
if code is None:
    raise ValueError("C++ code extraction failed")
result = judge(code, [{"input": "1 2\n", "output": "3\n"}])
print(result.to_dict())
```

执行器使用独立临时目录、进程组超时终止、CPU/内存/core dump/输出大小限制，并在判题结束后清理临时文件。这些措施用于本地与开发阶段的基础防护，不等同于容器、nsjail 或其他强安全隔离，不应在高权限主机上执行任意来源的不可信代码。

训练与生成的标准响应格式已冻结为 Output Protocol v1：`<think>...</think>` 后跟 `cpp` Markdown code block。Verifier 会先过滤完整的 `<think>` 区域，再从剩余输出中选择最长的显式 C++ code block；内部数据 schema、数据源适配和兼容提取规则见 [docs/data.md](docs/data.md)。

## Evaluation Pipeline

Milestone 2 的默认配置使用固定 Qwen revision、Output Protocol v1 prompt 和 7 个手工 toy problems：

```bash
bash scripts/cloud_eval_toy.sh
```

该入口会沿用云端 setup/smoke test 的环境检查和项目内缓存优先策略，并逐题输出进度。也可以直接调用 Python 模块并传入自定义配置。

每次运行生成 `outputs/eval/<experiment_id>/`，包含解析后的 `config.yaml`、`environment.json`、逐 generation 的 `generations.jsonl` 和汇总 `metrics.json`。第一版指标包括 code extraction success rate、compile rate、test pass rate、pass@1 和平均响应字符数；提取失败按该题全部 testcase 失败计入 test pass rate 分母。

## Fixed Eval Set

M3 固定使用 LiveCodeBench `release_v6` 的明确 revision，经 stdin/stdout 兼容过滤和难度分层后得到 399 题正式 eval、101 题 dev，以及 dev 内固定的 10 题 smoke。准备数据并运行云端 smoke：

```bash
.third_party/verl/.venv/bin/python scripts/prepare_eval.py
bash scripts/cloud_eval_fixed_smoke.sh
```

原始数据优先复用 `cache/huggingface`；有序 problem IDs 与选择参数固化在 `data/splits/eval_v1_problem_ids.json`。评测结果同时汇总 overall 与 easy/medium/hard 指标。详细来源、隔离规则和泄漏检查命令见 [docs/data.md](docs/data.md)。

## Base Baseline

M4 冻结模型原生 32K context 和 16K generation cap，正式评测使用 vLLM continuous batching。先在云端运行固定 10 题 smoke：

```bash
bash scripts/cloud_eval_m4.sh smoke
```

通过后运行 399 题正式 baseline：

```bash
bash scripts/cloud_eval_m4.sh full
```

每个请求批次完成后会立即追加 `generations.jsonl`。中断时使用同一配置和输出目录恢复，例如 `bash scripts/cloud_eval_m4.sh full --resume outputs/eval/m4-base-eval-v1-<timestamp>`。本地 WSL 可使用 `local-smoke` 做 2 题短生成技术检查；vLLM/Triton 仍要求可用的 CUDA 开发工具和 Python headers，该检查不产生可比较的 M4 指标。

## SFT Data Pipeline

M5 数据准备提前于 M4 baseline 执行，以实际 token 分布决定统一生成上限和训练上下文；此阶段不启动训练：

```bash
bash scripts/cloud_prepare_sft.sh
```

脚本从固定 revision 的 OpenCodeReasoning-2 C++ 数据中恢复原始题面，执行质量过滤、problem-level 去重、Eval contamination 排除、C++17 编译检查、嵌套 1K/5K/10K 构造和 Qwen tokenizer 长度统计。详细筛选规则与人工 audit 要求见 [docs/data.md](docs/data.md)。

## 里程碑

- M0：仓库与云端环境（已完成）
- M1：Code Verifier（已完成）
- M2：Evaluation Pipeline（已完成）
- M3：Fixed Eval Set（已完成）
- M5：SFT Data Pipeline（已完成并通过 100 条 audit）
- M4：Base Baseline（已完成；pass@1 0.0576）
- 后续阶段见项目方案文档
