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

`execute_code` 最多调用三次，初始只运行 feedback-visible tests 并返回受限的执行反馈。若这些
tests 全过但 private gate 失败，环境最多揭示一条真实失败反例，并将它加入后续 feedback tests；
仍至少保留一条 private test。`final` 不返回反馈，直接用全部 feedback + remaining private tests
判定最终结果。三次执行额度用完后再次请求 `execute_code`，该候选会被明确
记录为 auto-final。缺失或非法 action 使用预算感知的兼容回退，同时保留 parse status 供分析。

M8 的内部架构位于 `src/agent/`：模型动作协议、trajectory schema、feedback formatter、可替换
`ExecutionBackend`、同步 controller 和 Agent metrics。`LocalVerifierBackend` 只用于可信代码的
开发测试，不是强安全沙箱。

## M9 Agent baseline

M9 从冻结 LiveCodeBench dev 中确定性选择 60 题，其中原固定 10 题是 smoke 子集。每题按固定
seed 将 tests 拆为 visible/private；Agent final 始终重新验证二者的完整并集。

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
`trajectories.jsonl` 和 `metrics.json`。指标包含 first-attempt/Agent/repair success、final full-test
pass rate、action validity/fallback、主动 final、execution/token efficiency、termination 和难度分层。
即使某轮被截断或没有可提取代码，原始 response、token 数与 finish reason 也会写入 trajectory，
并在 final full-test 汇总中按失败计入分母。

当前 `LocalVerifierBackend` 延续项目原 verifier 的 resource limit，只能用于固定 benchmark 和受控
实验。它不提供文件系统或网络强隔离；正式扩大 rollout 前仍需在云平台接入 isolate/nsjail。

## M11 Agent SFT smoke

当前默认入口生成对齐真实 Agent loop 的 v3 smoke：初始错误代码作为 context-only assistant turn，
初始 execution feedback 作为独立 tool turn，只有 teacher repair/final 参与 loss。超过三次 execution
预算的轨迹拒绝，现有数据得到 31 train / 8 dev：

```bash
.third_party/verl/.venv/bin/python scripts/prepare_agent_sft_smoke.py
```

训练使用 Qwen3-1.7B-Base、原生多轮 chat template 和 assistant-only loss。两张 A100 40GB 上运行：

```bash
SFT_GPU_COUNT=2 bash scripts/cloud_train_sft.sh agent-smoke \
  2>&1 | tee /tmp/qwen3-m11-agent-sft-smoke.log
```

该 pilot 共 4 epochs，每个 epoch 保存 checkpoint 并计算 8 条固定 dev 的 loss。训练数据位于
`data/processed/agent_sft_v3/`，默认不提交 Git，上传云端时需同时包含 `train.jsonl` 与
`dev_8.jsonl`。

## M10 API repair data pilot

M10 使用官方 post-trained 4B GPU rollout 生成干净的 one-shot 与真实 failure，不再设置本地
teacher 修复层。后续 SFT 初始化仍先实验 Base；这是数据 producer 与训练 student 两个独立字段。
失败样本由阿里云百炼 OpenAI-compatible API 的 `qwen3-8b` 并发修复；每轮候选必须经过本地
verifier，模型只看到 feedback cases；private tests 仅用于 gate，除非按策略迁移一条失败反例。

M10 executable 主源现已迁移到固定 revision 的 CodeContests+。它提供 1x tests、testlib checker
和真实正误提交；published TPR/TNR 只做预筛选，正误标签仍必须通过本地 checker gate。真实错误
提交只作为 repair 起点，teacher 根据 execution feedback 生成 reasoning/action/code target：

```bash
bash scripts/cloud_prepare_codecontests_plus_repair.sh \
  2>&1 | tee /tmp/qwen3-codecontests-plus-prepare-smoke.log
```

当前 smoke 冻结 50 题、1,148 tests 和 50 个真实 failure。旧 TACO-native 200 题实验保留为历史
对照；其复现入口如下：

正式扩充的 300 题数据使用 compact v2，位于
`data/processed/codecontests_plus_repair_300_v2/`。API 配置使用 compact failure pool 时必须同时
提供 `input.problem_dataset` 与 `input.problem_index`；worker 会按 `problem_id` 读取单题环境，
不会将约 3.7 GB 的 checker/testcase 数据整体载入内存。冻结哈希见
`data/splits/codecontests_plus_repair_300_v2_manifest.json`。

构造正式 Agent SFT source 时，先设置百炼 Key，再运行统一入口：

```bash
export DASHSCOPE_API_KEY='...'
bash scripts/prepare_m11_agent_sources.sh \
  2>&1 | tee /tmp/qwen3-m11-agent-sources.log
```

第一阶段会重新通过 checker 验证 300 个 correct seeds，并生成
`one-shot → execute_code → passed feedback → final` 消息；第二阶段使用 `qwen3-8b` 对 300 个真实
错误提交进行可断点续跑的 Agent repair。若 API 任务中断，从输出目录继续：

```bash
bash scripts/cloud_generate_repair_api.sh \
  configs/data/m11_repair_api_codecontests_plus_300_8b.yaml \
  --resume outputs/data_generation/<run-directory>
```

对未解决 repair 更换主 teacher 前，使用固定 20 题 termination-stratified bake-off 比较普通
`qwen3-32b` 与代码专用 `qwen3-coder-next`：

```bash
bash scripts/run_m11_teacher_bakeoff.sh \
  2>&1 | tee /tmp/qwen3-m11-teacher-bakeoff.log
```

两侧共享完全相同的 problem IDs、初始候选、checker、Agent horizon 与 8K 单轮生成上限；Coder
首轮 Coder Next 关闭 thinking，32B 保持 thinking。Coder Next 首轮出现大量仅 action tag 或将长推理
写入 visible content 的响应，因此第二轮在同一20题上只打开 thinking，其他配置保持不变。选择以
full-checker success 为主，并同时比较代码输出率、重复、执行次数、visible response 长度和 API
token 消耗。

若 Coder Next 因自定义 action 协议或算法推理能力未通过 gate，使用相同20题测试高难推理 teacher：

```bash
bash scripts/cloud_generate_repair_api.sh \
  configs/data/m11_repair_api_bakeoff_235b_thinking.yaml
```

该实验保持8K单轮/32K总生成预算不变，仅替换模型为
`qwen3-235b-a22b-thinking-2507` 并启用独立 thinking。

同一20题还可用百炼直供 `deepseek-v4-pro` 做异构 teacher 对照：

```bash
bash scripts/cloud_generate_repair_api.sh \
  configs/data/m11_repair_api_bakeoff_deepseek_v4_pro.yaml
```

该配置使用16K thinking、8K visible output，并采用 DeepSeek V4 官方默认 temperature/top-p 1.0。

固定20题 teacher bake-off 最终选择 `qwen3-235b-a22b-thinking-2507`：16K thinking 下严格成功
11/20、实际 full-checker success 12/20，且全部调用均生成代码。对剩余147条 escalation 使用：

```bash
bash scripts/cloud_generate_repair_api.sh \
  configs/data/m11_repair_api_codecontests_plus_escalation_235b.yaml
```

```bash
HF_HOME="$PWD/cache/m10_source_audit" \
.third_party/verl/.venv/bin/python scripts/prepare_taco_native_sft.py \
  2>&1 | tee /tmp/qwen3-m10-taco-native-prepare.log
```

输出是 `data/processed/repair_sft_native_v1/train_agent_smoke_200.jsonl` 和
`data/splits/repair_train_native_smoke_v1_manifest.json`。当前冻结版本包含 200 个唯一问题和
19,396 个 tests，数据 SHA-256 为
`ec04aba7b92ddddedc500362c8d80c5aeddcb0566c7693143e29545218da73e2`。上传这两个文件后，在两张
A100 40GB 上运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/cloud_generate_m10_native_failures.sh \
  2>&1 | tee /tmp/qwen3-m10-native-4b-rollout.log
```

将生成的 `failure_pool_smoke.jsonl` 与 `one_shot_candidates_smoke.jsonl` 下载回本地后，不需要
GPU 即可对 native-validated failure 运行 8B API repair：

```bash
export DASHSCOPE_API_KEY='...'
bash scripts/cloud_generate_repair_api.sh \
  configs/data/m10_repair_api_native_smoke.yaml \
  2>&1 | tee /tmp/qwen3-m10-native-api-repair.log
```

也可以从同一份 200 题 frozen dataset 运行 verifier-gated one-shot 蒸馏。8B 先处理全部题目，
只有未通过 full tests、无代码、截断、过长或明显重复的响应才进入 32B escalation。API 的私有
`reasoning_content` 仅作为 provenance 保存，SFT target 只使用简短的 visible response：
启动 API 请求前会校验 frozen manifest、dataset SHA-256 和逐题 test split hash，避免对错误或被修改
的数据付费生成。

```bash
export DASHSCOPE_API_KEY='...'
bash scripts/local_generate_m10_distillation.sh \
  2>&1 | tee /tmp/qwen3-m10-native-distillation.log
```

中断后使用原 run 目录恢复，不会重复请求已完成任务：

```bash
bash scripts/local_generate_m10_distillation.sh \
  configs/data/m10_distillation_api_native_smoke.yaml \
  --resume outputs/data_generation/RUN_DIR
```

run 目录包含两个独立 SQLite 队列、各阶段 accepted/rejected、合并后的 `accepted.jsonl` 以及
`metrics.json`。硬生成上限为 8,192 tokens；visible target 估算超过 4,096 tokens 仍会拒绝，
避免把长推演蒸馏给 student。

先从已有 SFT provenance 解析固定 TACO train pilot。该步骤只补取 testcase，并再次执行 eval
fingerprint 检查：

候选必须同时通过 eval fingerprint gate 和 OCR2 reference code 的本地 full-test gate。若已运行
source audit，直接复用其隔离下载缓存继续扫描 600 个候选，直到补足 300 条：

```bash
HF_HOME="$PWD/cache/m10_source_audit" \
.third_party/verl/.venv/bin/python scripts/prepare_repair_train.py \
  2>&1 | tee /tmp/qwen3-m10-prepare-clean-300.log
```

该命令会覆盖 `train_agent_pilot.jsonl` 和对应 manifest；旧 rollout artifacts 保持不变，但不得与
新 manifest 混用。manifest 会记录 reference gate、检查数量及各拒绝原因。

在四张 GPU 上按 problem 分片运行官方 post-trained 4B：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/cloud_generate_m10_failures.sh \
  2>&1 | tee /tmp/qwen3-m10-4b-rollout.log
```

两张 A100 40GB 使用并发 8 的 throughput profile；脚本仍按两路 problem shard 运行：

```bash
M10_FAILURE_CONFIG=configs/eval/m10_qwen3_4b_posttrained_failure_pilot_40gb_v1.yaml \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/cloud_generate_m10_failures.sh \
  2>&1 | tee /tmp/qwen3-m10-4b-rollout.log
```

先将 GPU generation artifacts 和对应的、与 eval 隔离的训练题合并为 failure pool：

```bash
.third_party/verl/.venv/bin/python scripts/build_repair_failure_pool.py \
  --problems data/processed/repair_sft_v1/train_agent.jsonl \
  --generations outputs/failure_rollout/SHARD_*/generations.jsonl \
  --producer-model Qwen/Qwen3-4B \
  --exclude-manifest data/splits/agent_eval_v1_problem_ids.json \
  --output data/processed/repair_sft_v1/failure_pool_pilot.jsonl \
  --one-shot-output data/processed/repair_sft_v1/one_shot_candidates_pilot.jsonl
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

若怀疑 OCR2/TACO 本地缓存或 source testcase 损坏，先停止 API 调用。以下命令只删除 OCR2、
TACO 的 hub 缓存入口和专用审计缓存，不删除模型或其他数据集；随后在全新隔离缓存中重新下载固定
TACO revision、逐题比较 fresh/frozen testcase，并执行 OCR2 teacher code：

```bash
bash scripts/redownload_audit_m10_sources.sh --purge \
  2>&1 | tee /tmp/qwen3-m10-source-audit.log
```

审计报告和 reference-full-pass 子集分别写到
`data/processed/repair_sft_v1/source_audit/audit_report.json` 与
`train_agent_reference_full_pass.jsonl`。只有 fresh testcase 与 frozen 完全一致、且 reference code
本地 full-pass 的题目才允许进入后续 failure rollout。

8B pilot 完成后，将通过 full tests 但缺失 action 的输出确定性规范化；规范化只补齐 pipeline
实际采用的 action，不修改代码、reasoning 或执行结果，并显式记录 provenance。真正失败的轨迹以
可见测试表现最好的 8B 候选为起点交给 32B 接力：

```bash
.third_party/verl/.venv/bin/python scripts/postprocess_repair_api.py \
  --run outputs/data_generation/M10_8B_RUN \
  --failure-pool data/processed/repair_sft_v1/failure_pool_pilot.jsonl \
  --canonical-output data/processed/repair_sft_v1/repair_api_8b_canonical.jsonl \
  --escalation-output data/processed/repair_sft_v1/failure_pool_escalation_32b.jsonl

bash scripts/cloud_generate_repair_api.sh \
  configs/data/m10_repair_api_escalation_32b.yaml \
  2>&1 | tee /tmp/qwen3-m10-api-repair-32b.log
```

32B 后仍失败的任务先固定抽取 10 条、按 difficulty 轮转平衡，再用 code-specialized teacher
做小规模 bake-off，禁止直接在完整失败池上试错：

```bash
.third_party/verl/.venv/bin/python scripts/prepare_repair_bakeoff.py \
  --input data/processed/repair_sft_v1/failure_pool_after_32b.jsonl \
  --output data/processed/repair_sft_v1/failure_pool_bakeoff_10.jsonl \
  --size 10

bash scripts/cloud_generate_repair_api.sh \
  configs/data/m10_repair_api_bakeoff_coder_next.yaml \
  2>&1 | tee /tmp/qwen3-m10-coder-next-bakeoff.log
```

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
