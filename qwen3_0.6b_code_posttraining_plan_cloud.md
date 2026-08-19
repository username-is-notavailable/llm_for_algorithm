# Qwen3-0.6B Code Post-Training 项目计划

> 目标：以 **Qwen3-0.6B-Base** 为起点，在 **云端 Linux + NVIDIA GPU** 环境中完成训练与正式评测，在本地 Windows 环境中完成代码开发、数据处理和不依赖 GPU 的测试，建立一个可复现的代码推理后训练项目。第一阶段优先建立可靠的 **Eval → SFT → Eval → GRPO → Eval** 闭环，而不是追求大规模训练或 SOTA。
>
> 核心研究问题：**SFT 与 execution-verifiable
> RL（GRPO）分别能在多大程度上提升小型语言模型的算法题求解能力？这种提升与数据规模、题目难度和初始策略能力有什么关系？**

------------------------------------------------------------------------

## 1. 项目原则

Codex 在执行本项目时应遵循以下原则：

1.  **一次只完成一个 Milestone。** 每完成一个 Milestone
    后先运行测试、报告结果，再进入下一阶段。
2.  **先正确，再优化。** 第一版优先保证 pipeline
    正确、可复现、可测试，不提前做复杂性能优化。
3.  **Eval 优先。** 在任何训练之前先建立并冻结基础评测协议，并得到 Base
    Model baseline。
4.  **数据严格隔离。** SFT、GRPO、Eval 必须按照 `problem_id` 做
    problem-level 隔离，并尽可能进行跨数据源去重。
5.  **所有实验配置化。** 模型、数据路径、prompt、generation
    参数、训练参数等不得散落在代码中。
6.  **所有实验可复现。** 保存 config、seed、git
    commit、环境信息、checkpoint 和 metrics。
7.  **云端训练优先。** 不为了兼容本地 Windows 或 16GB 显存而牺牲训练方案。涉及 Qwen GPU 推理、SFT、verl、vLLM rollout、GRPO 和正式 Eval 时，默认使用云端 Linux NVIDIA GPU。
8.  **本地与云端职责分离。** 本地负责仓库开发、数据处理、静态检查、单元测试、结果分析；云端负责 GPU 相关训练与评测。
9.  **不要提前加入非必要功能。** Phase 1 暂不加入洛谷爬虫、LLM
    Judge、复杂 reasoning reward、大模型训练等。
10. **训练代码和评测代码解耦。** Eval pipeline 不依赖具体训练框架。
11. **Verifier 是核心基础设施。** GRPO reward 与 Eval
    尽可能复用同一套代码提取、编译和执行逻辑。

------------------------------------------------------------------------

# 2. 开发与运行环境划分

## 本地 Windows

本地机器只承担不依赖正式 GPU 训练环境的工作：

- 编写和 review Python / shell / config；
- 数据下载、清洗、schema 转换、dedup 和 contamination 检查；
- Verifier 的纯 CPU 单元测试；
- toy problem 的编译/执行测试；
- 分析云端产生的 JSONL、metrics 和训练日志；
- Git 提交和实验文档维护。

**不要要求 Codex 为 verl、vLLM 或训练框架增加 Windows 兼容层。**

## 云端 Linux GPU

以下工作默认只在云端 Linux NVIDIA GPU 实例执行：

- Qwen3-0.6B-Base GPU inference；
- Base benchmark；
- SFT smoke test / SFT-1K / SFT-5K / SFT-10K；
- verl 安装和验证；
- vLLM / rollout engine；
- GRPO difficulty filtering；
- GRPO smoke / pilot / full；
- 正式 LiveCodeBench evaluation。

## 云端实例原则

第一阶段不固定某个云厂商或 GPU 型号。选择实例时优先：

1. Linux + NVIDIA CUDA 环境成熟；
2. 显存能让当前实验以简单 recipe 运行；
3. 支持持久磁盘或方便同步 checkpoint / dataset；
4. 按量计费，先短时 smoke test 再长时间训练；
5. 不为了省极少费用引入复杂 offload 或 Windows 兼容工作。

每次创建新的训练实例后，必须记录：

```text
GPU model
GPU count
VRAM
OS
NVIDIA driver
CUDA runtime
Python
PyTorch
Transformers
verl commit/version
vLLM version
FlashAttention version（若使用）
```

并保存到 experiment metadata。

------------------------------------------------------------------------

# 3. 最终实验路线

``` text
Qwen3-0.6B-Base
       │
       ├──────────────→ Eval → Base Baseline
       │
       ├─ SFT-1K ─────→ Eval
       │
       ├─ SFT-5K ─────→ Eval
       │
       └─ SFT-10K ────→ Eval（仅当 5K 尚未明显饱和）
                              │
                              ▼
                      选择 SFT-best
                              │
                              ▼
                  GRPO Candidate Pool
                              │
                       rollout 难度估计
                              │
                              ▼
                       GRPO-Debug-100
                              │
                              ▼
                       GRPO-Pilot-500
                              │
                              ▼
                     GRPO-Full-2K~5K
                              │
                              ▼
                         Final Eval
```

主要对比：

  Model             目的
  ----------------- -----------------------------
  Qwen3-0.6B-Base   原始 pretrained baseline
  Base + SFT-1K     小规模 SFT 效果
  Base + SFT-5K     SFT data scaling
  Base + SFT-10K    可选，检查 SFT 是否继续受益
  SFT-best + GRPO   验证 RLVR 是否进一步提升

------------------------------------------------------------------------

# 4. 推荐项目结构

Codex 首先创建如下目录；后续如有充分理由可以调整，但需要保持职责清晰。

``` text
qwen3-code-posttraining/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── eval/
│   ├── sft/
│   └── grpo/
│
├── src/
│   ├── data/
│   │   ├── schemas.py
│   │   ├── preprocess_sft.py
│   │   ├── preprocess_rl.py
│   │   ├── dedup.py
│   │   └── contamination.py
│   │
│   ├── inference/
│   │   ├── generate.py
│   │   └── prompts.py
│   │
│   ├── verifier/
│   │   ├── extract_code.py
│   │   ├── compiler.py
│   │   ├── executor.py
│   │   └── judge.py
│   │
│   ├── eval/
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   └── pass_at_k.py
│   │
│   ├── training/
│   │   ├── sft/
│   │   └── grpo/
│   │
│   └── utils/
│       ├── logging.py
│       ├── reproducibility.py
│       └── experiment.py
│
├── scripts/
│   ├── prepare_eval.sh
│   ├── eval_base.sh
│   ├── prepare_sft.sh
│   ├── train_sft.sh
│   ├── prepare_grpo.sh
│   ├── train_grpo.sh
│   └── cloud_smoke_test.sh
│
├── tests/
│   ├── test_code_extraction.py
│   ├── test_executor.py
│   ├── test_metrics.py
│   └── test_data_pipeline.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── outputs/
│   ├── generations/
│   ├── eval/
│   ├── checkpoints/
│   └── experiments/
│
└── docs/
    ├── experiment_log.md
    ├── data.md
    └── findings.md
```

`data/`、`outputs/` 和模型 checkpoint 默认不提交 Git。

------------------------------------------------------------------------

# 5. 统一数据 Schema

## 5.1 SFT Schema

建议内部统一为 JSONL：

``` json
{
  "problem_id": "source:12345",
  "source": "OpenCodeReasoning2",
  "problem": "...",
  "difficulty": "medium",
  "tags": ["dp"],
  "reasoning": "...",
  "code": "...",
  "language": "cpp",
  "verified": true,
  "metadata": {}
}
```

训练前再转换成具体框架需要的 message/chat 格式。

------------------------------------------------------------------------

## 5.2 GRPO / Verifiable Problem Schema

``` json
{
  "problem_id": "source:12345",
  "source": "TACO-Verified",
  "problem": "...",
  "difficulty": "medium",
  "tags": ["greedy"],
  "language": "cpp",
  "tests": [
    {
      "input": "...",
      "output": "..."
    }
  ],
  "metadata": {}
}
```

GRPO 数据**不要求标准 reasoning response**。

核心是：

``` text
Problem + Reliable Tests
```

------------------------------------------------------------------------

## 5.3 Generation Result Schema

所有 Eval 和 rollout 都保存原始结果：

``` json
{
  "experiment_id": "...",
  "problem_id": "...",
  "sample_id": 0,
  "prompt": "...",
  "raw_response": "...",
  "extracted_code": "...",
  "compiled": true,
  "passed_tests": 7,
  "total_tests": 10,
  "reward": 0.7,
  "error_type": null,
  "response_tokens": 1024
}
```

禁止只保存最终 aggregate metric。

------------------------------------------------------------------------

# 6. Milestone 0 --- Repository & Cloud Environment

## Codex 任务

-   [ ] 初始化项目目录。
-   [ ] 创建 Python 环境配置。
-   [ ] 安装 PyTorch、Transformers、Datasets 等基础依赖。
-   [ ] 安装/配置 verl，但不要立即写复杂 GRPO 代码。
-   [ ] 验证 CUDA 可用。
-   [ ] 验证 Qwen3-0.6B-Base 可以加载。
-   [ ] 完成一次简单 inference。
-   [ ] 输出当前 GPU、CUDA、PyTorch、Transformers、verl 版本。
-   [ ] 创建统一 config 读取机制。
-   [ ] 设置随机种子。
-   [ ] 建立基础日志系统。

## Acceptance Criteria

``` text
python scripts/smoke_test_model.py
```

可以：

1.  在云端 Linux GPU 上加载 Qwen3-0.6B-Base；
2.  成功生成文本；
3.  打印 GPU 型号、显存占用和关键软件版本；
4.  verl/vLLM 的最小环境检查通过；
5.  无异常退出。

完成后**停止并报告**，不要自动进入下一个 Milestone。

------------------------------------------------------------------------

# 7. Milestone 1 --- Code Verifier

这是项目最重要的基础设施之一。

## 6.1 Code Extraction

实现：

``` text
model response
      ↓
extract_code()
      ↓
C++ source code
```

需要处理：

-   Markdown `cpp` code block；
-   `c++` code block；
-   普通 code block；
-   `<answer>` 等目标格式；
-   response 中没有 code block；
-   多个 code block；
-   malformed response。

## 6.2 Compiler

实现 C++17 编译：

``` text
source.cpp
   ↓
g++ -std=c++17
   ↓
binary / compilation error
```

记录：

-   compile success；
-   compiler stderr；
-   compilation timeout。

## 6.3 Executor

执行 binary：

-   输入 stdin；
-   捕获 stdout/stderr；
-   timeout；
-   runtime error；
-   non-zero exit code；
-   输出长度限制；
-   临时目录清理。

安全方面第一阶段至少保证：

-   独立临时目录；
-   timeout；
-   process kill；
-   resource limit；
-   不直接信任模型输出。

后续再考虑 Docker/nsjail 等更严格 sandbox。

## 6.4 Judge

统一接口：

``` python
result = judge(code, test_cases)
```

输出至少包括：

``` json
{
  "compiled": true,
  "passed": 7,
  "total": 10,
  "pass_rate": 0.7,
  "runtime_error": false,
  "timeout": false,
  "error_type": null
}
```

## Acceptance Criteria

建立单元测试覆盖：

-   [ ] 正确代码；
-   [ ] Compile Error；
-   [ ] Wrong Answer；
-   [ ] Runtime Error；
-   [ ] Infinite Loop / TLE；
-   [ ] 多 testcase；
-   [ ] code extraction failure。

所有测试通过后停止并报告。

------------------------------------------------------------------------

# 8. Milestone 2 --- Evaluation Pipeline

## Codex 任务

实现统一：

``` text
Dataset
   ↓
Prompt Builder
   ↓
Model Generation
   ↓
Code Extraction
   ↓
Verifier
   ↓
Metrics
   ↓
JSONL Results + Summary
```

## 第一版 Metrics

必须包含：

-   Compile Rate
-   Code Extraction Success Rate
-   Test Pass Rate
-   pass@1
-   Average Response Length

支持多 sample 后增加：

-   pass@k
-   pass@5

## Evaluation Protocol

所有实验尽量固定：

-   prompt template；
-   max generation tokens；
-   sampling 参数；
-   compiler flags；
-   timeout；
-   test comparison 方法；
-   eval subset；
-   random seed。

保存为：

``` text
configs/eval/default.yaml
```

后续修改 eval protocol 必须形成新版本，不得静默修改。

## Acceptance Criteria

使用 5～10 个手工 toy problems：

``` text
python -m src.eval.evaluator ...
```

能够完成：

``` text
generate → compile → execute → judge → metrics
```

并生成：

``` text
outputs/eval/<experiment_id>/
├── config.yaml
├── generations.jsonl
└── metrics.json
```

完成后停止并报告。

------------------------------------------------------------------------

# 9. Milestone 3 --- Fixed Eval Set

目标测试集优先使用 **LiveCodeBench**。

## Codex 任务

-   [ ] 获取/适配 LiveCodeBench coding problems。
-   [ ] 选择固定开发评测 subset，建议约 300～500 题。
-   [ ] 保留原始 problem ID 和发布日期等 metadata。
-   [ ] 将选中的 problem ID 固化到 split 文件。
-   [ ] 检查 verifier 是否兼容。
-   [ ] 不允许 Eval problem 进入后续 SFT/GRPO。

建议：

``` text
data/splits/eval_v1_problem_ids.json
```

从此所有主要实验都使用该固定 subset。

## Acceptance Criteria

可以在固定 subset 中随机取 10 题，完整跑通 Eval pipeline。

完成后停止并报告。

------------------------------------------------------------------------

# 10. Milestone 4 --- Base Baseline

在任何训练之前评测：

``` text
Qwen3-0.6B-Base
```

## Codex 任务

-   [ ] 使用固定 Eval config。
-   [ ] 首先跑 pass@1。
-   [ ] 保存全部 generation。
-   [ ] 保存 execution result。
-   [ ] 保存 aggregate metrics。
-   [ ] 统计错误类型。
-   [ ] 记录推理速度和显存峰值。

如果完整 300～500 题的云端评测成本暂时过高，可以先运行固定的小型 dev subset，但必须明确标注，正式对比时使用完全一致的固定 subset。

## 输出

至少得到：

``` text
Qwen3-0.6B-Base

pass@1 = ?
compile_rate = ?
extraction_rate = ?
test_pass_rate = ?
```

这是后续所有实验的 baseline。

完成后停止并报告。

------------------------------------------------------------------------

# 11. Milestone 5 --- SFT Data Pipeline

第一阶段主要使用：

``` text
OpenCodeReasoning-2
C++ subset
```

暂时**不要爬洛谷，不要生成 synthetic data**。

## 数据处理

-   [ ] 下载/读取数据；
-   [ ] 保留 C++；
-   [ ] 标准化字段；
-   [ ] 去除损坏样本；
-   [ ] 去除异常长度样本；
-   [ ] 去除无法解析的代码；
-   [ ] problem-level dedup；
-   [ ] 尽可能验证代码；
-   [ ] 与 Eval Set 做 contamination 检查；
-   [ ] 保存 source/original ID；
-   [ ] 输出数据统计。

人工检查：

-   [ ] 随机抽取至少 100 条；
-   [ ] 检查 problem；
-   [ ] 检查 reasoning；
-   [ ] 检查 code；
-   [ ] 检查格式；
-   [ ] 记录主要质量问题。

## Nested Dataset

构造：

``` text
SFT-1K ⊂ SFT-5K ⊂ SFT-10K
```

即 5K 必须包含 1K，10K 必须包含 5K。

尽量平衡：

-   difficulty；
-   tags；
-   source；
-   response length。

## Acceptance Criteria

生成：

``` text
data/processed/sft_1k.jsonl
data/processed/sft_5k.jsonl
data/processed/sft_10k.jsonl
data/processed/sft_stats.json
```

并生成 contamination/dedup report。

完成后停止并报告，不开始训练。

------------------------------------------------------------------------

# 12. Milestone 6 --- SFT Smoke Test

正式训练前必须先做极小规模 overfit。

## Codex 任务

从 Base 开始：

``` text
Qwen3-0.6B-Base
        ↓
32~100 samples
        ↓
SFT overfit
```

检查：

-   loss 是否明显下降；
-   输出格式是否开始匹配训练数据；
-   checkpoint 是否可保存；
-   checkpoint 是否可重新加载；
-   训练是否能在当前租用 GPU 上稳定运行，并记录峰值显存。

必要时使用：

-   bf16/fp16；
-   gradient checkpointing；
-   gradient accumulation；
-   LoRA（仅当 full fine-tuning 在当前云端资源上明显不经济或不可行时）。

但必须记录最终采用的训练方式。

## Acceptance Criteria

模型能明显 overfit 小数据集，训练和 checkpoint pipeline 正常。

完成后停止并报告。

------------------------------------------------------------------------

# 13. Milestone 7 --- SFT-1K

必须从：

``` text
Qwen3-0.6B-Base
```

开始。

训练：

``` text
Base → SFT-1K
```

保存：

-   config；
-   loss curve；
-   learning rate；
-   checkpoint；
-   peak VRAM；
-   wall-clock time；
-   tokens/s。

训练结束立即使用固定 Eval protocol：

``` text
SFT-1K → Eval
```

比较：

  Model      pass@1   Compile Rate   Test Pass Rate
  -------- -------- -------------- ----------------
  Base            ?              ?                ?
  SFT-1K          ?              ?                ?

另外随机检查失败案例。

完成后停止并报告结果。

------------------------------------------------------------------------

# 14. Milestone 8 --- SFT-5K

**重新从 Base checkpoint 开始。**

不要：

``` text
Base → 1K → 再追加 4K
```

而是：

``` text
Base → SFT-1K
Base → SFT-5K
```

这样才能形成较干净的数据规模对比。

保持训练设置尽量可比。

训练完成：

``` text
SFT-5K → Fixed Eval
```

比较：

  Model      SFT Samples   pass@1
  -------- ------------- --------
  Base                 0        ?
  SFT-1K              1K        ?
  SFT-5K              5K        ?

完成后停止并报告。

------------------------------------------------------------------------

# 15. Milestone 9 --- 是否训练 SFT-10K

根据结果决定。

如果：

``` text
Base << SFT-1K < SFT-5K
```

且 1K → 5K 仍有明显收益，则运行：

``` text
Base → SFT-10K → Eval
```

如果：

``` text
SFT-1K ≈ SFT-5K
```

则先不要增加数据，优先分析：

-   benchmark 是否过难；
-   数据质量；
-   模型容量；
-   training hyperparameters；
-   output format；
-   是否只提升了 instruction following 而非算法能力。

最终选择：

``` text
SFT-best
```

作为 GRPO initial policy。

------------------------------------------------------------------------

# 16. Milestone 10 --- GRPO Candidate Data

第一阶段优先考虑：

``` text
TACO-Verified
+
DeepCoder / 其他带可靠 testcase 的 coding dataset
```

目标不是立即得到最终 GRPO 数据，而是先建立：

``` text
10K~20K candidate problems
```

## Codex 任务

-   [ ] 转换统一 RL schema；
-   [ ] 验证 tests；
-   [ ] 删除 malformed tests；
-   [ ] problem-level dedup；
-   [ ] 与 SFT dataset dedup；
-   [ ] 与 Eval dataset contamination check；
-   [ ] 保存 source metadata；
-   [ ] 输出数据统计。

**特别注意 LiveCodeBench contamination。**

任何来源中若包含 Eval 使用的 LiveCodeBench 题目，必须剔除。

完成后停止并报告。

------------------------------------------------------------------------

# 17. Milestone 11 --- Offline Difficulty Filtering

使用：

``` text
SFT-best
```

对 GRPO candidate pool 进行 rollout。

第一版：

``` text
每题 4 rollouts
```

算力允许后：

``` text
每题 8 rollouts
```

计算：

``` text
empirical_pass_rate =
successful_rollouts / total_rollouts
```

例如：

``` text
Problem A → 0/8 = 0.000
Problem B → 1/8 = 0.125
Problem C → 3/8 = 0.375
Problem D → 6/8 = 0.750
Problem E → 8/8 = 1.000
```

重点保留存在探索和学习空间的问题，例如：

``` text
0 < pass_rate < 1
```

后续可以进一步尝试：

``` text
0.1 <= pass_rate <= 0.8
```

构造：

``` text
GRPO-Debug: 100 problems
GRPO-Pilot: 500 problems
GRPO-Full: 2K~5K problems
```

保存每道题的 rollout 和 empirical difficulty。

------------------------------------------------------------------------

# 18. Milestone 12 --- GRPO Reward

第一版 reward 必须简单。

核心：

``` text
execution correctness
```

推荐：

``` text
testcase_reward = passed_tests / total_tests
```

可加入非常轻量的：

``` text
format_reward
compile_reward
```

但不能让辅助 reward 压过 execution correctness。

暂时不要加入：

-   LLM-as-a-Judge；
-   reasoning quality judge；
-   复杂 style reward；
-   人工 reward model。

需要记录：

-   total reward；
-   testcase reward；
-   compile reward；
-   format reward；
-   reward variance。

------------------------------------------------------------------------

# 19. Milestone 13 --- GRPO Smoke Test

使用：

``` text
GRPO-Debug
100 problems
4 rollouts/problem
```

目标不是提升 benchmark，而是验证：

``` text
Prompt
  ↓
Rollout
  ↓
Code Extraction
  ↓
Compile
  ↓
Execute
  ↓
Reward
  ↓
Group Advantage
  ↓
Policy Update
```

检查：

-   [ ] verl pipeline 正常；
-   [ ] reward 正常；
-   [ ] group reward variance 非零；
-   [ ] loss 正常；
-   [ ] gradient 正常；
-   [ ] KL 正常；
-   [ ] entropy 可记录；
-   [ ] response length 可记录；
-   [ ] checkpoint 可保存；
-   [ ] 当前云端 GPU 配置下不 OOM，并记录峰值显存。

完成后停止并报告。

------------------------------------------------------------------------

# 20. Milestone 14 --- GRPO Pilot

运行：

``` text
SFT-best
   ↓
500 GRPO problems
   ↓
GRPO-Pilot
```

重点记录：

-   training reward；
-   validation reward；
-   KL；
-   entropy；
-   response length；
-   compile rate；
-   rollout success rate；
-   zero-variance group ratio；
-   peak VRAM；
-   training throughput。

训练结束：

``` text
GRPO-Pilot → Fixed Eval
```

比较：

  Model                     pass@1
  ----------------------- --------
  Base                           ?
  SFT-1K                         ?
  SFT-5K                         ?
  SFT-best                       ?
  SFT-best + GRPO-Pilot          ?

如果 reward 上升但 benchmark 不上升，**停止扩规模并先做 failure
analysis**。

------------------------------------------------------------------------

# 21. Milestone 15 --- GRPO Full

只有 Pilot 明确证明 pipeline 正常且存在有效学习信号后执行。

目标：

``` text
2K~5K problems
×
4/8 rollouts
```

根据云端 GPU 的实际吞吐、显存和租用成本决定规模。

保存多个 checkpoint，并定期在固定 dev eval 上评测。

最终输出：

``` text
Qwen3-0.6B-Base
        ↓
      SFT
        ↓
      GRPO
        ↓
   Final Evaluation
```

------------------------------------------------------------------------

# 22. Phase 1 必须产出的实验图

最终至少生成以下图表。

## Figure 1 --- SFT Data Scaling

``` text
SFT Samples
0 → 1K → 5K → 10K

vs

pass@1
```

回答：

> 增加 SFT 数据是否持续提升 0.6B 模型？

## Figure 2 --- GRPO Training Dynamics

``` text
training step
vs
reward / KL / entropy
```

## Figure 3 --- GRPO Benchmark Improvement

``` text
Base
SFT
SFT + GRPO

vs

pass@1
```

## Figure 4 --- Difficulty vs RL Gain

按照题目初始 empirical pass rate 分桶：

``` text
0~0.1
0.1~0.3
0.3~0.5
0.5~0.8
0.8~1.0
```

观察不同难度区间的 RL improvement。

## Figure 5 --- Initial Exploration vs RL Gain

研究：

``` text
initial rollout success rate
            ↓
       final improvement
```

这张图与项目核心研究问题直接相关。

------------------------------------------------------------------------

# 23. Phase 1 最终需要回答的问题

项目结束时 README / report 必须能够回答：

1.  SFT 是否显著提升 Qwen3-0.6B-Base 的 coding 能力？
2.  1K → 5K → 10K 数据增加是否持续有效？
3.  SFT 的收益主要来自代码格式改善，还是算法正确率改善？
4.  GRPO 是否能在 SFT checkpoint 上进一步提升？
5.  哪些难度的问题最适合 GRPO？
6.  对于 rollout 全错的问题，GRPO 是否基本无法获得有效信号？
7.  initial policy exploration capability 与 RL gain 是否相关？
8.  training reward 上升是否对应 held-out benchmark 提升？
9.  是否出现 reward hacking？
10. 是否出现 response length 异常增长？
11. 是否出现 entropy collapse？
12. 数据 contamination 是如何控制的？
13. SFT 与 GRPO 的提升分别来自什么？

------------------------------------------------------------------------

# 24. 暂时不要做的事情

在上述主线跑通前，不要主动扩展：

-   洛谷爬虫；
-   洛谷题解数据；
-   synthetic problem generation；
-   critique/self-repair 数据；
-   reasoning LLM Judge；
-   reward model；
-   DPO；
-   PPO；
-   多语言代码；
-   多模型 comparison；
-   4B/8B 训练；
-   Web UI；
-   Demo 网站；
-   过度工程化的分布式系统。

这些属于 Phase 2。

------------------------------------------------------------------------

# 25. Codex 工作协议

每次给 Codex 一个 Milestone 时，要求它遵循以下流程：

> **执行环境规则：** 如果当前 Milestone 涉及 GPU inference、Qwen 模型加载、SFT、verl、vLLM、GRPO 或正式 benchmark，Codex 应按 Linux 云端环境实现和执行；不要为 Windows 编写训练兼容层。如果当前会话无法直接访问云端机器，则应完成代码/config/脚本准备并明确给出需要在云端执行的命令，而不是退回设计 Windows workaround。

``` text
1. 阅读当前 repository 和已有实现。
2. 只实现当前指定 Milestone。
3. 不提前实现后续 Milestone。
4. 优先复用现有代码，不重复造轮子。
5. 为关键逻辑补充测试。
6. 运行测试。
7. 运行最小 smoke test。
8. 报告修改了哪些文件。
9. 报告测试结果。
10. 报告发现的问题、风险和下一步建议。
11. 等待确认后再继续。
```

每个阶段结束时要求 Codex 输出：

``` text
## Completed
- ...

## Files Changed
- ...

## Tests
- ...

## Results
- ...

## Known Issues
- ...

## Next Milestone
- ...
```

------------------------------------------------------------------------

# 26. 第一轮实际执行顺序

不要一次把整个计划交给 Codex 让它全部实现。

推荐逐步执行：

``` text
M0  Repository & Environment
 ↓
M1  Code Verifier
 ↓
M2  Evaluation Pipeline
 ↓
M3  Fixed Eval Set
 ↓
M4  Base Baseline
 ↓
M5  SFT Data Pipeline
 ↓
M6  SFT Smoke Test
 ↓
M7  SFT-1K
 ↓
M8  SFT-5K
 ↓
M9  Decide SFT-10K
 ↓
M10 GRPO Candidate Data
 ↓
M11 Difficulty Filtering
 ↓
M12 Reward
 ↓
M13 GRPO Smoke Test
 ↓
M14 GRPO Pilot
 ↓
M15 GRPO Full
```

------------------------------------------------------------------------

# 27. 当前最小目标

现在只关注：

``` text
Eval
 ↓
Base Baseline
 ↓
SFT Data
 ↓
SFT-1K
 ↓
Eval
 ↓
SFT-5K
 ↓
Eval
```

在得到 `Base / SFT-1K / SFT-5K` 三组可靠结果之前，**不需要开始正式
GRPO**。

第一阶段前半段成功的定义不是模型达到多高的 pass@1，而是：

> **建立了一套可信、可复现、数据隔离明确的实验系统，并能可靠测量 SFT 对
> Qwen3-0.6B-Base 的影响。**

GRPO 在此基础上继续。
