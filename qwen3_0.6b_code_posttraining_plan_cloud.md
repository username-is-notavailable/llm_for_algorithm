# Execution-Guided Agentic RL for Code Reasoning / Self-Debugging

> 项目目标：构建可复现的 execution-guided Code Agent 后训练系统，研究
> **execution feedback、Agentic SFT 与 Agentic GRPO 能否提升小模型利用环境反馈进行代码自我修复的能力**。
>
> 项目不再以“单轮 SFT + GRPO 提升 0.6B 模型算法题 pass@1”为核心。现有 verifier、eval、
> 数据隔离、实验记录、SFT、verl 和 vLLM 基础设施继续保留，后续围绕 multi-turn Code Agent
> 重构数据 schema、evaluation metrics、rollout pipeline 和训练目标。

---

## 1. 项目定位与研究问题

核心研究问题：

> Execution feedback + Agentic SFT/GRPO 能否显著提升小模型利用环境反馈进行
> self-correction 的能力？

项目需要明确区分两种能力：

1. **One-shot coding capability**：模型第一次生成代码时直接解决问题的能力；
2. **Feedback utilization capability**：第一次生成失败后，模型能否理解编译、运行和测试反馈并完成修复。

最终实验不仅追求最高 pass@1，还要测量后训练带来的增益、Agent 行为变化和工具使用效率。

### 1.1 简历导向的最终成果

- 有限 horizon、可复现、支持断点恢复的 Code Agent loop；
- 成熟 sandbox 的适配层，以及结构化 execution feedback；
- 从真实模型失败中构造并执行验证的 code-repair SFT 数据；
- multi-turn executable rollout 与 Agentic GRPO；
- one-shot、agent prompting、Agent + SFT、Agent + SFT + GRPO 的受控对照实验；
- repair success、success-vs-execution curve、token/execution efficiency 和错误类型迁移分析；
- 可复现实验配置、技术报告、失败案例分析和简历量化结果。

---

## 2. 项目原则

1. **一次完成一个 Milestone。** 每个阶段先测试、记录和验收，再进入下一阶段。
2. **Eval 与环境优先。** 在训练前冻结 Agent protocol、反馈格式、终止条件和评测指标。
3. **严格对照。** 同一模型、题集、生成预算下比较 one-shot 与 Agent；同一 Agent 环境下比较 Base、SFT 和 GRPO。
4. **能力解耦。** 同时报告首次成功率和首次失败后的修复率，不能只报告最终 pass@1。
5. **数据严格隔离。** SFT、RL 和 Eval 按 `problem_id` 隔离，并进行跨数据源去重和 contamination 检查。
6. **Verifier 逻辑复用。** Eval、数据验证和 RL reward 尽可能复用同一套代码提取、编译、执行与判题逻辑。
7. **Sandbox 接口解耦。** Agent 不直接依赖某个执行实现，通过统一 `ExecutionTool` 接口调用 backend。
8. **有限 Agent horizon。** 第一版最多 3 次 feedback-producing execution 和 1 次 final candidate，并设置 token budget 与重复终止条件。
9. **所有实验配置化、可恢复。** 保存配置、seed、git commit、环境、模型 revision、trajectory、checkpoint 和 metrics。
10. **先建立简单可靠的 reward。** 第一版以执行正确性和 testcase pass rate 为核心，确认有效后再加入工具成本或 progress shaping。
11. **正式 GPU 工作放在云端。** 本地 WSL 负责开发、数据处理、CPU 测试和结果分析。
12. **控制范围。** 第一阶段固定 C++17，不建设通用 Agent 框架，不同时支持多种语言或复杂 process reward。

---

## 3. 模型与运行环境

### 3.1 模型规划

| 模型 | 用途 | 正式训练 |
| --- | --- | ---: |
| Qwen3-1.7B-Base | pipeline、SFT/GRPO smoke、低成本实验和消融 | 是 |
| Qwen3-4B-Base | 正式 SFT、GRPO 和最终主实验 | 是 |
| Qwen3-8B 官方后训练模型 | inference upper-bound/reference | 否 |
| Qwen3-4B 官方后训练模型 | Agent prompting reference | 否 |
| 现有 Qwen3-0.6B 实验 | 容量与训练退化诊断历史 | 不继续训练 |

Base 模型用于测量本项目后训练增益；官方 post-trained 模型只作为能力参考，不能与 Base/SFT
结果当作公平训练对比。

### 3.2 本地与云端职责

本地 WSL 负责代码开发、数据处理、CPU 测试、结果分析、Git 和文档。云端 Linux GPU 负责
1.7B/4B/8B inference、Agent rollout、正式 evaluation、SFT、GRPO、vLLM、verl 和 sandbox。

云端实验继续记录 GPU、显存、驱动、CUDA、Python、PyTorch、Transformers、verl、vLLM、
FlashAttention 和 sandbox 版本。

---

## 4. Code Agent v1

### 4.1 Agent loop 与动作

```text
Problem
  ↓
Reasoning / Candidate Code
  ↓
Execute
  ↓
Observation
  ↓
Analyze / Fix / Complete Revised Code
  ↓
Execute again
  ↓
Final or Termination
```

第一版实现轻量、任务专用的有限状态循环，不引入通用 Agent 框架。

模型只需要选择两个动作，每个动作都必须附带一份完整 C++17 程序：

- `execute_code`：运行 visible tests 并返回执行反馈；
- `final`：结束交互，只运行 hidden tests，不返回反馈。

动作使用简单标签 `<action>execute_code</action>` 或 `<action>final</action>`，不要求 Base 模型
生成嵌套 JSON function call。缺失或非法标签采用预算感知回退，并记录 requested/effective action
和 parse status。

### 4.2 Horizon 与预算

- 最多 3 次会返回反馈的 `execute_code`；
- 额外保留 1 次 final candidate，总计最多生成 4 份候选代码；
- 每轮 generation token cap 配置化；
- 整条 trajectory 有 total token budget；
- `execute_code` 和 `final` 都必须提交可提取的完整 C++ 代码；
- 保存全部中间响应、代码、反馈和资源消耗；
- 支持按 trajectory 断点恢复。

### 4.3 Observation Protocol v1

模型可见反馈包含：

- compile success/failure；
- 有长度限制的 compiler stderr；
- passed tests / total tests；
- 第一个失败测试的 input、expected output 和 actual output；
- runtime error、timeout 或 output limit；
- 剩余 execution budget。

```text
Execution result:
- Status: WRONG_ANSWER
- Tests passed: 3/8
- First failing input:
...
- Expected output:
...
- Actual output:
...
- Executions remaining: 1
```

内部保存完整判题结果，模型可见 feedback 是经过裁剪和格式化的独立字段。后续可对比 detailed
feedback（失败输入、expected、actual）和 weak feedback（仅错误类型与通过率）。

### 4.4 Termination Reason v1

- `success`；
- `final_incorrect`；
- `execution_budget_exhausted_auto_final`；
- `token_budget_exhausted_auto_final`；
- `code_extraction_failed`；
- `repeated_code`；
- `sandbox_error`；
- `model_stop_without_code`。

第一版只记录 pass-rate improvement，不据此提前终止，因为相同通过率不代表代码没有实质改善。
即使提前终止，最后一次响应、代码和判题结果也必须落盘。

---

## 5. Sandbox 与 Tool Adapter

不从零实现底层安全沙箱。现有 verifier 继续承担代码提取、C++17 编译、testcase 判定、timeout、
output limit 和统一 `JudgeResult`，其上增加稳定接口：

```python
class ExecutionTool:
    def execute(
        self,
        code: str,
        tests: list[TestCase],
    ) -> ExecutionObservation:
        ...
```

```text
CodeAgent
   ↓
ExecutionTool
   ├── LocalVerifierBackend：开发、测试和可信数据
   └── Isolate/Nsjail/Judge0Backend：云端 rollout
```

优先调查 `isolate`，其次 `nsjail`，最后是服务化更方便但批量 rollout 开销可能更高的 Judge0。
当前 subprocess + resource limit 只能作为本地和可信数据 backend，不能描述为强安全隔离，也不能
在高权限主机上直接执行来源不明的任意代码。

---

## 6. 数据 Schema

### 6.1 Executable Problem

```json
{
  "problem_id": "source:id",
  "source": "taco",
  "problem": "...",
  "difficulty": "medium",
  "language": "cpp",
  "tests": [],
  "environment_id": "cpp17-v1",
  "split": "train",
  "metadata": {}
}
```

### 6.2 Agent Trajectory

```json
{
  "trajectory_id": "...",
  "problem_id": "...",
  "model": "...",
  "policy_version": "...",
  "initial_prompt": "...",
  "steps": [
    {
      "turn": 0,
      "response": "...",
      "reasoning": "...",
      "code": "...",
      "generation_tokens": 1234,
      "tool_call": {"name": "execute_cpp", "arguments": {}},
      "observation": {
        "status": "wrong_answer",
        "compiled": true,
        "passed": 3,
        "total": 8,
        "pass_rate": 0.375,
        "feedback": "..."
      }
    }
  ],
  "final_status": "success",
  "termination_reason": "success",
  "tool_calls": 2,
  "total_generation_tokens": 2150
}
```

数据中必须同时保留原始完整 observation 和实际展示给模型的 feedback，以支持反馈策略消融。

---

## 7. SFT 数据与训练目标

不再将原始超长 reasoning imitation 作为唯一或主要目标。SFT 数据由三类样本组成。

### 7.1 One-shot code reasoning

```text
Problem → Concise Reasoning + Complete Code
```

用于维持首次生成和基本 coding 能力。

### 7.2 Single-step repair

```text
Problem + Wrong Code + Execution Feedback
→ Analysis + Complete Fixed Code
```

Wrong code 来源优先级：Base 模型真实失败、正确代码的受控 mutation、开源错误提交、后续 Agent
rollout 失败。所有 repair target 必须通过可靠 tests，错误代码必须经过实际执行确认失败。

### 7.3 Multi-turn trajectory

```text
Problem
Assistant: Candidate Code
Tool: Failure Observation
Assistant: Analysis + Revised Code
Tool: ...
Assistant: Final Code
```

第一版仅构造少量高质量 trajectory 来学习交互协议，主要训练量仍由 one-shot 和 single-step
repair 提供。

### 7.4 初始混合比例

- 40% one-shot；
- 50% single-step repair；
- 10% multi-turn trajectory。

该比例只是 pilot 起点，后续通过消融调整。不能只训练 repair 数据，以免损害 first-attempt 能力。

---

## 8. Agentic GRPO

RL 样本只要求 problem、hidden reliable tests、executable environment，以及 execution/token budget。
训练时在线执行：

```text
Problem → Code → Execute → Observe → Repair → ... → Terminate
```

### 8.1 Reward v1

```text
if all_tests_pass:
    reward = 1.0
elif valid_execution:
    reward = 0.5 * final_test_pass_rate
else:
    reward = 0.0
```

工具次数和 token 数第一版只记录，不进入 reward。确认 correctness 能够学习后，再研究：

```text
reward = correctness - lambda * extra_tool_calls
```

后续消融可以加入 invalid action penalty、progress shaping、反馈强弱和不同 execution budget。在
correctness 尚未提升前，不加入复杂 style、reasoning 或 efficiency reward。

---

## 9. Evaluation Protocol 与核心指标

### 9.1 主实验矩阵

| Policy | Agent loop | Repair SFT | GRPO |
| --- | ---: | ---: | ---: |
| Base one-shot | 否 | 否 | 否 |
| Base agent prompting | 是 | 否 | 否 |
| SFT one-shot | 否 | 是 | 否 |
| SFT agent | 是 | 是 | 否 |
| SFT + GRPO agent | 是 | 是 | 是 |
| Official post-trained reference | 是 | 官方 | 官方 |

关键比较：

1. Base one-shot vs Base agent：测量 inference-time feedback 收益；
2. Base agent vs SFT agent：测量 repair SFT 收益；
3. SFT agent vs SFT + GRPO agent：测量 Agentic RL 额外收益；
4. 同一 policy 的 first attempt vs final attempt：区分 coding 能力与 repair policy。

### 9.2 核心指标

- `first_attempt_success_rate`；
- `agent_success_rate`；
- `repair_success_rate`：首次失败题目中最终修复成功的比例；
- `success_gain`：Agent success 减去 first-attempt success；
- `test_pass_rate`；
- `average_tool_calls` 和 `average_tool_calls_on_success`；
- `average_generation_tokens` 和 `tokens_per_success`；
- `pass_rate_curve[k]`：最多允许第 k 次提交时的累计成功率；
- `termination_reason` 分布；
- 按 difficulty、首轮错误类型和反馈类型分组的修复率；
- 修改后退化率与重复提交率。

所有指标同时保存 overall 和 easy/medium/hard 分层结果。最终正式实验尽量使用多个 seed；资源不足
时主实验至少两个 seed，并明确报告方差限制。

---

## 10. Milestones

### 已完成的前期工作：M0–M7

现有阶段作为项目基础与方向选择证据保留，不重写历史：

- M0：云端环境与可复现依赖；
- M1：C++ verifier；
- M2：evaluation pipeline；
- M3：固定 LiveCodeBench eval split；
- M4：Qwen3-0.6B-Base baseline；
- M5：OpenCodeReasoning-2 SFT 数据准备与 audit；
- M6：多卡 SFT pipeline 和吞吐验证；
- M7：0.6B SFT 退化诊断、数据格式消融和官方 0.6B/1.7B/4B/8B 容量诊断。

M7 用于说明：单纯模仿超长 reasoning 不能稳定提升 0.6B，模型容量、输出分布和训练目标都会影响
结果，因此后续转向 1.7B/4B execution-guided self-repair。

### M8：项目重构与协议冻结

产物：

- 更新 README、主计划、数据文档和 findings；
- 冻结 Action、Observation、Trajectory schema；
- 定义指标和 termination reason；
- 将 verifier 抽象为 `ExecutionTool` backend；
- 实现 Agent state machine 和单元测试。

验收：fake generator 能跑通“失败 → 反馈 → 修复 → 成功”；trajectory 可无损序列化；每种
termination reason 都有测试；Agent 与 sandbox backend 解耦。

### M9：Agent Evaluation Baseline

- 实现 Qwen3-1.7B/4B Agent loop；
- one-shot 与 agent prompting 使用相同题集和预算；
- 支持逐轮 metrics、trajectory artifacts、分片和断点恢复；
- 固定 50–100 题 `agent-dev`；
- 先通过固定 10 题 smoke，再得到 Base one-shot 与 Base agent 对照；
- 人工 audit 至少 30 条 trajectory。

实现协议固定为 10 题 smoke 和 60 题 agent-dev。one-shot 与 Agent 使用相同 hidden tests；最多三次
`execute_code` 仅使用 visible tests。支持完整 trajectory 落盘、单 GPU resume 和两 GPU
problem sharding。Qwen3-1.7B-Base 先完成云端 gate，再运行 4B 对照。

M9 已于 2026-08-24 验收。官方 post-trained Qwen3-4B 的 60 题能力参考中，Agent 首次/最终成功率
为 40.0%/53.3%，首次失败后的 repair success 为 25.0%；但 action validity 仅 0.8%，fallback
为 99.2%。全部 trajectory 完成结构化审计，并人工核验 9 条成功修复与 1 条退化轨迹。该结果确认
execution feedback 有效，同时为 Agent action、停止决策和修复效率留下明确后训练空间。

### M10：Repair SFT 数据构造

- 收集 Base 模型真实首次失败并按错误类型分桶；
- 用多 GPU problem sharding 运行官方 post-trained 4B；验证通过且无循环的短输出作为 one-shot
  teacher target，带完整错误代码的失败进入 repair pool；
- SFT 初始化仍先实验 1.7B/4B Base；若固定数据下出现循环、协议或 coding 能力 gate 失败，再用同一
  数据切换到官方 post-trained checkpoint，避免混淆数据 producer 与 student initialization；
- 不设置本地 teacher 修复层；默认用百炼 `qwen3-8b` API 并发生成修复候选，每轮重新执行验证；
- 8B 未解决的任务再路由到经固定小样本 bake-off 选出的更强 API 模型；
- 对代码已通过 full tests、但 teacher 缺失 action 标签的响应允许做确定性协议规范化；必须保留
  原 run provenance，并记录修改 turn，禁止修改 reasoning、代码或执行结果；
- 构造 one-shot、single-step repair 和 multi-turn 混合数据；
- 完成 problem-level 隔离、去重、数据卡和人工 audit；
- 先冻结 50 条 pipeline pilot，通过后扩大并冻结 500 条 SFT pilot。

验收要求 wrong code 实际失败、repair target 通过全部 tests、feedback 与 JudgeResult 一致且无 eval
problem 泄漏。API key 只从环境变量读取；保存 provider/model、请求 ID、usage、reasoning、content、
生成参数和时间，hidden tests 及其结果不得进入模型消息。

### M11：Qwen3-1.7B Agentic SFT Pilot

- 1.7B LoRA 或全参数 smoke；
- 1K mixed SFT pilot；
- one-shot-only SFT 对照；
- 固定 agent-dev 评测。

Go/no-go：first-attempt success 不显著退化；repair success 明显优于 Base agent；不出现重复循环和
格式崩溃。失败时先调整数据与 protocol，不直接扩大到 4B。

### M12：Qwen3-4B SFT

- 将验证过的 recipe 移植到 4B；
- 1K → 5K data scaling；
- 正式 SFT checkpoint；
- 完成 one-shot 与 Agent 两种评测。

模型选择依据 first-attempt、repair success 和 agent success，而不是只看 training loss。

### M13：Agentic GRPO Smoke

- 将 Code Agent environment 接入 verl rollout；
- group rollout 共用 problem/tests；
- reward 与 eval verifier 共用判定逻辑；
- 1.7B、几十题、少量 step 的端到端 smoke；
- 保存 reward、advantage、KL、execution/action 和 trajectory 日志。

验收：reward 与离线判题一致；无 eval 泄漏；multi-turn state 不串样本；checkpoint 可恢复；policy
不能通过非法格式利用 reward 漏洞。

### M14：Qwen3-4B GRPO Pilot / Formal

```text
Debug 32–64 problems
→ Pilot 256–500 problems
→ Formal 1K–2K executable problems
```

每一级先验证 reward、输出稳定性和 evaluation 增益，再扩大规模。

### M15：最终评测与项目总结

- 固定多难度正式 eval 和完整实验矩阵；
- 多 seed 结果；
- success-vs-execution curve；
- token/tool efficiency；
- termination 和错误类型迁移分析；
- 典型修复成功与失败案例；
- GPU 时间、训练成本和推理成本；
- README 展示、技术报告、复现实验命令和简历 bullet。

---

## 11. 当前不做的内容

- 通用 Agent 框架；
- 多语言执行；
- 自建完整容器调度系统；
- 洛谷爬虫；
- 无限 horizon；
- LLM-as-a-judge reward；
- 复杂 reasoning/process reward；
- 大规模自动课程学习；
- Qwen3-8B 正式训练；
- 同时实现多种 production sandbox backend。

第一阶段主线固定为：

> C++17 + 最多 3 次 execution + testcase feedback + Qwen3-1.7B 验证 + Qwen3-4B 正式 SFT/GRPO。

---

## 12. 待确认决策

开始实现 M8 不依赖以下答案，但进入云端 Agent rollout 和正式训练前必须确认：

1. 云平台是否允许安装或运行 `isolate`/`nsjail`，以及是否具有 root 权限；
2. 可接受的正式训练 GPU 类型、卡数与总 GPU-hour 预算；
3. 预计投递实习的时间窗口，以决定优先交付 1.7B 完整闭环还是同时完成 4B GRPO；
4. sandbox 未部署前，只允许使用可信、经过审查的数据运行 LocalVerifierBackend。

---

## 13. 下一步

下一阶段为 **M8：项目重构与协议冻结**：

1. 更新 README 和项目结构说明；
2. 定义 Agent action、observation、step 与 trajectory 类型；
3. 抽象 ExecutionTool，并适配现有 verifier；
4. 实现 feedback formatter 与 termination policy；
5. 实现同步 Code Agent loop；
6. 增加 fake generator、fake tool 和 LocalVerifierBackend 测试；
7. 固定 Agent metrics 与 artifact 格式；
8. 本地完整测试通过后提交 M8 第一阶段 commit。
