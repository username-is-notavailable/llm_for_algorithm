# Data

## Agent Trajectory v2

新方向将 executable problem 的 tests 固定拆成两组：

- `visible_tests`：由 `execute_code` 使用，可向模型展示第一个失败输入、expected 和 actual；
- `hidden_tests`：private gate；默认不进入模型上下文。当现有 visible tests 全过但 private gate
  失败时，最多选择一条真实失败 case，将其显式迁移为 `revealed_counterexample` 并加入后续
  visible tests。迁移后该 case 不再计作 private，且始终至少保留一条未揭示 private test；
- `final` 对当前 visible 与 remaining private 的完整并集重新验证，防止最后一次生成在已见 case
  上回归。

模型动作协议只有 `execute_code` 和 `final`。每个动作必须附带完整 C++17；系统不使用“沿用上次
代码”的隐式 final。单条 trajectory 保存模型原始响应、requested/effective action、action parse
status、代码与 hash、visible observation、不可见的 private audit、反例揭示 provenance、token 数和
termination reason。

默认预算为三次 feedback-producing execution 和一次 final candidate。超额 `execute_code` 作为
final 处理，但同时记录：

```json
{
  "requested_action": "execute_code",
  "effective_action": "final",
  "termination_reason": "execution_budget_exhausted_auto_final"
}
```

M8 开发阶段使用 `LocalVerifierBackend`；它只能执行可信、经过审查的代码。正式 rollout 接入强
sandbox 前不得将该 backend 用于任意来源的不可信代码。

### Agent Eval v1

M9 从冻结 `livecodebench_v1/dev_v1.jsonl` 选择 60 题；原有 10 题 smoke 保持同序并作为 dev
子集。其余题目按 difficulty 分桶后以 seed `20260825` 稳定排序、轮流取样。每题至少保留一个
visible 和一个 hidden testcase；visible 取约 20%，最多 5 个。

派生数据不提交 Git，可由下列命令在本地或云端重建：

```bash
.third_party/verl/.venv/bin/python scripts/prepare_agent_eval.py
```

提交的 `data/splits/agent_eval_v1_problem_ids.json` 固定 smoke/dev problem IDs、每题 visible/hidden
内容 hash 和总测试数。派生行中的 `tests` 是 `hidden_tests` 的副本，供现有 one-shot evaluator 使用；
Agent evaluator 会验证同一 manifest 后分别读取 `visible_tests` 和 `hidden_tests`。

Milestone 0 不引入数据集。后续数据均按 `problem_id` 做 problem-level 隔离，并在此记录来源、许可、版本、清洗和切分信息。

### Agent SFT smoke v2

M11 pilot 使用 M10 CodeContests+ 两阶段 teacher 生成后经完整 checker 验收的 42 条 repair
trajectory。按 seed `20260826` 和 problem ID 稳定划分为 34 train / 8 dev，开发集不参与梯度更新。
训练候选中唯一的 14,850-token 长尾在 A100 40GB full-parameter backward 时 OOM；该样本保留在
冻结来源及 manifest 的 excluded 记录中，不截断 target。实际 pilot 使用 33 train / 8 dev，训练
最长 9,837 tokens，上限固定为 10,240。
数据保存为原生 `system/user/assistant/tool` messages；每个 assistant turn 带显式 `trainable`
标记。训练通过 Qwen3 chat template 渲染，并只对 `trainable=true` 的 assistant 内容及其
`<|im_end|>` 终止符计算 loss，system、题面、初始错误代码、execution feedback、context-only
assistant 和模板控制部分全部 mask。v2 从 canonical step submission 重建 assistant turns，修复了
v1 中后处理只更新 submission、未同步后续 prompt snapshot，导致少量规范化前响应进入 target 的问题。

派生数据位于 `data/processed/agent_sft_v2/`（不提交 Git），提交的历史 manifest 为
`data/splits/agent_sft_smoke_v2_manifest.json`。v2 已由 v3 取代，不再由默认入口重建。

### Agent SFT smoke v3

v2 的训练 target 虽已 canonical，但首个 user message 仍把题面、初始错误代码和 execution feedback
拼成一段，这与真实 Agent rollout 的消息状态不一致。v3 将同一条 checker-backed repair 样本重建为：

```text
system → user(problem)
       → assistant(initial wrong code, trainable=false)
       → tool(initial execution feedback)
       → assistant(teacher repair, trainable=true)
       → tool(...)
       → assistant(final, trainable=true)
```

所有 feedback 的 execution budget 按包含初始失败在内的真实 Agent horizon 重新计算。总 execution
超过协议上限 3 的轨迹整条拒绝，不截断成功路径。现有 42 条中 3 条因此拒绝，剩余 39 条稳定划分为
31 train / 8 dev；共 39 个 context-only initial assistant turns 和 85 个可训练 teacher turns。
上述 42 条数据用于早期结构 smoke。正式 v3 随后使用 300 条 one-shot 和 238 条 repair 候选；5 条
repair 因总 execution 次数超过 3 而排除，按 problem 稳定划分后为 497 train / 36 dev。8K 完整
序列门槛排除 8 条，实际训练输入为 490 train / 35 dev。对应冻结 manifest 为
`data/splits/m11_agent_sft_v3_manifest.json`，派生 JSONL 位于 `data/processed/agent_sft_v3/`（不提交
Git）。这仍是 pilot 规模数据，不作为最终正式训练集。

### M12 增量 repair source pool

下一轮数据扩充保留上述 300 题，并从相同冻结 revision 增量筛选 1000 个新 problem。配置通过旧池
`problems.index.json` 排除已付费处理的 problem ID，继续保持 TPR/TNR 均不低于 0.9、完整 checker
本地门禁及 eval fingerprint 排除。原始 JSONL 生成后立即 compact，tests/checker 只在 indexed problem
store 保存一份。

Teacher 采用两阶段队列：全部新问题先由 `qwen3-8b` 处理，未通过完整 checker 的任务携带最佳候选
升级至 `qwen3-32b`。每阶段使用独立 SQLite queue，可用原输出目录 `--resume`，不会重放已完成请求。
API 接收率不是训练质量指标；最终仅冻结经过完整 checker、发生过实质修复且满足三次 execution
budget 的 rollout-aligned trajectory。新池与旧池合并后，目标约为 1000 条 repair 和 400 条
one-shot，实际数量以最终 checker audit 为准，不为凑数放宽门禁。

Source preparation 使用 streaming compact checkpoint：每个 accepted problem 立即写入 compact problem
store、failure pool 和 one-shot seed，并在三份文件 `fsync` 后原子更新已提交字节偏移；每个 rejected
candidate 也更新扫描位置但不触发无意义的数据文件同步。恢复时先按 checkpoint 截断可能存在的未
提交尾部，再从冻结 candidate index 继续。因而 WSL 重启不会产生半行、三文件错位或重复 problem，
内存占用也不随 accepted 数量增长。重复执行 `scripts/prepare_m12_repair_pool.sh` 会自动检测 checkpoint
并进入 resume 模式。

早期阶段曾在本地从同一固定 CodeContests+ revision 准备 300 题 checker-backed source pool，同时导出
真实错误提交 repair pool 与 checker full-pass one-shot seed。该阶段只需 CPU、磁盘和网络：

```bash
bash scripts/cloud_prepare_codecontests_plus_repair.sh \
  --config configs/data/m11_codecontests_plus_repair_300_v1.yaml
```

该批次最终得到 300 个严格 accepted problems。原始 producer 文件为了兼容旧 pipeline，在 problem、
failure 和 one-shot 三处重复保存 tests/checker，合计约 17 GB，仅作为可追溯中间产物。正式使用
compact v2：`problems.jsonl` 唯一保存题目环境，failure/one-shot 仅以 `problem_id` 引用，并将
`source_judge` 缩减为判题摘要；`problems.index.json` 提供 byte-offset 随机读取，worker 无需把题库
载入内存。最终四个文件约 3.71 GB，SHA-256 与计数冻结在
`data/splits/codecontests_plus_repair_300_v2_manifest.json`。转换命令：

```bash
.third_party/verl/.venv/bin/python scripts/compact_codecontests_plus_repair.py \
  --problems data/processed/codecontests_plus_repair_300_v1/problems_300.jsonl \
  --failure-pool data/processed/codecontests_plus_repair_300_v1/failure_pool_300.jsonl \
  --one-shot-seeds data/processed/codecontests_plus_repair_300_v1/one_shot_seeds_300.jsonl \
  --output-dir data/processed/codecontests_plus_repair_300_v2 \
  --manifest data/splits/codecontests_plus_repair_300_v2_manifest.json
```

compact v2 的第一批监督源由 `scripts/prepare_m11_agent_sources.sh` 构造。one-shot 正确代码会再次
通过 visible 与完整 checker gate，然后形成两个 trainable assistant turns（execute 与 pass 后的
final）；repair 部分使用 qwen3-8b 在线执行，SQLite queue 支持 `--resume`。

首版正式 Agent SFT 数据由 300 条 one-shot、153 条 qwen3-8b repair 和 85 条
qwen3-235b-a22b repair source 构成。235B source 包括 71 条严格成功、10 条协议规范化成功和 4 条
中间成功截断恢复；剩余 62 条保留为后续 hard/RL pool。运行
`bash scripts/prepare_m11_agent_sft.sh` 会把 repair 转换为
`Problem + Wrong Code + Execution Feedback -> execute/final` messages，并使用 Qwen3-4B-Base
tokenizer 计数。split 以 problem ID 为单位，确保同题的 one-shot 与 repair 不会跨 train/dev。

当前冻结结果为 537 条：train 499（280 one-shot、219 repair），dev 38（20 one-shot、18 repair），
覆盖 300 个唯一题目。唯一排除项是 17,379-token repair，超过 16,384-token 训练窗口；不截断代码
或成功轨迹。全体长度 P50/P90/P95/P99 为 1,650/5,760/6,530/10,323 tokens，最大 15,077。
文件 hash、20 个 dev problem ID 和完整计数记录在
`data/splits/m11_agent_sft_v1_manifest.json`；训练数据位于被 Git 忽略的
`data/processed/agent_sft_v1/{train,dev}.jsonl`。

### Agent SFT v2：final 引用已执行代码

v1 训练后出现了明显的代码重复：同一份完整程序同时作为 `execute_code` 和后续 `final` 的监督目标，
模型容易在长生成中反复续写程序。v2 将提交语义改为：只要当前轨迹已经执行过代码，
`<action>final</action>` 单独出现就提交最近一次已执行程序，不再重复输出代码。若此前没有执行记录，
仍允许并要求 `final` 携带一个完整程序，因此首轮直接作答和历史 final-only 教师轨迹保持可用。

新数据由 `configs/data/m11_agent_one_shot_300_v2.yaml` 与
`configs/data/m11_agent_sft_v2.yaml` 构造，写入
`data/processed/agent_sft_source_v2` 和 `data/processed/agent_sft_v2`。训练起点同时从
`Qwen/Qwen3-4B-Base` 改为官方后训练模型 `Qwen/Qwen3-4B`，以利用其已有的停止与指令遵循能力；
本项目继续训练的是 execution-feedback policy，而不是要求 Base 模型从少量样本中重新学习完整对话行为。

冻结的 v2 数据共有 538 条（300 one-shot、238 repair），按 problem ID 划分为 train 500 与 dev 38，
没有样本超过 16,384-token 数据窗口。全体长度 P50/P90/P95/P99 为
1,310/5,411/6,140/9,976，最大 15,431。四张 A100-40GB 的正式 8K 配置按长度选择
train 492 与 dev 37，仅排除 9 条超 8K 轨迹；冻结文件本身不裁剪。

首轮 `1e-5` 学习率实验虽然将 Agent action 有效率提升到 100%，但 3 卡同配置 smoke 中
Agent success 从官方模型的 50% 降至 20%，并出现 7/10 次相同代码重试。后续
`m11_agent_sft_4b_post_lr2e6_pilot.yaml` 从官方 Qwen3-4B 重新起训，将峰值学习率降至
`2e-6`，保持同一冻结数据和单 epoch，用于隔离 catastrophic forgetting 是否主要由更新过猛导致。

进一步审计发现 v2 repair 将错误代码和反馈嵌入初始 user prompt，与真实 rollout 的
`assistant execute -> tool feedback -> assistant repair` 状态不一致。v3 改为保留初始失败 assistant turn
作为不计 loss 的上下文，然后监督后续修复动作。233 条 repair 中 5 条因加上初始执行后超过 3 次
execute 总预算而排除；冻结结果为 533 条，8K 三卡 pilot 使用 train 490、dev 35。

## Output Protocol v1

第一阶段的 SFT、GRPO rollout 和正式评测统一使用以下 C++ 响应协议：

````text
<think>
{reasoning}
</think>

```cpp
{complete_cpp17_program}
```
````

约定如下：

- `<think>...</think>` 用于推理、算法说明、正确性分析和复杂度分析；其中允许出现代码或 code block，verifier 会先丢弃整个区域；
- 最终答案是唯一一个带 `cpp` 标记的 Markdown code block；
- code block 必须包含可独立编译执行的完整 C++17 程序；
- `<answer>...</answer>` 不是协议所需结构；提取时仅将其视为透明包装，不给予内部代码更高优先级；
- reasoning 缺失或标签损坏时，verifier 可以回退到最终 Markdown code block 或原始 C++，但这些不是标准训练 target；
- 修改标准格式时必须新增协议版本，不得静默修改 v1。

Qwen3-0.6B-Base 没有可依赖的内置 chat template，因此数据预处理代码负责将统一内部字段渲染为上述文本。不同来源的原始响应不得未经标准化直接混合训练。

### 统一内部表示

SFT 样本先转换为结构化字段，再由统一 renderer 生成 Output Protocol v1：

```json
{
  "problem_id": "source:stable-id",
  "source": "OpenCodeReasoning-2",
  "problem": "...",
  "reasoning": "...",
  "code": "...",
  "language": "cpp",
  "verified": true,
  "metadata": {}
}
```

不得把来源数据中的多个响应字段直接拼接。例如 OpenCodeReasoning-2 应从 `r1_generation` 中拆分 reasoning 和最终 code，并以 `solution`、`judgement`、`pass_rate` 等字段做验证或过滤，而不是盲目拼接 `r1_generation + solution`。

### 数据源适配

- OpenCodeReasoning-2：原始 `r1_generation` 通常已经是 `<think>...</think>` 加 `cpp` Markdown code block，解析后重新按 v1 渲染；
- TACO / TACO-verified：原始数据主要提供 problem、solutions 和 tests，不假定存在 reasoning 或 Markdown 包装；优先用于可执行验证、GRPO problem pool 和补充代码数据；
- CodeContests+：M10 起作为 Agent executable problem 主源，固定数据 revision，并保留题目级
  testlib checker、1x tests 和真实正误提交。published TPR/TNR 只用于预筛选，不能替代本地执行：
  correct submission 必须由 checker full-pass，incorrect submission 必须真实失败后才能进入 repair
  pool。错误提交是初始状态而非训练答案，因此不要求自带 reasoning；repair reasoning/action 由 teacher
  在真实反馈上生成并再次通过完整 checker gate。
- 自建数据：必须生成或转换为统一内部表示，并通过同一个 v1 renderer 输出。

### Verifier 提取优先级

提取器首先删除完整的 `<think>...</think>` 区域，再将 `<answer>` 标签视作透明包装。剩余内容的提取优先级为：

1. 最长的 `cpp` / `c++` / `cc` / `cxx` block；
2. 最长且可识别为 C++ 的无语言标记 code block；
3. 可恢复的未闭合 C++ fence；
4. 可识别为 C++ 的原始文本；
5. 提取失败。

在进入数据处理 Milestone 时，应为 renderer、协议解析、往返转换和异常样本建立独立测试。

## Fixed Eval Set v1

正式评测集使用 `livecodebench/code_generation_lite`，固定来源为官方 `release_v6` 和 Hugging Face revision `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`。这里的 `eval_v1` 是本项目 split 版本，与上游 `release_v6` 是两个不同的版本维度。

上游 1,055 题中仅接收带标准 stdin/stdout tests 且具有 `easy`、`medium` 或 `hard` 标签的题；LeetCode function-call 题在当前 verifier 不兼容时明确拒绝。对剩余候选按难度分层、固定 seed `20260821` 抽取 500 题，再按 80/20 分为：

- `eval_v1`：399 题，仅用于阶段性正式评分；
- `dev_v1`：101 题，可用于开发观察，与 eval 不重叠；
- `smoke_10`：dev 中固定的 10 题，用于流水线验收，不作为正式得分。

唯一可信 split 定义是提交到 Git 的 `data/splits/eval_v1_problem_ids.json`。它记录上游 revision、选择参数、难度计数、有序 problem IDs 和 split SHA-256；评测入口会拒绝与 manifest 不一致的 processed 文件。原始数据与转换后的 tests 体积较大，分别保存在项目 `cache/huggingface/` 和被 Git 忽略的 `data/processed/`。

构造任何 SFT 或 GRPO 数据后，必须在训练前执行 `scripts/check_data_leakage.py`。检查范围包括 problem ID、标准化题面 SHA-256 和长题面的 SimHash 近重复；命中时命令以非零状态退出。Eval 的题面、代码、推理和 tests 都不得作为训练或 reward 调试数据。

## SFT Data v1（已冻结）

第一阶段仅使用 `nvidia/OpenCodeReasoning-2` 的 C++ split。数据源固定到 revision `eadf535931451525f3e5621d0f960c240bc62fd9`；数据集卡许可为 CC BY 4.0，同时每条样本继续保留上游 `license`、`dataset`、`split` 和 `index`，因为原始题面仍受各上游数据源条款约束。

OCR2 的 `question` 字段是 `-` 占位符。准备脚本按照官方方法从固定 revision 的 TACO、APPS、CodeContests 或 open-r1/codeforces 回填题面，禁止使用占位题面训练。verl 当前 `datasets` 已不执行旧式 dataset scripts，因此 TACO 直接读取同一固定 revision 内的 `ALL/*.parquet`，APPS 固定使用官方 `refs/convert/parquet` commit，同时保留其源仓库 commit。真实准备结果显示：250,000 rows 得到 7,849 个严格候选，400,000 rows 只增加到 8,487 个，题目在相邻 rows 中高度重复。为了不使用 `pass_rate=-1` 的未验证样本凑数，v1 固定扫描完整的 1,174,475-row C++ split、最多保留 30,000 个 problem-level 候选。扫描每完成 10,000 rows 就更新候选与 checkpoint 元数据；网络或下游失败后可以从最近的 checkpoint 继续。准备入口默认最多重启失败进程 20 次，以规避流式 HTTP 客户端关闭后引发的不可恢复进程状态；可通过 `SFT_PREPARE_MAX_ATTEMPTS` 和 `SFT_PREPARE_RETRY_DELAY_SECONDS` 调整。随后执行：

1. `judgement=right` 且 `pass_rate >= 0.8`，不接收 `pass_rate=-1`；
2. 拆分 `r1_generation` 的 reasoning 与最终 C++，并与 `solution` 交叉检查；
3. 过滤损坏、极短、异常长、interactive 及缺失题面的样本；
4. 使用 Eval v1 的题面 SHA-256 与 SimHash 指纹排除精确和近重复；
5. 以 GNU C++17 编译检查完整程序；
6. 按 difficulty/platform 交错排序，生成严格嵌套的 SFT-1K、5K、10K；
7. 使用固定 Qwen tokenizer 统计 problem、实际 prompt、reasoning、code、response 和 prompt+response 的 P50/P90/P95/P99；`prompt+response` 超过 16,384 tokens 时整条剔除并从候选池补齐，禁止截断末尾代码。

输出位于 Git 忽略的 `data/processed/`，包括三个训练 JSONL、`sft_stats.json`、contamination/dedup reports 和固定随机抽取的 `sft_audit_100.jsonl`。100 条 audit 已完成，结果、发现、修复和最终文件 SHA-256 见 [sft_v1_audit.md](sft_v1_audit.md)。
