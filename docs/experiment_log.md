# Experiment Log

## M8 — Code Agent v1 protocol

- 冻结两个模型动作：`execute_code` 与 `final`；
- 最多三次 visible execution feedback，外加一次 final candidate；
- 超额 execute 明确转换为 auto-final，不静默丢弃候选；
- hidden tests 在每个候选上只做后台审计，从不进入模型反馈；
- 新增 action parsing、trajectory schema、feedback formatter、ExecutionBackend、同步 controller 和
  Agent metrics；
- `LocalVerifierBackend` 仅用于可信代码的本地开发，不宣称强安全隔离。

## M9 — Agent prompting baseline（完成）

- 固定 Qwen3-1.7B-Base revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`；
- 固定 Qwen3-4B-Base revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`；
- 官方 post-trained Qwen3-1.7B protocol reference 固定 revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`；
- 官方 post-trained Qwen3-4B protocol reference 固定 revision
  `1cfa9a7208912126459214e8b04321603b3df60c`；
- 从冻结 LiveCodeBench dev 构造 10 题 smoke / 60 题 agent-dev；
- 共 221 个 visible tests、1,104 个 hidden tests；
- one-shot 和 Agent final 共享 hidden tests，Agent execution 只使用 visible tests；
- 新增真实 tokenizer/chat + vLLM 多轮 generator、context-aware generation cap；
- 新增 trajectory artifacts、resume、两 GPU problem sharding 和严格 shard merge；
- 无可提取代码或 length 截断的生成同样写入 trajectory；hidden testcase 指标按完整题集计分；
- 支持用环境变量临时覆盖单轮/累计生成预算，长输出诊断与冻结 baseline 使用不同 experiment ID；
- 2026-08-24，官方 post-trained Qwen3-4B 在冻结 60 题 agent-dev 上完成 8K 诊断：
  - Git commit：`979a5fb`；one-shot experiment：
    `m9-oneshot-qwen3-4b-posttrained-dev-v1-long8k-20260824-211926`；
  - Agent experiment：`m9-agent-qwen3-4b-posttrained-dev-v1-long8k-20260824-221532`；
  - one-shot pass@1：28/60（46.7%）；Agent 同轨迹首次成功：24/60（40.0%），最终成功：
    32/60（53.3%），净增 8 题（+13.3 pp）；
  - 首次失败 36 题中 9 题修复成功，repair success 25.0%；其中 8 条从 compile error
    恢复、1 条从 wrong answer 恢复；另有 1 条首次正确后退化；
  - easy/medium/hard Agent success 分别为 90.0%/52.6%/19.0%，相对首次生成分别提升
    5.0/21.1/14.3 pp；
  - action validity 0.8%，fallback 99.2%，explicit final 1.7%；终止原因为 repeated code 31、
    code extraction failed 20、success 5、final incorrect 3、model stop without code 1；
  - 对全部 60 条做结构化审计，并人工核验全部 9 条 repair 与 1 条 regression；确认存在真实利用
    compiler/test feedback 的修复，同时暴露工具协议、停止决策和效率上的明确训练空间；
  - 结论：M9 验收完成。1.7B 保留为低成本训练 gate，4B 作为正式主模型；M10 优先构造
    Agent action/termination 与 execution-guided repair 数据。

每次云端实验记录：experiment ID、Git commit、配置路径、GPU/CUDA/依赖版本、命令、结果与异常。

## M10 — API repair data pipeline（进行中）

- 冻结架构为“多 GPU 官方 post-trained 4B data producer → CPU verifier → 百炼 API repair
  worker”；取消本地 teacher 修复层；
- producer 的 verified clean/short 输出进入 one-shot pool，完整错误代码进入 repair pool；截断无代码
  和明显循环输出不作为训练 target；
- 训练初始化仍先使用 Base，并将 producer/student provenance 分开记录；Base gate 失败时用完全相同
  数据切换到官方 post-trained initialization；
- primary teacher 为 `qwen3-8b`，更强模型由固定 failure 子集 bake-off 后选择；
- 百炼接口使用 OpenAI-compatible streaming API，默认 1M TPM / 600 RPM、16 workers；
- API key 仅从 `DASHSCOPE_API_KEY` 读取，reasoning/content/usage/request ID 分离保存；
- 新增 SQLite 持久化任务队列、lease 恢复、幂等 task ID、并发限流和 API 重试；
- repair 每轮都使用真实 visible execution feedback；hidden evaluation 只用于数据 gate；
- 50 条 pilot 只接收 full-test success、全部 action explicit 且最终显式 `final` 的 trajectory。
- M10 failure producer 提供独立 A100 40GB throughput profile：保留 16K context/8K generation，
  将每卡 `max_num_seqs` 与 request batch 提升到 8；24GB 默认配置保持为并发 2。
- API repair prompt 要求最多五条短分析、不复述题面、优先最小修改；visible tests 通过后必须直接
  `final` 并复用最后通过的代码，避免 teacher 产生新的退化候选。
- 2026-08-25，8B pilot 对 98 条 failure 完成 API repair：严格接收 9 条，但实际有 37 条通过
  full tests；其中 28 条仅因缺少显式 action 被拒。54 条真实修复失败，另有 6 条 visible-pass /
  hidden-fail 无可用反馈、1 条复验已正确。253 次响应中 explicit action 91 次、fallback 162 次。
- 数据阶段允许对 full-test verified 输出做可审计的 action canonicalization：只补齐实际执行的
  `execute_code/final` 标签，不修改 reasoning、代码或 JudgeResult。恢复 37 条 8B repair 数据；
  54 条真实失败以可见通过率最好的 8B 候选为起点路由到 `qwen3-32b`。
- 32B escalation 对 54 条输入新增 7 条 full-test success（严格接收 5、仅协议拒绝 2）；4 条
  visible-pass/hidden-fail 无反馈，43 条真实失败。失败以算法 WA 为主，且 thinking token 较长，
  不通过单纯放宽 token 重试；固定抽取 10 条 difficulty-balanced 子集测试 `qwen3-coder-next`。
- clean-cache source audit 表明 fresh TACO 与 frozen tests 为 300/300 完全一致，排除本地缓存
  损坏；OCR2 reference code 仅 217/300 full-pass，83 条不可信（81 WA、2 runtime error）。
  正式 M10 preparation 因此加入 reference full-pass gate，并继续扫描固定 600 候选直至补足
  300 条干净任务；旧 failure rollout 不得与新 manifest 混用。
- TACO native-solution 对照抽样中，OCR2-pass 组 9/9 有解答样本由 TACO solution full-pass；
  OCR2-fail 组 0/8 full-pass，且两组题面均 10/10 与 TACO row 精确一致。这否定了简单的本地
  index 错配假设，并暴露 TACO 部分 tests/solutions 内部不自洽。后续 Agent smoke 不再依赖
  OCR2，直接扫描 TACO，并要求至少一份 native solution 本地 full-pass；先冻结 200 条。
- 2026-08-25，首批 native-validated smoke 数据冻结完成：200 个唯一问题、19,396 个 tests，
  其中 visible 934、hidden 18,462，二者无重叠；全部 200 份用于 gate 的 native solution 均
  full-pass。每题 tests 数量 p50=101、p90=180、p95=200、max=200；difficulty 为 easy 53、
  medium 59、hard 47、unknown 41。数据 SHA-256 为
  `ec04aba7b92ddddedc500362c8d80c5aeddcb0566c7693143e29545218da73e2`。
- 官方 post-trained Qwen3-4B 在两张 A100 40GB 上完成 200/200 首次生成：57 条 full-pass、
  36 条生成了可执行但未通过全部测试的代码、107 条在 8,192 tokens 时仍未输出代码。最终冻结
  28 条不超过 4,096 tokens、正常 stop 且无明显重复的 one-shot，以及 36 条真实 failure。
  全体 pass@1=28.5%；126/200 因 length 结束。该结果表明主要损耗来自过长 thinking，而不是
  分片或 verifier；无代码输出不进入 repair SFT，后续另做短思考 producer 消融。
- 新增本地两阶段 teacher distillation：`qwen3-8b` 覆盖全部 200 题，任何未满足完整测试与简短
  target gate 的题自动交给 `qwen3-32b` 独立重做。两阶段均使用 SQLite 断点队列和同一 full-test
  verifier；私有 reasoning 与 visible target 分离保存，生成硬上限 8K、visible target gate 4K。
  原有 36 条 4B failure 继续走独立 execution-feedback repair pipeline，避免 one-shot 蒸馏取代
  self-repair 研究主线。
- 2026-08-25，两阶段 one-shot distillation 完成：8B full-pass 36/200（18.0%）；其余 164 条
  交给 32B 后 full-pass 95/164（57.9%），合计得到 131/200（65.5%）verified one-shot，剩余
  69 条均为 verification failure。两阶段所有响应均正常 stop、均成功抽取代码；32B 的 164/164
  响应均含单个标准 C++ fence，可见 target 最大估算 1,562 tokens。因此当前 one-shot 瓶颈是
  correctness 而非格式，无需切换 native tool calling。32B 私有 thinking 成本仍偏高：completion
  tokens p50=12,029、p95=21,597；后续单独做 thinking budget/关闭 thinking 消融。
- 2026-08-26，native 36-task repair smoke 中严格接收 15 条，实际 full-test success 17 条；79
  个 steps 中 77 个 explicit action，2 条仅因首步 action 缺失被协议拒绝。另有 4 条初始代码
  visible-pass/private-fail 且无反馈。由此冻结 adaptive counterexample policy：最多将一条失败
  private case 迁移为 feedback case、至少保留一条 private test，并在 `final` 对完整测试并集重新
  验证。trajectory schema 升级为 v2，记录 reveal index/count/private remaining；native tool calling
  暂不作为 one-shot 或 repair 的阻塞项。
- 2026-08-26，开始评估以 CodeContests+ 替换 TACO executable source。主仓库固定 revision
  `96c850540fade31d384a25766461e0da6b08f5fc`，独立 `Code-Contests-Plus-Verified` 仓库当前
  无法访问，因此按题内 `true_positive_rate >= 0.9` 且 `true_negative_rate >= 0.9` 本地重建
  verified 子集。1x 首 shard、首 row group 有 76 条同时具备 checker、tests 和 C++ 正误提交的
  eligible problems；确定性抽样 8 题后，8/8 checker 使用固定 testlib revision 编译成功，8/8
  正确提交通过全部 1x tests，8/8 错误提交被拒。仅使用每题前 8 个 tests 时错误提交拒绝率为
  5/8，说明 rollout 不应任意截断测试集。该 smoke 支持迁移，但正式决定前仍需跨 shard 扩大审计，
  并将 checker 执行接入隔离 sandbox。
- 随后扩展到 5 个 shards 各 1 个 row group，共得到 217 个 eligible problems，并确定性抽样 50
  题使用全部 1x tests。修正本地 judge parity（定义 `ONLINE_JUDGE`，输出上限由 1 MiB 调整为
  16 MiB）后：checker compile 50/50，已标记正确提交 full-pass 48/50，已标记错误提交被拒
  49/50。剩余 3 个 label/execution 异常与数据集公布的非完美 TPR/TNR 一致，说明不能直接相信
  submission label，正式数据仍须执行 gate。结论为迁移 go：CodeContests+ 作为 executable 主源，
  TACO 降为补充；下一步实现 problem-level split、sandbox checker adapter 与 locally-verified
  correct/incorrect pools。
- CodeContests+ 迁移 smoke 已落地：5 shards 中扫描候选并冻结 50 个 checker-backed problems，包含
  1,148 tests（visible 422、private 726）和 50 个经本地复验的真实错误起点；错误类型为 44 WA、
  3 runtime、3 compile。筛选过程中拒绝 56 个未达 TPR/TNR 阈值、1 个 incorrect/full-pass 冲突、
  3 个 correct/non-full-pass 冲突。checker contract 已接入 Agent visible execution、hidden gate 与
  final gate；数据重新加载后的端到端 backend smoke 正确复现 WA。TACO pipeline 保留但不再是 M10
  默认主线。
- CodeContests+ 50-task repair smoke 使用 `qwen3-8b` 完成：严格接收 28/50，真实 full-checker
  success 30/50；2 条仅首轮缺失显式 action，按既定规则规范化后冻结 30 条 trajectory。其余 20 条
  为真实失败：12 final incorrect、3 token budget exhausted、3 repeated code、2 execution budget
  exhausted。失败任务中 17/20 的最佳 visible pass rate 至少 0.5，表明 8B 通常取得部分修复进展。
  已从每条失败轨迹抽取最佳 visible 候选，导出 20 条 checker-backed `qwen3-32b` escalation tasks；
  抽样重新执行得到 visible 7/9、private 9/14，确认导出保留真实中间状态而非退回原始错误代码。
- 后处理规则升级为 success-preserving truncation：若任意 `execute_code` 同时通过全部 visible 与
  当时剩余 private tests，则截断后续退化 turns，并追加复用完全相同代码的显式 `final`；只合成
  action wrapper，不修改 reasoning、代码或 JudgeResult，并记录 normalization provenance。回放 8B
  run 恢复 3 条 success-before-regression，故 8B 冻结数由 30 增至 33、32B escalation 输入由 20
  降至 17。已完成的 32B run 中另有 9 条 full-checker success（严格 3、协议规范化 6），最终两阶段
  可用 repair trajectories 为 42/50，剩余真实失败 8 条。
- M10 官方 4B producer 使用 4090 24GB 安全配置：BF16、16K context、8K 最大输出、单卡
  `max_num_seqs=2`，按 problem 多卡分片，不使用 tensor parallel。
- M11 Agent SFT smoke 数据已冻结：M10 checker-backed trajectories 共 42 条，稳定划分为 34 train /
  8 dev；总计 143,474 chat tokens、32,610 assistant target tokens。样本总长度 train
  min/median/max 为 1,039/2,394/14,850，dev 为 1,365/2,069/6,086，均完整落在 16,384-token
  上限内。训练使用 Qwen3-1.7B-Base、assistant-only loss（含 `<|im_end|>`）、4 epochs，并按 epoch
  保存 checkpoint 和计算 dev loss；这是数据/协议可学性的低成本 pilot，不作为最终能力结论。
- M11 首次 4x A100-40GB DDP 在 step 6 遇到 14,850-token 样本，rank 0 backward 额外申请
  8.41 GiB 时 OOM；此前 5 steps 正常，确认是单条长尾而非总显存或 DDP 故障。pilot 不引入 FSDP
  且不截断 target，改为显式排除该 1 条并记录 provenance；实际训练集 33 条、最长 9,837 tokens、
  max length 10,240，8 条 dev 不变。
- M11 第二次运行已完整完成 epoch 1 的 9 个训练 steps，但进入 dev evaluation 后 OOM。根因是
  Transformers 默认 `per_device_eval_batch_size=8`，8 条不同长度序列一起 padding，causal-LM loss
  将 151K-vocabulary logits 转为 FP32 时额外申请 27.57 GiB；不是训练长度仍超限。训练入口现统一
  将 eval batch 默认设为 1，M11 配置显式冻结为 1，无需升级至 80GB。
- M11 v1 四个 checkpoint smoke 显示 valid action 均为 100%，但 repair success 均为 0%；epoch 2
  行为最好（explicit final 100%、无 repeated code、full-test case pass 26.5%），epoch 4 退化至
  agent success 0。数据复审确认初始错误程序位于 user context、原本已 mask；实际缺陷是 native
  structured action/canonicalization 更新了 step submission，却未同步历史 prompt snapshot，少量
  规范化前 assistant 文本进入 target。Agent SFT v2 改为从 canonical steps 重建所有 assistant
  turns，逐消息显式记录 `trainable`，并要求全部 93 个监督 turns 具有文本 action prefix。
- M11 v2 checkpoint Agent smoke 仍为 repair success 0/10；进一步审计发现更主要的 distribution
  mismatch：teacher 看到的是把错误代码与反馈拼入 user 的 repair request，而正式 Agent 在独立的
  assistant/tool turns 中接收自己的执行结果。v3 按真实 loop 重建消息，初始错误 assistant 设为
  context-only、初始 observation 独立为 tool，并将三次 execution budget 贯穿整条轨迹。42 条中
  3 条需要 4 次 execution 而拒绝，冻结候选为 31 train / 8 dev。正式训练前将扩充至至少
  300–500 条，并混入 one-shot correct 数据保护首次编码能力。
- M11 v2 首次运行训练与 eval 正常完成 2 epochs（dev loss 0.2578 → 0.2530），但保存
  checkpoint-18 的 optimizer state 时在约 5.22GB 处发生 filesystem iostream error；这是磁盘空间/
  配额问题，不是 CUDA OOM。该 pilot 的 epoch checkpoint 仅用于推理比较、不需要 resume，故 M11
  冻结为 `save_only_model=true`，保留四轮模型权重而不重复保存大体积 optimizer state。
- M11 正式扩充源池得到 300 个 CodeContests+ accepted problems。100-candidate 固定前缀审计中
  accepted 39、quality threshold rejected 56、correct local gate rejected 4、incorrect/full-pass
  conflict 1；quality rejection 主要来自 TNR 低于 0.9，因此继续保留 TPR/TNR 0.9 门槛。原始三个
  JSONL 各 300 行但因重复 tests/checker 合计约 17 GB；compact v2 将 problem 环境单独存储并以
  byte-offset index 按需读取，failure pool 缩至 1,053,070 bytes、one-shot seeds 缩至 758,924
  bytes，完整 problems 为 3,709,608,541 bytes。300 个引用、判题摘要和 problem IDs 全量校验通过，
  最终四文件 SHA-256 见 `data/splits/codecontests_plus_repair_300_v2_manifest.json`。
- M11 qwen3-8b repair 完成 300/300：严格接收 143；后处理另恢复 7 条 full-pass 但 action protocol
  不规范的轨迹，以及 3 条 success-before-regression 轨迹，故冻结 8B canonical 153 条。其余 147
  条全部进入 qwen3-32b escalation：42 repeated code、49 final incorrect、30 token budget、25
  execution budget、1 model stop without code；最后一条无 8B 候选代码，明确回退到数据集原始错误
  提交。原始 accepted/rejected 因 prompt snapshots 重复 checker/tests 合计约 6 GB，流式 compact 后
  canonical 为 6,394,184 bytes、escalation 为 624,489 bytes，哈希见
  `data/splits/m11_repair_8b_escalation_v1_manifest.json`。
- M11 正式 Agent SFT v3 将 300 条 one-shot 与 238 条 checker-backed repair 候选按真实在线状态
  重建；5 条 repair 因“初始失败 + teacher 修复”超过 3 次 execution budget 而排除，最终按 problem
  去重后冻结 497 train / 36 dev。8K 完整长度门槛再排除 8 条，实际训练为 490 train / 35 dev；
  repair 的初始错误 assistant turn 仅作为 mask 后的 context，execution feedback 使用独立 tool turn，
  teacher 修复和 final 才计算 assistant-only loss。
- 2026-08-27，Qwen3-4B 官方后训练模型起点、3x A100-40GB、LR `2e-6`、1 epoch 的 v3 pilot
  完成 41 steps（937,568 input tokens），train loss 0.4594、dev loss 0.4008，运行 797 秒；产物为
  `m11-agent-sft-v3-4b-post-rollout-lr2e6-pilot-20260827-220626`。固定 10 题三卡 Agent smoke 得到
  first-attempt/agent success 3/10、repair success 0/7、valid action 100%；termination 为 success 3、
  repeated code 5、execution budget 1、final incorrect 1。虽然最终成功率未超过官方模型的
  first-attempt 4/10、agent 5/10，但 v3 在 7 条首轮失败轨迹中有 4 条至少生成一次不同代码，
  相比 v2 LR `2e-6` 的 7/7 失败均直接重复已有明显行为改善。结论：rollout-aligned schema 有效，
  但 233 条 repair supervision 尚不足以产生成功修复；下一轮应优先扩充高质量 repair 状态并改善
  feedback/target 配对，而不是增加 epoch。
- M12 数据扩充入口已准备：现有 Agent eval 保持冻结的 smoke 10 / dev 60，不重新抽样；训练侧从
  同一 CodeContests+ revision 筛选 1000 个新问题，并通过旧 300 题 compact index 显式排重。
  生产顺序冻结为 checker-backed source → compact → qwen3-8b repair → 失败任务 qwen3-32b
  escalation → full-checker audit → rollout-aligned SFT。各 API 阶段均使用可恢复 SQLite queue。
- M12 首次 source preparation 在 accepted 约 53/1000 时因 WSL 重启丢失进程内结果，确认旧入口
  只在全部完成后写文件。新入口改为边筛选边 compact，并以原子 checkpoint 保存 candidate index、
  counters、problem byte-offset index 和三份输出的 committed sizes；重启后自动截断未提交尾部并
  续跑。同时移除随 accepted rows 线性增长的内存列表，适配 WSL 24GB 内存上限。

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
  - M7-v2 4,096-token response pilot（256 samples，1 epoch）greedy smoke：6/10 length、4/10 stop，extraction 0.5、compile 0.3、pass@1 0、平均 10,028.3 tokens；比 v1 改善但未通过 gate；
  - 同一 pilot 的 temperature 0.7 / top-p 0.95 诊断：4/10 length、6/10 stop，extraction 0.7、compile 0.4、pass@1 0、平均 7,150.6 tokens；采样能偶尔逃离循环但仍不稳定，不修改冻结的 greedy 正式协议；
  - 下一控制变量：保持训练与 greedy eval 设置不变，将完整样本 response cap 降至 2,048 tokens 后重新运行 256-sample pilot。
  - 2,048-token compact pilot 的 1-epoch greedy smoke：双卡 9/10 length、extraction 0.1、compile/pass@1 0；单卡 8/10 length、extraction 0.1、compile/pass@1 0。Base 同环境回归正常（extraction 1.0、compile 0.5、pass@1 0.2），因此排除评测环境与 DDP 为主因；
  - 补充评测原 M6 final（最短 64 samples，100 steps，约 12.5 epochs）：固定 dev smoke 得到 6/10 stop、4/10 length、extraction 0.8、compile 0.4、pass@1 0.1。多次接触完整 target 能从首 epoch collapse 中恢复格式，但仍弱于 Base；
  - 下一控制实验：compact 256 从 Base 训练 4 epochs，每个 epoch 保存并使用相同 greedy smoke，测量格式恢复与能力变化曲线，而不是继续缩短 response。
