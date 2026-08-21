# Data

Milestone 0 不引入数据集。后续数据均按 `problem_id` 做 problem-level 隔离，并在此记录来源、许可、版本、清洗和切分信息。

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
