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

- `<think>...</think>` 只包含自然语言推理、算法说明、正确性分析和复杂度分析；
- 最终答案是唯一一个带 `cpp` 标记的 Markdown code block；
- code block 必须包含可独立编译执行的完整 C++17 程序；
- v1 不使用 `<answer>...</answer>`；提取器对 `<answer>` 的支持仅用于兼容外部数据和异常生成；
- 不在 `<think>` 中放 fenced code block，避免 verifier 误选中间代码；
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

Output Protocol v1 的标准输出应直接命中 `cpp` fenced block。兼容非标准输出时，提取优先级为：

1. `<answer>` 内的 `cpp` / `c++` block（外部兼容）；
2. 整个响应中的 `cpp` / `c++` block；
3. 可识别为 C++ 的普通 code block；
4. `<answer>` 内的原始 C++；
5. 整个响应中的原始 C++；
6. 提取失败。

在进入数据处理 Milestone 时，应为 renderer、协议解析、往返转换和异常样本建立独立测试。
