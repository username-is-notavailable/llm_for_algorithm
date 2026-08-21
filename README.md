# Qwen3-0.6B Code Post-Training

以 `Qwen/Qwen3-0.6B-Base` 为起点，使用 `verl` 建立可复现的代码能力后训练实验。

当前进度：Milestone 0 至 Milestone 3 均已通过云端验收，当前进入 Milestone 4 Base Baseline。本地 Windows 仅运行 CPU 单元测试；模型、CUDA、vLLM 和 verl 验证在 Linux NVIDIA GPU 实例执行。

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

## 里程碑

- M0：仓库与云端环境（已完成）
- M1：Code Verifier（已完成）
- M2：Evaluation Pipeline（已完成）
- M3：Fixed Eval Set（已完成）
- M4：Base Baseline（当前）
- 后续阶段见项目方案文档
