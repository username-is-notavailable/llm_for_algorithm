# Qwen3-0.6B Code Post-Training

以 `Qwen/Qwen3-0.6B-Base` 为起点，使用 `verl` 建立可复现的代码能力后训练实验。

当前进度：Milestone 0（仓库与云端环境）正在云端原生 Linux GPU 环境重新验收。本地 Windows 仅运行 CPU 单元测试；模型、CUDA、vLLM 和 verl 验证在 Linux NVIDIA GPU 实例执行。

## 本地开发

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
pytest
```

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

依赖下载优先使用阿里云 PyPI，官方 PyPI 作为新版本 PyTorch/CUDA 包缺失时的回退。可在运行 setup 前通过 `UV_INDEX` 和 `UV_DEFAULT_INDEX` 覆盖。

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

## 里程碑

- M0：仓库与云端环境（云端原生环境重验中）
- M1：Code Verifier（等待确认后开始）
- M2：Evaluation Pipeline
- 后续阶段见项目方案文档
