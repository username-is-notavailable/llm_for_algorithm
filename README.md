# Qwen3-0.6B Code Post-Training

以 `Qwen/Qwen3-0.6B-Base` 为起点，使用 `verl` 建立可复现的代码能力后训练实验。

当前进度：Milestone 0（仓库与云端环境）已完成代码准备。本地 Windows 仅运行 CPU 单元测试；模型、CUDA、vLLM 和 verl 验证在 Linux NVIDIA GPU 实例执行。

## 本地开发

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
pytest
```

## 云端 smoke test

项目统一使用 Python 3.12，并先根据云实例 CUDA 版本安装匹配的 PyTorch。随后执行：

```bash
export VERL_REF=<verified-verl-tag-or-commit>
bash scripts/cloud_setup.sh
bash scripts/cloud_smoke_test.sh
```

`VERL_REF` 必须固定为 tag 或 commit，避免实验环境随 `main` 漂移。`cloud_setup.sh` 按 verl 官方 FSDP 安装路径调用其依赖安装脚本；首次租用 GPU 时应先核对所选镜像、CUDA 与该 revision 的兼容性。

也可直接运行：

```bash
python scripts/smoke_test_model.py --config configs/environment/smoke.yaml
```

运行记录写入 `outputs/experiments/<experiment_id>/`，包括解析后的配置、环境元数据和日志。

## 里程碑

- M0：仓库与云端环境（当前）
- M1：Code Verifier（等待确认后开始）
- M2：Evaluation Pipeline
- 后续阶段见项目方案文档
