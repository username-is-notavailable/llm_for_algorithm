#!/usr/bin/env bash
set -euo pipefail

: "${VERL_REF:?Set VERL_REF to a verified verl tag or commit for reproducibility}"

python - <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python 3.12 is required, got {sys.version.split()[0]}")
PY

python -m pip install --upgrade pip
python -m pip install -e ".[cloud]"

# Use verl's own installer because vLLM and PyTorch compatibility is revision-specific.
mkdir -p .third_party
if [[ ! -d .third_party/verl/.git ]]; then
  git clone https://github.com/verl-project/verl.git .third_party/verl
fi
git -C .third_party/verl fetch --tags origin
git -C .third_party/verl checkout --detach "${VERL_REF}"
(
  cd .third_party/verl
  USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
  python -m pip install --no-deps -e .
)

python -c "import verl, vllm; print('verl/vLLM import check passed')"
