#!/usr/bin/env bash
# Install the official WeVisOne-VL runtime next to this project (needs Python >= 3.10).
# Linux + NVIDIA GPU recommended; on CPU it runs but is slow (float32).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d third_party/WeVisOne-VL ]; then
  git clone --depth 1 https://github.com/Tencent/WeVisOne-VL.git third_party/WeVisOne-VL
fi
pip install -r third_party/WeVisOne-VL/requirements.txt
pip install -e third_party/WeVisOne-VL
pip install -r requirements.txt
echo "Done. Weights (tencent/WeVisOne-VL-4B or -2B) download from Hugging Face on first run."
