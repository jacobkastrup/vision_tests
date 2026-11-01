#!/usr/bin/env bash
# Install everything on a Linux machine (GPU VM, Colab, RunPod, ...). Needs Python >= 3.10.
#   ./setup.sh            WeVisOne-VL (GPU) + PP-OCRv6
#   ./setup.sh --ppocr    PP-OCRv6 only: runs on any laptop CPU (mode "full")
set -euo pipefail
cd "$(dirname "$0")"
if [ "${1:-}" != "--ppocr" ]; then
  if [ ! -d third_party/WeVisOne-VL ]; then
    git clone --depth 1 https://github.com/Tencent/WeVisOne-VL.git third_party/WeVisOne-VL
  fi
  # vllm/gradio are only for WeVisOne's fast backend / web demo; install vllm yourself if wanted
  grep -v -E '^(vllm|gradio)' third_party/WeVisOne-VL/requirements.txt > /tmp/wevisone-req.txt
  pip install -r /tmp/wevisone-req.txt
  pip install -e third_party/WeVisOne-VL
fi
pip install -r requirements.txt
echo "Done. Model weights download on first run (WeVisOne-VL from Hugging Face, PP-OCRv6 from Paddle's hosts)."
