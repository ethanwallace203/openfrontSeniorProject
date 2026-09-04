#!/usr/bin/env bash
# Render the latest current-stage showcase watches (see run_recent_watches.sh).
set -euo pipefail
cd /workspace
export PATH=/tmp/bin-stubs:$PATH
export BROWSER=none SKIP_BROWSER_OPEN=true PYTHONUNBUFFERED=1
export OF_FORCE_GPU=1 OF_REFUSE_SOFTGL=1
export DISPLAY=${DISPLAY:-:99}
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export OFSHOWCASE="${OFSHOWCASE:-/workspace/rust/target/release/ofshowcase}"
mkdir -p /workspace/artifacts
if [[ ! -f /tmp/.X99-lock ]]; then
  Xvfb :99 -screen 0 1920x1080x24 >/tmp/of-xvfb.log 2>&1 &
  sleep 1
fi

STATE=rust/checkpoints/ppo_v11/latest.state.json
UPD=$(python3 -c "import json;print(json.load(open('$STATE'))['update'])")
STAGE=$(python3 -c "import json;print(json.load(open('$STATE'))['stage'])")
PREFIX="ppo_v11_u${UPD}"

for TAG in "s${STAGE}_pangaea" "s${STAGE}_europe" "s${STAGE}_world" "s${STAGE}_asia"; do
  REC="showcase-clips/${PREFIX}_${TAG}.json"
  OUT="/workspace/artifacts/${PREFIX}_${TAG}.webm"
  LOG="showcase-clips/${PREFIX}_${TAG}.render.log"
  if [[ ! -f "$REC" ]]; then
    echo "MISSING $REC" | tee "$LOG"
    continue
  fi
  echo "=== RENDER $TAG ===" | tee "$LOG"
  uv run --no-project --with playwright python scripts/render_client_replay.py \
    --record "$REC" --out "$OUT" \
    --speed max --timeout 2400 --trim-gameplay --width 1280 --height 720 --device-scale-factor 1 \
    2>&1 | tee -a "$LOG"
  echo "RENDER_EC_${TAG}:${PIPESTATUS[0]}" | tee -a "$LOG"
  ls -lh "$OUT" | tee -a "$LOG"
done
echo ALL_RECENT_RENDERS_DONE
