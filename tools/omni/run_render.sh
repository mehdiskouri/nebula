#!/bin/bash
# Headless Omniverse RTX render of the Nebula burning tree -> orbit MP4 + hero PNG.
# Usage: tools/omni/run_render.sh [N_FRAMES] [W] [H] [ACC]
set -e
ROOT=/workspace/nebula
KIT="$ROOT/kit-app-template/_build/linux-x86_64/release/nebula.omniverse_usd_composer.kit.sh"
export NEBULA_USD="${NEBULA_USD:-$ROOT/demo_output/omniverse_scene/scene_render.usd}"
export NEBULA_OUT="${NEBULA_OUT:-$ROOT/demo_output/_omni_frames}"
export NEBULA_N="${1:-36}"
export NEBULA_W="${2:-900}"
export NEBULA_H="${3:-1150}"
export NEBULA_ACC="${4:-70}"
rm -rf "$NEBULA_OUT"; mkdir -p "$NEBULA_OUT"

echo "[run_render] launching Kit headless RTX..."
"$KIT" \
  --no-window \
  --/app/window/hideUi=true \
  --/app/extensions/registryEnabled=false \
  --/persistent/app/viewport/displayOptions=0 \
  --/rtx/rendermode="PathTracing" \
  --exec "$ROOT/tools/omni/render.py" \
  2>&1 | grep -E "\[nebula\]|error|Error|Fatal|Traceback" | tail -60 || true

NF=$(ls "$NEBULA_OUT"/f*.png 2>/dev/null | wc -l)
echo "[run_render] captured $NF frames"
if [ "$NF" -gt 0 ]; then
  ffmpeg -y -framerate 14 -i "$NEBULA_OUT/f%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 18 \
    "$ROOT/demo_output/omniverse_burning_tree.mp4" >/dev/null 2>&1
  # hero = an early frame
  cp "$(ls "$NEBULA_OUT"/f*.png | sed -n '6p')" "$ROOT/demo_output/omniverse_burning_tree.png" 2>/dev/null || \
    cp "$(ls "$NEBULA_OUT"/f*.png | head -1)" "$ROOT/demo_output/omniverse_burning_tree.png"
  echo "[run_render] wrote demo_output/omniverse_burning_tree.mp4 + .png"
else
  echo "[run_render] NO FRAMES captured — see Kit output above"
fi
