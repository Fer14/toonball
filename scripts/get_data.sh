#!/usr/bin/env bash
# Download clips and model weights. Usage: scripts/get_data.sh [dfl|kaggle|weights|all]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/clips" "$ROOT/weights"
GDOWN="uvx gdown"

weights() {
  # roboflow/sports pretrained YOLOv8 football models
  [[ -f "$ROOT/weights/football-player-detection.pt" ]] || $GDOWN -O "$ROOT/weights/football-player-detection.pt" "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"
  [[ -f "$ROOT/weights/football-ball-detection.pt" ]]   || $GDOWN -O "$ROOT/weights/football-ball-detection.pt"   "https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"
  [[ -f "$ROOT/weights/football-pitch-detection.pt" ]]  || $GDOWN -O "$ROOT/weights/football-pitch-detection.pt"  "https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"
  # body pose for avatar animation (Ultralytics YOLO11)
  [[ -f "$ROOT/weights/yolo11m-pose.pt" ]] || curl -sSL -o "$ROOT/weights/yolo11m-pose.pt" "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt"
  # 2D → 3D pose lifting: MotionAGFormer-B (default) and VideoPose3D (fallback), Human3.6M
  [[ -f "$ROOT/weights/motionagformer-b-h36m.pth.tr" ]] || $GDOWN -O "$ROOT/weights/motionagformer-b-h36m.pth.tr" "https://drive.google.com/uc?id=1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP"
  [[ -f "$ROOT/weights/videopose3d_h36m_detectron_coco.bin" ]] || curl -sSL -o "$ROOT/weights/videopose3d_h36m_detectron_coco.bin" "https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin"
}

dfl() {
  # DFL Bundesliga Data Shootout clips (mirrored by roboflow/sports)
  declare -A ids=(
    [08fd33_0]=1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-
    [0bfacc_0]=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF
    [2e57b9_0]=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf
    [573e61_0]=1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU
    [121364_0]=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu
  )
  for name in "${!ids[@]}"; do
    [[ -f "$ROOT/clips/$name.mp4" ]] || $GDOWN -O "$ROOT/clips/$name.mp4" "https://drive.google.com/uc?id=${ids[$name]}"
  done
}

kaggle() {
  # shreyamainkar/football-soccer-videos-dataset — public per-file download
  for f in ${KAGGLE_FILES:-106.mp4}; do
    out="$ROOT/clips/kaggle_${f%.mp4}.mp4"
    [[ -f "$out" ]] && continue
    tmp="$(mktemp -d)"
    curl -sSL -o "$tmp/f.zip" "https://www.kaggle.com/api/v1/datasets/download/shreyamainkar/football-soccer-videos-dataset/$f"
    unzip -q -o "$tmp/f.zip" -d "$tmp" && find "$tmp" -name '*.mp4' -exec mv {} "$out" \;
    rm -rf "$tmp"
  done
}

case "${1:-all}" in
  weights) weights ;; dfl) dfl ;; kaggle) kaggle ;;
  all) weights; dfl; kaggle ;;
esac
ls -lh "$ROOT/clips" "$ROOT/weights"
