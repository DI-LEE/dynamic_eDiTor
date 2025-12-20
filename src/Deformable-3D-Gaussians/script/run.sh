#!/usr/bin/env bash

DATA_DIR="/home/ubuntu/PATH/data/DyCheck/mochi-high-five"
BASE_OUTPUT_NAME="mochi-high-five"
BASE_OUTPUT_ROOT="/home/ubuntu/PATH/dynamic_editor/src/Deformable-3D-Gaussians/output/${BASE_OUTPUT_NAME}"

# 10 targets and corresponding prompts
TARGETS=(van_gogh)
PROMPTS=(
"Turn the cat into a Van Gogh painting"
)

for i in "${!TARGETS[@]}"; do
  OUTPUT_DIR=/home/ubuntu/PATH/dynamic_editor/src/Deformable-3D-Gaussians/output/${BASE_OUTPUT_NAME}-${TARGETS[$i]}
  PROMPT="${PROMPTS[$i]}"
  mkdir -p "${OUTPUT_DIR}"
  BASE_OUTPUT_ROOT="/home/ubuntu/PATH/dynamic_editor/src/Deformable-3D-Gaussians/output/${BASE_OUTPUT_NAME}"
  cp -r "${BASE_OUTPUT_ROOT}/deform" "${OUTPUT_DIR}/deform"
  cp -r "${BASE_OUTPUT_ROOT}/point_cloud" "${OUTPUT_DIR}/point_cloud"

  python /home/ubuntu/PATH/dynamic_editor/src/Deformable-3D-Gaussians/edit_mono.py \
    -s "${DATA_DIR}" \
    -m "${OUTPUT_DIR}" \
    --iterations 30000 \
    --load_checkpoint 30000 \
    --edit_iterations 40000 \
    --prompt "${PROMPT}" \
    --idx 42 \
    --layer_range "0,29" 

done