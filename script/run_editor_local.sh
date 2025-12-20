#!/bin/bash

declare -A prompts
prompts["red_shirt"]="Change his shirt to a red shirt"

prompts_order=(
    "red_shirt"
)

scene_names=(
    "sear_steak"
)

LAYER_RANGES="0,29"

mkdir -p logs

echo "Running Dynamic-eDiTor"

for ((i=0; i<${#prompts_order[@]}; i++)); do
    key=${prompts_order[$i]}
    prompt=${prompts[$key]}
    SCENE_NAME=${scene_names[$i]}  

    echo "======================================================"
    echo "===== Dynamic-eDiTor: $key ====="
    echo "======================================================"
    echo "Prompt: $prompt"
    echo "======================================================"

    output_root_name="editor_${DATE_SUFFIX}"

    CUDA_VISIBLE_DEVICES=0 python ../src/4DGaussians/edit_4dgs_local_caching.py \
        -s "../../data/DyNeRF/${SCENE_NAME}" \
        -m "../output/${output_root_name}/${SCENE_NAME}_${key}" \
        --expname "${SCENE_NAME}_${key}" \
        --checkpoint_path "../../data/DyNeRF/${SCENE_NAME}/output" \
        --configs "../src/4DGaussians/arguments/dynerf/${SCENE_NAME}.py" \
        --start_checkpoint 14000 \
        --editing_iterations 20000 \
        --prompt "$prompt" \
        --idx 62 \
        --editing_flag \
        --write_local_log \
        --densification_iterations 7000 \
        --edit_fps 1 \
        --layer_range "$LAYER_RANGES" \
        --rendering_flag \
        > logs/${key}_full.out 

done

