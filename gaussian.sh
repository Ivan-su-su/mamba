#!/bin/bash
source /home/chubin/miniconda3/etc/profile.d/conda.sh
conda activate mambafusion
MODEL_DIR="AirV2X-Perception-Checkpoints/airv2x_intermediate_gaussian"
gpu_id=${1:-0}

python opencood/tools/inference_multi_scenario.py \
    --model_dir "${MODEL_DIR}" \
    --eval_best_epoch \
    --gpu_id ${gpu_id} \