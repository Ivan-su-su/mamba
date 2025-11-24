#!/bin/bash
source /home/chubin/miniconda3/etc/profile.d/conda.sh
conda activate mambafusion
MODEL_DIR="AirV2X-Perception-Checkpoints/airv2x_intermediate_gaussian"
gpu_id=${1:-0}
train=${2:-test}
if [ "$train" = "test" ]; then
    python opencood/tools/inference_multi_scenario.py \
        --model_dir "${MODEL_DIR}" \
        --eval_best_epoch \
        --gpu_id ${gpu_id}
fi

if [ "$train" = "train" ]; then
    python opencood/tools/train.py \
        -y "${MODEL_DIR}/config.yaml" \
        --gpu_id ${gpu_id}
fi