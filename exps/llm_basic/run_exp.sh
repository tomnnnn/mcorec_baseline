#!/bin/bash

# This script is used to run the LLM basic experiment

MODELS=(
        "Qwen/Qwen3-8B-AWQ"
        "meta-llama/Llama-3.2-3B-Instruct"
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

mkdir -p logs

for MODEL in "${MODELS[@]}"; do
	MODEL_NAME=$(basename $MODEL)
	qsub -v "MODEL=$MODEL" job.sh -o "logs/${MODEL_NAME}.out" -e "logs/${MODEL_NAME}.err" -N "llm_basic_${MODEL_NAME}"

	echo "Submitted job for model: $MODEL"
done
