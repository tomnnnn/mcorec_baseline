#!/bin/bash

# This script is used to run the LLM basic experiment

OUTPUT_DIR="outputs/$EXP_NAME"
MODELS=(
        "Qwen/Qwen3-8B-AWQ",
        "meta-llama/Llama-3.2-3B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

mkdir -p $OUTPUT_DIR

for MODEL IN "${MODELS[@]}"; do
	qsub -v "MODEL=$MODEL" job.sh -o "logs/${MODEL}.out" -e "logs/${MODEL}.err" -N "llm_basic_$(basename $MODEL)"
	echo "Submitted job for model: $MODEL"
done
