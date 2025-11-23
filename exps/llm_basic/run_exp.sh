#!/bin/bash

# This script is used to run the LLM basic experiment

OUTPUT_DIR="outputs/$EXP_NAME"
MODEL=$1
LOG_FILE="$OUTPUT_DIR/exp-{MODEL}.log"

mkdir -p $OUTPUT_DIR

echo "Starting experiment with model: $MODEL"

cd ../../
python scripts/cluster_llm.py --model $MODEL
