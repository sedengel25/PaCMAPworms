#!/bin/bash

RUN_ID=$1

if [ -z "$RUN_ID" ]; then
    echo "Usage: bash run_cluster_array.sh <RUN_ID>"
    exit 1
fi

RUN_PATH="runs/${RUN_ID}"
GRID_CSV="${RUN_PATH}/grid_search_${RUN_ID}.csv"

NUM_TASKS=$(($(wc -l < "$GRID_CSV") - 1))

if [ "$NUM_TASKS" -le 0 ]; then
    echo "Grid CSV scheint leer oder fehlerhaft: $GRID_CSV"
    exit 1
fi

echo "🚀 Launching Slurm Array with ${NUM_TASKS} tasks for ${GRID_CSV}"

sbatch --array=0-$(($NUM_TASKS - 1)) run_job.sh "$RUN_ID"

