#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: ./submit_experiment.sh <RUN_ID>"
    exit 1
fi

RUN_ID=$1

CSV_FILE="runs/${RUN_ID}/grid_search_${RUN_ID}.csv"

if [ ! -f "$CSV_FILE" ]; then
    echo "CSV file $CSV_FILE not found."
    exit 1
fi

NUM_LINES=$(($(wc -l < "$CSV_FILE") - 1))

if [ "$NUM_LINES" -le 0 ]; then
    echo "No jobs to run: CSV file has no data rows."
    exit 1
fi

ARRAY_RANGE="0-$(($NUM_LINES - 1))"

echo "Submitting job with RUN_ID=$RUN_ID using array range $ARRAY_RANGE based on $NUM_LINES configs."

sbatch --array=$ARRAY_RANGE run_single_job.sh "$RUN_ID"

