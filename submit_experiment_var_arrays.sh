#!/bin/bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: ./submit_experiment_var_arrays.sh <RUN_ID>"
    exit 1
fi

RUN_ID=$1
CSV_FILE="runs/${RUN_ID}/grid_search_${RUN_ID}.csv"

[ -f "$CSV_FILE" ] || { echo "CSV file $CSV_FILE not found."; exit 1; }

NUM_LINES=$(($(wc -l < "$CSV_FILE") - 1))   
[ "$NUM_LINES" -gt 0 ] || { echo "No jobs to run."; exit 1; }

# Max pro Array: 70k (oder aus Slurm ziehen)
MAX_PER_ARRAY=$(scontrol show config 2>/dev/null | awk -F= '/MaxArraySize/ {gsub(/ /,"",$2); print $2}')
: "${MAX_PER_ARRAY:=70000}"

echo "Submitting $NUM_LINES configs in Blöcken à ≤ $MAX_PER_ARRAY…"

for ((offset=0; offset<NUM_LINES; offset+=MAX_PER_ARRAY)); do
    count=$(( NUM_LINES - offset )) # count how many lines (configurations) are left
    (( count > MAX_PER_ARRAY )) && count=$MAX_PER_ARRAY # if there are more lines left then allowed by maximal array size set count to that value
    end=$(( count - 1 ))  # substract 1 as array starts at 0

    echo " -> sbatch --array=0-$end (OFFSET=$offset)"
    sbatch --array="0-$end" \
           --export=ALL,RUN_ID="$RUN_ID",OFFSET="$offset",NUM_LINES="$NUM_LINES" \
           run_03_single_job_var_arrays.sh
done

