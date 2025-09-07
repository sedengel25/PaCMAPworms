#!/bin/bash
#SBATCH -J pacmap_cluster
#SBATCH -o /dev/null
#SBATCH -e /dev/null
set -euo pipefail 
# e = exit immediately, if any command fails, exit code unequal 0, script stops 
# u = unset variables are errors, so if a variable is used that is not set, script stops
# o = if script fails, return first error 
: "${RUN_ID:?missing RUN_ID}"
: "${OFFSET:?missing OFFSET}"
: "${NUM_LINES:?missing NUM_LINES}"
: "${SLURM_ARRAY_TASK_ID:?no array id}"

mkdir -p "logs/${RUN_ID}"

AJOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-unknown}}" # a fallback-chain of parameter expansions, if first var does not exist or is null go the second
TASK="${SLURM_ARRAY_TASK_ID}"
OUTFILE="logs/${RUN_ID}/job${AJOB}_array${TASK}.out"
ERRFILE="logs/${RUN_ID}/job${AJOB}_array${TASK}.err"

exec 1>"$OUTFILE"
exec 2>"$ERRFILE"

export PYTHONUNBUFFERED=1

module purge
module --ignore_cache load release/23.10
module --ignore_cache load GCCcore/11.3.0
module --ignore_cache load Python
source /data/horse/ws/sede829c-python_virtual_environment/bin/activate

# Index berechnen (CSV mit Header -> +2)
GLOBAL_IDX=$(( OFFSET + SLURM_ARRAY_TASK_ID ))
(( GLOBAL_IDX >= NUM_LINES )) && exit 0 # ensures that we dont search for a line in the csv file that doesn't exist
CSV_LINE=$(( GLOBAL_IDX + 2 ))
echo "DEBUG: GLOBAL_IDX=${GLOBAL_IDX}"    # für dich zum Prüfen
echo "DEBUG: CSV_LINE=${CSV_LINE}"        # für dich zum Prüfen
python 03_cluster.py --run_id "$RUN_ID" --index "$CSV_LINE"

