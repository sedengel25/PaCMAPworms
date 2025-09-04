#!/bin/bash
#SBATCH -J pacmap_cluster
set -euo pipefail

: "${RUN_ID:?missing RUN_ID}"
: "${OFFSET:?missing OFFSET}"
: "${NUM_LINES:?missing NUM_LINES}"
: "${SLURM_ARRAY_TASK_ID:?no array id}"

# Logs/<RUN_ID> anlegen
mkdir -p "logs/${RUN_ID}"

# Ein Logfile pro Array-Task (Array-ID + Task-ID); alles hinein (stdout+stderr)
AJOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-unknown}}"
TASK="${SLURM_ARRAY_TASK_ID}"
LOGFILE="logs/${RUN_ID}/job${AJOB}_array${TASK}.log"
exec >"$LOGFILE" 2>&1

# (optional) sofortiges Flushen von Python-Output
export PYTHONUNBUFFERED=1

# Module & venv
module purge
module --ignore_cache load release/23.10
module --ignore_cache load GCCcore/11.3.0
module --ignore_cache load Python
source /data/horse/ws/sede829c-python_virtual_environment/bin/activate

# Index berechnen (CSV mit Header -> +2)
GLOBAL_IDX=$(( OFFSET + SLURM_ARRAY_TASK_ID ))
(( GLOBAL_IDX >= NUM_LINES )) && exit 0
CSV_LINE=$(( GLOBAL_IDX + 2 ))
echo "DEBUG: GLOBAL_IDX=${GLOBAL_IDX}"    # für dich zum Prüfen
echo "DEBUG: CSV_LINE=${CSV_LINE}"        # für dich zum Prüfen
python 03_cluster.py --run-id "$RUN_ID" --index "$CSV_LINE"

