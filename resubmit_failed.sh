#!/bin/bash

# Usage: ./resubmit_failed.sh <SLURM_JOB_ID>
# Beispiel: ./resubmit_failed.sh 18147305

if [ $# -ne 1 ]; then
    echo "Usage: $0 <SLURM_JOB_ID>"
    exit 1
fi

JOBID=$1

echo "Suche fehlgeschlagene Array-Tasks für JobID: $JOBID"

# Hole alle FAILED-Zeilen, ignoriere .batch-Zeilen, extrahiere Array-IDs
ARRAY_IDS=$(sacct -u $USER --format=JobID%40,JobName,State,NodeList | \
    grep FAILED | grep "$JOBID" | \
    grep -v ".batch" | \
    awk '{print $1}' | \
    grep '_' | \
    sed "s/^${JOBID}_//" | \
    sort -n | uniq | paste -sd "," -)

if [ -z "$ARRAY_IDS" ]; then
    echo "Keine fehlgeschlagenen Array-Jobs gefunden."
    exit 1
fi

echo "Gefundene fehlgeschlagene Array-IDs: $ARRAY_IDS"
echo "Starte sbatch --array=$ARRAY_IDS run_single_job.sh"

# Starte sbatch mit genau diesen Array-IDs
#sbatch --array=$ARRAY_IDS run_single_job.sh


