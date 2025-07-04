#!/bin/bash

# Usage: ./resubmit_failed.sh <SLURM_JOB_ID> <RUN_ID>
# Beispiel: ./resubmit_failed.sh 18147305 20250703_1842

if [ $# -ne 2 ]; then
    echo "Usage: $0 <SLURM_JOB_ID> <RUN_ID>"
    exit 1
fi

JOBID=$1
RUN_ID=$2

echo "Suche fehlgeschlagene Array-Tasks für JobID: $JOBID"

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
echo "Starte: sbatch --array=$ARRAY_IDS run_single_job.sh $RUN_ID"

sbatch --array=$ARRAY_IDS run_single_job.sh $RUN_ID

