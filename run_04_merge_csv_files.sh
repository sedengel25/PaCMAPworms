#!/bin/bash

# Usage check
if [ $# -ne 1 ]; then
    echo "Usage: $0 <RUN_ID>"
    exit 1
fi

RUN_ID=$1

# Module-Umgebung bereinigen
module purge

# Module laden
module load release/23.10
module load GCCcore/11.3.0
module load Python

# venv aktivieren
source /data/horse/ws/sede829c-python_virtual_environment/bin/activate

# Python-Skript mit srun ausführen
srun python 04_merge_csv_files.py "$RUN_ID"

