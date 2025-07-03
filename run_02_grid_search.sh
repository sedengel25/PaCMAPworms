#!/bin/bash
#SBATCH --job-name=grid_search
#SBATCH --output=logs/grid_search_%j.out
#SBATCH --error=logs/grid_search_%j.err

# Stoppe bei Fehler
set -e

# Check ob run_id übergeben wurde
if [ -z "$1" ]; then
    echo "Usage: $0 <run_id>"
    exit 1
fi

RUN_ID="$1"

# Module laden
module load release/23.10
module load GCCcore/11.3.0
module load Python

# (Optional) venv aktivieren, falls benötigt:
source /data/horse/ws/sede829c-python_virtual_environment/bin/activate

# Kontrollausgaben
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Pip path: $(which pip)"

# Interaktive Ausführung mit sichtbarer Ausgabe
python 02_grid_search.py --run_id "$RUN_ID"

