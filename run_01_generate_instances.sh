#!/bin/bash
#SBATCH --job-name=gen_instances
#SBATCH --output=logs/gen_instances_%j.out
#SBATCH --error=logs/gen_instances_%j.err


# Stoppe bei Fehler
set -e

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
python 01_generate_instances.py

