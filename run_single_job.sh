#!/bin/bash
#SBATCH --job-name=pacmap_cluster
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=n1016,n1048,n1159

RUN_ID=$1

if [ -z "$RUN_ID" ]; then
    echo "Usage: sbatch --array=0-N run_single_job.sh <RUN_ID>"
    exit 1
fi

module purge
module --ignore_cache load release/23.10
module --ignore_cache load GCCcore/11.3.0
module --ignore_cache load Python
source /data/horse/ws/sede829c-python_virtual_environment/bin/activate

INDEX=${SLURM_ARRAY_TASK_ID}

#echo "Running 03_cluster.py with RUN_ID=$RUN_ID and INDEX=$INDEX"

python 03_cluster.py --run_id "$RUN_ID" --index "$INDEX"

