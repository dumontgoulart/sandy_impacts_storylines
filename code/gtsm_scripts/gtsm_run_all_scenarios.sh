#! /bin/bash
#SBATCH -p thin
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --job-name=gtsm_all_scen
#SBATCH --time=10:30:00
#SBATCH --mem=40G
# /gpfs/work3/0/einf4318 X Job Wall-clock time: 08:20:02

declare -a sn_runs=("counter_1" "counter_2" "counter_3" "factual_1" "factual_2" "factual_3" "plus2_1" "plus2_2" "plus2_3")

for sn_run in "${sn_runs[@]}"; do
    sbatch /gpfs/work3/0/einf4318/gtsm4.1/run_gtsm_xynthia_${sn_run}.sh

    # Wait for 60 seconds before proceeding to the next iteration
    sleep 300
done
