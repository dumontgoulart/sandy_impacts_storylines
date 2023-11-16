#!/bin/bash
# Runs the gtsm workflow as one script, based on last month
module load 2021
module load Anaconda3/2021.05

eval "$(conda shell.bash hook)"
conda activate /home/hmoreno/.conda/envs/gtsm3-era5-nrt-slm

set -e 

scriptsdir="/gpfs/home6/hmoreno/gtsm4.1/scripts_SN/"


# Folder paths
folder_sn="/gpfs/home6/hmoreno/gtsm4.1/meteo_ERA5"

# Prepare data and convert to FM format
conda run -n gtsm3-era5-nrt-slm python gtsm_preprocess.py

# # 1 Download daily ERA5 data and tides 
# downloadjobid=$(sbatch $scriptsdir/p1_sbatch_download.sh $lastym | awk 'match($0, /[0-9]+/) {print substr($0, RSTART, RLENGTH)}')
# echo "started download job with id: $downloadjobid"
# # 2 Prepare data and convert to FM format
# preprocjobid=$(sbatch --dependency=afterok:$downloadjobid $scriptsdir/p2_sbatch_preproc.sh $lastym | awk 'match($0, /[0-9]+/) {print substr($0, RSTART, RLENGTH)}')
# echo "started preproc job with id: $preprocjobid"
# # 3 Prepare run 
# preparejobid=$(sbatch --dependency=afterok:$preprocjobid $scriptsdir/p3_prepare_run.py $lastym | awk 'match($0, /[0-9]+/) {print substr($0, RSTART, RLENGTH)}')
# echo "started prepare job with id: $preparejobid"
# # # 4 Run GTSM --> check folder
# # runjobid=$(sbatch --dependency=afterok:$preparejobid $scriptsdir/p3_sbatch_gtsm_delft3dfm2022.01_xnodes.sh $lastym | awk 'match($0, /[0-9]+/) {print substr($0, RSTART, RLENGTH)}')
# # echo "started run job with id: $runjobid"
# # # 5 Postprocess and plot
# # postprocjobid=$(sbatch --dependency=afterok:$catchupjobid $scriptsdir/p4_sbatch_postprocess.sh $lastym | awk 'match($0, /[0-9]+/) {print substr($0, RSTART, RLENGTH)}')
# # echo "started postproc job with id: $postprocjobid"
