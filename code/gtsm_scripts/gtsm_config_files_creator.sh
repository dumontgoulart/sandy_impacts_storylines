#!/bin/bash

declare -a storm=("xynthia")
declare -a year=("2010")
declare -a sn_runs=("counter_1" "counter_2" "counter_3" "factual_1" "factual_2" "factual_3" "plus2_1" "plus2_2" "plus2_3")

# Set the name of the original .sh file
output_folder="/gpfs/home6/hmoreno/gtsm4.1/scripts_SN/ext_files/"
output_folder_mdu="/gpfs/home6/hmoreno/gtsm4.1/scripts_SN/mdu_files/"
original_ext_file="gtsm_forcing_template.ext"
original_mdu_file="gtsm_fine_template.mdu"
original_bash_file="run_gtsm_template.sh"


for sn_run in "${sn_runs[@]}"; do
    # Set the name of the new .sh file
    new_file="${output_folder}gtsm_forcing_${storm}_${sn_run}.ext"
    new_mdu_file="${output_folder_mdu}gtsm_fine_${storm}_${sn_run}.mdu"
    new_bash_file="/gpfs/home6/hmoreno/gtsm4.1/run_gtsm_${storm}_${sn_run}.sh"

    # Set the values for the parameters that you want to change
    windx_filename_new="/gpfs/home6/hmoreno/gtsm4.1/meteo_ECHAM6.1SN_fm/BOT_t_HR255_Nd_SU_1015_${sn_run}_${year}_${storm}_storm_u10_gtsm.nc"
    windy_filename_new="/gpfs/home6/hmoreno/gtsm4.1/meteo_ECHAM6.1SN_fm/BOT_t_HR255_Nd_SU_1015_${sn_run}_${year}_${storm}_storm_v10_gtsm.nc"
    atmosphericpressure_filename_new="/gpfs/home6/hmoreno/gtsm4.1/meteo_ECHAM6.1SN_fm/BOT_t_HR255_Nd_SU_1015_${sn_run}_${year}_${storm}_storm_msl_gtsm.nc"

    # Read in the original .sh file and replace the specific parameters with their new values
    sed -e "s|FILENAME =/path/to/windx/file.nc|FILENAME =$windx_filename_new|g" \
        -e "s|FILENAME =/path/to/windy/file.nc|FILENAME =$windy_filename_new|g" \
        -e "s|FILENAME =/path/to/atmosphericpressure/file.nc|FILENAME =$atmosphericpressure_filename_new|g" \
        $original_ext_file > $new_file

    # Make the new .sh file executable
    chmod +x $new_file

    sed -e "s|ExtForceFile         = ${output_folder}gtsm_forcing_tmp.ext|ExtForceFile         = ${new_file}|g" \
        -e "s|OutputDir            = output |OutputDir         = output_${sn_run}|g" \
        $original_mdu_file > $new_mdu_file
    
    chmod +x $new_mdu_file

    sed -e "s|mduFile=$original_mdu_file|mduFile=$new_mdu_file|g" \
        -e "s|#SBATCH --job-name=gtsm_template|#SBATCH --job-name=gtsm_${storm}_${sn_run}|g" \
        $original_bash_file > $new_bash_file

done