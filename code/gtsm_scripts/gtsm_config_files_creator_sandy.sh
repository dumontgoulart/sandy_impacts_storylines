#!/bin/bash

declare -a storm=("sandy")
declare -a year=("2012")
declare -a sn_runs=("counter_1" "counter_2" "counter_3" "factual_1" "factual_2" "factual_3" "plus2_1" "plus2_2" "plus2_3")

# main directory
main_dir="/gpfs/work3/0/einf4318/gtsm4.1/scripts_SN/${storm}/"
mkdir -p ${main_dir}
# Set the name of the original .sh file
output_folder="${main_dir}ext_files/"
output_folder_mdu="${main_dir}mdu_files/"
mkdir -p ${output_folder}
mkdir -p ${output_folder_mdu}

original_ext_file="/gpfs/work3/0/einf4318/gtsm4.1/scripts_SN/ext_files/gtsm_forcing_template.ext"
original_mdu_file="/gpfs/work3/0/einf4318/gtsm4.1/scripts_SN/ext_files/gtsm_fine_template.mdu"
original_bash_file="/gpfs/work3/0/einf4318/gtsm4.1/scripts_SN/ext_files/run_gtsm_template.sh"

refdate="20121013"
tstop="480"

for sn_run in "${sn_runs[@]}"; do
    # Set the name of the new .sh file
    new_file="${output_folder}gtsm_forcing_${storm}_${sn_run}.ext"
    new_mdu_file="/gpfs/work3/0/einf4318/gtsm4.1/gtsm_fine_${storm}_${sn_run}.mdu"
    new_bash_file="/gpfs/work3/0/einf4318/gtsm4.1/run_gtsm_${storm}_${sn_run}.sh"

    # Set the values for the parameters that you want to change
    windx_filename_new="/gpfs/work3/0/einf4318/gtsm4.1/meteo_ECHAM6.1SN_fm/${storm}/BOT_t_HR255_Nd_SU_1015_${sn_run}_${year}_${storm}_storm_u10_gtsm.nc"
    windy_filename_new="/gpfs/work3/0/einf4318/gtsm4.1/meteo_ECHAM6.1SN_fm/${storm}/BOT_t_HR255_Nd_SU_1015_${sn_run}_${year}_${storm}_storm_v10_gtsm.nc"
    atmosphericpressure_filename_new="/gpfs/work3/0/einf4318/gtsm4.1/meteo_ECHAM6.1SN_fm/${storm}/BOT_t_HR255_Nd_SU_1015_${sn_run}_${year}_${storm}_storm_msl_gtsm.nc"

    # Read in the original .sh file and replace the specific parameters with their new values
    sed -e "s|FILENAME =/path/to/windx/file.nc|FILENAME =$windx_filename_new|g" \
        -e "s|FILENAME =/path/to/windy/file.nc|FILENAME =$windy_filename_new|g" \
        -e "s|FILENAME =/path/to/atmosphericpressure/file.nc|FILENAME =$atmosphericpressure_filename_new|g" \
        $original_ext_file > $new_file

    # Make the new .sh file executable
    chmod +x $new_file

    sed -e "s|RefDate              = 20100208|RefDate              = ${refdate}|g" \
        -e "s|TStop                = 720.|TStop                = ${tstop}|g" \
        -e "s|ExtForceFile         = /gpfs/work3/0/einf4318/gtsm4.1/scripts_SN/ext_files/gtsm_forcing_tmp.ext |ExtForceFile         = ${new_file}|g" \
        -e "s|OutputDir            = output |OutputDir         = output_${storm}_${sn_run}|g" \
        -e "s|ObsFile              = /gpfs/work3/0/einf4318/gtsm4.1/selected_output_OR_xynthia_hg_obs.xyn  |ObsFile              = /gpfs/work3/0/einf4318/gtsm4.1/selected_output_OR_${storm}_hg_obs.xyn|g"\
        $original_mdu_file > $new_mdu_file
    
    chmod +x $new_mdu_file

    sed -e "s|mduFile=gtsm_fine_template.mdu|mduFile=$new_mdu_file|g" \
        -e "s|#SBATCH --job-name=gtsm_template|#SBATCH --job-name=gtsm_${storm}_${sn_run}|g" \
        $original_bash_file > $new_bash_file

done