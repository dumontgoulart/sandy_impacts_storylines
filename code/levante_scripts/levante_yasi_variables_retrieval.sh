#!/bin/bash
# INSTRUCTIONS #################################################################
# Personal folder inside levante: cd /work/gg0301/from_Mistral/gg0301/g260212
# Expected outputs: 9 files for the selected variable at the selected spatial extent - 3 GW levels (F,C,P2) with 3 members each.
# STEPS:
# 1) Send this script from pc to server <scp -r D:\paper_3\code\levante_xaver_variables_retrieval.sh g260212@levante.dkrz.de:/work/gg0301/from_Mistral/gg0301/g260212/>
# 2) Access Levante (id g260212@levante.dkrz.de)
# 3) Run on Levante terminal: <bash levante_hayan_time_sellonlat_precip.sh>
# 4) Local computer: Load files from server to pc <scp -r g260212@levante.dkrz.de:/work/gg0301/from_Mistral/gg0301/g260212/megi D:/paper_3/data/spectral_nudging_data>
################################################################################
# Load module
module load cdo
module load nco

# EDIT PARAMETERS HERE ########################################################
# Months and years and storm name to select:
declare -a storm=("yasi") # string format
declare -a months=("01" "02") # string format with a zero before for single digits
declare -a year=("2011") # string format
declare -a bbox=("140.449219,183.120117,-27.098254,-8.102739") # lon_min,lon_max,lat_min,lat_max 140.449219,-27.098254,183.120117,-8.102739

################################################################################
# Variables
declare -a var=("Ptot")
declare -a suffix_out=("mm3h")
old_var_name=142
suffix=Ptot
long_name=precipitation
################################################################################

# Folder locations
folder_in_factual=/work/gg0301/from_Mistral/gg0301/g260132/echam-6.1.00_modified/experiments/HR255_Nd_SU_1015/Glob/precip/
folder_in_plus2=/work/gg0301/from_Mistral/gg0301/g260132/echam-6.1.00_modified/experiments/HR255_Nd_SU_1015_plus2/Glob/Ptot/

folder_in=${folder_in_factual}
folder_out=/work/gg0301/from_Mistral/gg0301/g260212/${storm}/${var}_${suffix_out}/
mkdir -p ${folder_out}

# Clip region to boundaries selected and merge along time for each member and global warming level.
for scenario in factual counter plus2 ; do
	for ensemble_member in 1 2 3 ; do
		for month in ${months[@]} ; do
			if [[ "${scenario}" != "plus2" ]] ; then
				cdo -sellonlatbox,${bbox} ${folder_in}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}${month}.${suffix}_${suffix_out}.nc ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}${month}_${storm}.${suffix}_${suffix_out}.nc; 
			else
				folder_in=${folder_in_plus2}
				cdo -sellonlatbox,${bbox} ${folder_in}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}${month}.${suffix}.nc ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}${month}_${storm}.${suffix}.nc; 				
				cdo mulc,10800 ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}${month}_${storm}.${suffix}.nc ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}${month}_${storm}.${suffix}_${suffix_out}.nc;
				rm ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}${month}_${storm}.${suffix}.nc
			fi
		done;
		cdo -O mergetime ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}??_${storm}.${suffix}_${suffix_out}.nc ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_${suffix_out}_merge.nc;
		cdo -setattribute,${var}@units=${suffix_out} -chname,var${old_var_name},${var} ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_${suffix_out}_merge.nc ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_${suffix_out}.nc
		ncatted -O -a standard_name,${var},o,c,${long_name} ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_${suffix_out}.nc
		rm ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_${suffix_out}_merge.nc
        rm ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}??_${storm}.${suffix}_${suffix_out}.nc
	done;
done

