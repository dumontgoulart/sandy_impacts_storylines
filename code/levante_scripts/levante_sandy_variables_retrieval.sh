#!/bin/bash
#SBATCH --job-name=sandy_data			 # Specify job name
#SBATCH --output=sandy_data.%j                # name for standard output log file, same as job-name
#SBATCH --error=sandy_data.%j                 # name for standard error output log file, it is better it is the same as ouput
#SBATCH --partition=shared                                # Specify partition name, do not change
#SBATCH --ntasks=1                                        # Specify max. number of tasks to be invoked, do not change
#SBATCH --mem=6GB                                         # allocated memory for the script, do not change
#SBATCH --time=48:00:00                                   # Set a limit on the total run time, 48 hours for a retrieval job of 12 months of data is OK at this moment in time
#SBATCH --account=gg0301                                  # our project code, do not change   


# INSTRUCTIONS #################################################################
# Personal folder inside levante: cd /work/gg0301/from_Mistral/gg0301/g260212
# Expected outputs: 9 files for the selected variable at the selected spatial extent - 3 GW levels (F,C,P2) with 3 members each.
# STEPS:
# 1) Send this script from pc to server <scp -r D:\paper_3\code\levante_sandy_variables_retrieval.sh g260212@levante.dkrz.de:/work/gg0301/from_Mistral/gg0301/g260212/>
# 2) Access Levante (id g260212@levante.dkrz.de)
# 3) Run on Levante terminal: <bash levante_sandy_variables_retrieval.sh>
# 4) Local computer: Load files from server to pc <scp -r g260212@levante.dkrz.de:/work/gg0301/from_Mistral/gg0301/g260212/sandy D:/paper_3/data/spectral_nudging_data>
################################################################################
# Load module
module load cdo
module load nco

# EDIT PARAMETERS HERE ########################################################
# Months and years and storm name to select:
declare -a storm=("sandy") # string format
declare -a months=("10" "11") # string format with a zero before for single digits
declare -a year=("2012") # string format
declare -a bbox=("-180,180,-90,90")  # lon_min,lon_max,lat_min,lat_max -89.472656,12,-40.209961,48.458352
################################################################################
# Variables
suffix=151
declare -a var=("mslp")
declare -a suffix_out=("hPa")
declare -a long_name=("mean_sea_level_pressure")
################################################################################

# Folder locations
folder_in=/work/gg0301/from_Mistral/gg0301/g260132/echam-6.1.00_modified/experiments/Henrique/var${suffix}/

folder_out=/work/gg0301/from_Mistral/gg0301/g260212/${storm}/${var}_${suffix_out}/
mkdir -p ${folder_out}

# Clip region to boundaries selected and merge along time for each member and global warming level.
for scenario in factual counter plus2 ; do
	for ensemble_member in 1 2 3 ; do
		for month in ${months[@]} ; do
			cdo -sellonlatbox,${bbox} ${folder_in}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}${month}.${suffix}.nc ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}${month}_${storm}.${suffix}.nc; 
		done;
		cdo -O mergetime ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}??_${storm}.${suffix}.nc ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_merge.nc;
		cdo divc,100 ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_merge.nc ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_merge_hpa.nc;
		cdo -setattribute,${var}@units=${suffix_out} -chname,var${suffix},${var} ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_merge_hpa.nc ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${var}_${suffix_out}.nc
		ncatted -O -a standard_name,${var},o,c,${long_name} ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${var}_${suffix_out}.nc
		rm ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_merge.nc
        rm ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_merge_hpa.nc
        rm ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}??_${storm}.${suffix}.nc
	done;
done    

################################################################################
# Variables
declare -a var=("u10m")
suffix=165
declare -a suffix_out=("m_s")
declare -a long_name=("wind_speed_u")
################################################################################
# Folder locations
folder_in=/work/gg0301/from_Mistral/gg0301/g260132/echam-6.1.00_modified/experiments/Henrique/var${suffix}/

folder_out_u=/work/gg0301/from_Mistral/gg0301/g260212/${storm}/${var}_${suffix_out}/
mkdir -p ${folder_out_u}

# Clip region to boundaries selected and merge along time for each member and global warming level.
for scenario in factual counter plus2 ; do
	for ensemble_member in 1 2 3 ; do
		for month in ${months[@]} ; do
			cdo -sellonlatbox,${bbox} ${folder_in}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}${month}.${suffix}.nc ${folder_out_u}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}${month}_${storm}.${suffix}.nc; 
		done;
		cdo -O mergetime ${folder_out_u}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}??_${storm}.${suffix}.nc ${folder_out_u}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_merge.nc;
		cdo -setattribute,${var}@units=${suffix_out} -chname,var${suffix},${var} ${folder_out_u}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_merge.nc ${folder_out_u}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${var}_${suffix_out}.nc
		ncatted -O -a standard_name,${var},o,c,${long_name} ${folder_out_u}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${var}_${suffix_out}.nc
		rm ${folder_out_u}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_merge.nc
        rm ${folder_out_u}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}??_${storm}.${suffix}.nc
	done;
done    

# Variables
declare -a var=("v10m")
suffix=166
declare -a suffix_out=("m_s")
declare -a long_name=("wind_speed_v")
################################################################################

# Folder locations
folder_in=/work/gg0301/from_Mistral/gg0301/g260132/echam-6.1.00_modified/experiments/Henrique/var${suffix}/

folder_out_v=/work/gg0301/from_Mistral/gg0301/g260212/${storm}/${var}_${suffix_out}/
mkdir -p ${folder_out_v}

# Clip region to boundaries selected and merge along time for each member and global warming level.
for scenario in factual counter plus2 ; do
	for ensemble_member in 1 2 3 ; do
		for month in ${months[@]} ; do
			cdo -sellonlatbox,${bbox} ${folder_in}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}${month}.${suffix}.nc ${folder_out_v}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}${month}_${storm}.${suffix}.nc; 
		done;
		cdo -O mergetime ${folder_out_v}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}??_${storm}.${suffix}.nc ${folder_out_v}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_merge.nc;
		cdo -setattribute,${var}@units=${suffix_out} -chname,var${suffix},${var} ${folder_out_v}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_merge.nc ${folder_out_v}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${var}_${suffix_out}.nc
		ncatted -O -a standard_name,${var},o,c,${long_name} ${folder_out_v}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${var}_${suffix_out}.nc
		rm ${folder_out_v}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_merge.nc
		rm ${folder_out_v}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}??_${storm}.${suffix}.nc
	done;
done    

# Variables
declare -a var=("t2m")
suffix=167
declare -a suffix_out=("degC")
long_name=temperature
################################################################################

# Folder locations
folder_in_factual=/work/gg0301/from_Mistral/gg0301/g260132/echam-6.1.00_modified/experiments/HR255_Nd_SU_1015/Glob/var${suffix}/
folder_in_plus2=/work/gg0301/from_Mistral/gg0301/g260132/echam-6.1.00_modified/experiments/HR255_Nd_SU_1015_plus2/

folder_in=${folder_in_factual}
folder_out=/work/gg0301/from_Mistral/gg0301/g260212/${storm}//${var}_${suffix_out}/
mkdir -p ${folder_out}

# Clip region to boundaries selected and merge along time for each member and global warming level.
for scenario in factual counter plus2; do
	for ensemble_member in 1 2 3 ; do
		if [[ "${scenario}" == "plus2" ]] ; then
			folder_in=${folder_in_plus2}${ensemble_member}/
		fi
		for month in ${months[@]} ; do
			cdo -sellonlatbox,${bbox} ${folder_in}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}${month}.${suffix}.nc ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}${month}_${storm}.${suffix}.nc; 
		done;
		cdo -O mergetime ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}??_${storm}.${suffix}.nc ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}.nc;
		cdo -setattribute,${var}@units=${suffix_out} -addc,-273.15 -chname,var${suffix},${var} ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}.nc ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${var}_${suffix_out}.nc
		ncatted -O -a standard_name,${var},o,c,${long_name} ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${var}_${suffix_out}.nc
		rm ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}.nc
		rm ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}??_${storm}.${suffix}.nc
	done;
done

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

		# Divide by 3 to have the average of the hourly precipitation
		cdo divc,3 ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_${suffix_out}.nc ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_mmh.nc
		ncatted -O -a standard_name,${var},o,c,${long_name} ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_mmh.nc
		ncatted -O -a units,${var},o,c,"mm/h" ${folder_out}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.${suffix}_mmh.nc

	done;
done



# # Now get absolute wind speeds
# folder_windspeed=/work/gg0301/from_Mistral/gg0301/g260212//${storm}/U10m_${suffix_out}/
# mkdir -p ${folder_windspeed}

# for scenario in factual counter plus2 ; do
# 	for ensemble_member in 1 2 3 ; do
# 		cdo chname,u10m,U10m -sqrt -add -sqr ${folder_out_u}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.u10m_${suffix_out}.nc -sqr ${folder_out_v}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.v10m_${suffix_out}.nc ${folder_windspeed}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.U10m_${suffix_out}.nc
# 		ncatted -O -a standard_name,U10m,o,c,total_wind_speed ${folder_windspeed}BOT_t_HR255_Nd_SU_1015_${scenario}_${ensemble_member}_${year}_${storm}_storm.U10m_${suffix_out}.nc
# 	done;
# done;
