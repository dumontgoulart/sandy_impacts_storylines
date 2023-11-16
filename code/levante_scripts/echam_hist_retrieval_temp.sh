#!/bin/bash
# INSTRUCTIONS #################################################################
# find: cd /work/gg0301/from_Mistral/gg0301/g260070/ECHAM6/echam-6.1.00/experiments/echam6_t255l95_sn_ncep1/scripts
# Expected outputs: 9 files for the selected variable at the selected spatial extent - 3 GW levels (F,C,P2) with 3 members each.
# STEPS:
# 1) Send this script from pc to server <scp -r D:\paper_3\code\echam_hist_retrieval_temp.sh g260212@levante.dkrz.de:/work/gg0301/from_Mistral/gg0301/g260212/>
# 2) Access Levante (id g260212@levante.dkrz.de)
# 3) Run on Levante terminal: <bash echam_hist_retrieval_temp.sh>
# 4) Local computer: Load files from server to pc <scp -r g260212@levante.dkrz.de:/work/gg0301/from_Mistral/gg0301/g260212/sandy/mslp_Pa D:/paper_3/data/spectral_nudging_data/raed>
#   What this scrip does:
# 1) Converts file to .nc
# 2) Select years, latlon and variable;
# 2) Aggregate across month (correct values by the way)
################################################################################
# Load module
module load cdo
module load nco

# EDIT PARAMETERS HERE ########################################################
# Months and years and storm name to select:
declare -a folder=("hist") # string format
declare -a year=("2012") # string format
declare -a bbox=("-150,-30,-60,70") # lon_min,lon_max,lat_min,lat_max

# Variables
declare -a var=("t2max")
declare -a suffix_out=("K")
long_name=maximum_2m_temperature

# Folder locations
folder_in=/work/gg0301/from_Mistral/gg0301/g260070/ECHAM6/echam-6.1.00/experiments/echam6_t255l95_sn_ncep1/scripts/

folder_out=/work/gg0301/from_Mistral/gg0301/g260212/${folder}//${var}_${suffix_out}/
mkdir -p ${folder_out}

# Actual script:
cdo -r -f nc selvar,${var} ${folder_in}rerun_echam6_t255l95_sn_ncep1_echam ${folder_out}rerun_echam6_t255l95_sn_ncep1_echam_${var}.nc


# selyear,1980/2015
# -sellonlatbox,${bbox}
# cdo -O monmean AM_BOT_t_HR255_Nd_SU_1015_counter_1_201101.140.nc AM_BOT_t_HR255_Nd_SU_1015_counter_1_201101.140.monmean.nc\n