#! /bin/bash
#SBATCH -p thin
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --job-name=gtsm_all_scen
#SBATCH --time=10:30:00
#SBATCH --mem=40G

module load 2022
module load CDO/2.0.6-gompi-2022a

# Set input file name and output grid file name
input_file="/gpfs/work3/0/einf4318/echam61_ptot/BOT_t_HR255_Nd_SU_1015_factual_1_2010_xynthia_storm.Ptot_mm3h.nc"
output_grid_file="${input_file%.*}.grid"

# Create the grid file
cat << EOF > $output_grid_file
gridtype = lonlat
xsize = 1440
ysize = 720
xfirst = -180
xinc = 0.25
yfirst = -90
yinc = 0.25
EOF

# Set input and output directories
indir="/gpfs/work3/0/einf4318/echam61_ptot/"
outdir="/gpfs/work3/0/einf4318/echam61_ptot/regular/"
if [ ! -d "$outdir" ]; then
  mkdir "$outdir"
  echo "Created directory: $dir"
fi
# Loop over input files
for infile in "${indir}"/*Ptot_mm3h.nc; do
  # Set output filename
  outfile="${outdir}$(basename "$infile" .nc)_regular.nc"
  # Remap file
  cdo remapcon,"/gpfs/work3/0/einf4318/echam61_ptot/BOT_t_HR255_Nd_SU_1015_factual_1_2010_xynthia_storm.Ptot_mm3h.grid" "$infile" "$outfile"
done

