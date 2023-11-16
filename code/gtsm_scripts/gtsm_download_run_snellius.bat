@echo off

setlocal

set storm=sandy_shifted
rem Set the remote server and username
set server=hmoreno@snellius.surf.nl

rem Specify the remote folder
set folder=/gpfs/work3/0/einf4318/gtsm4.1/output_gtsm_sn_allruns_%storm%

rem Set the local folder to copy files to
set local_folder=D:\paper_3\data\gtsm_local_runs\%storm%_fine_grid

echo Copying files from %folder% to %local_folder%
scp -r %server%:%folder%/*his_waterlevel.nc %local_folder%

echo Done.

endlocal