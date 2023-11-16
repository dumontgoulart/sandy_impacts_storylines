@echo off

setlocal

rem Set the remote server and username
set server=hmoreno@snellius.surf.nl

set storm=sandy_shifted

rem Specify the remote folder
set folder=/gpfs/work3/0/einf4318/echam61_ptot/%storm%/regular/

rem Set the local folder to copy files to
set local_folder=D:\paper_3\data\spectral_nudging_data\regular_grid\%storm%

echo Copying files from %folder% to %local_folder%
scp -r %server%:%folder%/*_clip.nc %local_folder%

echo Done.

endlocal