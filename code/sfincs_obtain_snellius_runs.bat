@echo off
setlocal enabledelayedexpansion

rem Set the storm folder
set storm=sandy_slr71

rem Set the remote server and username
set server=hmoreno@snellius.surf.nl

rem Specify the remote folder
set folder=/gpfs/work3/0/einf4318/sfincs/sfincs_inundation_results/%storm%

rem Set the local folder to copy files to
set local_folder=D:\paper_3\data\sfincs_inundation_results\%storm%

echo Copying files from %folder% to %local_folder%

scp -r %server%:%folder%/* %local_folder%

rem Get a list of subdirectories in the external folder

@REM set command=ssh %server% "find %folder% -type d \( -name %storm%_factual* -o -name %storm%_counter* -o -name %storm%_plus2* \)"
@REM for /f "delims=" %%a in ('%command%') do (
@REM   rem Copy the file from the external server to the local computer
@REM   echo Copying file from %%a
@REM   start "" scp %server%:"%%a\sfincs_map.nc" "%local_folder%\%%~nxa\sfincs_map.nc"
@REM )


rem scp hmoreno@snellius.surf.nl:/gpfs/work3/0/einf4318/noaa_dem/northeast_sandy/cudem_wgs84_raster.tif D:\paper_3\data\us_dem

rem scp -r hmoreno@snellius.surf.nl:/gpfs/work3/0/einf4318/sfincs/sfincs_inundation_results/ D:\paper_3\data\