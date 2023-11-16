@echo off

setlocal

set server=hmoreno@snellius.surf.nl
set storm=sandy_shifted

rem Depends on the type of files we are uploading. Normal runs have multiple folders, shifted runs have one folder

rem set folders="D:\paper_3\data\spectral_nudging_data\sandy\mslp_hPa" "D:\paper_3\data\spectral_nudging_data\sandy\u10m_m_s" "D:\paper_3\data\spectral_nudging_data\sandy\v10m_m_s"

@REM for /d %%f in (%folders%) do (
@REM     echo Copying files from %%f to %remote_folder%
@REM     start "" scp %%f\*.nc %server%:/gpfs/work3/0/einf4318/gtsm4.1/meteo_ECHAM6.1SN/sandy_shifted/ || (
@REM         echo Error sending files from "%%f" to %server%
@REM         pause
@REM )

@REM echo Done.


rem if the files are in one folder, use this
set folder="D:\paper_3\data\spectral_nudging_data\%storm%"

rem Loop through each file in the folder
for %%f in ("%folder%\*.nc") do (
    rem Send the file to the cluster
    echo Sending "%%f" to %server%:/gpfs/work3/0/einf4318/gtsm4.1/meteo_ECHAM6.1SN/%storm%/
    start "" scp "%%f" %server%:/gpfs/work3/0/einf4318/gtsm4.1/meteo_ECHAM6.1SN/%storm%/
)

echo Done.


endlocal
