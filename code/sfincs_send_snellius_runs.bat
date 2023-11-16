@echo off
setlocal enabledelayedexpansion

rem Set the storm folder
set storm=sandy_shifted

rem Set the remote server and username
set server=hmoreno@snellius.surf.nl

rem Specify the remote folder
set folder=/gpfs/work3/0/einf4318/sfincs/%storm%/

rem Set the local folder to copy files to
set local_folder=D:\paper_3\data\sfincs_ini\spectral_nudging\

echo Copying files from %local_folder% to %folder%

echo Searching for folders starting with %storm% in %search_folder%

rem Loop through each subdirectory that starts with "sandy_shifted"
for /d %%a in ("%local_folder%\%storm%_factual*" "%local_folder%\%storm%_counter*" "%local_folder%\%storm%_plus2*" "%local_folder%\data_deltares_%storm%") do (
  rem Send the directory to the cluster
  echo Sending "%%a" to %server%:%folder%
  start "" scp -r "%%a" %server%:%folder% || (
    echo Error sending "%%a" to %server%:%folder%
    pause
  )
)

echo Done sending files to the cluster.