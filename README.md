# sandy_impacts_storylines

This repository contains the code for the reproduction of the paper:

Goulart, H.M.D., Benito Lazaro, I., van Garderen, L., van der Wiel, K., Le Bars, D., Koks, E., and van den Hurk, B., 2023. Compound flood impacts from Hurricane Sandy on New York City in climate-driven storylines

The paper aims to assess the compound coastal flooding impacts of Hurricane Sandy (2012) on critical infrastructure in New York City, using a storyline approach. The storyline approach is a qualitative method that allows to explore the cascading effects of extreme events and the interdependencies between critical infrastructures. The paper presents four storylines that describe different scenarios of Hurricane Sandy impacts, based on the literature review, expert interviews, and climate model projections. The storylines are then quantified using a modelling framework that connects meteorological conditions to local hazards and impacts, using the Global Tide and Surge Model (GTSM), the SFINCs model, and the Critical Infrastructure Exposure Analysis (CIEA).

The code is written in Python and requires the following packages:

- numpy
- pandas
- matplotlib
- scipy
- networkx
- pySD
- cdsapi
- hydromt

The code consists of the following files and directories:

- gtsm_scripts: the directory that contains the scripts for the GTSM model
- levante_scripts: the directory that contains the scripts for the Levante model
- cds_era5_data.py: the file that downloads and processes the ERA5 data from the Copernicus Data Store
- cds_extreme_precip_download.py: the file that downloads and processes the extreme precipitation data from the Copernicus Data Store
- cds_water_level_change.py: the file that downloads and processes the water level change data from the Copernicus Data Store
- ci_exposure_analysis.py: the file that performs the critical infrastructure exposure analysis using the CIEA
- ci_functions.py: the file that contains some functions for the critical infrastructure analysis
- cities_storms_creation_script.py: the file that creates the storm tracks for the different cities
- clip_deltares_data_to_region.py: the file that clips the Deltares data to the region of interest
- cmip6_load.py: the file that loads and processes the CMIP6 data
- coastal_sfincs_model_hydromt_notebooks.ipynb: the file that runs the SFINCs model using the hydromt package
- gtsm_plots_scatter.py: the file that plots the scatter plots for the GTSM model
- hydromt_sfincs_pipeline.py: the file that runs the SFINCs model using the hydromt package
- paper_3.code-workspace: the file that contains the code workspace for the paper
- plot_gtsm_surge.py: the file that plots the surge results for the GTSM model
- plot_storms_animation.py: the file that plots the animation of the storm tracks
- sfincs_creation_files_sn.py: the file that creates the input files for the SFINCs model on the Snellius cluster
- sfincs_file_creation_era5.py: the file that creates the input files for the SFINCs model using the ERA5 data
- sfincs_inundation_scenarios_quantification.py: the file that quantifies the inundation scenarios for the SFINCs model
- sfincs_obtain_snellius_runs.bat: the file that obtains the output files from the SFINCs model on the Snellius cluster
- sfincs_sealevelrise.py: the file that calculates the sea level rise scenarios for the SFINCs model
- sfincs_send_snellius_runs.bat: the file that sends the input files for the SFINCs model to the Snellius cluster
