BChoat 2022/12/09

Describe workflow related to HPC bash scripts and execution on the 
SUMMIT HPC.

For each clustering method (e.g., AggEcoregion, CAMELS, UMAP-HDBSCAN,
Hydrologic Landscapes Regions, etc.), there are five bash scripts used
for defining jobs and submitting them.

Using AggEcoregions as an example:

Within the submission_scripts folder there is a folder labeled
'aggecoregion'.

In the 'aggecoregion' folder the following files are present:
(first files saved as 'clustermethod_timescale_submit.sh')
(second files saved as 'callable_submission_timescale.sh')

aggecoregion_meanannual_submit.sh
aggecoregion_annual_submit.sh
aggecoregion_monthly_submit.sh
callable_submission_annual.sh
callable_submission_monthly.sh

The "clustermethod_timescale_submit.sh) is the main file that is executed,
and it calls the other files.

The aggecoregion_meanannual_submit.sh and aggecoregion_annual_submit.sh
files call the callable_submission_annual.sh file and the 
aggecoregion_monthly_submit.sh file calls the callable_submission_monthly.sh
file. 

The 'clustermethod_timescale_submit.sh' scripts define the labels used for
the regions resulting from the clustermethod (e.g., NorthEast, SECstPlain,
etc. for AggEcoregions). 

Those labels are then looped through and a job is submitted for each region by
calling the 'callable_submission_timescale.sh' file and providng a job name,
clustermethod label, and timescale (e.g., "annual" or "mean_annual").

The 'callable_submission_timescale.sh' script then makes appropriate directories
in the scratch directory, copies files into those directories, defines output/log
file names, executes the jobs, removes uneeded files created by the job, and copies
appropriate files to appropriate folders.

NOTE: This workflow expects the appropriate files to be located in the scratch
directory before executing them. That is to say, the scripts do not copy the files
over from the project directory to the scratch directory each time. This decision 
was made becuase some of the input data files are quite large (especially) when
I was using daily data, so it could take some time to copy them over.


The arguments passed in the 'clustermethod_timescale_submit.py' file 
(e.g., "AggEcoregion") should match column names in the 
ID csvs (e.g., ID_train.csv).








