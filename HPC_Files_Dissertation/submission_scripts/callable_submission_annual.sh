#!/bin/bash




#SBATCH --export=NONE
### #SBATCH --job-name=test_timing6 # commented out so job name can be passed from command line
#SBATCH -p shas
#SBATCH --qos condo
#SBATCH -A csu-summit-asb
#SBATCH -t 120:00:00
#SBATCH --ntasks=6
#SBATCH --mail-user=bchoat@colostate.edu
#SBATCH --mail-type=ALL
#SBATCH --output=./SLURM_Outputs/%x.txt
#SBATCH -N 1


# load modules and python interpreter
module load anaconda
conda activate gagesiienv

# source bash aliases specifically for use of SCRATCH_DIR and PROJECT_DIR
source ~/.bash_aliases

# define variable to hold cluster method descipter, e.g., AggEcoregion or UMAP_HDBSCAN
CLST_MTH=$1   # 'AggEcoregion' # 'None'
# define variable to hold region name, e.g., NorthEast or one
CLST_LBL=$2   #'WestXeric' # 'All'   #'NorthEast'
# define variable specifying the time-scale, e.g., daily, monthly, annual
TIME_LBL=$3   # 'annual'
# make copy of time lable with a capital first letter for use in identifying files
# TIME_LBLcap="${TIME_LBL^}"

# define number of cores to use for relevant processes
NCORES=6

# combine labels into single name/label
NAME="${CLST_MTH}_${CLST_LBL}"


# name of folder to hold working data and scripts in home GAGES_Work dir
HOME_WORK="${PROJECT_DIR}/data_out/${TIME_LBL}/${CLST_MTH}"


# name of folder to hold working data and scripts in scratch GAGES_Work dir
SCRATCH_WORK="${SCRATCH_DIR}/data_out/${TIME_LBL}/${CLST_MTH}"


# make directory to hold scripts, outputs, etc.
mkdir -p $HOME_WORK


# make directory to hold scripts, outputs, etc. in scratch directory
mkdir -p $SCRATCH_WORK



# go to scripts directory in scratch and execute main python script
cd "${SCRATCH_DIR}/scripts"



# execute appropriate code based in time scale and write output to a .txt file

if [[ $TIME_LBL == "mean_annual" ]]
then
 	python HPC_MeanAnnual_CallPred.py $CLST_MTH $CLST_LBL $NCORES > "${SCRATCH_WORK}/${NAME}_output.txt"

elif [[ $TIME_LBL == "annual" ]]
then
	python HPC_Annual_CallPred.py $CLST_MTH $CLST_LBL $NCORES > "${SCRATCH_WORK}/${NAME}_output.txt"

elif [[ $TIME_LBL == "monthly" ]]
then
	python HPC_Monthly_CallPred.py $CLST_MTH $CLST_LBL $NCORES > "${SCRATCH_WORK}/${NAME}_output.txt"

elif [[ $TIME_LBL == "daily" ]]
then
	python HPC_Daily_CallPred.py $CLST_MTH $CLST_LBL $NCORES > "${SCRATCH_WORK}/{$NAME}_output.txt"
fi

# remove *TEMP files
rm "$SCRATCH_DIR"/data_out/"$TIME_LBL"/*TEMP*

# move any csv's just created to SCRATCH WORK
mv "$SCRATCH_DIR"/data_out/"$TIME_LBL"/Results_*"$NAME"*.csv $SCRATCH_WORK

# copy results from scratch to project dir (called $HOME_WORK here)
cp "$SCRATCH_WORK"/*"$NAME"* $HOME_WORK/

# copy csv's holding removed variables from scratch to project
cp -u "$SCRATCH_DIR"/data_out/"$TIME_LBL"/VIF_Removed/*Removed_*"$NAME".csv "$PROJECT_DIR"/data_out/"$TIME_LBL"/VIF_Removed

# copy csv's holding the vif values for each model from scratch to project
cp -u "$SCRATCH_DIR"/data_out/"$TIME_LBL"/VIF_dfs/"$NAME"*csv "$PROJECT_DIR"/data_out/"$TIME_LBL"/VIF_dfs

# copy xgboost model written to model directory
cp -u "$SCRATCH_DIR"/data_out/"$TIME_LBL"/Models/*"$NAME"* "$PROJECT_DIR"/data_out/"$TIME_LBL"/Models/

