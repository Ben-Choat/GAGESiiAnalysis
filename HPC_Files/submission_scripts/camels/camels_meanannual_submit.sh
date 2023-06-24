#!/bin/bash

# Script to loop through defined variables/regions and submit a job for each



# define regions
regions=("y" "n")
# ("NorthEast") # "SECstPlain" "SEPlains" "EastHghlnds" "CntlPlains" \
# 	"MxWdShld" "WestMnts" "WestPlains" "WestXeric")


for i in ${regions[*]}
do
#	-J (names job), second option calls submission script, thrid option names
#	clustering/regionalization method and must match column name in ID files
#	fourth option defines region
#	fifith option defines time resolution, options include: "mean_annual", "annual", "monthly", "daily"
	sbatch -J mean_annual_CAMELS_${i} callable_submission_annual.sh "CAMELS" $i "mean_annual"
done



