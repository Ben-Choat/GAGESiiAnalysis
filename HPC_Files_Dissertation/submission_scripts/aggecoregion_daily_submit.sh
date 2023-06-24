#!/bin/bash

# Script to loop through defined variables/regions and submit a job for each



# define regions
regions=("NorthEast" "SECstPlain" "SEPlains" "EastHghlnds" "CntlPlains" \
	"MxWdShld" "WestMnts" "WestPlains" "WestXeric")


for i in ${regions[*]}
do
#	-J (names job), second option calls submission script, third option names
#	clustering/regionalization method and must match column name in ID files
#	fourth option defines region
#	fifth option defines time resolution; options include: "mean_annual", "annual", "monthly", "daily"
	sbatch -J daily_AggEcoregion_${i} callable_submission_daily.sh "AggEcoregion" $i "daily"
done



