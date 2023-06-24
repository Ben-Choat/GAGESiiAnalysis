#!/bin/bash

# Script to loop through defined variables/regions and submit a job for each



# define regions
regions=("Ref" "Non-ref")


for i in ${regions[*]}
do
#	-J (names job), second option calls submission script, thrid option names
#	clustering/regionalization method and must match column name in ID files
#	fourth option defines region
#	fifith option defines time resolution, options include: "mean_annual", "annual", "monthly", "daily"
	sbatch -J monthly_Class_${i} callable_submission_monthly.sh "Class" $i "monthly"
done



