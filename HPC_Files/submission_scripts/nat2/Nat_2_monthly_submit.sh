#!/bin/bash

# Script to loop through defined variables/regions and submit a job for each



# define regions
regions=(-1 0 1 2 3 4 5 6)

for i in ${regions[*]}
do
#	-J (names job), second option calls submission script, third option names
#	clustering/regionalization method and must match column name in ID files
#	fourth option defines region
#	fifth option defines time resolution; options include: "mean_annual", "annual", "monthly", "daily"
	sbatch -J monthly_Nat_2_${i} callable_submission_monthly.sh "Nat_2" $i "monthly"
done



