4/20/2022
Ben Choat

This folder (i.e., .../GAGESii_ANNstuff/R/scripts) holds R scripts used for ch. 4 of my disseration 
that is using GAGESii data.

The scripts are generally as follows:

GAGESii_Analysis.rmd:

GAGESii_Analysis2_Presentable.rmd: 
	Presents EDA of GAGESii data.
	PCA, Clustering, etc.

GAGESii_Cluster_Wrk.rmd: 
	k-means, k-medoids clustering in R

GAGESii_CombineOrganize_ExplVars.rmd: 
	Script to combine static and time-series GAGESii data and filter to only 
	those variables that will be considered in the final analysis.

GAGESii_DataWrangling.rmd:
	Data manipulation and organizing with the GAGESii data set

GAGESii_Daymet_AnnualWY_Scratch.rmd:
	A scratch file exploring the Daymet annual data and the annual WY data from USGS gages

GAGESii_Exploring_Scratch.Rmd:
	A scratch file with some exploratory work

GAGESii_GatherStreamflow.rmd:
	Gather streamflow data from USGS gages based on gage id's

GAGESii_Partition_Train_Val_Test.rmd: 
	Partition the gages into training, validation, and testing sets

GAGESii_Regression_MeanAnnualWY.rmd:
	Regression of mean annual WY (from USGS gages) against GAGESii data

GAGESii_Spatial.rmd:
	Some spatial epxloratory / visualization ... not really used for any 
	specific outputs at this point:


