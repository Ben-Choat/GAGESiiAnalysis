Ben Choat, 2022/05/24, ben.choat@colostate.edu

This repo holds scripts used in the processing and analysis of data related to the
fourth chapter of my dissertation.

Data used includes DAYMET climate data, USGS streamgauge data, and 
GAGESii catchment characteristics 


# File descriptions:
Python:
	DAYMET_Download_Methods.docx: A few notes on methods used to download DAYMET data

	Scripts:
		DownloadDAYMET_Annual.py: 
			Used to download annual daymet data
		DownloadDAYMET_Monthly.py: 
			Used to download monthly daymet data
		DownloadDAYMET_Daily.py: 
			Used to download daily daymet data

		GAGESii_Class.py: 
			Define Class object holding methods relevant to clustering methodologies
		
		UMAP_HDBSCAN.py:
			Scripts to perform HDBSCAN, UMAP, and combination 
			Note that core of this code is also in GAGESii_Class.py

R:
	MarkdownOutput:
		GAGESii_Analysis2_Presentable.html:
			Some early output from PCA and clustering efforts
		GAGESii_Partition_Train_Val_Test.html:
			Code and results from partitioning of catchments
	
	renv: 
		r environment folder used for version control
	
	scripts:
		GAGESiiAnalysis.Rmd:
			DELETE THIS FILE -- Old Delete - Mostly copy of GAGESii_Analysis.Rmd
		GAGESii_Analysis.Rmd:
			Some preliminary analysis exploring correlation, PCA, UMAP, k-means, and k-medoids
		GAGESii_Analysis2_Presentable:
			Cleaned up preliminary analysis of PCA and clustering for presenting updates at time of writing
		GAGESii_Cluster_Wrk:
			Some preliminary clustering exploration
		GAGESii_CombineOrganize_ExplVars.rmd:
			Script to combine and organize explanatory variables after initial data wrangling, 
			streamflow downloads, and data partitioning.
			Intended to be executed after GAGESii_DataWranglling.Rmd, GAGESii_GatherStreamflow.Rmd,
			GAGESii_Partition_Train_Val_Test.Rmd
		GAGESii_DataWrangling.Rmd:
			Primary data wrangling file used with GAGESii static and TS catchment characteristics
		GAGESii_Daymet_AnnualWY_Scratch.Rmd:
			DELETE THIS FILE
		GAGESii_GatherStreamflow.Rmd:
			Short script to download USGS daily streamflow data using the USGS dataretrieval package
		GAGESii_Partition_Train_Val_Test.Rmd:
			Explore data, the times represented by streamflow data, based on that exploration
			select time period to use and partition catchments into training, validation, and training
			partitions
		GAGESii_Regression_Daymet_AnnualWY.rmd:
			DELETE THIS FILE -- Switched to python before completing this script
		GAGESii_Regression_MeanAnnualWY.Rmd:
			Script used to develop regression analysis of mean annual water yield and catchment
			characteristics from GAGESii. GAGESii-TS variables were averaged over time
		GAGESii_Spatial.Rmd:
			Unddeveloped script at time of writing this. If performing spatial analysis/mapping in R,
			will use this file as basis
		ReadMe.txt:
			Mostly a repetition of information found in this file regarding R scripts
