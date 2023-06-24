2022/07/26 Ben Choat
Text file describing the files located in the Python/Scripts directory related to
GAGESii_ANNstuff (GAGESii work from my PhD dissertation).

GAGESii_Class.py: Defines the Clusterer and Regressor objects which contain nearly all
	methods and objects used in this analysis.

GAGESii_Class_Apply.py: This is the testground/playground for the GAGESii_Class.py script.
	It is also the best script to view for examples of applying the GAGESii_Class.py classes.

GAGESii_XGBOOSTcv_learn.py: Script applying XGBoost algorith from the xgboost library to mean
	annual water yield. Also code investigating poorly performing catchments from original
	paritioning.

GAGESii_AddAntecedent_All.py: Script to define antecedent variables for all 
	DAYMET variables. Opposed to summing or average, this script simply provides
	the actual reported values for each day, month, or year or a spefied lead time.

GAGESii_Mapping.py: Script to produce maps for GAGESii chapter of dissertation.

GAGESii_Plotting.py: Script used for plotting results from GAGESii work. E.g.,
	heatmap of shap values, barplots of shap values.

GAGESii_SHAP_CalcAll.py: Script to calculate all shapley values for each model specified. It also
	performs linear regression on standardized explanatory variables and shape values to get
	a general directional relationship.

GAGESii_ExploreTree.py: Explore information available in the XGB trees produced during 
	training.

GAGESii_ProcessWWTP.py: Code to read in WWTP effluent data, sum the total effluent
	within a catchment, and write out to explanatory variable csv.

GAGESii_AddCAMELS_ID.py: Script to identify CAMELS catchments, and add a column
	to ID***csv's noting if each catchment is a CAMELS catchment or not.

GAGESii_timeseries_plots.py: Create timeseries plots of results

Gather_Outputs.py: Script to gather outputs from individual models (e.g., ecoregions)
	and combine them into one of two pickle files. One pickle file holds the results
	for individual catchments and the other holds summary results. 

DownloadDAYMET_Annual.py: Script to download annual daymet data using the shape files that came 
	with the GAGESii data.

DownloadDAYMET_Monthly.py: Script to download monthly daymet data using the shape files that came 
	with the GAGESii data.

DownloadDAYMET_Daily.py: Script to download daily daymet data using the shape files that came 
	with the GAGESii data. 
	NOTE: (2022/07/26) There is an updated script on the campus PC where data is actively 
	being downloaded over several days.
	BE SURE TO REPLACE THE SCRIPT HERE WITH THAT ONE, AND DELETE THIS NOTE AFTER DOING SO

Daymet_CleanUP.py: The script used to download daily DAYMET data resulted in a couple
	of errors that are corrected in this script:
	1. There are about 6 catchments that are missing a day or two. They are removed
	2. There were some duplicated rows, so those are removed.

Plotting_Scratch.py: A plotting playground. Currently (2022/07/26) primarily used to investigate
	regression results from mean annual water yield data including all data and ecoregions

Regression_PerformanceMetrics_Functs.py: Script that defines several functions, such as 
	adjR2, MAE, VIF, etc. to be applied when analyzing the performance of predictive models.

UMAP_LASSO_CVwGS.py: # File specifically for finding best hyperparemters for the model pipeline of
	raw -> stdrd -> umap -> lasso
	same umap paramters identified here will be applied for the pipeline of
	raw -> stdrd -> umap -> ols-mlr
	using gridsearch

ExplDataReduction_VIF.py: Short script to remove variables that have a VIF greater than
	a defined threshold (.e.g., 10). Removes variables with largest VIF first. If 
	variables tie, then variables that tie are removed in alphabetical order.

Train_Val_Test_RePartitioning_AdvVal.py: After investigating original partitioning, decided to repartition
	data using adversarial paritioning up front. 

	This script also removes catchments for which daily DAYMET data failed to download.
	The relevant lines of code will be commented out if we decide to remove daily 
	data from analysis.

	NOTE: This is now (10/1/2022) the main script partitioning data into training, validation, and testing 
	partitions.

Train_Val_Test_RePartitioning.py: Similar to Train_Val_Test_RePartitioning_AdvPart.py, except
	does not consider multiple random states, but rather, uses one and then applies 
	adversarial validation after the fact to investigate if data was split in a way where
	the partitions are from the same distribution (i.e., xgboost is unable to identify
	which data is in training vs testing/validation partitions based on explantory vars.

ManualHierarchicalClustering.py: Script to perform manual hieararchical clustering - where the 
	user defines the type of variables to be used in each layer of clustering (e.g., use
	basin size as base layer, followed by climate, followed by physiography, followed by
	antrhopogenic variables.

CSVtoPickle.py: Script to read in multiple csv files and combine them into a single
	pickle file. Was used to combine DAYMET and USGS csv files into training
	testin, and valnit pickle files.

CSVtoParquet.py: Script to read in multiple csv files and combine them into a single
	parquet file. Was used to combine DAYMET and USGS csv files into training
	testin, and valnit pickle files. This was motivated by consideration of using
	DASK for parallel processing. It allows readinga and writing Parquet files
	but not pickle files.

HPC_$TIMESCALE_Callable.py: Script that defines function that applies specified models
	 at specified $TIMESCALE when called

HPC_$TIMESCALE_CallPred.py: Script that calls HPC_$TIMESCALE_Callable.py. This script
	is where data is developed for use in Callable script.

HydrologicLandscapes.py: Script to assign each study catchment to a hydrologic
	landscape region (Winter 2001)..

Load_Data.py: Defines a function that loads explanatory variables, response variables,
	and ID variables for specified time-scale

NGE_KGE_timeseries.py: Function to calculate NSE and KGE.

UMAP_HDBSCAN_ParamIdentify.py: Identify potential hyperparameter sets to be used 
	in UMAP -> HDBSCAN process. Outputs parameter values and resuling
	relative validity as measure of how well clusters are separated.

UMAP_HDBSCAN_GetLabels.py: After identifying potential parameters to be used,
	this script applies UMAP->HDBSCAN using those parameters sets and
	returns a label for each catchment and each paramter set identifying 
	to which cluster each catchment belongs.



###########
OldFiles: Directory to hold older scripts/files that did not end up being used, but
	may be of use in other context/applications.
###########

ExplDataReduction_r.py: Short script to remove variables that share a pearson correlation 
	greater than a defined threshold (e.g., 0.9) with another variable. Ended up 
	using VIF instead of r.

GAGESii_AddAntecedentPrecTemp_summ.py: Script to calculate antecedent forcings of precip
	and temperature. For precip it summs total precip over time periods of interest.
	For temperature, it takes the average.


###############
Likely okay to delete scripts below here (2022/07/26)
###############
UMAP_HDBSCAN: Likely okay to delete. Was used to explore application of UMAP with HDBSCAN. That work is
	mostly replicated in the GAGESii_Class_Apply.py script.

GAGESii_Master_Apply.py: Likely okay to delete. Was an intermediate script where the actual 
	anlaysis was being perfromed (results being produced), but I later decided to have a
	script for each time scale of analysis. For example for mean annual water yield, 
	annual water yield, monthly water yield, and daily water yield.

GAGESii_MeanAnnual_Apply.py: Also likely okay to delete later on. Began using this script, but
	later edited so that all of the analysis is a more automated callable function which
	prompts the user for input as needed.

Cluster_CallPredFirstPartitionin.py: Script to perform clustering and then call 
	GAGESii_MeanAnnual_Callable.py to apply regression methods to a given cluster (or all data)
	Written to work with the first partitioning, which has been redone.

Cluster_CallPredSecondPartitioning.py: Script to perform clustering and then call 
	GAGESii_MeanAnnual_Callable.py to apply regression methods to a given cluster (or all data)
	Written to work with the second partitioning, which is the working version.

CombineDataForPlottingCatchments.py: Short script to combine variables of interest that may be used 
	for labelling/coloring/defining shapes along with lat and long to be used as an input file 
	for plotting the location of the study catchments across the contiguous U.S.  

GAGESii_Master_Apply.py: Similar to GAGESii_Class_Apply.py, but cleaned up and more streamlined.

GAGESii_MeanAnnual_Callable.py: This is the script where the analysis for mean annual water yield
	has been automated to the greatest extent reasonable. - Replaced by HPC_MeanAnnual_Callable.py

InvestigateTrain_Val_Test_Partitioning.py: Script applying adversarial validation to
	understand if first partitioning resulted in partitions of training, validating, and
	training data with similar distributions of explanatory variables.

	NOTE: This script is originally where the ID output from 'CombineDataForPlottingCatchments.py'
	is read in, edited, and written back out as 'ID_all_avail98_12.csv'

