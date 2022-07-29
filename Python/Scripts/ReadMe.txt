2022/07/26 Ben Choat
Text file describing the files located in the Python/Scripts directory related to
GAGESii_ANNstuff (GAGESii work from my PhD dissertation).


GAGESii_Class.py: Defines the Clusterer and Regressor objects which contain nearly all
	methods and objects used in this analysis.

GAGESii_Class_Apply.py: This is the testground/playground for the GAGESii_Class.py script.
	It is also the best script to view for examples of applying the GAGESii_Class.py classes.

GAGESii_MeanAnuual_Callable.py: This is the script where the analysis for mean annual water yield
	has been automated to the greatest extent reasonable. 

Cluster_CallPred.py: Script to perform clustering and then call 
	GAGESii_MeanAnnual_Callable.py to apply regression methods to a given cluster (or all data)

CombineDataForPlottingCatchments.py: Short script to combine variables of interest that may be used 
	for labelling/coloring/defining shapes along with lat and long to be used as an input file 
	for plotting the location of the study catchments across the contiguous U.S.  

DownloadDAYMET_Annual.py: Script to download annual daymet data using the shape files that came 
	with the GAGESii data.

DownloadDAYMET_Monthly.py: Script to download monthly daymet data using the shape files that came 
	with the GAGESii data.

DownloadDAYMET_Daily.py: Script to download daily daymet data using the shape files that came 
	with the GAGESii data. 
	NOTE: (2022/07/26) There is an updated script on the campus PC where data is actively 
	being downloaded over several days.
	BE SURE TO REPLACE THE SCRIPT HERE WITH THAT ONE, AND DELETE THIS NOTE AFTER DOING SO

Plotting_Scratch.py: A plotting playground. Currently (2022/07/26) primarily used to investigate
	regression results from mean annual water yield data including all data and ecoregions

Regression_PerformanceMetrics_Functs.py: Script that defines several functions, such as 
	adjR2, MAE, VIF, etc. to be applied when analyzing the performance of predictive models.

UMAP_LASSO_CVwGS.py: # File specifically for finding best hyperparemters for the model pipeline of
	raw -> stdrd -> umap -> lasso
	same umap paramters identified here will be applied for the pipeline of
	raw -> stdrd -> umap -> ols-mlr
	using gridsearch


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
