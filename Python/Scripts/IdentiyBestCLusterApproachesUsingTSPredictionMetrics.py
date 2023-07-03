'''
2023/07/02 BChoat

Script to read in results from either CacluateComponentsOfKGE.py
or CalculatePerformanceMetrics_MeanAnnual.py, and uses those values
to identify best performing models.

Looking for best clustering approach or two out of each of the following
groups which consistently clustered shared > 0.6 AMI.

Consistent groups​
1. 
Class
CAMELS (0.68)​
2.
AggEcoRegion – ​
USDA_LRR_Site (0.62)​
3.
Eco3 – ​
USDA_LRR_Site (0.65)​
4.
All_0​
All_1 (0.91)​
All_2 (0.82)​
Nat_0 (0.75)​
Nat_1 (0.74)​
Nat_2 (0.75)​
5.
Nat_3​
Nat_4 (0.94)​
6.
Anth_0 ​
Anth_1 (0.87)
'''


# %% import libraries
###############################################


import pandas as pd
import numpy as np


# %% define directories, variables, and such
#############################################


# dir holding performance metrics
dir_work = 'd:/Projects/GAGESiiANNStuff/Data_Out/Results'

# define time_scale working with (monthly, annual, mean_annual)
time_scale = 'monthly'




# %% read in metrics to df
#####################################




