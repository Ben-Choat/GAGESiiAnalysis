# Ben Choat, 2022/09/07
# This script defines a function that applies Nash-Sutcliffe Efficiency and Kling-Gupta
# Efficieny to many catchments

# %% 
# Import Libraries
from time import perf_counter_ns
import pandas as pd
import numpy as np
from Regression_PerformanceMetrics_Functs import NSE
from Regression_PerformanceMetrics_Functs import KGE
from Regression_PerformanceMetrics_Functs import PercentBias
from sklearn.metrics import mean_squared_error


# %%
# NSE and KGE

def NSE_KGE_Apply(df_in): # , y_obs = 'y_obs', y_pred = 'y_pred', ID_in = 'ID_in'):
    '''
    Paramters
    ----------
    df_in: pandas dataframe including three columns named
        1. 'y_obs', 2. 'y_pred', 3. 'ID_in'
        y_obs: string
            column name holding observed response variable
        y_pred: string
            column name holding predicted response variables
        ID_in: string
            column name holding ID's by which to group response variables
            e.g.,. Catchment ID's (STAID)

    Output
    ----------
    df_NSE_KGE: pandas DataFrame
        Holds NSE and KGE for each catchment
    '''

    # NSE
    nse_out = df_in.groupby('ID_in').apply(
        lambda df: NSE(df['y_pred'], df['y_obs'])
        ) 
    
    # KGE
    kge_out = df_in.groupby('ID_in').apply(
        lambda df: KGE(df['y_pred'], df['y_obs'])
        )

    # percent bias
    pb_out = df_in.groupby('ID_in').apply(
        lambda df: PercentBias(df['y_pred'], df['y_obs'])
        )

    # RMSE
    rmse_out = df_in.groupby('ID_in').apply(
        lambda df: mean_squared_error(df['y_obs'], df['y_pred'], squared = False)
        )
    

    df_out = pd.DataFrame({
        'STAID': nse_out.index,
        'NSE': nse_out.values,
        'KGE': kge_out.values,
        'PercBias': pb_out.values,
        'RMSE': rmse_out.values
    })

    return(df_out)






# %%
