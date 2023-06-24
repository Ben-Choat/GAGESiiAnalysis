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

def NSE_KGE_Apply(df_in, return_comp = False): # , y_obs = 'y_obs', y_pred = 'y_pred', ID_in = 'ID_in'):
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
    return_comp: return the individual components of KGE (True or False)?

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
    # if return_comp:
    #     kge_out, rom_out, alpha_out, beta_out = df_in.groupby('ID_in').apply(
    #         lambda df: KGE(df['y_pred'], df['y_obs'], return_comp = return_comp)
    #         )
    # else:
    kge_out = df_in.groupby('ID_in').apply(
        lambda df: KGE(df['y_pred'], df['y_obs'], return_comp = return_comp)
        )

    # percent bias
    pb_out = df_in.groupby('ID_in').apply(
        lambda df: PercentBias(df['y_pred'], df['y_obs'])
        )

    # RMSE
    rmse_out = df_in.groupby('ID_in').apply(
        lambda df: mean_squared_error(df['y_obs'], df['y_pred'], squared = False)
        )
     

    if return_comp:
        df_out = pd.DataFrame({
            'STAID': nse_out.index,
            'NSE': nse_out.values,
            'KGE': [x[0] for x in kge_out],
            'r': [x[1] for x in kge_out],
            'alpha': [x[2] for x in kge_out],
            'beta': [x[3] for x in kge_out],
            'PercBias': pb_out.values,
            'RMSE': rmse_out.values
        })
    else:
        df_out = pd.DataFrame({
            'STAID': nse_out.index,
            'NSE': nse_out.values,
            'KGE': kge_out.values,
            'PercBias': pb_out.values,
            'RMSE': rmse_out.values
        })

    return(df_out)




# %%
# KGE as objective function

def KGE_Objective(y_obs, y_pred, ID_in): # , y_obs = 'y_obs', y_pred = 'y_pred', ID_in = 'ID_in'):
    '''
    Paramters
    ----------
    y_obs: array or pandas Series (float)
        column name holding observed response variable
    y_pred: array or pandas Series (float)
        column name holding predicted response variables
    ID_in: array or pandas Series (string)
        column name holding ID's by which to group response variables
        e.g.,. Catchment ID's (STAID)

    Output
    ----------
    KGE_mean: float
        the mean KGE 
    '''

    df_in = pd.DataFrame({
        'y_obs': y_obs,
        'y_pred': y_pred,
        'ID_in': ID_in
    })
  
    # KGE
    kge_out = df_in.groupby('ID_in').apply(
        lambda df: KGE(df['y_pred'], df['y_obs'])
        )

    kge_out = np.mean(kge_out)


    return(kge_out)
