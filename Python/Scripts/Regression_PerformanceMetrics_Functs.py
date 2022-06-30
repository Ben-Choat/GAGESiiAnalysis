# %% Intro
# Ben Choat, 2022/6/29, bchoat@rams.colostate.edu

# This script defines several functions to quantify the performance and attributes
# of predictive regression models
# for example: AIC, BIC, Mallows' Cp, and adjusted R2 can all be calculated

# NOTE:
# unadjusted R2 can be calculated using r2_score() enabled with
# "from sklearn.metrics import r2_score"
# Mean absolute error (MAE) can be calculated with an sklearn function as well ...
# "from sklearn.metrics import mean_absolute_error"
# note that the mean_absolute_error function will actually return the negative
# of the mean absolute error

# %% Import libraries
import numpy as np
import pandas as pd

# %% Define functions for calucating prediction performance metrics
# sum of squared residuals
def ssr(y_pred, y_obs):
    """
    Parameters
    ----------
    y_pred: list(double check type) of predicted y (Water yield) values
    y_obs: list(double doublce check type) of observed y (Water yield) values
    Attributes
    ----------
    ssr: float
        sum of squared residuals
    """
    ssr = round(np.sum((y_pred - y_obs)**2), 2)
    return(ssr)
# AIC
def AIC(n_k, ssr, n_f):
    """
    Parameters
    ----------
    n_k: sample size
    n_f: number of features
    ssr: sum of squared residuals
    
    Attributes
    ----------
    aic: float
        Akaike Information Criterion; Smaller is better
    """
    aic = round(n_k * np.log(ssr/n_k) + 2 * n_f, 2)
    return(aic)

# BIC
def BIC(n_k, ssr, n_f):
    """
    Parameters
    ---------
    n_k: sample size
    n_f: number of features
    ssr: sum of squared residuals
    
    Attributes
    ----------
    bic: float
        Bayesian Information Criterion; Smaller is better
    """
    bic = round(n_k * np.log(ssr/n_k) + n_f * np.log(n_k), 2)
    return(bic)

# Mallows' Cp
def M_Cp(ssr_all, ssr, n_k, p_k):
    """
    Mallows' Cp. Theory suggests the best-fitting model should have Cp ~= p,
    where p = k + 1 and k = number of features
    
    NOTE: ssr_all/n_k = MSE (mean squared error of the model using all features)

    Parameters
    ----------
    MSE_all: float
        Mean squared error from regression equation with all features
    ssr: float
        sum of squared residuals from model with p paramters (including intercept)
    n_k: integer
        sample size
    p_k: integer
        total number of variables in model being measured, including intercept.
        If number of features = k, then p = k + 1, where 1 is the intercept
    """
    M_Cp = round(ssr/(ssr_all/n_k) - (n_k - 2*p_k), 2)
    return(M_Cp)


# Adjusted R2
def R2adj(n_k, n_f, r2):
    """
    Parameters
    ----------
    n_k: sample size
    n_f: number of features
    r2: unadjusted r-squared (NOTE: that r2_score() function from
        sklearn.metrics can be used to calculate unadjusted r2)
    """
    r2adj = round(1 - ((n_k - 1)/(n_k - n_f - 1)) * (1 - r2), 2)
    return(r2adj)

# Variance Inflation Factor (VIF)
def VIF(X):
    """
    This if taken from,
    https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python
    with few edits by me.
    It is a very concise syntax for calculating VIF
    I compared it againt the car::vif command in R and it produced idential results
    It essentially takes the diagonal of the inverse of the correlation matrix, which
    gives the values of 1/(1 = Rj2) = VIF

    NOTE: The function expects pandas to be imported as pd, and numpy as np
    Paramters
    ---------
    X: pandas DataFrame of explantory variables
    """
    df_cor = X.corr()
    vifs = pd.Series(np.linalg.inv(df_cor.values).diagonal(), index=df_cor.index)
    return(vifs) 

# Nash-Sutcliffe Efficiency
def NSE(y_pred, y_obs):
    """
    Nash-Sutcliffe Efficiency for time series performance
    
    Parameters
    ----------
    y_pred: array of predicted time-series values
    y_obs: array of observed time-series values
    """
    nse = round(1 - (np.sum((y_obs - y_pred)**2)/np.sum((y_obs - np.mean(y_obs))**2)), 3)
    return(nse)

# Kling Gupta Efficiency
def KGE(y_pred, y_obs):
    """
    Kling-Gupta Efficiency for time series performance
    
    Parameters
    ----------
    y_pred: array of predicted time-series values
    y_obs: array of observed time-series values
    """
    # get correlation coef between observed and predicted values
    r_o_m = np.corrcoef(y_pred, y_obs)[0, 1]
    # calc alpha as StDev_pred/StDev_obs
    alpha = np.std(y_pred)/np.std(y_obs)
    # calc bias term beta as mean_pred/mean_obs
    beta = np.mean(y_pred)/np.mean(y_obs)
    kge = round(1 - np.sqrt((r_o_m - 1)**2 + (alpha - 1)**2 + (beta - 1)**2), 3)
    
    return(kge)

# Percent Bias
def PercentBias(y_pred, y_obs):
    """
    calculate percent bias: The average tendency of the simulated values
    to be larger or smaller than observed values

    Parameters
    ----------
    y_pred: array of predicted time-series values
    y_obs: array of observed time-series values
    """

    # calc percent bias
    perc_bias = round(100 * (np.sum(y_pred - y_obs)/sum(y_obs)), 1)
    return(perc_bias)