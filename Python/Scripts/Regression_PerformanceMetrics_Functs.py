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
# import plotnine as p9
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
# import matplotlib.pyplot as plt


# %% Define functions for calucating prediction performance metrics
# sum of squared residuals
def ssr(y_pred, y_obs):
    """
    Parameters
    ----------
    y_pred: numpy array of predicted y (Water yield) values
    y_obs: numpy array of observed y (Water yield) values
    Attributes
    ----------
    ssr: float
        sum of squared residuals
    """
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    
    ssr = np.sum((y_pred - y_obs)**2)
    return(ssr)
# AIC
def AIC(n_k, ssr_k, n_f):
    """
    Parameters
    ----------
    n_k: sample size
    n_f: number of features
    ssr_k: sum of squared residuals
    
    Attributes
    ----------
    aic: float
        Akaike Information Criterion; Smaller is better
    """
    # aic = n_k * np.log(ssr_k/n_k) + 2 * n_f
    # below formulation from Helsel et al. 2020 - Statistical Methods in Water Resources
    aic = n_k + n_k * np.log(2 * np.pi) + n_k * np.log(ssr_k/n_k) + 2 * (n_f + 2)
    return(aic)

# BIC
def BIC(n_k, ssr_k, n_f):
    """
    Parameters
    ---------
    n_k: sample size
    n_f: number of features
    ssr_k: sum of squared residuals
    
    Attributes
    ----------
    bic: float
        Bayesian Information Criterion; Smaller is better
    """
    # bic = n_k * np.log(ssr_k/n_k) + n_f * np.log(n_k)
    # below formulation from Helsel et al. 2020 - Statistical Methods in Water Resources
    bic = n_k + n_k * np.log(2 * np.pi) + n_k * np.log(ssr_k/n_k) + np.log(n_k) * (n_f + 1)
    return(bic)

# Mallows' Cp
def M_Cp(ssr_all, ssr_k, n_k, n_f):
    """
    Mallows' Cp. Theory suggests the best-fitting model should have Cp ~= p,
    where p = k + 1 and k = number of features
    
    NOTE: ssr_all/n_k = MSE (mean squared error of the model using all features)

    Parameters
    ----------
    ssr_all: float
        sum of squared residuals from regression equation with all features
    ssr_k: float
        sum of squared residuals from model with p paramters (including intercept)
    n_k: integer
        sample size
    n_f: integer
        n_f: number of features
    """
    # M_Cp = ssr_k/(ssr_all/(n_k - n_f - 1)) - n_k + 2 * (n_f + 1)
    # below formulation from Helsel et al. 2020 - Statistical Methods in Water Resources
    # define p; p = number of features + 1 (1 for the intercept)
    p_in = n_f + 1
    # define MSE_k (mean squared error for model with p coefficients (including intercept as coefficeint))
    MSE_k = ssr_k/(n_k - p_in)
    # define MSE for model including all candidate variables
    MSE_all = ssr_all/(n_k  - p_in)
    M_Cp = (p_in) + ((n_k - p_in) *(MSE_k - MSE_all))/MSE_all
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
    try:
        r2adj = 1 - ((n_k - 1)/(n_k - n_f - 1)) * (1 - r2)
    except:
        r2adj = -9999

    return(r2adj)

# Variance Inflation Factor (VIF)
def VIF(df_X):
    """
    This if taken from,
    https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python
    with few edits by me.
  
    NOTE: The function expects pandas to be imported as pd, and numpy as np
    Parameters
    ---------
    df_X: pandas DataFrame of explantory variables
    """

    # try to use statsmodels VIF function first. It may fail if VIF approaches Inf
    # or RSS is 0 (which rarely occurs).
    # If the statsmodel VIF function fails, then take the diagonal which will return 
    # a NA or NaN leaving a blank space in the results.
    try:
        df = df_X
    
        X = add_constant(df, has_constant = 'add')
        vifs = pd.Series([variance_inflation_factor(X.values, i)
                for i in range(X.shape[1])], 
                    index=X.columns)

        vifs = vifs.drop('const')
    
    except:
        df_cor = df_X.corr()
        vifs = pd.Series(np.linalg.inv(df_cor.values).diagonal(), index=df_cor.index)

    

    return(vifs)


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
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    # calc percent bias
    perc_bias = 100 * (np.sum(y_pred - y_obs)/np.sum(y_obs))
    return(perc_bias)

# Nash-Sutcliffe Efficiency
def NSE(y_pred, y_obs):
    """
    Nash-Sutcliffe Efficiency for time series performance
    
    Parameters
    ----------
    y_pred: array of predicted time-series values
    y_obs: array of observed time-series values
    """
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    nse = 1 - (np.sum((y_obs - y_pred)**2)/np.sum((y_obs - np.mean(y_obs))**2))
    return(nse)

# Kling Gupta Efficiency
def KGE(y_pred, y_obs, return_comp = False):
    """
    Kling-Gupta Efficiency for time series performance
    
    Parameters
    ----------
    y_pred: array of predicted time-series values
    y_obs: array of observed time-series values
    return_comp: return the individual components of KGE (True or False)?

    Returns:
    ----------
    array of kge and if return_comp == True, r, alpha, and beta
        r: correlation between observed and predicted values
        alpha: variability bias (ratio of stdevs)
        beta: mean bias

    """
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    # get correlation coef between observed and predicted values
    r_o_m = np.corrcoef(y_pred, y_obs)[0, 1]
    # calc alpha as StDev_pred/StDev_obs
    alpha = np.std(y_pred)/np.std(y_obs)
    # calc bias term beta as mean_pred/mean_obs
    beta = np.mean(y_pred)/np.mean(y_obs)
    kge = 1 - np.sqrt((r_o_m - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    if return_comp:
        return(kge, r_o_m, alpha, beta)
    else:
        return(kge)

# plot performance metrics against number of features
def PlotPM(df_in, timeseries = False):
    """
    Assumes the columns of input Pandas DataFrame are 
    'n_features', 'ssr', 'r2', 'r2adj', 'mae', 'rmse', 'AIC', 'BIC', 'M_Cp',
    'VIF', 'percBias', 
    or if timeseries = True, then also including 'NSE' and 'KGE'
    
    Parameters
    -----------
    df_in: Pandas DataFrame
        see note above about columns
    """

    for i in range(1, (df_in.shape[1]), 1):
        # define y_in as response variable
        # if VIF, then take max
        y_in = df_in[df_in.columns[i]]
        
        # if column = VIF, then take max
        if df_in.columns[i] == 'VIF':
            y_in = df_in['VIF'].apply(max)
            # color = []
            # for j in y_in:
            #     if y_in <= 10:
            #         color = [color, 'green']
            #     e



        # define plot
        if df_in.columns[i] == 'VIF' or df_in.columns[i] == 'M_Cp':
            p = (
                p9.ggplot(data = df_in) +
                p9.geom_point(p9.aes(x = 'n_features', y = y_in)) +
                p9.scale_y_log10(breaks = [1, 10, 50, 100, 1000]) +
                p9.theme_light() 
            )

        p = (
                p9.ggplot(data = df_in) +
                p9.geom_point(p9.aes(x = 'n_features', y = y_in)) +
                p9.theme_light()
            )
        print(p)


    # analyze residuals and return diagnostic plots
def test_reg_assumpts(residuals, y_pred):

    #create Q-Q plot with 1:1 line added to plot
    # fig = sm.qqplot(residuals, line='45')
    # return(fig)
    fig_qq = (
                p9.ggplot(p9.aes(sample = residuals)) +
                p9.stat_qq() +
                p9.stat_qq_line(color = 'red') +
                p9.ggtitle('Q-Q Plot')
    )
    
    

    # plot residuals vs fitted values
    # input dataframe
    df_in = pd.DataFrame({
        'residuals': residuals,
        'predicted': y_pred
    })

    fig_res_pred = (
                    p9.ggplot(df_in, p9.aes('predicted', 'residuals')) +
                    p9.geom_point() +
                    p9.stat_smooth(method = 'lowess', color = 'red') +
                    p9.ggtitle('Residuals vs Fitted')
    ) 

    # return(fig_qq)
    return(fig_res_pred, fig_qq)

# %%
