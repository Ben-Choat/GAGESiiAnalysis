'''
Investigating use of custom loss fuction in XGBoost.
Specifically, trying to use average NSE as loss function.
'''
# %%
# import libraries
import pandas as pd
import numpy as np
import xgboost as xgb
import plotnine as p9


# %% load data

# main working directory
# NOTE: may need to change '8' below to a different value
# dir_Work = '/media/bchoat/2706253089/GAGES_Work' 
# dir_Work = os.getcwd()[0:(len(os.getcwd()) - 8)]
# dir_Work = '/scratch/bchoat'
# dir_Work = '/scratch/summit/bchoat@colostate.edu/GAGES'
dir_Work = 'D:/Projects/Gagesii_ANNstuff/HPC_Files/GAGES_Work'

# water yield directory
# dir_WY = 'D:/DataWorking/USGS_discharge/train_val_test'
dir_WY = f'{dir_Work}/data_work/USGS_discharge'

# DAYMET directory
# dir_DMT = 'D:/DataWorking/Daymet/train_val_test'
dir_DMT = f'{dir_Work}/data_work/Daymet'

# explantory var (and other data) directory
# dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out/AllVars_Partitioned'
dir_expl = f'{dir_Work}/data_work/GAGESiiVariables'

# directory to write csv holding removed columns (due to high VIF)
# dir_VIF = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Results/VIF_Removed'
dir_VIF = f'{dir_Work}/data_out/annual/VIF_Removed'


# GAGESii explanatory vars
# training
df_train_expl = pd.read_csv(
    f'{dir_expl}/Expl_train.csv',
    dtype = {'STAID': 'string'}
)
# test_in
df_testin_expl = pd.read_csv(
    f'{dir_expl}/Expl_testin.csv',
    dtype = {'STAID': 'string'}
)
# val_nit
df_valnit_expl = pd.read_csv(
    f'{dir_expl}/Expl_valnit.csv',
    dtype = {'STAID': 'string'}
)


# Water yield variables
# Annual Water yield
# training
df_train_anWY = pd.read_csv(
    f'{dir_WY}/annual/WY_Ann_train.csv',
    dtype = {"site_no":"string"}
    )

# val_in
df_testin_anWY = pd.read_csv(
    f'{dir_WY}/annual/WY_Ann_testin.csv',
    dtype = {"site_no":"string"}
    )

# val_nit
df_valnit_anWY = pd.read_csv(
    f'{dir_WY}/annual/WY_Ann_valnit.csv',
    dtype = {"site_no":"string"}
    )


# DAYMET
# training
df_train_anDMT = pd.read_csv(
    f'{dir_DMT}/annual/DAYMET_Annual_train.csv',
    dtype = {"site_no":"string"}
    )

# val_in
df_testin_anDMT = pd.read_csv(
    f'{dir_DMT}/annual/DAYMET_Annual_testin.csv',
    dtype = {"site_no":"string"}
    )

# val_nit
df_valnit_anDMT = pd.read_csv(
    f'{dir_DMT}/annual/DAYMET_Annual_valnit.csv',
    dtype = {"site_no":"string"}
    )

# ID vars (e.g., ecoregion)

# training ID
df_train_ID = pd.read_csv(f'{dir_expl}/ID_train.csv',
    dtype = {'STAID': 'string'})
# val_in ID
df_testin_ID = df_train_ID
# val_nit ID
df_valnit_ID = pd.read_csv(f'{dir_expl}/ID_valnit.csv',
    dtype = {'STAID': 'string'})
# %%

X_train = df_train_expl.drop(['STAID', 'year'], axis = 1)
y_train = df_train_anWY['Ann_WY_ft']


# Custom loss
def gradient_nse(y_true, y_pred, groupby = df_train_anWY['site_no']):
    # compute gradient of NSE

    df_in = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'ID_in': groupby
    })

    # print(df_in)

    return np.mean(df_in.groupby('ID_in').apply(
        lambda df: 2*(df['y_true'] - df['y_pred'])/(df['y_true'] - np.mean(df['y_true']))**2
    ))

def hessian_nse(y_true, y_pred, groupby = df_train_anWY['site_no']):
    # compute gradient of NSE

    df_in = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'ID_in': groupby
    })

    # print(df_in)

    return np.mean(df_in.groupby('ID_in').apply(
        lambda df: 2/(df['y_true'] - np.mean(df['y_true']))**2
    ))

def custom_nse(y_true, y_pred, groupby = df_train_anWY['site_no']):
    # mean NSE as objective function
    grad = float(gradient_nse(y_true, y_pred))
    hess = float(hessian_nse(y_true, y_pred))

    print(type(grad), type(hess))

    return grad, hess


# define xgboost model
xgb_reg = xgb.XGBRegressor(
    objective = custom_nse, # 'reg:squarederror',
    tree_method = 'hist', # 'gpu_hist',
    verbosity = 1, # 0 = silent, 1 = warning (default), 2 = info, 3 = debug
    sampling_method = 'uniform', # 'gradient_based', # default is 'uniform'
    # nthread = 4 defaults to maximum number available, only use for less threads
    n_estimators = 250,
    colsample_bytree = 0.8,
    max_depth = 4,
    gamma = 0.1,
    reg_lambda = 0.1,
    learning_rate = 0.01
)

xgb_reg.fit(
    X_train,
    y_train
    )

y_pred = xgb_reg.predict(X_train)

data_in = pd.DataFrame({
    'pred': y_pred,
    'obs': y_train,
    'region': pd.merge(df_train_ID, df_train_expl, on = 'STAID')['AggEcoregion']
})

p = (
    p9.ggplot(data_in, p9.aes(x = 'obs', y = 'pred', color = 'region')) +
    p9.geom_point() +
    p9.geom_abline()
)

p



# %%
