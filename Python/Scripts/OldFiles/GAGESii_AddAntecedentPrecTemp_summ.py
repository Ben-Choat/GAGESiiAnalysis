'''
BChoat 10/14/2022

Script to add antecedent precipitation and temperature to 
weather explanatory variables for trianing,
valnit, and testing data.

Because we will use the first year of weather data as antecedent
predictors, we will remove the first year of data from water yield
data as well.

First for daily data, then monthly, then annual
'''


# %%
# Import libraries
##############

import pandas as pd # data manipulate
# import shutil # for copying new files to HPC working directory
import numpy as np

# %% define variables, directories, etc.
##############

# water yield directory
dir_WY = 'D:/DataWorking/USGS_discharge/train_val_test'

# DAYMET directory
dir_DMT = 'D:/DataWorking/Daymet/train_val_test'

# directory to copy files to
# dir_HPC = 'D:/Projects/GAGESiiANNStuff/HPC_Files/GAGES_work/data_work'


# %% daily data
############
# load data
###########


# water yield
# training
df_trainWY = pd.read_pickle(
    f'{dir_WY}/daily/pickle/WY_daily_train.pkl'
)
# testin
df_testinWY = pd.read_pickle(
    f'{dir_WY}/daily/pickle/WY_daily_testin.pkl'
)
# valnit
df_valnitWY = pd.read_pickle(
    f'{dir_WY}/daily/pickle/WY_daily_valnit.pkl'
)


# DAYMET
# training
df_trainDMT = pd.read_pickle(
    f'{dir_DMT}/daily/DAYMET_daily_train.pkl'
)
# testin
df_testinDMT = pd.read_pickle(
    f'{dir_DMT}/daily/DAYMET_daily_testin.pkl'
)
# valnit
df_valnitDMT = pd.read_pickle(
    f'{dir_DMT}/daily/DAYMET_daily_valnit.pkl'
)


# combine training and testin data into single datasets for calculations
# of DAYMET data
df_tDMT = pd.concat([df_trainDMT, df_testinDMT])
df_tDMT.reset_index(drop = True, inplace = True)

##########
# calculate previous 1, 7, 28, and 365 days of precip and temp

# 1-day
# training
d1_trainprecip = df_tDMT['prcp'].rolling(
    window = 1,
    closed = 'left'
    ).sum()

d1_traintmax = df_tDMT['tmax'].rolling(
    window = 1,
    closed = 'left'
    ).mean()

d1_traintmin = df_tDMT['tmin'].rolling(
    window = 1,
    closed = 'left'
    ).mean()

# valnit
d1_valnitprecip = df_valnitDMT['prcp'].rolling(
    window = 1,
    closed = 'left',
    ).sum()

d1_valnittmax = df_valnitDMT['tmax'].rolling(
    window = 1,
    closed = 'left'
    ).mean()

d1_valnittmin = df_valnitDMT['tmin'].rolling(
    window = 1,
    closed = 'left'
    ).mean() 


# 7-day
# training
d7_trainprecip = df_tDMT['prcp'].rolling(
    window = 7,
    closed = 'left'
    ).sum()

d7_traintmax = df_tDMT['tmax'].rolling(
    window = 7,
    closed = 'left'
    ).mean()

d7_traintmin = df_tDMT['tmin'].rolling(
    window = 7,
    closed = 'left'
    ).mean()

# valnit
d7_valnitprecip = df_valnitDMT['prcp'].rolling(
    window = 7,
    closed = 'left',
    ).sum()

d7_valnittmax = df_valnitDMT['tmax'].rolling(
    window = 7,
    closed = 'left'
    ).mean()

d7_valnittmin = df_valnitDMT['tmin'].rolling(
    window = 7,
    closed = 'left'
    ).mean() 


# 28-day
# training
d28_trainprecip = df_tDMT['prcp'].rolling(
    window = 28,
    closed = 'left'
    ).sum()

d28_traintmax = df_tDMT['tmax'].rolling(
    window = 28,
    closed = 'left'
    ).mean()

d28_traintmin = df_tDMT['tmin'].rolling(
    window = 28,
    closed = 'left'
    ).mean()

# valnit
d28_valnitprecip = df_valnitDMT['prcp'].rolling(
    window = 28,
    closed = 'left',
    ).sum()

d28_valnittmax = df_valnitDMT['tmax'].rolling(
    window = 28,
    closed = 'left'
    ).mean()

d28_valnittmin = df_valnitDMT['tmin'].rolling(
    window = 28,
    closed = 'left'
    ).mean() 


# 365-day
# training
d365_trainprecip = df_tDMT['prcp'].rolling(
    window = 365,
    closed = 'left'
    ).sum()

d365_traintmax = df_tDMT['tmax'].rolling(
    window = 365,
    closed = 'left'
    ).mean()

d365_traintmin = df_tDMT['tmin'].rolling(
    window = 365,
    closed = 'left'
    ).mean()

# valnit
d365_valnitprecip = df_valnitDMT['prcp'].rolling(
    window = 365,
    closed = 'left',
    ).sum()

d365_valnittmax = df_valnitDMT['tmax'].rolling(
    window = 365,
    closed = 'left'
    ).mean()

d365_valnittmin = df_valnitDMT['tmin'].rolling(
    window = 365,
    closed = 'left'
    ).mean() 


# assign new variables to DAYMET dfs
# train
df_tDMT = df_tDMT.assign(**{
    'd1_prcp': d1_trainprecip,
    'd1_tmax': d1_traintmax,
    'd1_tmin': d1_traintmin,
    'd7_prcp': d7_trainprecip,
    'd7_tmax': d7_traintmax,
    'd7_tmin': d7_traintmin,
    'd28_prcp': d28_trainprecip,
    'd28_tmax': d28_traintmax,
    'd28_tmin': d28_traintmin,
    'd365_prcp': d365_trainprecip,
    'd365_tmax': d365_traintmax,
    'd365_tmin': d365_traintmin,
})

del(
    d1_trainprecip,
    d1_traintmax,
    d1_traintmin,
    d7_trainprecip,
    d7_traintmax,
    d7_traintmin,
    d28_trainprecip,
    d28_traintmax,
    d28_traintmin,
    d365_trainprecip,
    d365_traintmax,
    d365_traintmin
)
# valnit
df_valnitDMT = df_valnitDMT.assign(**{
    'd1_prcp': d1_valnitprecip,
    'd1_tmax': d1_valnittmax,
    'd1_tmin': d1_valnittmin,
    'd7_prcp': d7_valnitprecip,
    'd7_tmax': d7_valnittmax,
    'd7_tmin': d7_valnittmin,
    'd28_prcp': d28_valnitprecip,
    'd28_tmax': d28_valnittmax,
    'd28_tmin': d28_valnittmin,
    'd365_prcp': d365_valnitprecip,
    'd365_tmax': d365_valnittmax,
    'd365_tmin': d365_valnittmin,
})

del(
    d1_valnitprecip,
    d1_valnittmax,
    d1_valnittmin,
    d7_valnitprecip,
    d7_valnittmax,
    d7_valnittmin,
    d28_valnitprecip,
    d28_valnittmax,
    d28_valnittmin,
    d365_valnitprecip,
    d365_valnittmax,
    d365_valnittmin
)


#####
# repartition data to appropriate datasets
# get rid of year 1998
#####

# DMT
# training
df_trainDMT = df_tDMT[
    df_tDMT['year'].isin(np.arange(1999, 2008, 1))
    ]
df_trainDMT.to_pickle(
    f'{dir_DMT}/W_AntecedentForcings/DAYMET_daily_train.pkl'
    )
# testin
df_testinDMT = df_tDMT[
    df_tDMT['year'].isin(np.arange(2008, 2013, 1))
    ]
df_testinDMT.to_pickle(
    f'{dir_DMT}/W_AntecedentForcings/DAYMET_daily_testin.pkl'
    )
# valnit
df_valnitDMT = df_valnitDMT[
    df_valnitDMT['year'] != 1998
    ]
df_valnitDMT.to_pickle(
    f'{dir_DMT}/W_AntecedentForcings/DAYMET_daily_valnit.pkl'
    )


# water yield
# training
df_trainWY = df_trainWY[
    df_trainWY['yr'].isin(np.arange(1999, 2008, 1))
    ]
df_trainWY.to_pickle(
    f'{dir_WY}/W_AntecedentForcings/WY_daily_train.pkl'
)
# testin
df_testinWY = df_testinWY[
    df_testinWY['yr'].isin(np.arange(2008, 2013, 1))
    ]
df_testinWY.to_pickle(
    f'{dir_WY}/W_AntecedentForcings/WY_daily_testin.pkl'
)
# valnit
df_valnitWY = df_valnitWY[
    df_valnitWY['yr'] != 1998
    ]
df_valnitWY.to_pickle(
    f'{dir_WY}/W_AntecedentForcings/WY_daily_valnit.pkl'
)

del(df_tDMT,
    df_testinDMT,
    df_testinWY,
    df_trainDMT,
    df_trainWY,
    df_valnitDMT,
    df_valnitWY)

# %% monthly data
############
# load data
###########


# water yield
# training
df_trainWY = pd.read_csv(
    f'{dir_WY}/monthly/WY_Mnth_train.csv',
    dtype = {'site_no': 'string'}
)
# testin
df_testinWY = pd.read_csv(
    f'{dir_WY}/monthly/WY_Mnth_testin.csv',
    dtype = {'site_no': 'string'}
)
# valnit
df_valnitWY = pd.read_csv(
    f'{dir_WY}/monthly/WY_Mnth_valnit.csv',
    dtype = {'site_no': 'string'}
)


# DAYMET
# training
df_trainDMT = pd.read_csv(
    f'{dir_DMT}/monthly/DAYMET_monthly_train.csv',
    dtype = {'site_no': 'string'}
)
# testin
df_testinDMT = pd.read_csv(
    f'{dir_DMT}/monthly/DAYMET_monthly_testin.csv',
    dtype = {'site_no': 'string'}
)
# valnit
df_valnitDMT = pd.read_csv(
    f'{dir_DMT}/monthly/DAYMET_monthly_valnit.csv',
    dtype = {'site_no': 'string'}
)


# combine training and testin data into single datasets for calculations
# of DAYMET data
df_tDMT = pd.concat([df_trainDMT, df_testinDMT])
df_tDMT.reset_index(drop = True, inplace = True)

##########
# calculate previous 1, 7, 28, and 365 days of precip and temp

# 1-month
# training
d1_trainprecip = df_tDMT['prcp'].rolling(
    window = 1,
    closed = 'left'
    ).sum()

d1_traintmax = df_tDMT['tmax'].rolling(
    window = 1,
    closed = 'left'
    ).mean()

d1_traintmin = df_tDMT['tmin'].rolling(
    window = 1,
    closed = 'left'
    ).mean()

# valnit
d1_valnitprecip = df_valnitDMT['prcp'].rolling(
    window = 1,
    closed = 'left',
    ).sum()

d1_valnittmax = df_valnitDMT['tmax'].rolling(
    window = 1,
    closed = 'left'
    ).mean()

d1_valnittmin = df_valnitDMT['tmin'].rolling(
    window = 1,
    closed = 'left'
    ).mean() 


# 12-month
# training
d12_trainprecip = df_tDMT['prcp'].rolling(
    window = 12,
    closed = 'left'
    ).sum()

d12_traintmax = df_tDMT['tmax'].rolling(
    window = 12,
    closed = 'left'
    ).mean()

d12_traintmin = df_tDMT['tmin'].rolling(
    window = 12,
    closed = 'left'
    ).mean()

# valnit
d12_valnitprecip = df_valnitDMT['prcp'].rolling(
    window = 12,
    closed = 'left',
    ).sum()

d12_valnittmax = df_valnitDMT['tmax'].rolling(
    window = 12,
    closed = 'left'
    ).mean()

d12_valnittmin = df_valnitDMT['tmin'].rolling(
    window = 12,
    closed = 'left'
    ).mean() 


# assign new variables to DAYMET dfs
# train
df_tDMT = df_tDMT.assign(**{
    'd1_prcp': d1_trainprecip,
    'd1_tmax': d1_traintmax,
    'd1_tmin': d1_traintmin,
    'd12_prcp': d12_trainprecip,
    'd12_tmax': d12_traintmax,
    'd12_tmin': d12_traintmin,

})

del(
    d1_trainprecip,
    d1_traintmax,
    d1_traintmin,
    d12_trainprecip,
    d12_traintmax,
    d12_traintmin,
)
# valnit
df_valnitDMT = df_valnitDMT.assign(**{
    'd1_prcp': d1_valnitprecip,
    'd1_tmax': d1_valnittmax,
    'd1_tmin': d1_valnittmin,
    'd12_prcp': d12_valnitprecip,
    'd12_tmax': d12_valnittmax,
    'd12_tmin': d12_valnittmin,
})

del(
    d1_valnitprecip,
    d1_valnittmax,
    d1_valnittmin,
    d12_valnitprecip,
    d12_valnittmax,
    d12_valnittmin,
)


#####
# repartition data to appropriate datasets
# get rid of year 1998
#####

# DMT
# training
df_trainDMT = df_tDMT[
    df_tDMT['year'].isin(np.arange(1999, 2008, 1))
    ]
df_trainDMT.to_csv(
    f'{dir_DMT}/W_AntecedentForcings/DAYMET_monthly_train.csv',
    index = False
    )
# testin
df_testinDMT = df_tDMT[
    df_tDMT['year'].isin(np.arange(2008, 2013, 1))
    ]
df_testinDMT.to_csv(
    f'{dir_DMT}/W_AntecedentForcings/DAYMET_monthly_testin.csv',
    index = False
    )
# valnit
df_valnitDMT = df_valnitDMT[
    df_valnitDMT['year'] != 1998
    ]
df_valnitDMT.to_csv(
    f'{dir_DMT}/W_AntecedentForcings/DAYMET_monthly_valnit.csv',
    index = False
    )


# water yield
# training
df_trainWY = df_trainWY[
    df_trainWY['yr'].isin(np.arange(1999, 2008, 1))
    ]
df_trainWY.to_csv(
    f'{dir_WY}/W_AntecedentForcings/WY_monthly_train.csv',
    index = False
)
# testin
df_testinWY = df_testinWY[
    df_testinWY['yr'].isin(np.arange(2008, 2013, 1))
    ]
df_testinWY.to_csv(
    f'{dir_WY}/W_AntecedentForcings/WY_monthly_testin.csv',
    index = False
)
# valnit
df_valnitWY = df_valnitWY[
    df_valnitWY['yr'] != 1998
    ]
df_valnitWY.to_csv(
    f'{dir_WY}/W_AntecedentForcings/WY_monthly_valnit.csv',
    index = False
)


del(df_tDMT,
    df_testinDMT,
    df_testinWY,
    df_trainDMT,
    df_trainWY,
    df_valnitDMT,
    df_valnitWY)



# %% monthly data
############
# load data
###########


# water yield
# training
df_trainWY = pd.read_csv(
    f'{dir_WY}/annual/WY_Ann_train.csv',
    dtype = {'site_no': 'string'}
)
# testin
df_testinWY = pd.read_csv(
    f'{dir_WY}/annual/WY_Ann_testin.csv',
    dtype = {'site_no': 'string'}
)
# valnit
df_valnitWY = pd.read_csv(
    f'{dir_WY}/annual/WY_Ann_valnit.csv',
    dtype = {'site_no': 'string'}
)


# DAYMET
# training
df_trainDMT = pd.read_csv(
    f'{dir_DMT}/annual/DAYMET_Annual_train.csv',
    dtype = {'site_no': 'string'}
)
# testin
df_testinDMT = pd.read_csv(
    f'{dir_DMT}/annual/DAYMET_Annual_testin.csv',
    dtype = {'site_no': 'string'}
)
# valnit
df_valnitDMT = pd.read_csv(
    f'{dir_DMT}/annual/DAYMET_Annual_valnit.csv',
    dtype = {'site_no': 'string'}
)


# combine training and testin data into single datasets for calculations
# of DAYMET data
df_tDMT = pd.concat([df_trainDMT, df_testinDMT])
df_tDMT.reset_index(drop = True, inplace = True)

##########
# calculate previous 1, 7, 28, and 365 days of precip and temp

# 1-year
# training
d1_trainprecip = df_tDMT['prcp'].rolling(
    window = 1,
    closed = 'left'
    ).sum()

d1_traintmax = df_tDMT['tmax'].rolling(
    window = 1,
    closed = 'left'
    ).mean()

d1_traintmin = df_tDMT['tmin'].rolling(
    window = 1,
    closed = 'left'
    ).mean()

# valnit
d1_valnitprecip = df_valnitDMT['prcp'].rolling(
    window = 1,
    closed = 'left',
    ).sum()

d1_valnittmax = df_valnitDMT['tmax'].rolling(
    window = 1,
    closed = 'left'
    ).mean()

d1_valnittmin = df_valnitDMT['tmin'].rolling(
    window = 1,
    closed = 'left'
    ).mean() 



# assign new variables to DAYMET dfs
# train
df_tDMT = df_tDMT.assign(**{
    'd1_prcp': d1_trainprecip,
    'd1_tmax': d1_traintmax,
    'd1_tmin': d1_traintmin
})

del(
    d1_trainprecip,
    d1_traintmax,
    d1_traintmin
)

# valnit
df_valnitDMT = df_valnitDMT.assign(**{
    'd1_prcp': d1_valnitprecip,
    'd1_tmax': d1_valnittmax,
    'd1_tmin': d1_valnittmin
})

del(
    d1_valnitprecip,
    d1_valnittmax,
    d1_valnittmin
)


#####
# repartition data to appropriate datasets
# get rid of year 1998
#####

# DMT
# training
df_trainDMT = df_tDMT[
    df_tDMT['year'].isin(np.arange(1999, 2008, 1))
    ]
df_trainDMT.to_csv(
    f'{dir_DMT}/W_AntecedentForcings/DAYMET_Annual_train.csv',
    index = False
    )
# testin
df_testinDMT = df_tDMT[
    df_tDMT['year'].isin(np.arange(2008, 2013, 1))
    ]
df_testinDMT.to_csv(
    f'{dir_DMT}/W_AntecedentForcings/DAYMET_Annual_testin.csv',
    index = False
    )
# valnit
df_valnitDMT = df_valnitDMT[
    df_valnitDMT['year'] != 1998
    ]
df_valnitDMT.to_csv(
    f'{dir_DMT}/W_AntecedentForcings/DAYMET_Annual_valnit.csv',
    index = False
    )


# water yield
# training
df_trainWY = df_trainWY[
    df_trainWY['yr'].isin(np.arange(1999, 2008, 1))
    ]
df_trainWY.to_csv(
    f'{dir_WY}/W_AntecedentForcings/WY_Ann_train.csv',
    index = False
)
# testin
df_testinWY = df_testinWY[
    df_testinWY['yr'].isin(np.arange(2008, 2013, 1))
    ]
df_testinWY.to_csv(
    f'{dir_WY}/W_AntecedentForcings/WY_Ann_testin.csv',
    index = False
)
# valnit
df_valnitWY = df_valnitWY[
    df_valnitWY['yr'] != 1998
    ]
df_valnitWY.to_csv(
    f'{dir_WY}/W_AntecedentForcings/WY_Ann_valnit.csv',
    index = False
)


del(df_tDMT,
    df_testinDMT,
    df_testinWY,
    df_trainDMT,
    df_trainWY,
    df_valnitDMT,
    df_valnitWY)


# %%
