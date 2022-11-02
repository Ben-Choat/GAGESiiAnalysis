'''
BChoat 2022/10/03

This script accepts a working directory and time-scale (e.g., mean_annual, annual, 
monthly, daily) and loads the explanatory and response variables for the
GAGESii work completed for my dissertation.

Working directory is expectd to be the directory where data_work, data_out, and 
scripts folders are located
'''

#####
# Load libraries
#####

import pandas as pd
from sklearn.preprocessing import StandardScaler # standardizing data


##### 
# define function that accepts working directory and time-scale as inputs
#####
def load_data_fun(dir_work = 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work', 
    time_scale = 'mean_annual',
    train_val = 'train',
    clust_meth = 'None',
    region = 'All',
    standardize = True):

    # print input variables to check if as expected
    print(f' working directory: {dir_work} \n time scale: {time_scale}')

    # define sub-directories
    # water yield directory
    # dir_WY = 'D:/DataWorking/USGS_discharge/train_val_test'
    dir_WY = f'{dir_work}/data_work/USGS_discharge'

    # explantory var (and other data) directory
    # dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out/AllVars_Partitioned'
    dir_expl = f'{dir_work}/data_work/GAGESiiVariables'

    # DAYMET directory
    # dir_DMT = 'D:/DataWorking/Daymet/train_val_test'
    dir_DMT = f'{dir_work}/data_work/Daymet'

    # Define features not to transform
    not_tr_in = ['GEOL_REEDBUSH_DOM_granitic', 
                'GEOL_REEDBUSH_DOM_quarternary', 
                'GEOL_REEDBUSH_DOM_sedimentary', 
                'GEOL_REEDBUSH_DOM_ultramafic', 
                'GEOL_REEDBUSH_DOM_volcanic',
                'GEOL_REEDBUSH_DOM_gneiss',
                'year',
                'month',
                'day',
                'date',
                'STAID']
    

    # # directory to write csv holding removed columns (due to high VIF)
    # # dir_VIF = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Results/VIF_Removed'
    # dir_VIF = f'{dir_work}/data_out/mean_annual/VIF_Removed'

    # training expl vars for standardizing data
    df_trainexpl = pd.read_csv(
        f'{dir_expl}/Expl_train.csv',
        dtype = {'STAID': 'string'}
    )
    
    # GAGESii explanatory vars
    df_expl = pd.read_csv(
        f'{dir_expl}/Expl_{train_val}.csv',
        dtype = {'STAID': 'string'}
    )

    # ID vars (e.g., ecoregion)
    df_ID = pd.read_csv(f'{dir_expl}/ID_{train_val}.csv',
        dtype = {'STAID': 'string'})

    # ID vars (e.g., ecoregion)
    df_trainID = pd.read_csv(f'{dir_expl}/ID_train.csv',
        dtype = {'STAID': 'string'})

    # mean annual time-scale
    if(time_scale == 'mean_annual'):
        
        # water yield variables
        # Annual Water yield
        df_anWY = pd.read_csv(
            f'{dir_WY}/annual/WY_Ann_{train_val}.csv',
            dtype = {"site_no":"string"}
        )

        # mean annual water yield
        # training
        df_WY = df_anWY.groupby(
            'site_no', as_index = False
        ).mean().drop(columns = ["yr"])
        
        # drop in training data used for fitting standardscaler
        df_trainexpl = df_trainexpl.groupby(
            'STAID', as_index = False
        ).mean()

        df_trainexpl.drop('year', axis = 1, inplace = True)
        

        # mean GAGESii explanatory vars
        df_expl = df_expl.groupby(
            'STAID', as_index = False
        ).mean()
        
        df_expl = df_expl.drop(columns = ['year'])

        
        # DAYMET
        # train for training standscaler
        df_trainDMT = pd.read_csv(
            f'{dir_DMT}/annual/DAYMET_Annual_train.csv',
            dtype = {"site_no":"string"}
        )
        df_trainDMT = df_trainDMT.groupby(
            'site_no', as_index = False
        ).mean().drop('year', axis = 1)

        df_DMT = pd.read_csv(
            f'{dir_DMT}/annual/DAYMET_Annual_{train_val}.csv',
            dtype = {"site_no":"string"}
        )
        df_DMT = df_DMT.groupby(
            'site_no', as_index = False
        ).mean().drop('year', axis = 1)


        # Add DAYMET to explanatory vars
        # train
        df_trainexpl = pd.merge(
            df_trainexpl, df_trainDMT, left_on = ['STAID'], right_on = ['site_no']
            ).drop('site_no', axis = 1)

        df_expl = pd.merge(
            df_expl, df_DMT, left_on = ['STAID'], right_on = ['site_no']
            ).drop('site_no', axis = 1)
        
        
        # reassign mean values to df's
        # explanatory vars
        # df_expl = df_mnexpl
        # water yield
        # df_WY = df_mnanWY


    #######################


    # annual time-scale
    if(time_scale == 'annual'):

        # DAYMET
        # train for training standscaler
        df_trainDMT = pd.read_csv(
            f'{dir_DMT}/annual/DAYMET_Annual_train.csv',
            dtype = {"site_no":"string"}
        )

        df_DMT = pd.read_csv(
            f'{dir_DMT}/annual/DAYMET_Annual_{train_val}.csv',
            dtype = {"site_no":"string"}
        )

        # Add DAYMET to explanatory vars
        # train
        df_trainexpl = pd.merge(
            df_trainexpl, df_trainDMT, left_on = ['STAID', 'year'], right_on = ['site_no', 'year']
            ).drop('site_no', axis = 1)

        df_expl = pd.merge(
            df_expl, df_DMT, left_on = ['STAID', 'year'], right_on = ['site_no', 'year']
            ).drop('site_no', axis = 1)

        # water yield variables
        # Annual Water yield
        df_WY = pd.read_csv(
            f'{dir_WY}/annual/WY_Ann_{train_val}.csv',
            dtype = {"site_no":"string"}
        )

      


    #######################


    # monthly time-scale
    if(time_scale == 'monthly'):
 
        # DAYMET
        # train for training standard scaler
        df_trainDMT = pd.read_csv(
            f'{dir_DMT}/{time_scale}/DAYMET_{time_scale}_train.csv',
            dtype = {"site_no":"string"}
        )
        df_DMT = pd.read_csv(
            f'{dir_DMT}/{time_scale}/DAYMET_{time_scale}_{train_val}.csv',
            dtype = {"site_no":"string"}
        )

        # Add DAYMET to explanatory vars
        # train
        df_trainexpl = pd.merge(
            df_trainexpl, df_trainDMT, left_on = ['STAID', 'year'], right_on = ['site_no', 'year']
            ).drop(['site_no'], axis = 1)

        df_expl = pd.merge(
            df_expl, df_DMT, left_on = ['STAID', 'year'], right_on = ['site_no', 'year']
            ).drop('site_no', axis = 1)

        # water yield variables
        # monthly Water yield
        df_WY = pd.read_csv(
            f'{dir_WY}/{time_scale}/WY_Mnth_{train_val}.csv',
            dtype = {"site_no":"string"}
        )






    #######################


    # daily time-scale
    if(time_scale == 'daily'):
 

        # DAYMET
        # training for fitting standardscaler
        df_trainDMT = pd.read_pickle(
            f'{dir_DMT}/{time_scale}/DAYMET_{time_scale}_train.pkl'
        )

        df_DMT = pd.read_pickle(
            f'{dir_DMT}/{time_scale}/DAYMET_{time_scale}_{train_val}.pkl'
        )

        # Add DAYMET to explanatory vars
        # training
        df_trainexpl = pd.merge(
            df_trainexpl, df_trainDMT, left_on = ['STAID', 'year'], right_on = ['site_no', 'year']
            ).drop('site_no', axis = 1)

        df_expl = pd.merge(
            df_expl, df_DMT, left_on = ['STAID', 'year'], right_on = ['site_no', 'year']
            ).drop('site_no', axis = 1)

        # water yield variables
        # monthly Water yield
        df_WY = pd.read_pickle(
            f'{dir_WY}/{time_scale}/WY_daily_{train_val}.pkl'
        )

        # ID vars (e.g., ecoregion)
        df_ID = pd.read_csv(f'{dir_expl}/ID_{train_val}.csv',
            dtype = {'STAID': 'string'})




    #############

    # if using all gages (None, All), then return dataframes as are
    if clust_meth == 'None':

        # transform data
        if standardize:

            # subset not_tr_in to those columns remaining in dataframe
            # in case any were removed due to high VIFs
            not_tr_in = df_expl.columns.intersection(not_tr_in)
            # hold untransformed not_tr_in columns
            hold_untr = df_expl[not_tr_in]
            # define standardizer
            stdsc = StandardScaler()
            # fit scaler
            scaler = stdsc.fit(df_trainexpl)
            # transform data
            temp_expl = pd.DataFrame(
                scaler.transform(df_expl)
            )
            # reassign column names
            temp_expl.columns = df_expl.columns
            # reassign untransformed data
            temp_expl[not_tr_in] = hold_untr
            # redefine df_expl and delete temp_expl and hold untr
            df_expl = temp_expl
            del(temp_expl, hold_untr)

        # return output dfs
        return(df_expl, df_WY, df_ID)

    # otherwise, subset the data to region/cluster of interest
    else:
        # working data
        cid_in = df_ID[df_ID[clust_meth] == region]

        cid_in.drop('DRAIN_SQKM', axis = 1, inplace = True)
        # training data
        cidtrain_in = df_trainID[df_trainID[clust_meth] == region]

        cidtrain_in.drop('DRAIN_SQKM', axis = 1, inplace = True)

        # Water yield
        df_WY = pd.merge(
            df_WY, cid_in, left_on = 'site_no', right_on = 'STAID'
            )

        # explanatory variables
        # work
        df_expl = pd.merge(df_expl, cid_in, left_on = 'STAID', right_on = 'STAID').drop(
            columns = cid_in.columns[1:len(cid_in.columns)]
            # ['Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site',
            #             'LAT_GAGE', 'LNG_GAGE', 'HUC02', 'CAMELS']
            )
        # train
        df_trainexpl = pd.merge(df_trainexpl, cidtrain_in, left_on = 'STAID', right_on = 'STAID').drop(
            columns = cidtrain_in.columns[1:len(cidtrain_in.columns)]
            # ['Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site',
            #             'LAT_GAGE', 'LNG_GAGE', 'HUC02', 'CAMELS']
            )

        # ID dataframes
        # working
        df_ID = pd.merge(
            df_ID, cid_in, on = ['STAID', 'Class', 'AggEcoregion', 'ECO3_Site', 
                                'USDA_LRR_Site', 'CAMELS'])[clust_meth] # ['ECO3_Site']
        # training
        df_trainID = pd.merge(
            df_trainID, cidtrain_in, on = ['STAID', 'Class', 'AggEcoregion',
                                     'ECO3_Site', 'USDA_LRR_Site', 'CAMELS'])[clust_meth] # ['ECO3_Site']

        # transform data
        if standardize:

            # subset not_tr_in to those columns remaining in dataframe
            # in case any were removed due to high VIFs
            not_tr_in = df_expl.columns.intersection(not_tr_in)
            # hold untransformed not_tr_in columns
            hold_untr = df_expl[not_tr_in]
            # define standardizer
            stdsc = StandardScaler()
            # fit scaler
            scaler = stdsc.fit(df_trainexpl)
            
            # transform data
            temp_expl = pd.DataFrame(
                scaler.transform(df_expl)
            )
            # reassign column names
            temp_expl.columns = df_expl.columns
            # reassign untransformed data
            temp_expl[not_tr_in] = hold_untr

            # redefine df_expl and delete temp_expl and hold untr
            df_expl = temp_expl
            del(temp_expl, hold_untr)
        
        # return output dfs
        return(df_expl, df_WY, df_ID)










