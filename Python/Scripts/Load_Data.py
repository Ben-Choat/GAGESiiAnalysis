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
                'year',
                'month',
                'day',
                'date',
                'STAID']
                # 'GEOL_REEDBUSH_DOM_gneiss', 
    

    # # directory to write csv holding removed columns (due to high VIF)
    # # dir_VIF = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Results/VIF_Removed'
    # dir_VIF = f'{dir_work}/data_out/mean_annual/VIF_Removed'

    # mean annual time-scale
    if(time_scale == 'mean_annual'):
 
        # GAGESii explanatory vars
        df_expl = pd.read_csv(
            f'{dir_expl}/Expl_{train_val}.csv',
            dtype = {'STAID': 'string'}
        )

        # water yield variables
        # Annual Water yield
        df_anWY = pd.read_csv(
            f'{dir_WY}/annual/WY_Ann_{train_val}.csv',
            dtype = {"site_no":"string"}
        )

        # mean annual water yield
        # training
        df_mnanWY = df_anWY.groupby(
            'site_no', as_index = False
        ).mean().drop(columns = ["yr"])
        
        # mean GAGESii explanatory vars
        df_mnexpl = df_expl.groupby(
            'STAID', as_index = False
        ).mean().drop(columns = ['year'])

        # ID vars (e.g., ecoregion)
        df_ID = pd.read_csv(f'{dir_expl}/ID_{train_val}.csv',
            dtype = {'STAID': 'string'})

        # reassign mean values to df's
        # explanatory vars
        df_expl = df_mnexpl
        # water yield
        df_WY = df_mnanWY


    #######################


    # annual time-scale
    if(time_scale == 'annual'):
 
        # GAGESii explanatory vars
        df_expl = pd.read_csv(
            f'{dir_expl}/Expl_{train_val}.csv',
            dtype = {'STAID': 'string'}
        )

        # DAYMET
        df_DMT = pd.read_csv(
            f'{dir_DMT}/annual/DAYMET_Annual_{train_val}.csv',
            dtype = {"site_no":"string"}
        )

        # Add DAYMET to explanatory vars
        df_expl = pd.merge(
            df_expl, df_DMT, left_on = ['STAID', 'year'], right_on = ['site_no', 'year']
            ).drop('site_no', axis = 1)

        # water yield variables
        # Annual Water yield
        df_WY = pd.read_csv(
            f'{dir_WY}/annual/WY_Ann_{train_val}.csv',
            dtype = {"site_no":"string"}
        )

        # ID vars (e.g., ecoregion)
        df_ID = pd.read_csv(f'{dir_expl}/ID_{train_val}.csv',
            dtype = {'STAID': 'string'})
      


    #######################


    # monthly time-scale
    if(time_scale == 'monthly'):
 
        # GAGESii explanatory vars
        df_expl = pd.read_csv(
            f'{dir_expl}/Expl_{train_val}.csv',
            dtype = {'STAID': 'string'}
        )

        # DAYMET
        df_DMT = pd.read_csv(
            f'{dir_DMT}/{time_scale}/DAYMET_{time_scale}_{train_val}.csv',
            dtype = {"site_no":"string"}
        )

        # Add DAYMET to explanatory vars
        df_expl = pd.merge(
            df_expl, df_DMT, left_on = ['STAID', 'year'], right_on = ['site_no', 'year']
            ).drop('site_no', axis = 1)

        # water yield variables
        # monthly Water yield
        df_WY = pd.read_csv(
            f'{dir_WY}/{time_scale}/WY_Mnth_{train_val}.csv',
            dtype = {"site_no":"string"}
        )

        # ID vars (e.g., ecoregion)
        df_ID = pd.read_csv(f'{dir_expl}/ID_{train_val}.csv',
            dtype = {'STAID': 'string'})




    #######################


    # monthly time-scale
    if(time_scale == 'daily'):
 
        # GAGESii explanatory vars
        df_expl = pd.read_csv(
            f'{dir_expl}/Expl_{train_val}.csv',
            dtype = {'STAID': 'string'}
        )

        # DAYMET
        df_DMT = pd.read_pickle(
            f'{dir_DMT}/{time_scale}/DAYMET_{time_scale}_{train_val}.pkl'
        )

        # Add DAYMET to explanatory vars
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
            scaler = stdsc.fit(df_expl)
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

        cid_in = df_ID[df_ID[clust_meth] == region]
        cid_in.drop('DRAIN_SQKM', axis = 1, inplace = True)

        # Water yield
        df_WY = pd.merge(
            df_WY, cid_in, left_on = 'site_no', right_on = 'STAID'
            )

        # explanatory variables
        df_expl = pd.merge(df_expl, cid_in, left_on = 'STAID', right_on = 'STAID').drop(
            columns = ['Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site',
                        'LAT_GAGE', 'LNG_GAGE', 'HUC02']
            )

        # ID dataframes
        df_ID = pd.merge(
            df_ID, cid_in, on = ['STAID', 'Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site']
            )[clust_meth] # ['ECO3_Site']

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
            scaler = stdsc.fit(df_expl)
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










