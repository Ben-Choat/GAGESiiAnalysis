# %%
# # BChoat 20220410

# Script to download DAYMET version 4 data using pydaymet api via HyRiver library

# %% Import libraries
import matplotlib.pyplot as plt
import pydaymet as daymet
from pynhd import NLDI
import pandas as pd
import geopandas as gpd
import datetime as dt
import numpy as np
import glob # for unix like interface to directories
import re # for replacing parts of strings
import gc # for controlling garbage collection
from os.path import exists # to check if file already exists
#import netcdf4


# %% Read in data

# Previously, computer shut off during download, so it didn't complete 
# 4/17/2022
# Here I read in the data that did get downloaded, so I can pick up where I left off
# if exists('D:/Projects/GAGESii_ANNstuff/Data_Out/Daymet_Monthly.csv'):
#     Daymet_Monthly = pd.read_csv(
#         # 'D:/Projects/GAGESii_ANNstuff/Data_Out/Daymet_Monthly.csv', 
#         'D:/DataWorking/Daymet/Daymet_Monthly.csv',
#         # 'C:/Users/bench/Desktop/Daymet_Monthly.csv',
#         dtype = {'site_no': 'string'})
#         # dtype = {'gages_retreived': 'string'}) # I changed column names after initial
#         # download.

# water yield directory
dir_WY = 'D:/DataWorking/USGS_discharge/train_val_test'
# explantory var (and other data) directory
# dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out'

# Monthly Water yield

# # training data
# df_train_sites = pd.read_csv(
#     f'{dir_WY}/yrs_98_12/monthly_WY/Mnthly_WY_train.csv',
#     dtype = {"site_no":"string"}
#     )['site_no']

# # test basins used in training
# df_test_in_sites = pd.read_csv(
#     f'{dir_WY}/yrs_98_12/monthly_WY/Mnthly_WY_test_in.csv',
#     dtype = {"site_no":"string"}
#     )['site_no']

# # test basins not used in training
# df_test_nit_sites = pd.read_csv(
#     f'{dir_WY}/yrs_98_12/monthly_WY/Mnthly_WY_test_nit.csv',
#     dtype = {"site_no":"string"}
#     )['site_no']

# # validate basins used in training
# df_val_in_sites = pd.read_csv(
#     f'{dir_WY}/yrs_98_12/monthly_WY/Mnthly_WY_val_in.csv',
#     dtype = {"site_no":"string"}
#     )['site_no']

# # validation basins not used in training
# df_val_nit_sites = pd.read_csv(
#     f'{dir_WY}/yrs_98_12/monthly_WY/Mnthly_WY_val_nit.csv',
#     dtype = {"site_no":"string"}
#     )['site_no']


# # all sites
# df_all_sites = pd.concat([
#     df_train_sites,
#     df_test_in_sites,
#     df_test_nit_sites,
#     df_val_in_sites,
#     df_val_nit_sites
# ]).drop_duplicates()

# del(df_train_sites,
#     df_test_in_sites,
#     df_test_nit_sites,
#     df_val_in_sites,
#     df_val_nit_sites)

# # df_all_sites.to_csv('C:/Users/bchoat/desktop/DAYMET/all_sites.csv')
# df_all_sites.to_csv('D:/Projects/GAGESii_ANNstuff/Data_Out/all_sites.csv')

# df_all_sites = pd.read_csv('F://DAYMET/all_sites.csv',
#     dtype = {'site_no': 'string'})
df_all_sites = pd.read_csv('D:/Projects/GAGESii_ANNstuff/Data_Out/all_sites.csv',
    dtype = {'site_no': 'string'}).drop(columns = 'Unnamed: 0')
# %% get list of shape file names

list_shp = glob.glob(r"D:/DataWorking/GAGESii/boundaries-shapefiles-by-aggeco/*.shp")
# list_shp = glob.glob(r'C:/Users/bchoat/Desktop/DAYMET/boundaries-shapefiles-by-aggeco/*.shp')
# list_shp = glob.glob(r'F:/DAYMET/boundaries-shapefiles-by-aggeco/*.shp')

# %% Loop through shape files and use geometry of catchments being used in study
# to extract DAYMET data

# Define dates to extract between, including the specified days
dates_get = ("1998-01-01", "2012-12-31")
# dates_get = ("1998-01-01", "1998-01-02")

# Write dataframe to csv to later append restults to
# if not exists('D:/Projects/GAGESii_ANNstuff/Data_Out/Daymet_Monthly.csv'):
#     pd.DataFrame({
#         'year': [],
#         'month': [],
#         'site_no': [], # 'gages_retreived': [], changed name
#         'tmin_retreived': [],
#         'tmax_retreived': [],
#         'prcp_retreived': [],
#         'vp_retreived': [],
#         'swe_retreived': []
#         # 'srad_retreived': [],
#         # 'dayl_retreived': [],
#     }).to_csv(
#         'D:/DataWorking/Daymet/Daymet_Monthly.csv',
#         #'D:/Projects/GAGESii_ANNstuff/Data_Out/Daymet_Monthly.csv',
#         # 'C:/Users/bench/Desktop/Daymet_Monthly.csv',
#         index = False
#     )



# Daymet_Monthly = pd.read_csv(
#         #'D:/Projects/GAGESii_ANNstuff/Data_Out/Daymet_Monthly.csv', 
#         'D:/DataWorking/Daymet/Daymet_Monthly.csv',
#         # 'C:/Users/bench/Desktop/Daymet_Monthly.csv',
#         dtype = {'site_no': 'string'})    

# Define variables to retrieve. Note that I'm retreiving all 7 available vars

# tmin: min daily temp [C]
# tmax: max daily temp [C]
# prcp: Daily total precipitation [mm/day],
# vp: water vapor pressure deficit [Pa] - Daily average partial pressur of water vapor
# swe: snow water equivalent [kg/m2]
# NOTE: the next two are only available for daily download
# srad: shortwave radiation [W/m2] - Average over daylight period - 
#   daily total radiation (MJ/m2/day) can be calculated as srad * dayl / 1,000,000
# dayl: day length [s/day] (s = seconds)
# NOTE: consider inclusion of latitute since srad and dayl are not available for 
##  monthly and annual time scales.
vars_get = ("dayl", "prcp", "srad","swe", "tmax", "tmin", "vp")

# disable garbage collector to keep loop from slowing as size of data grows
# gc.disable()
for shape in list_shp:

    # if shape is the Alaska, Hawaii, Peurto Rice file, then skip it
    if 'AKHIPR' in shape:
        continue

    # if not exists('D:/Projects/GAGESii_ANNstuff/Data_Out/Daymet_Monthly.csv'):
    # if not exists(f'C:/Users/bchoat/Desktop/DAYMET/Daily/daily_{shape[55:len(shape) - 4]}.csv'):
    # if not exists(f'C:/Users/bchoat/Desktop/DAYMET/Daily/Daymet_daily_{shape[67:len(shape) - 4]}.csv'):
    # if not exists(f'F:/DAYMET/Daily/Daymet_daily_{shape[46:len(shape) - 4]}.csv'):
    if not exists(f'D:/DataWorking/Daymet/Daily/Daymet_daily_{shape[55:len(shape) - 4]}.csv'):
        pd.DataFrame({
            'site_no': [], # 'gages_retreived': 
            'date': [], # will hold yyyy-mm-dd dates
            'dayl': [],
            'prcp': [],
            'srad': [],
            'swe': [],
            'tmax': [],
            'tmin': [],                   
            'vp': []
        }).to_csv( # NOTE that shape[indices] will need to be edited based on 
                    # directory from which shape variable objects are read
            # f'C:/Users/bench/Desktop/daily_{shape[55:len(shape) - 4]}.csv',
            # f'F:/DAYMET/Daily/Daymet_daily_{shape[46:len(shape) - 4]}.csv',
            f'D:/DataWorking/Daymet/Daily/Daymet_daily_{shape[55:len(shape) - 4]}.csv',
            index = False
        )

    daily_work = pd.read_csv(
        f'D:/DataWorking/Daymet/Daily/Daymet_daily_{shape[55:len(shape) - 4]}.csv',
        dtype = {'site_no': 'string'}
    )

    # read in file and convert to coordinates consistant with default pydaymet crs
    print(f'Reading {shape}')
    shp_in = gpd.read_file(
        shape 
        ).to_crs("EPSG:4326")

    for gg_id in shp_in['GAGE_ID']:

        # if gg_id already in downloaded data, then continue to next gage
        # if Daymet_Monthly['site_no'].str.contains(gg_id).any():
        #     continue
        if daily_work['site_no'].str.contains(gg_id).any():
            continue
        
        # if station is in data being used for this study, then subset the shape file
        # to that catchment and use it to extract DAYMET data

        # if df_all_sites.str.contains(gg_id).any():
        if df_all_sites['site_no'].str.contains(gg_id).any():
            
            # Get index of which shp_in row is equal to current gage id (gg_id)
            shp_single_index = np.where(shp_in['GAGE_ID'] == gg_id)[0][0]

            shp_in_single = shp_in['geometry'][shp_single_index]

            # Plot
            # shp_in.plot(column = 'GAGE_ID')

            # Get coords, for extracting form pixel instead of entire watershed
            # cor_x = np.array(shp_in_single.exterior.coords.xy)[0,].mean().round(5)
            # cor_y = np.array(shp_in_single.exterior.coords.xy)[1,].mean().round(5)

            # retrieve monthly daymet data
            # set up so if fails once, will try again (added because sometimes the connection
            # either timed out or failed to be established while working on campus)
            
            try:
                daily = daymet.get_bygeom(shp_in_single, 
                    dates = dates_get,
                    time_scale = "daily",
                    variables = vars_get
                    )
            except:
                daily = daymet.get_bygeom(shp_in_single, 
                    dates = dates_get,
                    time_scale = "daily",
                    variables = vars_get
                    )

            # Calculate mean across basin
            daily_mean = daily.mean(dim = ['x', 'y'])

            # write to csv via pd.DataFrame
            pd.DataFrame({
                'site_no': np.repeat(gg_id, len(daily_mean.time)),
                'date': pd.to_datetime(daily_mean.time).strftime('%m/%d/%Y').tolist(),
                'dayl': daily_mean.dayl.values.tolist(),
                'prcp': daily_mean.prcp.values.tolist(),
                'srad': daily_mean.srad.values.tolist(),
                'swe': daily_mean.swe.values.tolist(),
                'tmax': daily_mean.tmax.values.tolist(),
                'tmin': daily_mean.tmin.values.tolist(),
                'vp': daily_mean.vp.values.tolist()
            }).to_csv(
                # f'F:/DAYMET/Daily/Daymet_daily_{shape[46:len(shape) - 4]}.csv',
                f'D:/DataWorking/Daymet/Daily/Daymet_daily_{shape[55:len(shape) - 4]}.csv',
                mode = 'a',
                index = False,
                header = False
            )       
    
            # # save daymet data stored in daily to raw_data and working_data directories
            # daily.to_netcdf(f'D:/DataWorking/Daymet/Daily/Daymet_Daily_{gg_id}.nc')
            # daily.to_netcdf(f'D:/DataRaw/Daymet/Daily/Daymet_Daily_{gg_id}.nc')
            
            print(gg_id)

# enable garbage collector
# gc.enable()
#



# %%
