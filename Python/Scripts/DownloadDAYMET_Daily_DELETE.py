# %%
# # BChoat 20220410

# Script to download DAYMET version 4 data using pydaymet api via HyRiver library

# %% Import libraries
import matplotlib.pyplot as plt
import pydaymet as daymet
# from pynhd import NLDI
import pandas as pd
import geopandas as gpd
import numpy as np
import glob # for unix like interface to directories
# import re # for replacing parts of strings
# import gc # for controlling garbage collection
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
#         # dtype={'gages_retreived': 'string'}) # I changed column names after initial
#         # download.

# water yield directory
dir_WY = 'D:/DataWorking/USGS_discharge/train_val_test'
# explantory var (and other data) directory
# dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out'

# Monthly Water yield

# training data
df_train_sites = pd.read_csv(
    f'{dir_WY}/yrs_98_12/monthly_WY/Mnthly_WY_train.csv',
    dtype={"site_no":"string"}
    )['site_no']

# test basins used in training
df_test_in_sites = pd.read_csv(
    f'{dir_WY}/yrs_98_12/monthly_WY/Mnthly_WY_test_in.csv',
    dtype={"site_no":"string"}
    )['site_no']

# test basins not used in training
df_test_nit_sites = pd.read_csv(
    f'{dir_WY}/yrs_98_12/monthly_WY/Mnthly_WY_test_nit.csv',
    dtype={"site_no":"string"}
    )['site_no']

# validate basins used in training
df_val_in_sites = pd.read_csv(
    f'{dir_WY}/yrs_98_12/monthly_WY/Mnthly_WY_val_in.csv',
    dtype={"site_no":"string"}
    )['site_no']

# validation basins not used in training
df_val_nit_sites = pd.read_csv(
    f'{dir_WY}/yrs_98_12/monthly_WY/Mnthly_WY_val_nit.csv',
    dtype={"site_no":"string"}
    )['site_no']


# all sites
df_all_sites = pd.concat([
    df_train_sites,
    df_test_in_sites,
    df_test_nit_sites,
    df_val_in_sites,
    df_val_nit_sites
]).drop_duplicates()

del(df_train_sites,
    df_test_in_sites,
    df_test_nit_sites,
    df_val_in_sites,
    df_val_nit_sites)


# %% get list of shape file names

list_shp = glob.glob(r"D:/DataWorking/GAGESii/boundaries-shapefiles-by-aggeco/*.shp")

# %% Loop through shape files and use geometry of catchments being used in study
# to extract DAYMET data

# Define dates to extract between, including the specified days
# dates_get = ("1998-01-01", "2012-12-31")
dates_get = ("1998-01-01", "1998-01-02")

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

    # Define vectors to append results to for a joining to a final output table
    gage_out = []
    date_out = []
    dayl_out = []
    prcp_out = []
    srad_out = []
    swe_out = []
    tmax_out = []
    tmin_out = []
    vp_out = []

 
    # read in file and convert to coordinates consistant with default pydaymet crs
    print(f'Reading {shape}')
    shp_in = gpd.read_file(
        shape 
        ).to_crs("EPSG:4326")

    for gg_id in shp_in['GAGE_ID']:

        # if gg_id already in downloaded data, then continue to next gage
        # if Daymet_Monthly['site_no'].str.contains(gg_id).any():
        #     continue
        
        # if station is in data being used for this study, then subset the shape file
        # to that catchment and use it to extract DAYMET data

        if df_all_sites.str.contains(gg_id).any():
            
            # Get index of which shp_in row is equal to current gage id (gg_id)
            shp_single_index = np.where(shp_in['GAGE_ID'] == gg_id)[0][0]

            shp_in_single = shp_in['geometry'][shp_single_index]

            # Plot
            # shp_in.plot(column = 'GAGE_ID')

            # Get coords, for extracting form pixel instead of entire watershed
            cor_x = np.array(shp_in_single.exterior.coords.xy)[0,].mean().round(5)
            cor_y = np.array(shp_in_single.exterior.coords.xy)[1,].mean().round(5)

            # retrieve monthly daymet data
            # set up so if fails once, will try again (added because sometimes a connection
            # failed to be made)

            # NOTE: I compared results using get_bycoords using the mean lat long, 
            # and using get_bygeom. I looked at dates 5/2/2000-5/22/2000 in gage 
            # '03145000'. Results were generally similar for most days, but even in
            # this small sample, the get_bycoords method missed two days of relative low
            # precip (i.e., 1.41 mm and 0.61 mm), and on one day the bycoords methods
            # showed 10.49 mm of precip and bygeom showed 5.55 mm. 
            # Timing wise: ...
            # try:
            #     daily_crd = daymet.get_bycoords(coords = (cor_x, cor_y),
            #         dates = ("2000-05-02", "2000-05-22"),
            #         variables = vars_get,
            #         time_scale = "daily"
            #         )

            # except:
            #     daily = daily = daymet.get_bycoords(coords = (cor_x, cor_y),
            #         dates = "2000-05-02",
            #         variables = vars_get,
            #         time_scale = "daily"
            #     )
            try:
                daily = daymet.get_bygeom(shp_in_single, 
                    dates = ("2000-05-02", "2000-05-22"), #dates_get,
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

            # create vectors holding 
            gage_out = gage_out.append(np.repeat(gg_id, len(daily_mean.time)))
            date_out = date_out.append(pd.to_datetime(daily_mean.time).normalize())
            dayl_out = dayl_out.append(daily_mean.dayl.values)
            prcp_out = prcp_out.append(daily_mean.prcp.values)
            srad_out = srad_out.append(daily_mean.srad.values)
            swe_out = swe_out.append(daily_mean.swe.values)
            tmax_out = tmax_out.append(daily_mean.tmax.values)
            tmin_out = tmin_out.append(daily_mean.tmin.values)
            vp_out = vp_out.append(daily_mean.vp.values)




            # if not exists('D:/Projects/GAGESii_ANNstuff/Data_Out/Daymet_Monthly.csv'):
            if not exists(f'C:/Users/bench/Desktop/daily_{shape[55:len(shape) - 4]}.csv'):
                pd.DataFrame({
                    'site_no': [], # 'gages_retreived': 
                    'date': [], # will hold yyyy-mm-dd dates
                    'dayl_retreived': [],
                    'prcp_retreived': [],
                    'srad_retreived': [],
                    'swe_retreived': [],
                    'tmax_retreived': [],
                    'tmin_retreived': [],                   
                    'vp_retreived': []
                }).to_csv( # NOTE that shape[indices] will need to be edited based on 
                            # directory from which shape variable objects are read
                    f'C:/Users/bench/Desktop/daily_{shape[55:len(shape) - 4]}.csv',
                    #'D:/DataWorking/Daymet/Daymet_Monthly.csv',
                    #'D:/Projects/GAGESii_ANNstuff/Data_Out/Daymet_Monthly.csv',
                    # 'C:/Users/bench/Desktop/Daymet_Monthly.csv',
                    index = False
                )

            

            # # save daymet data stored in daily to raw_data and working_data directories
            # daily.to_netcdf(f'D:/DataWorking/Daymet/Daily/Daymet_Daily_{gg_id}.nc')
            # daily.to_netcdf(f'D:/DataRaw/Daymet/Daily/Daymet_Daily_{gg_id}.nc')
            

            print(gg_id)


# enable garbage collector
# gc.enable()
#



# %%
