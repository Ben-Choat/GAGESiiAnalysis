# %%
# # BChoat 20220410

# Script to download DAYMET version 4 data using pydaymet api via HyRiver library

# %% Import libraries
import matplotlib.pyplot as plt
import pydaymet as daymet
from pynhd import NLDI
import pandas as pd
import geopandas as gpd
import numpy as np
import glob # for unix like interface to directories
import gc # for controlling garbage collection
from os.path import exists # to check if file already exists
#import netcdf4


# %% Read in data

# Previously, computer shut off during download, so it didn't complete 
# 4/17/2022
# Here I read in the data that did get downloaded, so I can pick up where I left off
if exists('D:/Projects/GAGESii_ANNstuff/Data_Out/Daymet_Monthly.csv'):
    Daymet_Monthly = pd.read_csv(
        # 'D:/Projects/GAGESii_ANNstuff/Data_Out/Daymet_Monthly.csv', 
        'D:/DataWorking/Daymet/Daymet_Monthly.csv',
        # 'C:/Users/bench/Desktop/Daymet_Monthly.csv',
        dtype = {'site_no': 'string'})
        # dtype={'gages_retreived': 'string'}) # I changed column names after initial
        # download.

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
dates_get = ("1998-01-01", "2012-12-31")

# Write dataframe to csv to later append restults to
if not exists('D:/Projects/GAGESii_ANNstuff/Data_Out/Daymet_Monthly.csv'):
    pd.DataFrame({
        'year': [],
        'month': [],
        'site_no': [], # 'gages_retreived': [], changed name
        'tmin_retreived': [],
        'tmax_retreived': [],
        'prcp_retreived': [],
        'vp_retreived': [],
        'swe_retreived': []
        # 'srad_retreived': [],
        # 'dayl_retreived': [],
    }).to_csv(
        'D:/DataWorking/Daymet/Daymet_Monthly.csv',
        #'D:/Projects/GAGESii_ANNstuff/Data_Out/Daymet_Monthly.csv',
        # 'C:/Users/bench/Desktop/Daymet_Monthly.csv',
        index = False
    )

Daymet_Monthly = pd.read_csv(
        #'D:/Projects/GAGESii_ANNstuff/Data_Out/Daymet_Monthly.csv', 
        'D:/DataWorking/Daymet/Daymet_Monthly.csv',
        # 'C:/Users/bench/Desktop/Daymet_Monthly.csv',
        dtype = {'site_no': 'string'})    

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
vars_get = ("tmin", "tmax", "prcp", "vp", "swe") #"srad", , "dayl"

# disable garbage collector to keep loop from slowing as size of data grows
# gc.disable()
for shape in list_shp:

    # if shape is the Alaska, Hawaii, Peurto Rice file, then skip it
    if 'AKHIPR' in shape:
        continue

    # Define vectors to append results to for a joining to a final output table
    gages_retreived = []
    tmin_retreived = []
    tmax_retreived = []
    prcp_retreived = []
    vp_retreived = []
    swe_retreived = []
    # srad_retreived = []
    # dayl_retreived = []
 
    # read in file and convert to coordinates consistant with default pydaymet crs
    print(f'Reading {shape}')
    shp_in = gpd.read_file(
        shape 
        ).to_crs("EPSG:4326")

    for gg_id in shp_in['GAGE_ID']:

        # if gg_id already in downloaded data, then continue to next gage
        if Daymet_Monthly['site_no'].str.contains(gg_id).any():
            continue
        
        # if station is in data being used for this study, then subset the shape file
        # to that catchment and use it to extract DAYMET data

        if df_all_sites.str.contains(gg_id).any():
            
            # Get index of which shp_in row is equal to current gage id (gg_id)
            shp_single_index = np.where(shp_in['GAGE_ID'] == gg_id)[0][0]

            shp_in_single = shp_in['geometry'][shp_single_index]

            # Plot
            # shp_in.plot(column = 'GAGE_ID')

            # retrieve monthly daymet data
            # set up so if fails once, will try again (added because sometimes a connection
            # failed to be made)
            try:
                monthly = daymet.get_bygeom(shp_in_single, 
                    dates = dates_get,
                    time_scale = "monthly",
                    variables = vars_get
                    )
            except:
                monthly = daymet.get_bygeom(shp_in_single, 
                    dates = dates_get,
                    time_scale = "monthly",
                    variables = vars_get
                    )

            # save daymet data stored in annual to raw_data and working_data directories
            # annual.to_netcdf(f'D:/DataWorking/Daymet/Annual/Daymet_Annual_{gg_id}.nc')
            # annual.to_netcdf(f'D:/DataRaw/Daymet/Annual/Daymet_Annual_{gg_id}.nc')
            
            # update vectors holding updated results
            # gages_retreived.append(np.repeat(gg_id, len(annual.time)))
            # tmin_retreived.append(annual.tmin.mean(['x', 'y']).values)
            # tmax_retreived.append(annual.tmax.mean(['x', 'y']).values)
            # prcp_retreived.append(annual.prcp.mean(['x', 'y']).values)
            # vp_retreived.append(annual.vp.mean(['x', 'y']).values)
            # swe_retreived.append(annual.swe.mean(['x', 'y']).values)
            # srad_retreived.append(annual.srad.mean(['x', 'y']).values)
            # dayl_retreived.append(annual.dayl.mean(['x', 'y']).values)

            print(gg_id)

            pd.DataFrame({
                'year': np.repeat(
                    np.array(range(1998, 2013, 1)), 12
                ), # this line may need to be edited a bit
                'month': np.concatenate([np.array(range(1, 13, 1))] * 15),
                'site_no': np.repeat(gg_id, len(monthly.time)).tolist(),
                'tmin_retreived': monthly.tmin.mean(['x', 'y']).values.tolist(),
                'tmax_retreived': monthly.tmax.mean(['x', 'y']).values.tolist(),
                'prcp_retreived': monthly.prcp.mean(['x', 'y']).values.tolist(),
                'vp_retreived': monthly.vp.mean(['x', 'y']).values.tolist(),
                'swe_retreived': monthly.swe.mean(['x', 'y']).values.tolist()
                # 'srad_retreived': [monthly.srad.mean(['x', 'y']).values.tolist()],
                # 'dayl_retreived': [monthly.dayl.mean(['x', 'y']).values.tolist())]
            }).to_csv(
                'D:/DataWorking/Daymet/Daymet_Monthly.csv',
                # 'D:/Projects/GAGESii_ANNstuff/Data_Out/Daymet_Monthly.csv',
                # 'C:/Users/bench/Desktop/Daymet_Monthly.csv',
                mode = 'a',
                index = False,
                header = False)


# enable garbage collector
# gc.enable()
#
#all_results = pd.DataFrame(
#    {
#        'gage_id': np.concatenate(gages_retreived),
#        'tmin': np.concatenate(tmin_retreived),
#        'tmax': np.concatenate(tmax_retreived),
#        'prcp': np.concatenate(prcp_retreived),
#        'vp': np.concatenate(vp_retreived),
#        'swe': np.concatenate(swe_retreived)
#        # srad: srad_retreived,
#        # dayl: dayl_retreived
#    }
#)

# all_results.to_csv("D:/Projects/GAGESii_ANNstuff/Data_Out/Daymet_Monthly.csv")

del(gages_retreived,
    tmin_retreived,
    tmax_retreived,
    prcp_retreived,
    vp_retreived,
    swe_retreived,
    # srad_retreived,
    # dayl_retreived
)


# print("Number of gages not accounted for:", 
#    len(set(df_all_sites).intersection(gages_retreived)) - len(df_all_sites)
#    )


# head
# shp_in.head()

# print(shp_in)

# Subset to single catchmet



# %%
