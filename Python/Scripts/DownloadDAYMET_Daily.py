# %%
# # BChoat 20220410

# Script to download DAYMET version 4 data using pydaymet api via HyRiver library

# NOTE:

# if running on campus pc on windows, then all references subsetting a portion of shape
# in order to call or save a file should read:
# shape[58:len(shape) - 4]

# HOWEVER, if running on LINUX on campus pc, then it should read:
# shape[58:len(shape) - 4]

# use cntrl + h to find and replace

# %% Import libraries
# import matplotlib.pyplot as plt
from __future__ import nested_scopes
from tkinter.tix import COLUMN
import pydaymet as daymet
# from pynhd import NLDI
import pandas as pd
import geopandas as gpd
# import datetime as dt
import numpy as np
import glob # for unix like interface to directories
# import re # for replacing parts of strings
# import gc # for controlling garbage collection
import os
from os.path import exists # to check if file already exists
import netCDF4 # I don't think this library is necessary
# For emailing when script is complete
import smtplib
import ssl
from email.message import EmailMessage

# %%
# NOTE: See lines 240 - 254 (subject to change) to comment/uncomment lines of code to 
# either investigate all catchments, or only those that have previously failed

# %% set up email options for when script is complete
subject = "DAYMET script is complete"
body = "The daily DAYMET download script has completed!!"
sender_email = "ben.choat@gmail.com"
receiver_email = "bchoat@rams.colostate.edu"
# password = input("Enter your GMail password: ")
# app password stored in google drive under Misc/Python_App_Password.csv
# read it into variable 'password'
# with open('G:/My Drive/Misc/Python_App_Password.txt') as f:
with open('/home/bchoat/Python/Python_App_Password.txt') as f:
    password = f.readlines()
# convert to string for input as password
password = ''.join(password)

# pd.read_csv('G:/My Drive/Misc/Python_App_Password.csv',
pd.read_csv('/home/bchoat/Python/Python_App_Password.csv',
    dtype = object)
message = EmailMessage()
message["From"] = sender_email
message["To"] = receiver_email
message["Subject"] = subject
message.set_content(body)

context = ssl.create_default_context()

# code to actually send email is below the DAYMET script


# %% Read in data

# Previously, computer shut off during download, so it didn't complete 
# 4/17/2022
# Here I read in the data that did get downloaded, so I can pick up where I left off
# if exists('D:/Projects/GAGESii_ANNstuff/Data_Out/Daymet_Monthly.csv'):
#     Daymet_Monthly=pd.read_csv(
#         # 'D:/Projects/GAGESii_ANNstuff/Data_Out/Daymet_Monthly.csv', 
#         'D:/DataWorking/Daymet/Daymet_Monthly.csv',
#         # 'C:/Users/bench/Desktop/Daymet_Monthly.csv',
#         dtype={'site_no': 'string'})
#         # dtype={'gages_retreived': 'string'}) # I changed column names after initial
#         # download.

# water yield directory
# dir_WY='E:/DataWorking/USGS_discharge/train_val_test'
# # explantory var (and other data) directory
# # dir_expl='D:/Projects/GAGESii_ANNstuff/Data_Out'

# # Monthly Water yield

# # training data
# df_train_sites=pd.read_csv(
#     f'{dir_WY}/yrs_98_12/monthly_WY/Mnthly_WY_train.csv',
#     dtype={"site_no":"string"}
#     )['site_no']

# # test basins used in training
# df_test_in_sites=pd.read_csv(
#     f'{dir_WY}/yrs_98_12/monthly_WY/Mnthly_WY_test_in.csv',
#     dtype={"site_no":"string"}
#     )['site_no']

# # test basins not used in training
# df_test_nit_sites=pd.read_csv(
#     f'{dir_WY}/yrs_98_12/monthly_WY/Mnthly_WY_test_nit.csv',
#     dtype={"site_no":"string"}
#     )['site_no']

# # validate basins used in training
# df_val_in_sites=pd.read_csv(
#     f'{dir_WY}/yrs_98_12/monthly_WY/Mnthly_WY_val_in.csv',
#     dtype={"site_no":"string"}
#     )['site_no']

# # validation basins not used in training
# df_val_nit_sites=pd.read_csv(
#     f'{dir_WY}/yrs_98_12/monthly_WY/Mnthly_WY_val_nit.csv',
#     dtype={"site_no":"string"}
#     )['site_no']


# # all sites
# df_all_sites=pd.concat([
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

# df_all_sites.to_csv('C:/Users/bchoat/desktop/DAYMET/all_sites.csv')

df_all_sites=pd.read_csv('/media/bchoat/Data/DAYMET/all_sites.csv',
    dtype={'site_no': 'string'})

# if list of catchments where daymet fails to retrieve the data does not exit,
# then create one
# this list is kept so that the script can continue running even if data fails to be
# downloaded for a given catchment
# define directory to store fails in
fails_csv = '/media/bchoat/Data/DAYMET/Daily/Daymet_daily_fails.csv'
if not exists(fails_csv):
    pd.DataFrame({
        'site_no': [], # gages that failed
    }).to_csv(
        fails_csv,
        index = False
    )

df_daymet_fails = pd.read_csv(fails_csv,
    dtype = {'site_no': 'string'})


# loop back through failed catchments and retry 


# %% get list of shape file names

# list_shp=glob.glob(r"D:/DataWorking/GAGESii/boundaries-shapefiles-by-aggeco/*.shp")
# list_shp=glob.glob(r'C:/Users/bchoat/Desktop/DAYMET/boundaries-shapefiles-by-aggeco/*.shp')
list_shp=glob.glob(r'/media/bchoat/Data/DAYMET/boundaries-shapefiles-by-aggeco/*.shp')

# %% Loop through shape files and use geometry of catchments being used in study
# to extract DAYMET data

# Define dates to extract between, including the specified days
dates_get=("1998-01-01", "2012-12-31")
# dates_get=("1998-01-01", "1998-01-02")


# Define variables to retrieve. Note that I'm retreiving all 7 available vars

# tmin: min daily temp [C]
# tmax: max daily temp [C]
# prcp: Daily total precipitation [mm/day],
# vp: water vapor pressure deficit [Pa] - Daily average partial pressur of water vapor
# swe: snow water equivalent [kg/m2]
# NOTE: the next two are only available for daily download
# srad: shortwave radiation [W/m2] - Average over daylight period - 
#   daily total radiation (MJ/m2/day) can be calculated as srad * dayl / 1,000,000
# dayl: day length [s/day] (s=seconds)
# NOTE: consider inclusion of latitute since srad and dayl are not available for 
##  monthly and annual time scales.
vars_get=("dayl", "prcp", "srad","swe", "tmax", "tmin", "vp")

# loop through shape files in list_shp
for shape in list_shp:

    # if shape is the Alaska, Hawaii, Peurto Rice file, then skip it
    if 'AKHIPR' in shape:
        continue

    # if not exists('D:/Projects/GAGESii_ANNstuff/Data_Out/Daymet_Monthly.csv'):
    # if not exists(f'C:/Users/bchoat/Desktop/DAYMET/Daily/daily_{shape[55:len(shape) - 4]}.csv'):
    # if not exists(f'C:/Users/bchoat/Desktop/DAYMET/Daily/Daymet_daily_{shape[67:len(shape) - 4]}.csv'):
    if not exists(f'/media/bchoat/Data/DAYMET/Daily/Daymet_daily_{shape[58:len(shape) - 4]}.csv'):
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
        }).to_csv( # NOTE that 'shape[indices]' below will need to be edited based on 
                    # directory from which shape variable objects are read
            # f'C:/Users/bench/Desktop/daily_{shape[55:len(shape) - 4]}.csv',
            f'/media/bchoat/Data/DAYMET/Daily/Daymet_daily_{shape[58:len(shape) - 4]}.csv',
            index=False
        )

    daily_work=pd.read_csv(
        f'/media/bchoat/Data/DAYMET/Daily/Daymet_daily_{shape[58:len(shape) - 4]}.csv',
        dtype={'site_no': 'string'}
    )
    
    # read in file and convert to coordinates consistant with default pydaymet crs
    print(f'Reading {shape}')
    shp_in=gpd.read_file(
        shape 
        ).to_crs("EPSG:4326")

    # find last gage id added to the daily work csv to use as a starting point
    # find intersection between gages in shp_in and in daily_work
    c = set(shp_in['GAGE_ID']).intersection(daily_work['site_no'])
    # if c is not empty then find last index for starting the new processing
    if c != set():
        last_c = pd.DataFrame(c).loc[len(c) - 1][0]
        # return indices of daily_work where site_no == last_c
        ind_work = np.where(shp_in['GAGE_ID'] == last_c)[0] + 1

        # try to define next_gg as the next index, if that index is out of range
            # assumes this shape is complete and continues to next
        try:
            # next gage id to use
            next_gg = shp_in['GAGE_ID'][ind_work]
            # define start index for gg_id in shp_in['GAGE_ID'] loop
            strt_ind = ind_work[0]
        except:
            print(f'{shape} appears to be complete, next shape')
            continue

    # if set is empty, then set strt_ind to 0
    if c == set():
        strt_ind = 0

    # switch out the next two for statements depending on 
    # if you want to only investigate gauges that previously failed

    # for statement to loop through all catchments
    # for gg_id in shp_in['GAGE_ID'][strt_ind: len(shp_in['GAGE_ID']) - 1]:
    #     print(f'begin processing {gg_id}')

    # for statement to loop through failed catchments only
    for gg_id in df_daymet_fails.values:
        gg_id = gg_id[0]
        if gg_id not in shp_in['GAGE_ID'][strt_ind: len(shp_in['GAGE_ID']) - 1].values:
            print(f'{gg_id} is not in this ecoregion - next')
            continue
        else:
            print(f'begin processing {gg_id}')
    


        # if gg_id already in downloaded data, then continue to next gage
        if daily_work['site_no'].str.contains(gg_id).any():
            print(f'{gg_id} previously processed - next')

            # also remove gg_id from failed list if it is in it
            df_daymet_fails = df_daymet_fails[df_daymet_fails['site_no'] != gg_id]

            continue
        

    
        # if station is in data being used for this study, then subset the shape file
        # to that catchment and use it to extract DAYMET data

        # if df_all_sites.str.contains(gg_id).any():
        if df_all_sites['site_no'].str.contains(gg_id).any():
            
            # Get index of which shp_in row is equal to current gage id (gg_id)
            shp_single_index=np.where(shp_in['GAGE_ID'] == gg_id)[0][0]

            shp_in_single=shp_in['geometry'][shp_single_index]

            # Plot
            # shp_in.plot(column='GAGE_ID')

            # Get coords, for extracting form pixel instead of entire watershed
            # cor_x=np.array(shp_in_single.exterior.coords.xy)[0,].mean().round(5)
            # cor_y=np.array(shp_in_single.exterior.coords.xy)[1,].mean().round(5)

            # retrieve monthly daymet data
            # set up so if fails once, will try again (added because sometimes the connection
            # either timed out or failed to be established while working on campus)
            
            # try to get daymet data for shp_in_single
            print('first try to retrieve data for input shape file')
            try:
                daily=daymet.get_bygeom(shp_in_single, 
                    dates=dates_get,
                    time_scale="daily",
                    variables=vars_get
                    )
            except:
                # removing second try to speed up downloads
                # # if fails try again
                # print('second try to retrieve data for input shape file')
                # try:
                #     daily=daymet.get_bygeom(shp_in_single, 
                #         dates=dates_get,
                #         time_scale="daily",
                #         variables=vars_get
                #         )
                # if still fails, then log the catchment id and continue to next catchment
            # except:
                print('both tries failed, so logging and moving on to next shape file')
                # if gage is already in failed list then go to next gage
                if df_daymet_fails['site_no'].str.contains(gg_id).any():
                    print(f'{gg_id} already in failed list - next')
                    continue
                # if gage is not already in failed list, then add it and 
                # save an updated csv
                # append failed gage to list and continue
                df_daymet_fails = pd.concat([df_daymet_fails, pd.DataFrame([{'site_no': gg_id}])],
                    ignore_index = True)
                df_daymet_fails.to_csv(fails_csv,
                    index = False)
                print(f'appended {gg_id} to failed list - next')
                
                # go to next gage (gg_id)
                continue



            # Calculate mean across basin
            daily_mean=daily.mean(dim=['x', 'y'])

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
                f'/media/bchoat/Data/DAYMET/Daily/Daymet_daily_{shape[58:len(shape) - 4]}.csv',
                mode='a',
                index=False,
                header=False
            )       
    
            # # save daymet data stored in daily to raw_data and working_data directories
            # daily.to_netcdf(f'D:/DataWorking/Daymet/Daily/Daymet_Daily_{gg_id}.nc')
            # daily.to_netcdf(f'D:/DataRaw/Daymet/Daily/Daymet_Daily_{gg_id}.nc')
        

            # if a previous failed catchment succeeds this time, then remove if from 
            # the failed list
            df_daymet_fails = df_daymet_fails[df_daymet_fails['site_no'] != gg_id]

            # print that the gg_id has been processed
            print(f'{gg_id} has been processed')

            # print deleting cache
            print('deleteing cache')
            # delete cache (in retrospect, there was an option to avoid keeping the cache)
            os.remove('/media/bchoat/Data/DAYMET/cache/aiohttp_cache.sqlite')
        else:
            print(f'{gg_id} is not included in the study catchments')

        




# %% Send email confirming script is completed

print("Sending Email")

with smtplib.SMTP_SSL("smtp.gmail.com", 465, context = context) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message.as_string())

print("Email sent")

# %% Clean up
# I accidnetally switched the names that were being written so have split the individual 
# csvs for each region into two ... here I go through those and combine them into one


# loop through aggecoregions (and reference catchments) and combine what were
# accidentally written to two separate csvs

# name_list = ['nonref_CntlPlains', 'nonref_EastHghlnds', 'nonref_MxWdShld', 
#                 'nonref_NorthEast', 'nonref_SECstPlain', 'nonref_SEPlains', 
#                 'nonref_WestMnts', 'nonref_WestPlains', 'nonref_WestXeric', 'ref_all']

# # loop through list
# for name in name_list:

#     # read in files located in Daily directory
#     list_csvs = glob.glob(rf'/media/bchoat/Data/DAYMET/Daily/*{name}*')

#     print(list_csvs)

#     # read in two reference files and combine (it was an accident to write to two files)
#     ref1 = pd.read_csv(list_csvs[0],
#         dtype = {'site_no': 'string'})
#     ref2 = pd.read_csv(list_csvs[1],
#         dtype = {'site_no': 'string'})

#     # concatenate
#     ref_all = pd.concat([ref1, ref2])

#     # write all references to single csv
#     ref_all.to_csv(f'/media/bchoat/Data/DAYMET/Daily/combined/Daymet_daily_{name}.csv',
#         index = False)


# # loop through all csv's and if a staid was collected and is also listed in the 
# # list of failed staid's in Daymet_daily_fails.csv, then remove it from the failed list
# # read in fails
# df_fails = pd.read_csv(
#     '/media/bchoat/Data/DAYMET/Daily/accidentallySplitFiles/Daymet_daily_fails.csv',
#     dtype = {'site_no': 'string'})

# df_work = df_fails.copy()

# # loop through list
# for name in name_list:
#     df_good= pd.read_csv(f'/media/bchoat/Data/DAYMET/Daily/Daymet_daily_{name}.csv',
#     dtype = {'site_no': 'string'})

#     # identify which catchments in failed list were actually retrieved
#     temp_int = set(df_good['site_no']).intersection(df_fails['site_no'])
    
#     # if temp_int is empty set, then next
#     if temp_int == set():
#         continue

#     else:
#         temploc = np.where(df_work['site_no'].isin(temp_int))

#         df_work.drop(df_work.index[temploc], inplace = True)


# df_work.reset_index(inplace = True, drop = True)

# df_work.to_csv('/media/bchoat/Data/DAYMET/Daily/Daymet_daily_fails.csv',
#     index = False)


# %% Investigate stations not downloaded and related agecoregions

# # read in all id vars for all ecoregions
# df_GAGES_idVars_all = pd.read_csv(
#     'GAGES_idVars_All.csv',
#     dtype = {'STAID': 'string'}
#     )

# # merge to failed catchments
# df_fails_id = pd.merge(
#     df_daymet_fails,
#     df_GAGES_idVars_all,
#     left_on = 'site_no',
#     right_on = 'STAID'
# )

# # read in id vars with continuous streamflow data from '98-'12
# df_GAGES_idVars_cont = pd.read_csv(
#     'ID_all_avail98_12.csv',
#     dtype = {'STAID': 'string'}
# )

# # Plot histogogram of fails by ecoregion
# df_fails_id['AggEcoregion'].hist(xrot = 90)
# # print number of stations that failed by ecoregion
# fail_counts = df_fails_id['AggEcoregion'].value_counts().sort_index()

# # print total number of catchments in each aggecoregion
# # for those catchments w/continuous data form '98-'12
# df_GAGES_idVars_cont['AggEcoregion'].hist(xrot = 90)
# # print number of catchments in each ecoregion
# cont_counts = df_GAGES_idVars_cont['AggEcoregion'].value_counts().sort_index()

# sum(cont_counts - fail_counts)


# %%
# list_shp=glob.glob(r'/media/bchoat/Data/DAYMET/boundaries-shapefiles-by-aggeco/*.shp')
# # plot some of failed basins

# # read in states shape file
# states = gpd.read_file(
#     '/media/bchoat/Data/GAGESii_Plotting/cb_2018_us_state_20m/cb_2018_us_state_20m.shp'
# )

# # print head of states shape file df
# states.head()

# # print shapes coordinate system
# states.crs

# # convert to WGSS84 coordinate system
# shape = states.to_crs("EPSG:4326")

# # plot states
# shape.boundary.plot()        

# # exclude Alaska, Hawaii, and Puerto Rico
# shape = shape.loc[~shape['NAME'].isin(['Alaska', 'Hawaii', 'Puerto Rico'])]

# # plot 
# shape.boundary.plot()

# # read in West Mountain (7) shape files
# shp_in=gpd.read_file(
#         list_shp[7] 
#         ).to_crs("EPSG:4326")

# # subset to failed basins (basins that failed to download daily DAYMET data)
# shp_in_fail = shp_in[shp_in['GAGE_ID'].isin(df_fails_id['STAID'])]


# # Plot all plots
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize = (20, 10))
# shape.boundary.plot(ax = ax)
# shp_in.boundary.plot(ax = ax)
# shp_in_fail.boundary.plot(ax = ax, color = 'red')


# %%
