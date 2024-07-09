# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 09:56:28 2023

@author: Sarah_Berk
"""

##############################################################################
########## VI MEANS CALCULATOR WITH CITY PIXEL THRESHOLD #####################
##############################################################################


##############################################################################
########## TO RUN ON JASMIN WITH DATA FROM CEDA CATALOGUE ####################
##############################################################################

#Import packages
import os
import re  # regular expressions for getting lat lon grid
#import pathlib
#import warnings
#from osgeo import gdal
#import matplotlib.pyplot as plt
import numpy as np
#import numpy.ma as ma
import rasterio as rio # for extracting subsets
#from rasterio.plot import plotting_extent #for plotting
#import earthpy as et
#import earthpy.plot as ep
#import earthpy.spatial as es
#import earthpy.mask as em
import pandas as pd
import pickle
#import matplotlib.patches as mpatches
#import fnmatch  #for finding other file when city+rural isn't all in the main one
# for matching up the qc info filenames
import sys

#for the reprojecting
#import pyproj
#from pyproj import CRS
from pyproj import Transformer
#import cartopy
#import cartopy.crs as ccrs
#from shapely.geometry import Point, LineString, Polygon
from pyhdf.SD import SD, SDC
#import datetime as dt
#import scipy
#from scipy import interpolate
#import geopandas as gpd

#for finding the mode
#from collections import Counter


#load up the city info table 
CITY_COUNTRY_lat_lon = pd.read_csv('CITY_COUNTRY_lat_lon_mean.csv', encoding='latin-1')    
CITY_COUNTRY_lat_lon = CITY_COUNTRY_lat_lon.rename(columns={"ï»¿CITY_COUNTRY": "CITY_COUNTRY"})

print("CITY_COUNTRY: " + sys.argv[1])
CityID = int(sys.argv[1])

CITY_COUNTRY = CITY_COUNTRY_lat_lon['CITY_COUNTRY'][CityID]
print(CITY_COUNTRY)

CITY_COUNTRY_lat_lon = CITY_COUNTRY_lat_lon[CITY_COUNTRY_lat_lon['CITY_COUNTRY'] == CITY_COUNTRY].reset_index(drop = True)

#read in table with the EVI filenames, years and months
pickle_name = 'evi_filepaths_grid_ref_df.pkl'
with open(pickle_name, 'rb') as f:
    evi_filepaths_grid_ref_df = pickle.load(f)

#function to extract the subdatasets of interest and return a dataframe 
#NDVI and EVI are vegetation indices
#pixel reliability and VI quality are the quality checks

def vi_dataframe_create(SATELLITE_NDVI, vi_file_name):#, city_top):
    data_path = os.path.join(vi_file_name)
    with rio.open(data_path) as dataset:
    # Loop through each subdataset in HDF4 file
        for name in dataset.subdatasets:
                
            # Use regular expression to identify if subdataset has EVI in the name
            if re.search("1 km 16 days EVI", name):
        
                # Open the band subdataset
                with rio.open(name) as subdataset:
                    modis_meta = subdataset.profile
                
                    # Read band data as a 2 dim arr and append to list
                    EVI = subdataset.read(1)

            # Use regular expression to identify if subdataset has reliability in the name (for pixel reliability)
            if re.search("1 km 16 days pixel reliability", name):
        
                # Open the band subdataset
                with rio.open(name) as subdataset:
                    modis_meta = subdataset.profile
                
                    # Read band data as a 2 dim arr and append to list
                    pixel_reliability = subdataset.read(1)
                
                
              # Use regular expression to identify if subdataset has quality in the name (for VI Quality)
            if re.search("1 km 16 days VI Quality", name):
        
                # Open the band subdataset
                with rio.open(name) as subdataset:
                    modis_meta = subdataset.profile
                
                    # Read band data as a 2 dim arr and append to list
                    VI_quality = subdataset.read(1)       
                
                
    #Create the coordinate grid
    # Identify the data field- use the NDVI field but grid is same for all data
    DATAFIELD_NAME = '1 km 16 days NDVI'

    #if SATELLITE_NDVI == 'MOD13A3':
    #    GRID_NAME = 'MOD_Grid_monthly_1km_VI'
   # else:
    #    GRID_NAME = 'MYD_Grid_monthly_1km_VI'
        
    hdf = SD(vi_file_name, SDC.READ)

    # Read dataset.
    data2D = hdf.select(DATAFIELD_NAME)
    data = data2D[:,:].astype(np.float64)

    # Read global attribute.
    fattrs = hdf.attributes(full=1)
    ga = fattrs["StructMetadata.0"]
    gridmeta = ga[0]

    # Construct the grid.  Required information in global attribute called 'StructMetadata.0'

    ul_regex = re.compile(r'''UpperLeftPointMtrs=\(
                                  (?P<upper_left_x>[+-]?\d+\.\d+)
                                  ,
                                  (?P<upper_left_y>[+-]?\d+\.\d+)
                                  \)''', re.VERBOSE)
    match = ul_regex.search(gridmeta)
    x0 = float(match.group('upper_left_x')) 
    y0 = float(match.group('upper_left_y')) 

    lr_regex = re.compile(r'''LowerRightMtrs=\(
                                  (?P<lower_right_x>[+-]?\d+\.\d+)
                                  ,
                                  (?P<lower_right_y>[+-]?\d+\.\d+)
                                  \)''', re.VERBOSE)
    match = lr_regex.search(gridmeta)
    x1 = float(match.group('lower_right_x')) 
    y1 = float(match.group('lower_right_y')) 
    ny, nx = data.shape
    xinc = (x1 - x0) / nx
    yinc = (y1 - y0) / ny

    x = np.linspace(x0, x0 + xinc*nx, nx)
    y = np.linspace(y0, y0 + yinc*ny, ny)
    xv, yv = np.meshgrid(x, y)

    # convert the grid back to lat/lons.
    transformer = Transformer.from_crs("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext", "EPSG:4326")
    lat, lon = transformer.transform(xv, yv)

    #Apply scale factors
    scale_factor_EVI = 0.0001

    EVI = EVI*scale_factor_EVI

    #Create the lists to be combined to create a dataframe
    EVI_list = EVI.flatten()
    pixel_reliability_list = pixel_reliability.flatten()
    VI_quality_list = VI_quality.flatten()
    Lon_list = lon.flatten()
    Lat_list = lat.flatten()

    #Create the dataframe

    df = pd.DataFrame(list(zip(EVI_list, pixel_reliability_list, VI_quality_list, Lon_list, Lat_list)), 
               columns =['EVI','pixel_reliability', 'VI_quality','Longitude', 'Latitude']) 

    #Create dataframe of the required area
    df_subset = df[(df.Latitude > min_lat) & (df.Latitude < max_lat) & (df.Longitude > min_lon) & (df.Longitude < max_lon)]
    df_subset = df_subset.sort_values(by=['Latitude', 'Longitude']).reset_index(drop = True)
    return df_subset

    #functions for checking the VI quality, can include slightly lower quality pixels to increase the number of images available for use 
def check_bits(x,n):
    if (x & (1<<n)): 
  ## n-th bit is 1 
        flag = 1
    else:
  ## n-th bit is 0
        flag = 0
    return flag

#function to check bytes 0 and 2 and return 0 if they are both 0
def quality_control(x):
    if ((check_bits(x,0) == 0) and (check_bits(x,2) == 0)):
  ##if bit 1 and bit 2 are 0 then 0 
        flag = 0
    else:
  ##otherwise flag is 1
        flag = 1
    return flag    

    
#create a list of the urban mean evi and the rural mean evi
for m in range(len(CITY_COUNTRY_lat_lon)):

    CITY_COUNTRY = CITY_COUNTRY_lat_lon.CITY_COUNTRY[m]
    #Area to look at 
    min_lat = CITY_COUNTRY_lat_lon[CITY_COUNTRY_lat_lon['CITY_COUNTRY'] == CITY_COUNTRY]['min_lat'].values[0]
    max_lat = CITY_COUNTRY_lat_lon[CITY_COUNTRY_lat_lon['CITY_COUNTRY'] == CITY_COUNTRY]['max_lat'].values[0]
    min_lon = CITY_COUNTRY_lat_lon[CITY_COUNTRY_lat_lon['CITY_COUNTRY'] == CITY_COUNTRY]['min_lon'].values[0]
    max_lon = CITY_COUNTRY_lat_lon[CITY_COUNTRY_lat_lon['CITY_COUNTRY'] == CITY_COUNTRY]['max_lon'].values[0]
    Grid_Ref = CITY_COUNTRY_lat_lon[CITY_COUNTRY_lat_lon['CITY_COUNTRY'] == CITY_COUNTRY]['Grid_Ref'].values[0]
    
    #load in the data with the list of the city vi files
    #get lists of all the VI files and their months/ years for the required grid ref
            
    vi_means_df = evi_filepaths_grid_ref_df[evi_filepaths_grid_ref_df.Grid_Ref == Grid_Ref].reset_index(drop= True).copy()
    
    #create the df to be filled with the mean values
    vi_means_df['rur_mean_evi'] = np.nan
    vi_means_df['urb_mean_evi'] = np.nan     

    #Read in the LULC dataframe for the selected city
    LULC_df = pd.read_csv('LULC_df_{}.csv'.format(CITY_COUNTRY))
    
    #make sure LULC sorted by latitude and longitude
    LULC_df = LULC_df.sort_values(by=['Latitude', 'Longitude']).reset_index(drop = True)   
        
    for n in range(len(vi_means_df)):     
        #add the evi to the base
        #create the aqua and terra dataframes
        aqua_vi_file_name_0 = vi_means_df.Aqua_Filename_0[n]
        aqua_vi_file_name_1 = vi_means_df.Aqua_Filename_1[n]
        terra_vi_file_name_0 = vi_means_df.Terra_Filename_0[n]
        terra_vi_file_name_1 = vi_means_df.Terra_Filename_1[n]
        mon =  vi_means_df.Month[n]
        year = vi_means_df.Year[n]

        bad_list = []
        bad_list_year = []
        bad_list_month = []
    
        SATELLITE_NDVI = 'MYD13A3'
        if pd.isnull(aqua_vi_file_name_0): # if the file doesn't exist don't try to open 
            pass    
        else:
            try:
                aqua_vi_df_0 = vi_dataframe_create('MYD13A3', aqua_vi_file_name_0)
            except:
                bad_list.append(aqua_vi_file_name_0)
                bad_list_month.append(mon)
                bad_list_year.append(year)
                pass
            else:
                aqua_vi_df_0 = vi_dataframe_create('MYD13A3', aqua_vi_file_name_0)
                if CITY_COUNTRY in(["BULAWAYO_ZIMBABWE",'RIO_BRANCO_BRAZIL','SANHE_CHINA','DAYTON_USA']):
                    if CITY_COUNTRY == "BULAWAYO_ZIMBABWE":
                        Grid_Ref_top = 'h20v10'
                    elif CITY_COUNTRY == 'RIO_BRANCO_BRAZIL':
                        Grid_Ref_top = 'h11v10'  
                    elif CITY_COUNTRY == 'SANHE_CHINA':
                        Grid_Ref_top = 'h26v04'
                    elif CITY_COUNTRY == 'DAYTON_USA':
                        Grid_Ref_top = 'h11v04'
                    #add in other files for cities where the rural extent goes outside of the grid box
    
                    #extract the julian date of the main filename
                    yeardoy = aqua_vi_file_name_0.split('.')[1][1:] 

                    #find the filename which contains this in the top of city files
                    try:
                        top_file_name =evi_filepaths_grid_ref_df[(evi_filepaths_grid_ref_df.Grid_Ref == Grid_Ref_top)&(evi_filepaths_grid_ref_df.aqua_YearDOY_0 == yeardoy
                                                                                                                  )].Aqua_Filename_0.values[0]  
                    except:
                        aqua_vi_file_name_0 = np.nan
                        pass
                    else:
                        top_file_name =evi_filepaths_grid_ref_df[(evi_filepaths_grid_ref_df.Grid_Ref == Grid_Ref_top)&(evi_filepaths_grid_ref_df.aqua_YearDOY_0 == yeardoy
                                                                                                                  )].Aqua_Filename_0.values[0]    
        
                        try:
                            top_file_df = vi_dataframe_create('MYD13A3',top_file_name)
                        except:
                            aqua_vi_file_name_0 = np.nan
                            pass
                        else:          
                            top_file_df = vi_dataframe_create('MYD13A3',top_file_name)

                            aqua_vi_df_0 = aqua_vi_df_0.append(top_file_df).reset_index(drop = True)
                            aqua_vi_df_0 = aqua_vi_df_0.sort_values(by=['Latitude', 'Longitude']).reset_index(drop = True)

        if pd.isnull(aqua_vi_file_name_1): # if the file doesn't exist don't try to open 
            pass
        else:
            try:
                aqua_vi_df_1 = vi_dataframe_create('MYD13A3', aqua_vi_file_name_1)
            except:
                bad_list.append(aqua_vi_file_name_1)
                bad_list_month.append(mon)
                bad_list_year.append(year)
                pass
            else:
                aqua_vi_df_1 = vi_dataframe_create('MYD13A3', aqua_vi_file_name_1)
                if CITY_COUNTRY in(["BULAWAYO_ZIMBABWE",'RIO_BRANCO_BRAZIL','SANHE_CHINA','DAYTON_USA']):
                    if CITY_COUNTRY == "BULAWAYO_ZIMBABWE":
                        Grid_Ref_top = 'h20v10'
                    elif CITY_COUNTRY == 'RIO_BRANCO_BRAZIL':
                        Grid_Ref_top = 'h11v10'  
                    elif CITY_COUNTRY == 'SANHE_CHINA':
                        Grid_Ref_top = 'h26v04'
                    elif CITY_COUNTRY == 'DAYTON_USA':
                        Grid_Ref_top = 'h11v04'
                    #add in other files for cities where the rural extent goes outside of the grid box
            
                    #extract the julian date of the main filename
                    yeardoy = aqua_vi_file_name_1.split('.')[1][1:] 

                    #find the filename which contains this in the top of city files
                    try:
                        top_file_name =evi_filepaths_grid_ref_df[(evi_filepaths_grid_ref_df.Grid_Ref == Grid_Ref_top)&(evi_filepaths_grid_ref_df.aqua_YearDOY_1 == yeardoy
                                                                                                                  )].Aqua_Filename_1.values[0]   
                    except:
                        aqua_vi_file_name_1 = np.nan 
                        pass
                    else:
                        top_file_name =evi_filepaths_grid_ref_df[(evi_filepaths_grid_ref_df.Grid_Ref == Grid_Ref_top)&(evi_filepaths_grid_ref_df.aqua_YearDOY_1 == yeardoy
                                                                                                                  )].Aqua_Filename_1.values[0]     
                        
                        try:
                            top_file_df = vi_dataframe_create('MYD13A3',top_file_name)
                        except:
                            aqua_vi_file_name_1 = np.nan
                            pass
                        else:  
                            top_file_df = vi_dataframe_create('MYD13A3',top_file_name)

                            aqua_vi_df_1 = aqua_vi_df_1.append(top_file_df).reset_index(drop = True)
                            aqua_vi_df_1 = aqua_vi_df_1.sort_values(by=['Latitude', 'Longitude']).reset_index(drop = True)
                
            
        if pd.isnull(terra_vi_file_name_0): # if the file doesn't exist don't try to open 
            pass
        else:
            try:
                terra_vi_df_0 = vi_dataframe_create('MYD13A3', terra_vi_file_name_0)
            except:
                bad_list.append(aqua_vi_file_name_1)
                bad_list_month.append(mon)
                bad_list_year.append(year)
                pass 
            else:
                terra_vi_df_0 = vi_dataframe_create('MOD13A3', terra_vi_file_name_0)
                if CITY_COUNTRY in(["BULAWAYO_ZIMBABWE",'RIO_BRANCO_BRAZIL','SANHE_CHINA','DAYTON_USA']):
                    if CITY_COUNTRY == "BULAWAYO_ZIMBABWE":
                        Grid_Ref_top = 'h20v10'
                    elif CITY_COUNTRY == 'RIO_BRANCO_BRAZIL':
                        Grid_Ref_top = 'h11v10'  
                    elif CITY_COUNTRY == 'SANHE_CHINA':
                        Grid_Ref_top = 'h26v04'
                    elif CITY_COUNTRY == 'DAYTON_USA':
                        Grid_Ref_top = 'h11v04'
                    #add in other files for cities where the rural extent goes outside of the grid box

                    #extract the julian date of the main filename
                    yeardoy = terra_vi_file_name_0.split('.')[1][1:] 

                    #find the filename which contains this in the top of city files
                    try:
                        top_file_name =evi_filepaths_grid_ref_df[(evi_filepaths_grid_ref_df.Grid_Ref == Grid_Ref_top)&(evi_filepaths_grid_ref_df.terra_YearDOY_0 == yeardoy
                                                                                                                      )].Terra_Filename_0.values[0]  
                    except:
                        terra_vi_file_name_0 = np.nan
                        pass
                    else:
                        top_file_name =evi_filepaths_grid_ref_df[(evi_filepaths_grid_ref_df.Grid_Ref == Grid_Ref_top)&(evi_filepaths_grid_ref_df.terra_YearDOY_0 == yeardoy
                                                                                                                      )].Terra_Filename_0.values[0]    

                        try:
                            top_file_df = vi_dataframe_create('MOD13A3',top_file_name)
                        except:
                            terra_vi_file_name_0 = np.nan
                            pass
                        else:  
                            top_file_df = vi_dataframe_create('MOD13A3',top_file_name)

                            terra_vi_df_0 = terra_vi_df_0.append(top_file_df).reset_index(drop = True)
                            terra_vi_df_0 = terra_vi_df_0.sort_values(by=['Latitude', 'Longitude']).reset_index(drop = True)

        if pd.isnull(terra_vi_file_name_1): # if the file doesn't exist don't try to open 
            pass
        else:
            try:
                terra_vi_df_1 = vi_dataframe_create('MOD13A3', terra_vi_file_name_1)
            except:
                bad_list.append(terra_vi_file_name_1)
                bad_list_month.append(mon)
                bad_list_year.append(year)
                pass 
            else:
                terra_vi_df_1 = vi_dataframe_create('MOD13A3', terra_vi_file_name_1)
                if CITY_COUNTRY in(["BULAWAYO_ZIMBABWE",'RIO_BRANCO_BRAZIL','SANHE_CHINA','DAYTON_USA']):
                    if CITY_COUNTRY == "BULAWAYO_ZIMBABWE":
                        Grid_Ref_top = 'h20v10'
                    elif CITY_COUNTRY == 'RIO_BRANCO_BRAZIL':
                        Grid_Ref_top = 'h11v10'  
                    elif CITY_COUNTRY == 'SANHE_CHINA':
                        Grid_Ref_top = 'h26v04'
                    elif CITY_COUNTRY == 'DAYTON_USA':
                        Grid_Ref_top = 'h11v04'
                    #add in other files for cities where the rural extent goes outside of the grid box
    
                    #extract the julian date of the main filename
                    yeardoy = terra_vi_file_name_1.split('.')[1][1:] 

                    #find the filename which contains this in the top of city files
                    try:
                        top_file_name =evi_filepaths_grid_ref_df[(evi_filepaths_grid_ref_df.Grid_Ref == Grid_Ref_top)&(evi_filepaths_grid_ref_df.terra_YearDOY_1 == yeardoy
                                                                                                                      )].Terra_Filename_1.values[0]   
                    except:
                        terra_vi_file_name_1 = np.nan
                        pass
                    else:    
                        top_file_name =evi_filepaths_grid_ref_df[(evi_filepaths_grid_ref_df.Grid_Ref == Grid_Ref_top)&(evi_filepaths_grid_ref_df.terra_YearDOY_1 == yeardoy
                                                                                                                      )].Terra_Filename_1.values[0]    
                        try:
                            top_file_df = vi_dataframe_create('MOD13A3',top_file_name)
                        except:
                            terra_vi_file_name_1 = np.nan
                            pass
                        else:  
                            top_file_df = vi_dataframe_create('MOD13A3',top_file_name)
                        
                            terra_vi_df_1 = terra_vi_df_1.append(top_file_df).reset_index(drop = True)
                            terra_vi_df_1 = terra_vi_df_1.sort_values(by=['Latitude', 'Longitude']).reset_index(drop = True)        
        
        if (pd.isnull(aqua_vi_file_name_0)) & (pd.isnull(aqua_vi_file_name_1)) & (pd.isnull(terra_vi_file_name_0)) & (pd.isnull(terra_vi_file_name_1)):
            rur_mean_evi = np.nan
            urb_mean_evi = np.nan
        else:
            # situation where file doesnt exist
            #aqua 0 doesn't exist
            if pd.isnull(aqua_vi_file_name_0)& pd.isnull(aqua_vi_file_name_1) & pd.isnull(terra_vi_file_name_0):
                aqua_vi_df_0 = terra_vi_df_1.copy()
            elif pd.isnull(aqua_vi_file_name_0)& pd.isnull(aqua_vi_file_name_1):   
                aqua_vi_df_0 = terra_vi_df_0.copy()
            elif pd.isnull(aqua_vi_file_name_0):
                aqua_vi_df_0 = aqua_vi_df_1.copy()
                         
            #aqua 1 doesn't exist
            if pd.isnull(aqua_vi_file_name_1)& pd.isnull(aqua_vi_file_name_0) & pd.isnull(terra_vi_file_name_1):
                aqua_vi_df_1 = terra_vi_df_0.copy()
            elif pd.isnull(aqua_vi_file_name_1)& pd.isnull(aqua_vi_file_name_0):   
                aqua_vi_df_1 = terra_vi_df_1.copy()
            elif pd.isnull(aqua_vi_file_name_1):
                aqua_vi_df_1 = aqua_vi_df_0.copy()
              
            #terra 0 doesn't exist
            if pd.isnull(terra_vi_file_name_0)& pd.isnull(terra_vi_file_name_1) & pd.isnull(aqua_vi_file_name_0):
                terra_vi_df_0 = aqua_vi_df_1.copy()
            elif pd.isnull(terra_vi_file_name_0)& pd.isnull(terra_vi_file_name_1):   
                terra_vi_df_0 = aqua_vi_df_0.copy()
            elif pd.isnull(terra_vi_file_name_0):
                terra_vi_df_0 = terra_vi_df_1.copy()
                                    
            #terra 1 doesn't exist
            if pd.isnull(terra_vi_file_name_1)& pd.isnull(terra_vi_file_name_0) & pd.isnull(aqua_vi_file_name_1):
                terra_vi_df_1 = aqua_vi_df_0.copy()
            elif pd.isnull(terra_vi_file_name_1)& pd.isnull(terra_vi_file_name_0):   
                terra_vi_df_1 = aqua_vi_df_1.copy()
            elif pd.isnull(terra_vi_file_name_1):
                terra_vi_df_1 = terra_vi_df_0.copy()     

            #create a df containing final evi values (if aqua not reliable, use terra)
            LULC_df2 = LULC_df.copy()
            LULC_df2['aqua_evi_0'] = aqua_vi_df_0.EVI.values
            LULC_df2['aqua_pixel_reliability_0'] = aqua_vi_df_0.pixel_reliability.values
            LULC_df2['aqua_evi_1'] = aqua_vi_df_1.EVI.values
            LULC_df2['aqua_pixel_reliability_1'] = aqua_vi_df_1.pixel_reliability.values
            LULC_df2['terra_evi_0'] = terra_vi_df_0.EVI.values
            LULC_df2['terra_pixel_reliability_0'] = terra_vi_df_0.pixel_reliability.values
            LULC_df2['terra_evi_1'] = terra_vi_df_1.EVI.values
            LULC_df2['terra_pixel_reliability_1'] = terra_vi_df_1.pixel_reliability.values

            # add in the 2nd quality check flag, checks for slightly lower but still useful quality
            LULC_df2['aqua_0_VI_quality2'] = np.array([quality_control(int(i)) for i in aqua_vi_df_0.VI_quality])
            LULC_df2['aqua_1_VI_quality2'] = np.array([quality_control(int(i)) for i in aqua_vi_df_1.VI_quality])
            LULC_df2['terra_0_VI_quality2'] = np.array([quality_control(int(i)) for i in terra_vi_df_0.VI_quality])
            LULC_df2['terra_1_VI_quality2'] = np.array([quality_control(int(i)) for i in terra_vi_df_1.VI_quality])
                
                
            for ix in range(len(LULC_df2)): 
                aqua_p0 = LULC_df2.iloc[ix].aqua_pixel_reliability_0
                aqua_p1 = LULC_df2.iloc[ix].aqua_pixel_reliability_1
                terra_p0 = LULC_df2.iloc[ix].terra_pixel_reliability_0
                terra_p1 = LULC_df2.iloc[ix].terra_pixel_reliability_1
                    
                aqua_vi2_p0 = LULC_df2.iloc[ix].aqua_0_VI_quality2
                aqua_vi2_p1 = LULC_df2.iloc[ix].aqua_1_VI_quality2
                terra_vi2_p0 = LULC_df2.iloc[ix].terra_0_VI_quality2
                terra_vi2_p1 = LULC_df2.iloc[ix].terra_1_VI_quality2

                #update the 2nd flags to be 0 is the pixel quality flag is 0 (as some of the QA is then classed as good and the 2nd QA flags don't need to be assessed)
                if aqua_p0 == 0:
                    aqua_vi2_p0 = 0
                if aqua_p1 == 0:
                    aqua_vi2_p1 = 0
                if terra_p0 == 0:
                    terra_vi2_p0 = 0
                if terra_p1 == 0:
                    terra_vi2_p1 = 0
                    
                aqua_evi_0 = LULC_df2.iloc[ix].aqua_evi_0
                aqua_evi_1 = LULC_df2.iloc[ix].aqua_evi_1
                terra_evi_0 = LULC_df2.iloc[ix].terra_evi_0
                terra_evi_1 = LULC_df2.iloc[ix].terra_evi_1
                if aqua_p0 == 0:
                    evi_sum = aqua_evi_0
                    bad_count = 0
                else:
                    evi_sum = 0
                    bad_count = 1
                if aqua_p1 == 0:
                    evi_sum = evi_sum + aqua_evi_1
                else:
                    bad_count = bad_count + 1
                
                if terra_p0 == 0:
                    evi_sum = evi_sum + terra_evi_0
                else:
                    bad_count = bad_count + 1
                    
                if terra_p1 == 0:
                    evi_sum = evi_sum + terra_evi_1
                else:
                    bad_count = bad_count + 1
                
                if bad_count == 4:
                    evi_final = np.nan
                    pixel_reliablity_final = 1
                else:
                    evi_final = evi_sum/ (4-bad_count)
                    pixel_reliablity_final = 0
                LULC_df2.loc[ix, 'evi_final'] = evi_final
                LULC_df2.loc[ix, 'pixel_reliablity_final'] = pixel_reliablity_final

                if aqua_vi2_p0 == 0:
                    evi_vi2_sum = aqua_evi_0
                    bad_vi2_count = 0
                else:
                    evi_vi2_sum = 0
                    bad_vi2_count = 1
                if aqua_vi2_p1 == 0:
                    evi_vi2_sum = evi_vi2_sum + aqua_evi_1
                else:
                    bad_vi2_count = bad_vi2_count + 1
                
                if terra_vi2_p0 == 0:
                    evi_vi2_sum = evi_vi2_sum + terra_evi_0
                else:
                    bad_vi2_count = bad_vi2_count + 1
                    
                if terra_vi2_p1 == 0:
                    evi_vi2_sum = evi_vi2_sum + terra_evi_1
                else:
                    bad_vi2_count = bad_vi2_count + 1
                
                if bad_vi2_count == 4:
                    evi_vi2_final = np.nan
                    pixel_reliablity_final_2 = 1
                else:
                    evi_vi2_final = evi_vi2_sum/ (4-bad_vi2_count)
                    pixel_reliablity_final_2 = 0
                LULC_df2.loc[ix, 'evi_final_2'] = evi_vi2_final
                LULC_df2.loc[ix, 'pixel_reliablity_final_2'] = pixel_reliablity_final_2

            #calculate the average rur/ urb evi and pixel reliability percent
            rur_mean_evi = LULC_df2[(LULC_df2['is_urban_overall_{}'.format(year)] == 0)].evi_final.mean()
            urb_mean_evi = LULC_df2[(LULC_df2['is_urban_overall_{}'.format(year)] == 1)].evi_final.mean()
            
            rur_mean_evi_2 = LULC_df2[(LULC_df2['is_urban_overall_{}'.format(year)] == 0)].evi_final_2.mean()
            urb_mean_evi_2 = LULC_df2[(LULC_df2['is_urban_overall_{}'.format(year)] == 1)].evi_final_2.mean()

            pixel_reliability_percent = 100* len(LULC_df2[(LULC_df2['pixel_reliablity_final'] == 0)])/ len(LULC_df2)
            urban_only_pixel_reliability_percent = 100* len(LULC_df2[(LULC_df2['pixel_reliablity_final'] == 0)&(LULC_df2['is_urban_overall_{}'.format(year)] == 1)]
                                                            )/ len(LULC_df2[(LULC_df2['is_urban_overall_{}'.format(year)] == 1)])

            pixel_reliability_percent_2 = 100* len(LULC_df2[(LULC_df2['pixel_reliablity_final_2'] == 0)])/ len(LULC_df2)
            urban_only_pixel_reliability_percent_2 = 100* len(LULC_df2[(LULC_df2['pixel_reliablity_final_2'] == 0)&(LULC_df2['is_urban_overall_{}'.format(year)] == 1)]
                                                            )/ len(LULC_df2[(LULC_df2['is_urban_overall_{}'.format(year)] == 1)])
                
            #add to the overall dataframe with the list of files and the means
            vi_means_df.loc[n,'rur_mean_evi'] = rur_mean_evi
            vi_means_df.loc[n,'urb_mean_evi'] = urb_mean_evi
            vi_means_df.loc[n,'pixel_reliability_percent'] = pixel_reliability_percent
            vi_means_df.loc[n,'urban_only_pixel_reliability_percent'] = urban_only_pixel_reliability_percent
            vi_means_df.loc[n,'rur_mean_evi_2'] = rur_mean_evi_2
            vi_means_df.loc[n,'urb_mean_evi_2'] = urb_mean_evi_2
            vi_means_df.loc[n,'pixel_reliability_percent_2'] = pixel_reliability_percent_2
            vi_means_df.loc[n,'urban_only_pixel_reliability_percent_2'] = urban_only_pixel_reliability_percent_2
    
        vi_means_df['CITY_COUNTRY'] = CITY_COUNTRY
        #save the df
        pickle_name = 'vi_means_df_{}_jasmin.pkl'.format(CITY_COUNTRY)
        with open(pickle_name, 'wb') as f:
            pickle.dump(vi_means_df, f)
        
        #save the bad data
        bad_data_df = pd.DataFrame({'Year': bad_list_year, 'Month': bad_list_month, 'Filename': bad_list})
        pickle_name = 'evi_bad_data_info_{}.pkl'.format(CITY_COUNTRY)
        with open(pickle_name, 'wb') as f:
            pickle.dump(bad_data_df, f)
        
        