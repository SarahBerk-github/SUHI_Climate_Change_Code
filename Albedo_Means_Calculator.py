# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 18:37:25 2022

@author: Sarah_Berk
"""

#load up the required packages 

import pandas as pd
import geopandas
import numpy as np
import pickle 
from pyhdf.SD import SD, SDC #for extracting hdf files
import re  # regular expressions for getting lat lon grid
import rasterio as rio # for extracting subsets
import os
#for the reprojecting
import pyproj
from pyproj import CRS
from pyproj import Transformer
#import cartopy
#import cartopy.crs as ccrs
from shapely.geometry import Point, LineString, Polygon
#import datetime as dt
#import scipy
#from scipy import interpolate
import geopandas as gpd
#for finding the nearest points (to merge LULC data)
from scipy import spatial
# for matching up the qc info filenames
#import fnmatch
import sys

#load up the city info table 
CITY_COUNTRY_lat_lon = pd.read_csv('CITY_COUNTRY_lat_lon_mean.csv', encoding='latin-1')    
CITY_COUNTRY_lat_lon = CITY_COUNTRY_lat_lon.rename(columns={"ï»¿CITY_COUNTRY": "CITY_COUNTRY"})

print("CITY_COUNTRY: " + sys.argv[1])
CityID = int(sys.argv[1])

CITY_COUNTRY = CITY_COUNTRY_lat_lon['CITY_COUNTRY'][CityID]
print(CITY_COUNTRY)

CITY_COUNTRY_lat_lon = CITY_COUNTRY_lat_lon[CITY_COUNTRY_lat_lon['CITY_COUNTRY'] == CITY_COUNTRY]

# define the function to create the dataframe
def dataframe_create(file_name):
    #Create the coordinate grid
    # Identify the data field
    DATAFIELD_NAME = 'Albedo_WSA_shortwave'
        
    hdf = SD(file_name, SDC.READ)

    # Read dataset.
    data2D = hdf.select(DATAFIELD_NAME)
    data = data2D[:,:].astype(np.float64)

    # Read global attribute.
    fattrs = hdf.attributes(full=1)
    ga = fattrs["StructMetadata.0"]
    gridmeta = ga[0]

    
    # Construct the grid.  The needed information is in global attribute 'StructMetadata.0'
    # Use regular expressions (re) to find the extents of the grid in metadata

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
    sinu = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
    wgs84 = pyproj.Proj("+init=EPSG:4326") 
    lon, lat= pyproj.transform(sinu, wgs84, xv, yv)

    #Extract the QC_day subset

    data_path = os.path.join(file_name)

    with rio.open(data_path) as dataset:
    # Loop through each subdataset in HDF4 file
        for name in dataset.subdatasets:
        
        # Use regular expression to identify WSA 
            if re.search('Albedo_WSA_shortwave', name):
        
                # Open the band subdataset
                with rio.open(name) as subdataset:
                    modis_meta = subdataset.profile
                
                    # Read band data as a 2 dim arr and append to list
                    WSA = subdataset.read(1)
                
            # Use regular expression to identify BSA 
            if re.search('Albedo_BSA_shortwave', name):
        
                # Open the band subdataset
                with rio.open(name) as subdataset:
                    modis_meta = subdataset.profile
                
                    # Read band data as a 2 dim arr and append to list
                    BSA = subdataset.read(1)
                
                
              # Use regular expression to identify quality check 
            if re.search("BRDF_Albedo_Band_Mandatory_Quality_shortwave", name):
        
                # Open the band subdataset
                with rio.open(name) as subdataset:
                    modis_meta = subdataset.profile
                
                    # Read band data as a 2 dim arr and append to list
                    albedo_quality = subdataset.read(1)          



    #Create the coordinate grid
    # Identify the data field
    DATAFIELD_NAME = 'Albedo_WSA_shortwave'
        
    hdf = SD(file_name, SDC.READ)

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
    scale_factor= 0.001
   
    WSA = WSA*scale_factor
    BSA = BSA*scale_factor

    #Create the lists to be combined to create a dataframe
    WSA_list = WSA.flatten()
    BSA_list = BSA.flatten()
    albedo_quality_list = albedo_quality.flatten()
    Lon_list = lon.flatten()
    Lat_list = lat.flatten()

    df = pd.DataFrame(list(zip(WSA_list, BSA_list, albedo_quality_list, Lon_list, Lat_list)), 
               columns =['Albedo_WSA_shortwave', 'Albedo_BSA_shortwave','BRDF_Albedo_Band_Mandatory_Quality_shortwave','Longitude', 'Latitude']) 

    df_subset = df[(df.Latitude >min_lat) & (df.Latitude < max_lat) & (df.Longitude > min_lon) & (df.Longitude < max_lon)]
    
    df_subset = df_subset.sort_values(by=['Latitude', 'Longitude']).reset_index(drop = True)
    
    return df_subset

### The below functions have been added to optimise the code - the coordinate grid is only generated once #####
### as it is the same for all the images                                                                  #####

def data_extract(file_name, Lat_list, Lon_list):
    #Set path to chosen satellite
    #path to the file
    data_path = os.path.join(file_name)
    #LST daytime
    with rio.open(data_path) as dataset:
        # Loop through each subdataset in HDF4 file
        for name in dataset.subdatasets:
        
        # Use regular expression to identify if subdataset has LST:LST in the name
            if re.search("Albedo_WSA_shortwave*", name):
        
                # Open the band subdataset
                with rio.open(name) as subdataset:
                    modis_meta = subdataset.profile
                
                    # Read band data as a 2 dim arr and append to list
                    WSA = subdataset.read(1)
                
    #LST night       
    with rio.open(data_path) as dataset:
        # Loop through each subdataset in HDF4 file
        for name in dataset.subdatasets:
        
            # Use regular expression to identify if subdataset has LST:LST in the name
            if re.search("Albedo_BSA_shortwave*", name):
        
                # Open the band subdataset
                with rio.open(name) as subdataset:
                    modis_meta = subdataset.profile
                
                    # Read band data as a 2 dim arr and append to list
                    BSA = subdataset.read(1)                
                  
                
    #QC       
    with rio.open(data_path) as dataset:
        # Loop through each subdataset in HDF4 file
        for name in dataset.subdatasets:
        
            # Use regular expression to identify if subdataset has LST:LST in the name
            if re.search("BRDF_Albedo_Band_Mandatory_Quality_shortwave*", name):
        
                # Open the band subdataset
                with rio.open(name) as subdataset:
                    modis_meta = subdataset.profile
                
                    # Read band data as a 2 dim arr and append to list
                    albedo_quality = subdataset.read(1)   

                    
    #Apply scale factors
    scale_factor= 0.001
   
    WSA = WSA*scale_factor
    BSA = BSA*scale_factor

    #Create the lists to be combined to create a dataframe
    WSA_list = WSA.flatten()
    BSA_list = BSA.flatten()
    albedo_quality_list = albedo_quality.flatten()

    df = pd.DataFrame(list(zip(WSA_list, BSA_list, albedo_quality_list, Lon_list, Lat_list)), 
               columns =['Albedo_WSA_shortwave', 'Albedo_BSA_shortwave','BRDF_Albedo_Band_Mandatory_Quality_shortwave','Longitude', 'Latitude']) 

    df_subset = df[(df.Latitude >min_lat) & (df.Latitude < max_lat) & (df.Longitude > min_lon) & (df.Longitude < max_lon)]
    
    df_subset = df_subset.sort_values(by=['Latitude', 'Longitude']).reset_index(drop = True)
    
    return df_subset

def coord_grid_create(file_name):      
    #Create the coordinate grid
    # Identify the data field- use the LST day but grid is same for all data
    DATAFIELD_NAME = 'Albedo_WSA_shortwave'
    
    hdf = SD(file_name, SDC.READ)

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

    #Create the lists to be combined to create a dataframe

    Lon_list = lon.flatten()
    Lat_list = lat.flatten()
    
    return Lon_list, Lat_list

#function for finding the UTM projection
def utm_zoner(lon, lat):
    utm_lon = lon+180
    utm_zone = int(np.ceil(utm_lon/6))
    south_hem =''
    if lat<0:
        south_hem = ' +south'
    proj_str = f'+proj=utm +zone={utm_zone}{south_hem}'
    return proj_str



def qc_data_extract(file_name, Lat_list, Lon_list):
    #Set path to chosen satellite
    #path to the file
    data_path = os.path.join(file_name)
    # 7 bands
    with rio.open(data_path) as dataset:
        # Loop through each subdataset in HDF4 file
        for name in dataset.subdatasets:
        
        # Use regular expression to identify if subdataset has LST:LST in the name
            if re.search("BRDF_Albedo_Band_Quality_Band1*", name):
        
                # Open the band subdataset
                with rio.open(name) as subdataset:
                    modis_meta = subdataset.profile
                
                    # Read band data as a 2 dim arr and append to list
                    band1 = subdataset.read(1)

    with rio.open(data_path) as dataset:
        # Loop through each subdataset in HDF4 file
        for name in dataset.subdatasets:
        
        # Use regular expression to identify if subdataset has LST:LST in the name
            if re.search("BRDF_Albedo_Band_Quality_Band2*", name):
        
                # Open the band subdataset
                with rio.open(name) as subdataset:
                    modis_meta = subdataset.profile
                
                    # Read band data as a 2 dim arr and append to list
                    band2 = subdataset.read(1)


    with rio.open(data_path) as dataset:
        # Loop through each subdataset in HDF4 file
        for name in dataset.subdatasets:
        
        # Use regular expression to identify if subdataset has LST:LST in the name
            if re.search("BRDF_Albedo_Band_Quality_Band3*", name):
        
                # Open the band subdataset
                with rio.open(name) as subdataset:
                    modis_meta = subdataset.profile
                
                    # Read band data as a 2 dim arr and append to list
                    band3 = subdataset.read(1)

    with rio.open(data_path) as dataset:
        # Loop through each subdataset in HDF4 file
        for name in dataset.subdatasets:
        
        # Use regular expression to identify if subdataset has LST:LST in the name
            if re.search("BRDF_Albedo_Band_Quality_Band4*", name):
        
                # Open the band subdataset
                with rio.open(name) as subdataset:
                    modis_meta = subdataset.profile
                
                    # Read band data as a 2 dim arr and append to list
                    band4 = subdataset.read(1)


    with rio.open(data_path) as dataset:
        # Loop through each subdataset in HDF4 file
        for name in dataset.subdatasets:
        
        # Use regular expression to identify if subdataset has LST:LST in the name
            if re.search("BRDF_Albedo_Band_Quality_Band5*", name):
        
                # Open the band subdataset
                with rio.open(name) as subdataset:
                    modis_meta = subdataset.profile
                
                    # Read band data as a 2 dim arr and append to list
                    band5 = subdataset.read(1)

    with rio.open(data_path) as dataset:
        # Loop through each subdataset in HDF4 file
        for name in dataset.subdatasets:
        
        # Use regular expression to identify if subdataset has LST:LST in the name
            if re.search("BRDF_Albedo_Band_Quality_Band6*", name):
        
                # Open the band subdataset
                with rio.open(name) as subdataset:
                    modis_meta = subdataset.profile
                
                    # Read band data as a 2 dim arr and append to list
                    band6 = subdataset.read(1)

    with rio.open(data_path) as dataset:
        # Loop through each subdataset in HDF4 file
        for name in dataset.subdatasets:
        
        # Use regular expression to identify if subdataset has LST:LST in the name
            if re.search("BRDF_Albedo_Band_Quality_Band7*", name):
        
                # Open the band subdataset
                with rio.open(name) as subdataset:
                    modis_meta = subdataset.profile
                
                    # Read band data as a 2 dim arr and append to list
                    band7 = subdataset.read(1)  
                
    #Apply scale factors
    scale_factor= 1
   
    BRDF_Albedo_Band_Quality_Band1 = band1*scale_factor
    BRDF_Albedo_Band_Quality_Band2 = band2*scale_factor
    BRDF_Albedo_Band_Quality_Band3 = band3*scale_factor
    BRDF_Albedo_Band_Quality_Band4 = band4*scale_factor
    BRDF_Albedo_Band_Quality_Band5 = band5*scale_factor
    BRDF_Albedo_Band_Quality_Band6 = band6*scale_factor
    BRDF_Albedo_Band_Quality_Band7 = band7*scale_factor
    
    #Create the lists to be combined to create a dataframe
    BRDF_Albedo_Band_Quality_Band1_list = BRDF_Albedo_Band_Quality_Band1.flatten()
    BRDF_Albedo_Band_Quality_Band2_list = BRDF_Albedo_Band_Quality_Band2.flatten()
    BRDF_Albedo_Band_Quality_Band3_list = BRDF_Albedo_Band_Quality_Band3.flatten()
    BRDF_Albedo_Band_Quality_Band4_list = BRDF_Albedo_Band_Quality_Band4.flatten()
    BRDF_Albedo_Band_Quality_Band5_list = BRDF_Albedo_Band_Quality_Band5.flatten()
    BRDF_Albedo_Band_Quality_Band6_list = BRDF_Albedo_Band_Quality_Band6.flatten()
    BRDF_Albedo_Band_Quality_Band7_list = BRDF_Albedo_Band_Quality_Band7.flatten()

    df = pd.DataFrame(list(zip(BRDF_Albedo_Band_Quality_Band1_list,BRDF_Albedo_Band_Quality_Band2_list,BRDF_Albedo_Band_Quality_Band3_list,BRDF_Albedo_Band_Quality_Band4_list,
                               BRDF_Albedo_Band_Quality_Band5_list,BRDF_Albedo_Band_Quality_Band6_list, BRDF_Albedo_Band_Quality_Band7_list, Lon_list, Lat_list)), 
               columns =['BRDF_Albedo_Band_Quality_Band1','BRDF_Albedo_Band_Quality_Band2','BRDF_Albedo_Band_Quality_Band3','BRDF_Albedo_Band_Quality_Band4',
                         'BRDF_Albedo_Band_Quality_Band5','BRDF_Albedo_Band_Quality_Band6','BRDF_Albedo_Band_Quality_Band7','Longitude', 'Latitude']) 

    df_subset = df[(df.Latitude >min_lat) & (df.Latitude < max_lat) & (df.Longitude > min_lon) & (df.Longitude < max_lon)]
    
    df_subset = df_subset.sort_values(by=['Latitude', 'Longitude']).reset_index(drop = True)
    
    return df_subset

#Function to check the bands for quality, as the main qc flag is too strict
max_qc_flag = 3
#0 = best quality, full inversion (WoDs, RMSE majority good) 
#1 = good quality, full inversion (also including the cases that no clear sky observations over the day of interest or the Solar Zenith Angle is too large even WoDs, RMSE majority good)                           
#2 = Magnitude inversion (numobs >=7)                        
#3 = Magnitude inversion (numobs >=2&<7)                     
#4 = Fill value                                      

# define the function 
def qc_all_bands_check(row):
    if (row['BRDF_Albedo_Band_Quality_Band1'] <= max_qc_flag and row['BRDF_Albedo_Band_Quality_Band2'] <= max_qc_flag and row['BRDF_Albedo_Band_Quality_Band3'] <= max_qc_flag
        and row['BRDF_Albedo_Band_Quality_Band4'] <= max_qc_flag and row['BRDF_Albedo_Band_Quality_Band5'] <= max_qc_flag and row['BRDF_Albedo_Band_Quality_Band6'] <= max_qc_flag
        and row['BRDF_Albedo_Band_Quality_Band7'] <= max_qc_flag):
        qc = 0
    else:ssh sberk
        qc = 1
    return qc

# Read in grid ref lookups
pickle_name = 'filepaths_grid_ref_df.pkl'
with open(pickle_name, 'rb') as f:
    filepaths_grid_ref_df = pickle.load(f)

pickle_name = 'qc_filepaths_grid_ref_df.pkl'
with open(pickle_name, 'rb') as f:
    qc_filepaths_grid_ref_df = pickle.load(f)
    
# merge the qc filenames onto the albedo data filenames 
filepaths_grid_ref_df = filepaths_grid_ref_df.merge(qc_filepaths_grid_ref_df, on = ['Month', 'Year', 'YearDOY', 'Grid_Ref'], how = 'left').reset_index(drop = True)
filepaths_grid_ref_df = filepaths_grid_ref_df[~filepaths_grid_ref_df.QC_Filename.isnull()].reset_index(drop = True)

#set thresholds for the number of overall and city only pixels which have to be good quality
#the threshold is the number of bad pixels
qc_threshold = 0.3
urban_qc_threshold = 0.5

#supress depreciated function warnings
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


for count, CITY_COUNTRY in enumerate(CITY_COUNTRY_lat_lon.CITY_COUNTRY):
    
    City_Lat = CITY_COUNTRY_lat_lon[CITY_COUNTRY_lat_lon['CITY_COUNTRY'] == CITY_COUNTRY]['Lat'].values[0]
    City_Lon = CITY_COUNTRY_lat_lon[CITY_COUNTRY_lat_lon['CITY_COUNTRY'] == CITY_COUNTRY]['Lon'].values[0]
    min_lat = CITY_COUNTRY_lat_lon[CITY_COUNTRY_lat_lon['CITY_COUNTRY'] == CITY_COUNTRY]['min_lat'].values[0]
    max_lat = CITY_COUNTRY_lat_lon[CITY_COUNTRY_lat_lon['CITY_COUNTRY'] == CITY_COUNTRY]['max_lat'].values[0]
    min_lon = CITY_COUNTRY_lat_lon[CITY_COUNTRY_lat_lon['CITY_COUNTRY'] == CITY_COUNTRY]['min_lon'].values[0]
    max_lon = CITY_COUNTRY_lat_lon[CITY_COUNTRY_lat_lon['CITY_COUNTRY'] == CITY_COUNTRY]['max_lon'].values[0]
    grid_ref = CITY_COUNTRY_lat_lon[CITY_COUNTRY_lat_lon['CITY_COUNTRY'] == CITY_COUNTRY]['Grid_Ref'].values[0]
    City_bound_xmin = CITY_COUNTRY_lat_lon[CITY_COUNTRY_lat_lon['CITY_COUNTRY'] == CITY_COUNTRY]['City_Bound_xmin'].values[0]
    City_bound_xmax = CITY_COUNTRY_lat_lon[CITY_COUNTRY_lat_lon['CITY_COUNTRY'] == CITY_COUNTRY]['City_Bound_xmax'].values[0]
    City_bound_ymin = CITY_COUNTRY_lat_lon[CITY_COUNTRY_lat_lon['CITY_COUNTRY'] == CITY_COUNTRY]['City_Bound_ymin'].values[0]
    City_bound_ymax = CITY_COUNTRY_lat_lon[CITY_COUNTRY_lat_lon['CITY_COUNTRY'] == CITY_COUNTRY]['City_Bound_ymax'].values[0]

    #Read in the LULC dataframe for the selected city
    LULC_df = pd.read_csv('LULC_df_{}.csv'.format(CITY_COUNTRY))
    
    #file to create base from chosen at random, all coord grid are the same so doesn't matter
    file_name = filepaths_grid_ref_df[filepaths_grid_ref_df.Grid_Ref == grid_ref].iloc[7].Filename
    Lon_list, Lat_list = coord_grid_create(file_name)
    df = data_extract(file_name, Lat_list, Lon_list)
    
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
        yeardoy = file_name.split('.')[1][1:] 
        #find the filename which contains this in the top of city files
        top_file_name =filepaths_grid_ref_df[(filepaths_grid_ref_df.Grid_Ref == Grid_Ref_top)&(filepaths_grid_ref_df.YearDOY == yeardoy)].Filename.values[0]    
    
        Lon_list_top, Lat_list_top = coord_grid_create(top_file_name)
        top_df = data_extract(top_file_name, Lat_list_top, Lon_list_top)
        df = df.append(top_df).reset_index(drop = True)
        df = df.sort_values(by=['Latitude', 'Longitude']).reset_index(drop = True)

    #Get the UTM of the city
    local_utm = CRS.from_proj4(utm_zoner(City_Lon, City_Lat))

    #get geometry of the subset
    df_geometry = [Point(xy) for xy in zip(df.Longitude, df.Latitude)]

    #transform into a geopandas dataframe
    gdf = gpd.GeoDataFrame(df, geometry=df_geometry)
    gdf.crs= {'init': 'epsg:4326', 'no_defs': True}

    gdf2 = gdf.to_crs(epsg=local_utm.to_epsg()).copy() #create a new geo dataframe, with units in m, cartesian 2D axis 

    #origin coordinates - take to be city centre
    #create a data frame with origin points
    origin_df = pd.DataFrame({'City': [CITY_COUNTRY], 'Latitude': [City_Lat], 'Longitude': [City_Lon]})

    origin_df_geometry = [Point(xy) for xy in zip(origin_df.Longitude, origin_df.Latitude)]

    origin_gdf = gpd.GeoDataFrame(origin_df, geometry=origin_df_geometry)

    origin_gdf.crs= {'init': 'epsg:4326', 'no_defs': True}
    origin_gdf = origin_gdf.to_crs(epsg=local_utm.to_epsg())

    #Now change the grid so the the origin is the city centre 

    #Extract the x and y coordinates 
    x_points = gdf2['geometry'].x
    y_points = gdf2['geometry'].y

    #and add the extracted coordinates to dataframe
    gdf2['xpoints'] = x_points
    gdf2['ypoints'] = y_points

    #Determine the origin (city centre)
    origin_x = origin_gdf['geometry'].x
    origin_y = origin_gdf['geometry'].y

    #Subtract the orgin from the points
    gdf2['x_points'] = (gdf2['xpoints'] - origin_x.values)
    gdf2['y_points'] = (gdf2['ypoints'] - origin_y.values)

    #Drop the xpoints and ypoints columns as they are no longer needed
    gdf2 = gdf2.drop('xpoints', axis = 1)
    gdf2 = gdf2.drop('ypoints', axis = 1)

    #create a base of the x points and y points so don't have to generate this for every individual file, as will be the same
    x_points = gdf2.x_points.values
    y_points = gdf2.y_points.values

    ##For each point in LULC, the code runs compares all points in albedo ds. 
    ##It then extracts the closest 4 (res 500m square to res 1km square) points, saving their indexes and distances
    ##The indexes will be the same for all the albedo datasets so just need to do this onces per city 

    basepoints = [[x, y] for x, y in LULC_df[['x_points','y_points']].values]
    albedo_points =  [[x, y] for x, y in gdf2[['x_points','y_points']].values]

    for i in range(len(LULC_df)):   
        pt = basepoints[i]
        distance,index = spatial.cKDTree(albedo_points).query(pt, k=4)
        #distance # <-- The distances to the nearest neighbors
        #index # <-- The locations of the neighbors
        LULC_df.loc[i, 'index_point1'] = index[0]
        LULC_df.loc[i, 'index_point2'] = index[1]
        LULC_df.loc[i, 'index_point3'] = index[2]
        LULC_df.loc[i, 'index_point4'] = index[3]

        LULC_df.loc[i, 'distance_point1'] = distance[0]
        LULC_df.loc[i, 'distance_point2'] = distance[1]
        LULC_df.loc[i, 'distance_point3'] = distance[2]
        LULC_df.loc[i, 'distance_point4'] = distance[3]


    # loop through all the albedo files for the city     
    city_file_info = filepaths_grid_ref_df[filepaths_grid_ref_df.Grid_Ref == grid_ref].copy()
    # create empty columns to be filled
    city_file_info['qc_fail'] = np.nan
    city_file_info['Albedo_WSA_shortwave_rural'] = np.nan
    city_file_info['Albedo_WSA_shortwave_urban'] = np.nan
    city_file_info['Albedo_BSA_shortwave_rural'] = np.nan
    city_file_info['Albedo_BSA_shortwave_urban'] = np.nan

    # lists where the files which could not be opened are stored
    bad_list = []
    bad_list_year = []
    bad_list_month = []
    
    for file_name in city_file_info.Filename:
        try:
            df = data_extract(file_name, Lat_list, Lon_list)
            month = city_file_info[city_file_info.Filename == file_name].Month.values[0]
            year = city_file_info[city_file_info.Filename == file_name].Year.values[0]
        except:
            bad_list.append(file_name)
            bad_list_month.append(month)
            bad_list_year.append(year)
            city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_WSA_shortwave_rural'] = np.nan
            city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_WSA_shortwave_urban'] = np.nan
            city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_BSA_shortwave_rural'] = np.nan
            city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_BSA_shortwave_urban'] = np.nan
            pass
        else:
            df = data_extract(file_name, Lat_list, Lon_list)
            #get the qc info 
            qc_file_name = filepaths_grid_ref_df[(filepaths_grid_ref_df.Filename == file_name)].QC_Filename.values[0]
            try:
                qc_df = qc_data_extract(qc_file_name, Lat_list, Lon_list)  
                month = city_file_info[city_file_info.Filename == file_name].Month.values[0]
                year = city_file_info[city_file_info.Filename == file_name].Year.values[0]
            except:
                bad_list.append(file_name)
                bad_list_month.append(month)
                bad_list_year.append(year)
                city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_WSA_shortwave_rural'] = np.nan
                city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_WSA_shortwave_urban'] = np.nan
                city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_BSA_shortwave_rural'] = np.nan
                city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_BSA_shortwave_urban'] = np.nan
                qc_file_name = 'No_QC_file'
                pass
            else:
                qc_df = qc_data_extract(qc_file_name, Lat_list, Lon_list)
                
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
                    yeardoy = file_name.split('.')[1][1:] 
                    #find the filename which contains this in the top of city files
                    try: 
                        top_file_name = filepaths_grid_ref_df[(filepaths_grid_ref_df.Grid_Ref == Grid_Ref_top)&(filepaths_grid_ref_df.YearDOY == yeardoy)].Filename.values[0]
                    except:
                        bad_list.append(file_name)
                        bad_list_month.append(month)
                        bad_list_year.append(year)
                        city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_WSA_shortwave_rural'] = np.nan
                        city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_WSA_shortwave_urban'] = np.nan
                        city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_BSA_shortwave_rural'] = np.nan
                        city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_BSA_shortwave_urban'] = np.nan
                        top_file_name = 'none'
                        pass
                    else:
                        top_file_name =filepaths_grid_ref_df[(filepaths_grid_ref_df.Grid_Ref == Grid_Ref_top)&(filepaths_grid_ref_df.YearDOY == yeardoy)].Filename.values[0] 
                        try:
                            top_df = data_extract(top_file_name, Lat_list_top, Lon_list_top)
                        except:
                            bad_list.append(file_name)
                            bad_list_month.append(month)
                            bad_list_year.append(year)
                            city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_WSA_shortwave_rural'] = np.nan
                            city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_WSA_shortwave_urban'] = np.nan
                            city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_BSA_shortwave_rural'] = np.nan
                            city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_BSA_shortwave_urban'] = np.nan
                            top_file_name = 'none'
                            pass
                        else:   
                            top_df = data_extract(top_file_name, Lat_list_top, Lon_list_top)
                            df = df.append(top_df).reset_index(drop = True)
                            df = df.sort_values(by=['Latitude', 'Longitude']).reset_index(drop = True)

                            qc_top_file_name =filepaths_grid_ref_df[(filepaths_grid_ref_df.Grid_Ref == Grid_Ref_top)&(filepaths_grid_ref_df.YearDOY == yeardoy)].QC_Filename.values[0]
                            try:
                                qc_top_df = qc_data_extract(qc_top_file_name, Lat_list_top, Lon_list_top)
                            except:
                                bad_list.append(file_name)
                                bad_list_month.append(month)
                                bad_list_year.append(year)
                                city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_WSA_shortwave_rural'] = np.nan
                                city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_WSA_shortwave_urban'] = np.nan
                                city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_BSA_shortwave_rural'] = np.nan
                                city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_BSA_shortwave_urban'] = np.nan
                                top_file_name = 'none'
                                pass
                            else:
                                qc_top_df = qc_data_extract(qc_top_file_name, Lat_list_top, Lon_list_top)
                                qc_df = df.append(qc_top_df).reset_index(drop = True)
                                qc_df = qc_df.sort_values(by=['Latitude', 'Longitude']).reset_index(drop = True)

                if CITY_COUNTRY in(["BULAWAYO_ZIMBABWE",'RIO_BRANCO_BRAZIL','SANHE_CHINA','DAYTON_USA']) and (top_file_name == 'none'):
                    pass
                else:
                    if qc_file_name == 'No_QC_file':
                         df['all_bands_qc'] = df['BRDF_Albedo_Band_Mandatory_Quality_shortwave']
                    else:
                        df['all_bands_qc'] = qc_df.apply (lambda row: qc_all_bands_check(row), axis=1)        # add in the quality check for all the qc bands 
                        # set the band to -1 if the mandatory flag is -1 (bad quality, do not use)
                        df['all_bands_qc'] = df.apply(lambda x: x['BRDF_Albedo_Band_Mandatory_Quality_shortwave'] if x['BRDF_Albedo_Band_Mandatory_Quality_shortwave']==-1 else x['all_bands_qc'],axis=1) 
                        # set the band to 0 if the mandatory flag is 0 (good quality)
                        df['all_bands_qc'] = df.apply(lambda x: x['BRDF_Albedo_Band_Mandatory_Quality_shortwave'] if x['BRDF_Albedo_Band_Mandatory_Quality_shortwave']== 0 else x['all_bands_qc'],axis=1)
                    
                    year = city_file_info[city_file_info.Filename == file_name].Year.values[0]
                    LULC_df_base = LULC_df.copy()
        
                    for ix in range(len(LULC_df_base)): 
                        WSA_1 = df.Albedo_WSA_shortwave[LULC_df_base.index_point1[ix]]
                        BSA_1 = df.Albedo_BSA_shortwave[LULC_df_base.index_point1[ix]]
                        QC_1 = df.all_bands_qc[LULC_df_base.index_point1[ix]]
                        bad_count = 0 
                        if QC_1 != 0:
                            bad_count = bad_count + 1
                            WSA_1 = 0
                            BSA_1 = 0

                        WSA_2 = df.Albedo_WSA_shortwave[LULC_df_base.index_point2[ix]]
                        BSA_2 = df.Albedo_BSA_shortwave[LULC_df_base.index_point2[ix]]
                        QC_2 = df.all_bands_qc[LULC_df_base.index_point2[ix]]
                        if QC_2 != 0:
                            bad_count = bad_count + 1
                            WSA_2 = 0
                            BSA_2 = 0

                        WSA_3 = df.Albedo_WSA_shortwave[LULC_df_base.index_point3[ix]]
                        BSA_3 = df.Albedo_BSA_shortwave[LULC_df_base.index_point3[ix]]
                        QC_3 = df.all_bands_qc[LULC_df_base.index_point3[ix]]
                        if QC_3 != 0:
                            bad_count = bad_count + 1
                            WSA_3 = 0
                            BSA_3 = 0
    
                        WSA_4 = df.Albedo_WSA_shortwave[LULC_df_base.index_point4[ix]]
                        BSA_4 = df.Albedo_BSA_shortwave[LULC_df_base.index_point4[ix]]
                        QC_4 = df.all_bands_qc[LULC_df_base.index_point4[ix]]
                        if QC_4 != 0:
                            bad_count = bad_count + 1
                            WSA_4 = 0
                            BSA_4 = 0

                        num_good_pixels = 4 - bad_count
                        if num_good_pixels == 0:
                            Albedo_WSA_shortwave = np.nan
                            Albedo_BSA_shortwave = np.nan
                        else:
                            Albedo_WSA_shortwave = (WSA_1 + WSA_2 + WSA_3 + WSA_4) / num_good_pixels
                            Albedo_BSA_shortwave = (BSA_1 + BSA_2 + BSA_3 + BSA_4) / num_good_pixels

                        LULC_df_base.loc[ix, 'Albedo_WSA_shortwave'] = Albedo_WSA_shortwave
                        LULC_df_base.loc[ix, 'Albedo_BSA_shortwave'] = Albedo_BSA_shortwave

                    # drop areas which are urban and outside of the city
                    df_drop = LULC_df_base[((LULC_df_base['x_points'] < City_bound_xmin) | (LULC_df_base['x_points'] > City_bound_xmax) 
                             | (LULC_df_base['y_points'] < City_bound_ymin) | (LULC_df_base['y_points'] > City_bound_ymax)
                                  ) & (LULC_df_base['is_urban_overall_{}'.format(year)] == 1)]

                    LULC_df_base = LULC_df_base.drop(df_drop.index)
                    #remove water bodies 
                    df_drop = LULC_df_base[(LULC_df_base['lccs_class_overall_2015'] == 210)]
                    LULC_df_base = LULC_df_base.drop(df_drop.index)

                    #do the quality check and calculate the means from the usable pixels when the image is of sufficent quality
                    qc_percent = 100* len(LULC_df_base[~np.isnan(LULC_df_base.Albedo_WSA_shortwave)])/len(LULC_df_base)
                    qc_urban_percent = 100*len(LULC_df_base[(~np.isnan(LULC_df_base.Albedo_WSA_shortwave)) & (LULC_df_base['is_urban_overall_{}'.format(year)] == 1)]) / len(
                                                            LULC_df_base[(LULC_df_base['is_urban_overall_{}'.format(year)] == 1)])
                    
                    if len(LULC_df_base[np.isnan(LULC_df_base.Albedo_WSA_shortwave)])/len(LULC_df_base) > qc_threshold:
                        city_file_info.loc[(city_file_info['Filename'] == file_name), 'qc_fail'] = 1
                    elif len(LULC_df_base[(np.isnan(LULC_df_base.Albedo_WSA_shortwave)) & (LULC_df_base['is_urban_overall_{}'.format(year)] == 1)]) / len(
                                                            LULC_df_base[(LULC_df_base['is_urban_overall_{}'.format(year)] == 1)]) > urban_qc_threshold:
                        city_file_info.loc[(city_file_info['Filename'] == file_name), 'qc_fail'] = 1
                    else:
                        city_file_info.loc[(city_file_info['Filename'] == file_name), 'qc_fail'] = 0
                        Albedo_WSA_shortwave_rural = LULC_df_base[(~np.isnan(LULC_df_base.Albedo_WSA_shortwave)) & (LULC_df_base['is_urban_overall_{}'.format(year)]==0)].Albedo_WSA_shortwave.mean()
                        Albedo_WSA_shortwave_urban = LULC_df_base[(~np.isnan(LULC_df_base.Albedo_WSA_shortwave)) & (LULC_df_base['is_urban_overall_{}'.format(year)]==1)].Albedo_WSA_shortwave.mean()
                        Albedo_BSA_shortwave_rural = LULC_df_base[(~np.isnan(LULC_df_base.Albedo_BSA_shortwave)) & (LULC_df_base['is_urban_overall_{}'.format(year)]==0)].Albedo_BSA_shortwave.mean()
                        Albedo_BSA_shortwave_urban = LULC_df_base[(~np.isnan(LULC_df_base.Albedo_BSA_shortwave)) & (LULC_df_base['is_urban_overall_{}'.format(year)]==1)].Albedo_BSA_shortwave.mean()

                        city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_WSA_shortwave_rural'] = Albedo_WSA_shortwave_rural
                        city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_WSA_shortwave_urban'] = Albedo_WSA_shortwave_urban
                        city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_BSA_shortwave_rural'] = Albedo_BSA_shortwave_rural
                        city_file_info.loc[(city_file_info['Filename'] == file_name), 'Albedo_BSA_shortwave_urban'] = Albedo_BSA_shortwave_urban
                        city_file_info.loc[(city_file_info['Filename'] == file_name), 'qc_percent'] = qc_percent
                        city_file_info.loc[(city_file_info['Filename'] == file_name), 'qc_urban_percent'] = qc_urban_percent
            
            #save the city dataframe as a pickle
            pickle_name = 'albedo_info_{}.pkl'.format(CITY_COUNTRY)
            with open(pickle_name, 'wb') as f:
                pickle.dump(city_file_info, f)      

            #save the bad data
            bad_data_df = pd.DataFrame({'Year': bad_list_year, 'Month': bad_list_month, 'Filename': bad_list})
            pickle_name = 'albedo_bad_data_info_{}.pkl'.format(CITY_COUNTRY)
            with open(pickle_name, 'wb') as f:
                pickle.dump(bad_data_df, f)
