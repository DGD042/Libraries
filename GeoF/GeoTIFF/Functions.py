# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 15/11/2017
#______________________________________________________________________________
#______________________________________________________________________________

# Data Managment
import numpy as np
# Map Managment
from osgeo import gdal
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
from pyproj import Proj, transform
# System managment
import sys
import warnings

# ------------------
# Personal Modules
# ------------------
# Importing Modules
from Utilities import Utilities as utl

def GeoTIFFEx(Ar,band=1):
    '''
    DESCRIPTION:
    
        This function extracts one band a GeoTIFF format file.
    _________________________________________________________________________

    INPUT:
    :param Ar:   A str, GeoTIFF File.
    :param Band: An int, band number in the raster that needs to be 
                 extracted.
    _________________________________________________________________________
    
    OUTPUT:
    :return Data: A dict, Dictionary with all the data related to the 
                  GeoTIFF.
    '''

    # -----------------
    # Error Managment
    # -----------------
    # this allows GDAL to throw Python Exceptions
    gdal.UseExceptions()
    try:
        src_ds = gdal.Open(Ar)
    except RuntimeError:
        raise Exception('Unable to open '+Ar)

    try:
        srcband = src_ds.GetRasterBand(band)
    except RuntimeError:
        raise Exception('Band ( %i ) not found' % band)

    # ------------------
    # Data Extraction
    # ------------------
    Data = dict()
    Data['Data']= BandReadAsArray(srcband).astype(float)
    # Verify NaN data
    NoDataValue = srcband.GetNoDataValue()
    x = np.where(Data['Data'] == NoDataValue)
    Data['Data'][x] = np.nan
    # Coordinate System
    Data['Prj'] = src_ds.GetProjection()
    # Coordinates
    geoTrans = src_ds.GetGeoTransform()
    Data['geoTrans'] = geoTrans
    _shape = Data['Data'].shape
    Lat = np.empty(_shape[0])*np.nan
    Lon = np.empty(_shape[1])*np.nan
    # Latitude
    Clat = geoTrans[-1]
    for ilat in range(len(Lat)):
        if ilat == 0:
            Lat[ilat] = geoTrans[3]
        else:
            Lat[ilat] = Lat[ilat-1]+Clat
    # Longitude
    Clon = geoTrans[1]
    for ilon in range(len(Lon)):
        if ilon == 0:
            Lon[ilon] = geoTrans[0]
        else:
            Lon[ilon] = Lon[ilon-1]+Clon

    # Save Coordinates
    Data['latitude'] = Lat
    Data['longitude'] = Lon

    return Data

def GeoTIFFSave(T,geoTrans,Projection=None,Name='Results',Pathout=''):
    '''
    DESCRIPTION:
        This Functions saves one a raster in GeoTIFF format (.tif) in one
        one band.
    _________________________________________________________________________

    INPUT:
    :param T:          A ndArray, 2x2 Matrix with the values of the raster.
    :param geoTrans:   A tuple, tuple with the following values.
                       (x_min,cellsize_x,0.0,y_max,0.0,cellsize_y)
    :param Projection: A str, wkt projection. Is set WGS84.
    :param Name:       A str, Name of the file. Is set to 'Results'.
    :param Pathout:    A str, Path to save the file. Is set to ''.
    _________________________________________________________________________
    
        OUTPUT:
    Se guarda un archivo GeoTiff.
    '''

    # ------------------
    # Values
    # ------------------
    x_pixels = T.shape[1]  # number of pixels in x
    y_pixels = T.shape[0]  # number of pixels in y

    # ------------------
    # Save File
    # ------------------
    # Create Folder
    utl.CrFolder(Pathout)
    Nameout = Pathout + Name + '.tif'
    
    # Pojection
    if Projection == None:
        # WGS84
        wkt_projection = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
    else:
        wkt_projection = Projection

    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        Nameout,
        x_pixels,
        y_pixels,
        1,
        gdal.GDT_Float32, )

    dataset.SetGeoTransform(geoTrans)

    dataset.SetProjection(wkt_projection)
    dataset.GetRasterBand(1).WriteArray(T)
    Band = dataset.GetRasterBand(1)
    Band.SetNoDataValue(-9999.0)
    dataset.FlushCache()  # Write to disk.
    return 
