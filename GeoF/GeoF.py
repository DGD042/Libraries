# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 26/02/2016
#______________________________________________________________________________
#______________________________________________________________________________

#______________________________________________________________________________
#
# DESCRIPCIÓN DE LA CLASE:
#   En esta clase se incluyen las rutinas para tratar información de GIS.
#   Esta librería necesita del módulo GDAL, se deben instalar
#
#   Esta libreria es de uso libre y puede ser modificada a su gusto, si tienen
#   algún problema se pueden comunicar con el programador al correo:
#   dagonzalezdu@unal.edu.co
#______________________________________________________________________________

import numpy as np
import sys
import csv
import xlrd # Para poder abrir archivos de Excel
import xlsxwriter as xlsxwl
import warnings

from osgeo import gdal
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *

# Se importan los paquetes para manejo de fechas
from datetime import date, datetime, timedelta

# ------------------
# Personal Modules
# ------------------
# Importing Modules
from Utilities import Utilities as utl

class GeoF:

    def __init__(self):

        '''
            DESCRIPTION:

        This is a build up function.
        '''
    def GEOTiffEx(self,Ar,FlagGTD=False):
        '''
            DESCRIPTION:
        
        Esta función pretende extraer la información de una banda de un GeoTiff
        _________________________________________________________________________

            INPUT:
        + Ar: Archivo GeoTiff.
        + FlagGTD: Bandera para obtener los valores de tamaño de celda del GeoTiff.
        _________________________________________________________________________
        
            OUTPUT:
        - data1: Raster data
        - xllcorner: Coordenadas de la esquina occidental.
        - yllcorner: Coordenadas de la esquina sur.
        - cellsize: Tamaño de la celda.
        - geoTrans: Información de las coordenadas y el tamaño de celda.
        '''

        # this allows GDAL to throw Python Exceptions
        gdal.UseExceptions()

        try:
            src_ds = gdal.Open(Ar)
        except RuntimeError:
            print('Unable to open INPUT.tif')
            sys.exit(1)

        try:
            srcband = src_ds.GetRasterBand(1)
        except RuntimeError:
            # for example, try GetRasterBand(10)
            print ('Band ( %i ) not found' % band_num)
            sys.exit(1)

        # Se extrae la información
        data1 = BandReadAsArray(srcband).astype(float)
        NoDataValue = srcband.GetNoDataValue()
        
        x = np.where(data1 == NoDataValue)
        
        data1[x] = np.nan

        # Se encuentran las coordenadas
        geoTrans = src_ds.GetGeoTransform()

        # Se arregla el yllcorner porque se encuentra en el norte.
        a = data1.shape[0]
        yllcorner = geoTrans[3]+(a*geoTrans[-1])

        if FlagGTD:
            return data1,(geoTrans[0])-(geoTrans[1]/2),yllcorner+(geoTrans[-1]/2),geoTrans[1],geoTrans
        else:
            return data1,(geoTrans[0])-(geoTrans[1]/2),yllcorner+(geoTrans[-1]/2),geoTrans[1]

    def GEOTiffSave(self,T,cellsize_x,cellsize_y,x_min,y_max,Name,Pathout,Projection=0):
        '''
            DESCRIPTION:
        
        Esta función pretende guardar un archivo GEOTiff con información en una
        sola banda.
        _________________________________________________________________________

            INPUT:
        + T: Matriz con los valores que se quieren guardar.
        + cellsize_x: Tamaño de celda en la latitud.
        + cellsize_y: Tamaño de celda en la longitud.
        + x_min: Longitud en el punto occidental.
        + y_max: Latitud en el punto norte.
        + Name: Nombre del documento.
        + Pathout: Ruta de salida.
        + Projection: Proyección de la información 
                      0: Magna Bogotá.
                      1: WGS84.
        _________________________________________________________________________
        
            OUTPUT:
        Se guarda un archivo GeoTiff.
        '''

        # Se crea la carpeta
        utl.CrFolder(Pathout)

        Nameout = Pathout + Name + '.tif'

        x_pixels = T.shape[1]  # number of pixels in x
        y_pixels = T.shape[0]  # number of pixels in y
        
        if Projection == 0:
            # MAGNA SIRAS Bogotá
            wkt_projection = 'PROJCS["MAGNA-SIRGAS / Colombia Bogota zone",GEOGCS["MAGNA-SIRGAS",DATUM["Marco_Geocentrico_Nacional_de_Referencia",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6686"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4686"]],UNIT["metre",1,AUTHORITY["EPSG","9001"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",4.596200416666666],PARAMETER["central_meridian",-74.07750791666666],PARAMETER["scale_factor",1],PARAMETER["false_easting",1000000],PARAMETER["false_northing",1000000],AUTHORITY["EPSG","3116"],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
        elif Projection == 1:
            # WGS84
            wkt_projection = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'


        driver = gdal.GetDriverByName('GTiff')

        dataset = driver.Create(
            Nameout,
            x_pixels,
            y_pixels,
            1,
            gdal.GDT_Float32, )

        dataset.SetGeoTransform((
            x_min,    # 0
            cellsize_x,  # 1
            0,                      # 2
            y_max,    # 3
            0,                      # 4
            -cellsize_y))

        dataset.SetProjection(wkt_projection)
        dataset.GetRasterBand(1).WriteArray(T)
        Band = dataset.GetRasterBand(1)
        Band.SetNoDataValue(-9999.0)
        dataset.FlushCache()  # Write to disk.
