# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 15/11/2017
#______________________________________________________________________________
#______________________________________________________________________________


# Data Management
import numpy as np
# System Management
import sys
import warnings
import re
# Map Management
from osgeo import gdal
from osgeo import osr
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
from pyproj import Proj, transform
# ------------------
# Personal Modules
# ------------------
# Importing Modules
try:
    from GeoF.GeoTIFF import Functions as GF
    from GeoF.NetCDF import Functions as NF
except ImportError:
    from GeoTIFF import Functions as GF
    from NetCDF import Functions as NF

class GeoF:
    '''
    DESCRIPTION:
        This class opens and manipulates data related to the raster 
        data.
    _________________________________________________________________
    INPUT:
        :param Data:  A dict, a dictionary with the data, must have:
                      'Data': A ndArray, 3x3 matrix with time in the
                        first dimension.
                      'longitude': Longitude.
                      'latitude': Longitude.
                      'EPSG': EPSG number.
                      'Prj': wkt Projection.
                      'geoTrans': tuple with 
                        (xmin,xcellsize,0.0,ymax,0.0,ycellsize)
    '''
    def __init__(self,Data=None):
        '''
        '''
        # ----------------
        # Variables
        # ----------------
        Var = ['Data','longitude','latitude','EPSG','Prj','geoTrans']
        # ----------------
        # Error Managment
        # ----------------
        if not(isinstance(Data,dict)) or not(Data == None):
            self.ShowError('__init__','GeoF','Data has to be a dictionary or None.')

        if Data != None:
            for v in Var:
                try:
                    a = Data[Var]
                except KeyError:
                    self.ShowError('__init__','GeoF','%s is not in the dictionary, review the information for parameter Data' %v)

        self.Data = Data
        return

    def OpenNetCDFData(self,File,VarDict=None,VarRangeDict=None,EPSG=4326)
        '''
        DESCRIPTION:
            This class opens and manipulates data related to the raster 
            data.
        ________________________________________________________________________
        INPUT:
            :param File:         A str, File that would be extracted including 
                                 the path.
            :param VarDict:      A dict, List of variables that would be 
                                 extracted from the netCDF file. Defaulted 
                                 to None.
            :param VarRangeDict: A dict, Range of data that would be extracted 
                                 per variable. It is defaulted to None if all 
                                 the Range wants to be extracted.
                                 It must be a list with two values for each 
                                 variable.
        ________________________________________________________________________
        OUTPUT:
           :return Data: A dict, dictionary with projection ('Ptj' and 'EPSG'). 
        '''
        self.Data = NetF.EDnetCDFFile(File,VarDict=VarDict,VarRangeDict=VarRangeDict)
        if isinstance(self.Data,dict):
            self.SetProj(EPSG=EPSG)
        return

    def OpenGeoTIFFData(self,File,band=1)
        '''
        DESCRIPTION:
            This class opens and manipulates data related to the raster 
            data.
        _________________________________________________________________
        INPUT:
            :param File: A str, File that would be open.
            :param band: An int, band number in the raster that would
                         be extracted.
        '''
        # ----------------
        # Error Managment
        # ----------------
        if not(isinstance(File,str)):
            self.ShowError('OpenGeoTIFFData','GeoF','File has to be a string')
        if not(isinstance(band,int)):
            self.ShowError('OpenGeoTIFFData','GeoF','band has to be an intenger')
        # ----------------
        # Open File
        # ----------------
        self.File = File
        if File[-3:].lower() == 'tif' or File[-4:].lower() == 'tiff':
            self.Data = GF.GeoTIFFEx(File,band=band)
            a = re.compile('"EPSG",')
            b = re.finditer(a,self.Data['Prj'])
            c = [i.end() for i in b]
            self.Data['EPSG'] = 'epsg:'+self.Data['Prj'][c[-1]+1:-3]
        return

    def SetProj(self,EPSG):
        '''
        DESCRIPTION:

            This method projects the information of the raster to another 
            coordinate system.
        _______________________________________________________________________
        INPUT:
            :param EPSG: A int, epsg number of the new projection.
        _______________________________________________________________________
        OUTPUT:
           :return Data: A dict, dictionary with projection ('Ptj' and 'EPSG'). 
        '''
        # ----------------
        # Error Managment
        # ----------------
        if not(isinstance(EPSG,int)):
            self.ShowError('ProjData','GeoF','EPSG has to be an integer')

        # ----------------
        # Projection
        # ----------------
        spatialRef = osr.SpatialReference()
        spatialRef.ImportFromEPSG(EPSG)
        self.Data['Prj'] = spatialRef.ExportToWkt()
        self.Data['EPSG'] = 'epsg:'+str(EPSG)
        return

    def ProjData(self,EPSG):
        '''
        DESCRIPTION:

            This method projects the information of the raster to another 
            coordinate system.
        _______________________________________________________________________
        INPUT:
            :param EPSG: A int, epsg number of the new projection.
        _______________________________________________________________________
        OUTPUT:
           :return Data: A dict, Dict with the latitude and longitude 
                         projected.
        '''
        # ----------------
        # Error Managment
        # ----------------
        if not(isinstance(EPSG,int)):
            self.ShowError('ProjData','GeoF','EPSG has to be an integer')
        if len(Data['Data'].shape) > 4:
            self.ShowError('ProjData','GeoF',''Data' key has too much indices')

        # ----------------
        # Constants
        # ----------------
        Data = self.Data
        if len(Data['Data'].shape) == 3:
            Data['Data'] = Data['Data'][0]
        elif len(Data['Data'].shape) == 4:
            Data['Data'] = Data['Data'][0]
            Data['Data'] = Data['Data'][0]
        OrProj = Proj(init=Data['EPSG'])
        PrProj = Proj(init='epsg:'+str(EPSG))

        # ----------------
        # Project
        # ----------------
        # Projection
        spatialRef = osr.SpatialReference()
        spatialRef.ImportFromEPSG(EPSG)
        Data['Prj'] = spatialRef.ExportToWkt()
        Data['EPSG'] = 'epsg:'+str(EPSG)

        longitude0, latitude0 = transform(OrProj,PrProj,Data['longitude'][0],Data['latitude'][0])
        longitude1, latitude1 = transform(OrProj,PrProj,Data['longitude'][1],Data['latitude'][1])
        Clat = latitude1-latitude0
        Clon = longitude1-longitude0
        geoTrans = (longitude0,Clon,0.0,latitude0,0.0,Clat)

        # New Coordinates
        _shape = Data['Data'].shape
        latitude = np.empty(_shape[0])*np.nan
        longitude = np.empty(_shape[1])*np.nan
        # latitudeitude
        for ilat in range(len(latitude)):
            if ilat == 0:
                latitude[ilat] = geoTrans[3]
            else:
                latitude[ilat] = latitude[ilat-1]+Clat
        # longitudegitude
        for ilon in range(len(longitude)):
            if ilon == 0:
                longitude[ilon] = geoTrans[0]
            else:
                longitude[ilon] = longitude[ilon-1]+Clon

        # Save Coordinates
        Data['latitude'] = latitude
        Data['longitude'] = longitude
        Data['geoTrans'] = geoTrans
        self.Data = Data
        return

    def __str__(self):
        '''
        '''
        a  = '\nFILE OPENED: '+self.File
        b = '\nCOORDINATE SYSTEM INFO:\n'+self.Data['Prj']
        return a+b

    def ShowError(self,fn,cl,msg):
        '''
        DESCRIPTION:

            This method manages errors, and shows them. 
        _______________________________________________________________________
        INPUT:
            :param fn:  A str, Function that produced the error.
            :param cl:  A str, Class that produced the error.
            :param msg: A str, Message of the error.
        _______________________________________________________________________
        OUTPUT:
           :return: An int, Error managment -1. 
        '''

        raise Exception('ERROR: Method <'+fn+'> Class <'+cl+'>: '+msg)


