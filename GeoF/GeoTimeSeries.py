# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 16/11/2017
#______________________________________________________________________________
#______________________________________________________________________________


# Data Management
import numpy as np
import scipy.io as sio
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
# Dates
from datetime import date, datetime, timedelta
# ------------------
# Personal Modules
# ------------------
# Importing Modules
try:
    from GeoF.GeoTIFF import Functions as GF
except ImportError:
    from GeoTIFF import Functions as GF
from EMSD.Dates.DatesC import DatesC
from EMSD.Dates import DatesFunctions as DUtil
from Utilities import Utilities as utl
class GeoTimeSeries(object):
    '''
    DESCRIPTION:
        This class manipulates data related to the raster 
        data in time.
    _________________________________________________________________
    INPUT:
        :param Dates: A ndArray, Dates of the Data.
        :param Data:  A dict, a dictionary with the data, must have:
                      'Data': A ndArray, 3x3 matrix with time in the
                              first dimension.
                      'lonigute': Longitude.
                      'Latitude': Longitude.
                      'EPSG': EPSG number.
                      'Prj': wkt Projection.
                      'geoTrans': tuple with 
                        (xmin,xcellsize,0.0,ymax,0.0,ycellsize)
        :param var: A str, key with the name of the Data in the
                    Data dictionary, it is set to 'Data'. 
    '''
    def __init__(self,Dates,Data,Date_Format=None,var='Data'):
        '''
        '''
        # ----------------
        # Error Managment
        # ----------------
        if not(isinstance(Dates,np.ndarray)) and not(isinstance(Dates,list)):
            self.ShowError('__init__','GeoTimeSeries','Dates has to be on a list or a ndArray')
        if not(isinstance(Data,dict)):
            self.ShowError('__init__','GeoTimeSeries','Data has to be a dictionary')
        # ----------------
        # Constants
        # ----------------
        self.Var = var
        self._operations = {'mean':np.nanmean,'sum':np.nansum}
        # ----------------
        # Dates
        # ----------------
        self.Dates = DatesC(Dates,Date_Format=Date_Format)
        # ----------------
        # Data
        # ----------------
        self.Data = Data
        return
    
    def MCalc(self,oper='mean'):
        '''
        DESCRIPTION:

            This method calculates the Monthly data from the daily data of 
            the different map fields.
        _______________________________________________________________________
        INPUT:
            :param oper: a str, String with the operation that is requiered to
                         calculate the monthly means, for now the operation
                         could be: 
                         
                                 String | Operation
                                        |
                                 'mean' | np.nanmean
                                 'sum'  | np.nansum

                         If more operations need to be added use the AddOper
                         method.

            :param var: a str, name of the variable.

        _______________________________________________________________________
        OUTPUT:
           :return Data: A dict, Dict with the Monthly Data.
        '''
        # ----------------
        # Error Managment
        # ----------------
        if not(isinstance(oper,str)):
            self.ShowError('MCalc','GeoTimeSeries','Dates has to be on a list or a ndArray')
        try: 
            Oper = self._operations[oper.lower()]
        except KeyError:
            self.ShowError('MCalc','GeoTimeSeries','oper not added in the _operations attribute, try adding the operation with AddOper')
        # ----------------
        # Data
        # ----------------
        var = self.Var
        T = self.Data[var]
        # Dialy Dates vector
        Ai = self.Dates.datetime[0].year
        Af = self.Dates.datetime[-1].year

        DatesM = []
        for i in range(Ai,Af+1):
            for j in range(1,13):
                DatesM.append(date(i,j,1))

        DatesM = np.array(DatesM)
        DatesDD = np.array([i.strftime('%Y/%m/') for i in DatesM])
        Dates2 = np.array([i.strftime('%Y/%m/') for i in self.Dates.datetime])

        TM = np.empty((len(DatesM),T.shape[1],T.shape[2]))*np.nan
        TmaxM = np.empty((len(DatesM),T.shape[1],T.shape[2]))*np.nan
        TminM = np.empty((len(DatesM),T.shape[1],T.shape[2]))*np.nan

        # Monthly calculations
        for ii,i in enumerate(DatesDD):
            # Dates to calculate
            x = np.where(Dates2 == i)[0]
            if len(x) != 0:
                TM[ii,:,:] = Oper(T[x,:,:],axis=0)
                TmaxM[ii,:,:] = np.nanmax(T[x,:,:],axis=0)
                TminM[ii,:,:] = np.nanmin(T[x,:,:],axis=0)

        self.Data['DatesM'] = DUtil.Dates_datetime2str(DatesM,Date_Format='%Y/%m')
        self.Data[var+'M'] = TM
        self.Data[var+'maxM'] = TmaxM
        self.Data[var+'minM'] = TminM
        return 

    def AnnualCycle(self,oper='mean'):
        '''
        DESCRIPTION:

            This method calculates the annual cycle of the map. 
        _______________________________________________________________________
        INPUT:
        _______________________________________________________________________
        OUTPUT:
           :return DataM: A dict, Dict with the Annual Cycle of the data.
        '''
        # ----------------
        # Constans
        # ----------------
        var = self.Var
        try: 
            Oper = self._operations[oper.lower()]
        except KeyError:
            self.ShowError('MCalc','GeoTimeSeries','oper not added in the _operations attribute, try adding the operation with AddOper')
        # ----------------------
        # Monthly Calculations
        # ----------------------
        try:
            Data = self.Data[var+'M']
        except KeyError:
            self.MCalc(oper=oper)
            Data = self.Data[var+'M']
        # ----------------------
        # Cycle Calculations
        # ----------------------
        DM = np.reshape(Data,(-1,12,Data.shape[1],Data.shape[2]))
        # Annual Cycle
        DMM = np.nanmean(DM,axis=0)
        # Annual Agreggation
        AM = Oper(DM,axis=1)
        # Muliannual Mean
        AMM = np.nanmean(AM,axis=0)

        self.Data[var+'MM'] = DMM
        self.Data[var+'AM'] = AM
        self.Data[var+'AMM'] = AMM
        return 

    def AddOper(self,stroper,oper):
        '''
        DESCRIPTION:

            This method adds new operations in the _operations attribute.
        _______________________________________________________________________
        INPUT:
            :param stroper: a Str, Key to call the operation.
            :param oper:    a function, function to do the operation, it is 
                            preferable to use the functions of numpy or 
                            some function that can make matrix calculus.

        _______________________________________________________________________
        OUTPUT:
           :return DataM: A dict, Dict with the latitude and longitude 
                         projected.
        '''
        self._operation[stroper] = oper
        return

    def SaveMat(self,Name='Results',Pathout=''):
        '''
        DESCRIPTION:

            This method saves the data in a .mat format file.
        _______________________________________________________________________
        INPUT:
            :param Name:    a str, Name of the file.
            :param Pathout: a str, Path to save the file.
        _______________________________________________________________________
        OUTPUT:
           :return: File in .mat format.
        '''
        utl.CrFolder(Pathout)
        Data = self.Data
        Data['DatesC'] = self.Dates.str
        Nameout = Pathout + Name + '.mat'
        sio.savemat(Nameout,self.Data)
        return

    def __str__(self):
        '''
        '''
        a = '\nDATA INFORMATION:'
        b = '\n DATES ADDED FROM: '+self.Dates.str[0] +' TO '+ self.Dates.str[-1]
        c = '\n SHAPE OF THE DATA ADDED: '+str(self.Data['Data'].shape)
        return a+b+c

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


