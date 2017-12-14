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
from Utilities import Utilities as utl
from Hydro_Analysis import Hydro_Plotter as HyPl; HyPl=HyPl()
class KrigingClass(object):
    '''
    DESCRIPTION:
        This class develops the interpolations.
    _________________________________________________________________
    INPUT:
        :param Data:  A dict, a dictionary with the data, must have:
                      'Data': A ndArray, with the Data
                      'lonigute': Longitude.
                      'Latitude': Longitude.
    '''
    def __init__(self,Data):
        '''
        '''
        # ----------------
        # Error Managment
        # ----------------
        if not(isinstance(Data,dict)):
            self.ShowError('__init__','KrigingClass','Data has to be a dictionary')
        # ----------------
        # Constants
        # ----------------
        self.Data = Data
        return
    
    def Histogram(self,Bins=30,PathImg='',Name='',Var=''):
        '''
        DESCRIPTION:

            This method plots the histogram of the Data.
        _______________________________________________________________________
        INPUT:
        _______________________________________________________________________
        OUTPUT:
        '''
        Data = self.Data['Data']
        HyPl.HistogramNP(Data,Bins,Title='Histograma',Var=Var,Name=Name,PathImg=PathImg,
                M='porcen',FEn=False,Left=True,FlagHour=False,flagEst=False,
                FlagTitle=False,FlagBig=False,vmax=None,FlagMonths=False)
        return

    def Distribution_Data(self,PathImg='',Name=''):
        '''
        DESCRIPTION:

            This method plots the spatial distribution of the Data.
        _______________________________________________________________________
        INPUT:
        _______________________________________________________________________
        OUTPUT:
        '''
        V1 = self.Data['Longitude']
        V2 = self.Data['Latitude']
        HyPl.ScatterGen(V1,V2,Fit='',Title='Distribución Espacial',
                xLabel='Longitud',yLabel='Latitude',Name=Name,
                PathImg=PathImg,FlagA=False,FlagAn=False,FlagInv=False,
                FlagInvAxis=False,Annotations=None)
        return

    def SVh( P, h, bw ):
        '''
        DESCRIPTION:

            Experimental semivariogram for a single lag
        _______________________________________________________________________
        INPUT:
        _______________________________________________________________________
        OUTPUT:
        '''
        pd = squareform( pdist( P[:,:2] ) )
        N = pd.shape[0]
        Z = list()
        for i in range(N):
            for j in range(i+1,N):
                if( pd[i,j] >= h-bw )and( pd[i,j] <= h+bw ):
                    Z.append( ( P[i,2] - P[j,2] )**2.0 )
        return np.sum( Z ) / ( 2.0 * len( Z ) )
     
    def SV( P, hs, bw ):
        '''
        Experimental variogram for a collection of lags
        '''
        sv = list()
        for h in hs:
            sv.append( SVh( P, h, bw ) )
        sv = [ [ hs[i], sv[i] ] for i in range( len( hs ) ) if sv[i] > 0 ]
        return np.array( sv ).T


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


