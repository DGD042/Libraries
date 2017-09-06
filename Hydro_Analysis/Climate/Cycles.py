# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 11/03/2016
#______________________________________________________________________________
#______________________________________________________________________________

'''


 CLASS DESCRIPTION:
   This class have different routines for hydrological analysis. 

   This class do not use Pandas in any function, it uses directories and save
   several images in different folders. It is important to include the path 
   to save the images.
   
______________________________________________________________________________
'''
# ------------------------
# Importing Modules
# ------------------------ 
# Manipulate Data
import numpy as np
import scipy.io as sio
from scipy import stats as st
from datetime import date, datetime, timedelta
import time
# Graph
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates
import matplotlib.mlab as mlab
# System
import sys
import os
import glob as gl
import re
import operator
import warnings

# ------------------
# Personal Modules
# ------------------
# Importing Modules
from Utilities import Utilities as utl
from Utilities import Data_Man as DM
from Hydro_Analysis.Hydro_Plotter import Hydro_Plotter as HyPl; HyPl=HyPl()
from Utilities import DatesUtil as DUtil; DUtil=DUtil()
from Hydro_Analysis.Gen_Functions.Functions import *
from Hydro_Analysis.Meteo import Cycles as MCy

def CiclA(VMes,Years,flagA=False,oper='mean'):
    '''
    DESCRIPTION:
        This function calculates the annual cycle of a variable, also
        calculates the annual series of requiered.

        Additionally, this function can makes graphs of the annual cycle 
        and the annual series if asked.
    _______________________________________________________________________

    INPUT:
        :param VMes:  A ndarray, Variable with the monthly data.
        :param Years: A lisr or ndarray, Vector with the initial and 
                                         final year.
        :param flagA: A boolean, flag to know if the annual series 
                                 is requiered.
        :param oper:  A ndarray, Operation for the annual data.
    _______________________________________________________________________
    
    OUTPUT:
        This function returns a direcory with all the data.
    '''

    # --------------------
    # Error Managment
    # --------------------
    if len(Years) > 2:
        return utl.ShowError('CiclA','Hydro_Analysis','Years index vector larger than 2, review vector')
    
    # --------------------
    # Years Managment
    # --------------------
    Yi = int(Years[0])
    Yf = int(Years[1])
    VarM = np.reshape(VMes,(-1,12))
    # --------------------
    # Annual Cycle 
    # --------------------
    # Verify NaN data from the cycle
    MesM = np.empty(12)
    VarMNT = []
    for i in range(12):
        q = sum(~np.isnan(VarM[:,i]))
        VarMNT.append(sum(~np.isnan(VarM[:,i])))
        if q <= len(VarM[:,i])*0.70:
            MesM[i] = np.nan
        else:
            MesM[i] = np.nanmean(VarM[:,i]) # Multianual Mean
    
    MesD = np.nanstd(VarM,axis=0) # annual strandard deviation.
    MesE = np.array([k/np.sqrt(VarMNT[ii]) for ii,k in enumerate(MesD)]) # annual Error
    # Graph
    if FlagG:
        HyPl.AnnualCycle(MesM,MesE,VarL,VarLL,Name,NameArch,PathImg,flagAH,color=C)
    # --------------------
    # Annual Series
    # --------------------
    if flagA:
        # Determine operation
        Operation = utl.Oper_Det(oper)
        # ----------------
        # Error managment
        # ----------------
        if Operation == -1:
            return -1
        # Calculations
        AnM = np.empty(VarM.shape[0])
        AnMNT = []
        for i in range(VarM.shape[0]):
            q = sum(~np.isnan(VarM[i,:]))
            if q <= len(VarM[i,:])*0.70:
                AnM[i] = np.nan
                AnMNT.append(np.nan)
            else:
                AnM[i] = Operation(VarM[i,:])
                AnMNT.append(q)

        AnD = np.nanstd(VarM,axis=1) # Annual deviation
        AnE = np.array([k/np.sqrt(AnMNT[ii]) for ii,k in enumerate(AnD)]) # Annual Error 

        x = [date(i,1,1) for i in range(Yi,Yf+1)]
        xx = [i for i in range(Yi,Yf+1)]

        if FlagG:
            HyPl.AnnualS(x,AnM,AnE,VarL,VarLL,Name,NameArch,PathImg+'Anual/',color=C)

    # Return values
    results = dict()

    if flagA:
        results['MesM'] = MesM 
        results['MesD'] = MesD 
        results['MesE'] = MesE 
        results['AnM'] = AnM
        results['AnD'] = AnD
        results['AnE'] = AnE
        return results
    else:
        results['MesM'] = MesM 
        results['MesD'] = MesD 
        results['MesE'] = MesE 
        return results



