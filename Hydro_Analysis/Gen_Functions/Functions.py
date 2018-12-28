# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 11/03/2016
#______________________________________________________________________________
#______________________________________________________________________________

'''

This are general functions used to make different statistics calculus
inside the Hydro_Analysis Package.
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

# ---------------
# Functions
# ---------------
def MeanError(VM,axis=0):
    '''
    DESCRIPTION:
        
        Function that calculates the mean error of a set of data.
    _____________________________________________________________________
    INPUT:
        :param VM:   A ndarray, Array with values.
        :param axis: A int, Number of the axis to make the calculations.
    _____________________________________________________________________
    OUTPUT:
        :retun MM: A ndarray, Vector with the mean values.
        :retun MD: A ndarray, Vector with the std of the values.
        :retun ME: A ndarray, Vector with the mean errors of the values.
    '''
    # Mean
    MM = np.nanmean(VM,axis=axis)
    # Std
    MD = np.nanstd(VM,axis=axis)
    if axis == 0:
        # Data number
        DataN = np.array([sum(~np.isnan(VM[:,j])) for j in range(len(VM[0]))])
        # Error
        ME = np.array([MD[j]/(np.sqrt(DataN[j])) for j in range(len(MD))])
    elif axis == 1:
        # Data number
        DataN = np.array([sum(~np.isnan(VM[j,:])) for j in range(len(VM))])
        # Error
        ME = np.array([MD[j]/(np.sqrt(DataN[j])) for j in range(len(MD))])
    
    return MM, MD, ME

def PrecPor(VM):
    '''
    DESCRIPTION:
        
        Function that calculates the porcentage of precipitation in  
        hour time scale.
    _____________________________________________________________________
    INPUT:
        :param VM: A ndarray, Array with values.
    _____________________________________________________________________
    OUTPUT:
        :retun VP: A ndarray, Array with the percentage values.
    '''
    VP = np.empty(VM.shape) * np.nan
    for i in range(len(VM)):
        VS = np.nansum(VM[i])
        for j in range(len(VM[i])):
            VP[i,j] = VM[i,j]/VS
    return VP

def FreqPrec(Prec):
    '''
    DESCRIPTION:
        
        Function that calculates the frequency of precipitation from
        a series of intensities or a series of total prec.
    _____________________________________________________________________
    INPUT:
        :param Prec: A ndarray, Array with values.
    _____________________________________________________________________
    OUTPUT:
        :retun R: A dict, Dictionary with the following keys:
                  argsorted values:   'argsort'
                  Percentiles values: 'percentiles'
                  prec sorted values: 'precsort'
    '''
    x = np.argsort(Prec)
    xx = np.arange(1,Prec.shape[0]+1)/Prec.shape[0]*100
    return {'argsort':x,'percentiles':xx,'precsort':Prec[x]}

