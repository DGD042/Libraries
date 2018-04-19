# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 31/08/2017
#______________________________________________________________________________
#______________________________________________________________________________
# ------------------------
# Importing Modules
# ------------------------ 
# Manipulate Data
import numpy as np
import scipy.io as sio
from datetime import date, datetime, timedelta
# Open and saving data
import csv
import xlrd
import xlsxwriter as xlsxwl
# System
import sys
import os
import glob as gl
import re
import warnings
import platform

# ------------------
# Personal Modules
# ------------------
# Importing Modules
from Utilities import Utilities as utl
from BPumpL.Dates import DatesFunctions as DUtil; 

# ------------------------
# Functions
# ------------------------

'''

    This Package manages the data inside, like clip data and other 
    functions.
_________________________________________________________________________

'''

# Carga de información
def ClipData(DateSt,DateEnd,DateVar,Var):
    '''
    DESCRIPTION:
    
        This Function clips the Variable in the segment given by a starting
        date and a ending date.
    _________________________________________________________________________

    INPUT:
        :param DateSt: A datetime, Starting Date.
        :param DateEnd: A datetime, Ending Date.
        :param Var: A datetime ndarray, Vector with dates.
        :param Var: A ndarray, Vector.
    _________________________________________________________________________
    
    OUTPUT:
        :return ClipData: A ndarray, Cliped vector.
    '''
    # ---------------------
    # Finding Dates
    # ---------------------
    xi = np.where(DateVar == DateSt)[0]
    if len(xi) == 0:
        raise ValueError('No DateSt found in DateVar')
    xe = np.where(DateVar == DateEnd)[0]
    if len(xi) == 0:
        raise ValueError('No DateEnd found in DateVar')
    # ---------------------
    # Clipping
    # ---------------------
    ClipVar = Var[xi:xe+1]
    return ClipVar
