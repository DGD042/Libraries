# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 18/04/2018
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
from BPumpL.Dates import DatesFunctions as DUtil

# ------------------------
# Functions
# ------------------------

'''

    
    This package contains the functions to manipulate geographical
    data.
_________________________________________________________________________

'''

# Carga de información
def Degree2km(Lat1,Lon1,Lat2,Lon2):
    '''
    DESCRIPTION:

        Calculate the distance from the latitude and longitude.
    _________________________________________________________________________

    INPUT:
        :param Lat1: A float, Latitude of the point 1.
        :param Lon1: A float, Longitude of the point 1.
        :param Lat2: A float, Latitude of the point 2.
        :param Lon2: A float, Longitude of the point 2.
    _________________________________________________________________________
    
    OUTPUT:

        :return Distance: A float, Distance between the 2 points.
    '''
    # ---------------------
    # Constants
    # ---------------------
    # Earth Radius
    rE = 6371.0 # km

    # ---------------------
    # Distance in radians
    # ---------------------
    x = Lat1-Lat2
    y = Lon1-Lon2
    DD = np.sqrt(x**2+y**2)
    DR = (np.pi/180)*DD
    # ---------------------
    # Distance in Km
    # ---------------------
    Distance = rE*DR

    return Distance

