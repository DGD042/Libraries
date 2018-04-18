# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 31/10/2017
#______________________________________________________________________________
#______________________________________________________________________________

# --------------------
# Importing Packages
# --------------------
# Data managment
import numpy as np
import scipy.io as sio 
from scipy import stats as st 
# Documents
import xlsxwriter as xlsxwl
from xlsxwriter.utility import xl_range_abs
# Graphics
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates 
import matplotlib.mlab as mlab
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from mpl_toolkits.mplot3d import Axes3D
# Time
from datetime import date, datetime, timedelta
import time
# Sistem
import os
import sys
import warnings

# ------------------
# Personal Modules
# ------------------
# Importing Modules
from Utilities import Utilities as utl
from Utilities import DatesUtil as DUtil; DUtil=DUtil()
from Utilities import Data_Man as DM
from AnET.CFitting import CFitting as CF;
from EMSD.Dates.DatesC import DatesC
from Hydro_Analysis.Models.Evaporation import EvapFunctions as EvapM

class Model_EvapCover(object):
    def __init__(self)
        '''
        DESCRIPTION:
            
            This model calculates the evaporation with and without
            cover from a water body. Using the methodology described
            by Cielter & Terre (2015).

        _________________________________________________________________
        INPUT:
            :param Lat:        A float, Latitude of the region.
        _________________________________________________________________

        '''
        # ------------------
        # Error Managment
        # ------------------


        # ------------------
        # Atributes
        # ------------------
        # Latent heath of vaporization
        self._Lambda = 2.26 # MJ/kg

