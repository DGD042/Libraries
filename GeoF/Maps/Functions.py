# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 20/11/2017
#______________________________________________________________________________
#______________________________________________________________________________
# ------------------------
# Importing Modules
# ------------------------ 
# Manipulate Data
import numpy as np
import scipy.io as sio
from datetime import date, datetime, timedelta
# Graphs
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates
import matplotlib.mlab as mlab
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
import matplotlib.image as mpimg
from matplotlib import animation
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
from EMSD.Data_Man import Data_Man as DMan
from EMSD.Functions import Gen_Functions as GFun

# ------------------------
# Class
# ------------------------


def MapsShape_Raster(Raster,Shape=None,Var='Data',Name='Results',PathImg='',*args):
    '''
    DESCRIPTION:

        With this function the information of an netCDF type file can 
        be extracted.
    _______________________________________________________________________

    INPUT:
        :param Raster:         A str, File that would be extracted including 
                             the path.
        :param Shape:      A dict, List of variables that would be 
                             extracted from the netCDF file. Defaulted 
                             to None.
        :param Var: A dict, Range of data that would be extracted 
        :param Name: A dict, Range of data that would be extracted 
        :param PathImg: A dict, Range of data that would be extracted 
    _______________________________________________________________________
    
    OUTPUT:
        :return Data: A dict, Extracted Data Dictionary.    
    '''
    # ------------------
    # Error Managment
    # ------------------
    if not(isinstance(Raster,np.ndarray)):
        raise Exception('Raster must be a ndArray')

    # ------------------
    # Parameters
    # ------------------

    # Graph parameters
    fH = 30 
    fV = 20
    plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': 'Arial'\
            ,'xtick.labelsize': 15,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': 15,'ytick.major.size': 12,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})

    fig = plt.figure(figsize=DM.cm2inch(fH,fV))

    # Folder to save
    utl.CrFolder(PathImg)

    return






