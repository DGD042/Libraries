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
    def __init__(self,Rn,U,Ta,RH,Pa):
        '''
        DESCRIPTION:
            
            This model calculates the evaporation with and without
            cover from a water body. Using the methodology described
            by Cielter & Terre (2015).
        _________________________________________________________________
        INPUT:
            :param Rn: A ndarray, Solar Radiation MJ/m^2/day.
            :param U: A ndarray, Wind Velocity at 10 m m/s.
            :param Ta: A ndarray, Air Temperature at 2 m K.
            :param Pa: A ndarray, Pressure kPa.
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

        # Variables
        self._Rn = Rn
        self._U = U
        self._Ta = Ta
        self._RH = RH
        self._Pa = Pa

        return

    def CalculateEvaporation(self,A,CoverPer=0,PerRad=0.8):
        '''
        DESCRIPTION:
            
            This model calculates the evaporation with and without
            cover from a water body. Using the methodology described
            by Cielter & Terre (2015).

        _________________________________________________________________
        INPUT:
            :param A: A float, Area of the water body in m^2.
            :param CoverPer: A float, Percentage of cover in the water 
                             body goes between 0 and 1.
            :param PerRad: A float, Percentage of deficit in water.
        _________________________________________________________________
        OUTPUT:
            :return E: A ndarray, Evaporation in m^3/day.
        '''
        rho = EvapM.PsicometricC(self._Pa,self._Lambda)
        es = EvapM.esEq(self._Ta)
        m = EvapM.mEq(self._Ta,es)
        DeltaE = EvapM.DeltaEEq(self._RH,self._Ta,es)
        self.E = EvapM.Penman_Shuttleworth(self._Rn,self._U,m,rho,DeltaE,self._Lambda)

        # Area Cover
        self.E = self.E/1000 # m^3/m^2/day
        self.ET = self.E*A # m^3/day
        self.EC = self.E*A*(1-CoverPer) # m^3/day
        self.EC = self.EC + ((1-PerRad)*self.E*(A*CoverPer)) # m^3/day
        return


