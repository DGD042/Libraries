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
from Utilities import DatesUtil as DUtil; DUtil=DUtil()
from Utilities import Data_Man as DM
from AnET.CFitting import CFitting as CF;CF=CF()
from EMSD.Dates.DatesC import DatesC
from Hydro_Analysis.Models.Radiation import RadFunctions as RadM



class Model_AngstromPrescott(object):
    '''
    DESCRIPTION:

        This class transforms the information of sunshine duration
        to radiation using the Angostrom-Prescott model described in
        Almorox, Benito, Hontoria (2004) "Estimation of monthly
        Angstrom-Prescott equation coefficient from mesured daily
        data in Todelo, Spain".

        The model uses the equation:

        H/H0 = a+b(n/N),

        where H is the global radiation, H0 is global extraterrestial
        radiation, n is the sunshine duration and N is the maximum
        daily sunshine duration. a and b are parameters that can be 
        adjusted having the information of Sunshine duration and 
        Radiation.

    _________________________________________________________________
    INPUT:
        :param Lat:        A float, Latitude of the region.
        :param Dates:      A list or ndArray, Vector with datetime 
                           or string dates.
        :param SD:         A list or ndArray, 
                           Vector with Sunshine duration data in 
                           hours.
        :param Rad:        A list or ndArray, Vector with Radiation 
                           data in MJ/m^2/day.
        :param Parameters: A dict, Dictionary with parameters of 
                           the equation needed to adjust new series.
                           a and b are the parameters of the 
                           equation.
    _________________________________________________________________

    '''

    def __init__(self,Lat,Dates,SD,Rad=None,Parameters=None):
        '''
        DESCRIPTION:

            This class transforms the information of sunshine duration
            to radiation using the Angostrom-Prescott model described in
            Almorox, Benito, Hontoria (2004) "Estimation of monthly
            Angstrom-Prescott equation coefficient from mesured daily
            data in Todelo, Spain".

            The model uses the equation:

            H/H0 = a+b(n/N),

            where H is the global radiation, H0 is global extraterrestial
            radiation, n is the sunshine duration and N is the maximum
            daily sunshine duration. a and b are parameters that can be 
            adjusted having the information of Sunshine duration and 
            Radiation.

        _________________________________________________________________
        INPUT:
            :param Lat:        A float, Latitude of the region.
            :param Dates:      A list or ndArray, Vector with datetime 
                               or string dates.
            :param SD:         A list or ndArray, 
                               Vector with Sunshine duration data in 
                               hours.
            :param Rad:        A list or ndArray, Vector with Radiation 
                               data in MJ/m^2/day.
            :param Parameters: A dict, Dictionary with parameters of 
                               the equation needed to adjust new series.
                               a and b are the parameters of the 
                               equation.
        _________________________________________________________________

        '''
        # ------------------
        # Error Managment
        # ------------------
        try:
            assert isinstance(Dates[0],date) or isinstance(Dates[0],str)
        except AssertionError:
            self.Error('Model_AngstromPrescott','__init__','Wrong date format')
        try:
            assert len(Dates) == len(SD)
        except AssertionError:
            self.Error('Model_AngstromPrescott','__init__','Data and Dates must have the same lenght')
        try:
            assert not(Rad is None) or not(Parameters is None)
        except AssertionError:
            self.Error('Model_AngstromPrescott','__init__','Rad and Parameters are None, at least one has to have data')
        # ------------------
        # Parameters
        # ------------------
        # Dates
        Dates = DatesC(Dates)
        # Calculate Julian Day
        self.Julian_Day = np.array([i.timetuple().tm_yday for i in Dates.datetime])
        # Latitude
        self.Lat = Lat
        # Parameters
        self.Parameters = Parameters
        # ---------------------------
        # Calculations
        # ---------------------------
        self.Data = dict()
        self.Data['H0'],self.Data['N'] = self.CalcRadN()
        self.Data['H'] = Rad
        self.Data['n'] = SD

        return

    def CalcRadN(self,Lat=None,J=None):
        '''
        DESCRIPTION:

            This method calculates the extraterrestial radiation in the
            hole year and the day lenght in hours.
        _________________________________________________________________
        INPUT:
            :param Lat:        A float, Latitude of the region.
            :param J:          An int, Julian Day.
        _________________________________________________________________
        OUTPUT:
            :return H0: Extraterrestial radiation in a definde latitude.
            :return N: Day length in hours.

        '''
        if Lat is None:
            Lat = self.Lat
        if J is None:
            J = self.Julian_Day

        H0 = np.empty(len(J))
        N = np.empty(len(J))
        H0 = RadM.ExRad2(Lat,J)
        N = RadM.DaylenH(Lat,J)

        return H0,N

    def AdjustModel(self):
        '''
        DESCRIPTION:

            This method Adjusts the model to calculate the parameters
        _________________________________________________________________

        '''
        fun = RadM.AngstromPrescottEq(self.Data['H0'],self.Data['n'],self.Data['N'])
        CF.addfun(fun,key='AnPrEq',funeq=r'$H/H0 = %.4f + %.4f(n/N)')
        RelN = self.Data['n']/self.Data['N']
        RelRad = self.Data['H']/self.Data['H0']
        self.R = CF.FF(RelN,RelRad,F='AnPrEq')
        return

    def Error(self,Cl,method,msg):
        raise Exception('ERROR: in class <'+Cl+'> in method <'+method+'>\n'+msg)

    def __str__(self):
        L1 = '\nAngostrom-Prescott Model'
        if self.Parameters is None:
            L2 = '\nNo parameters added or calculated yet'
        else:
            L2 = '\nParameters: '+str(self.Parameters)
        return L1+L2

