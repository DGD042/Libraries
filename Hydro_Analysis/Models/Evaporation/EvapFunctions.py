# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 26/02/2018
#______________________________________________________________________________
#______________________________________________________________________________
#
# DESCRIPTION:
#   This Functions allows to make several calculus relation with evaporation.
#______________________________________________________________________________

# Data Managment
import numpy as np
import scipy.io as sio 
from scipy import stats as st 
import matplotlib.pyplot as plt
# Dates Managment
from datetime import date, datetime, timedelta
import time
# System Managment
import sys
import os
import warnings


# Personal Libraries
try:
    from Hydro_Analysis.Models.Atmos_Thermo import Thermo_Fun as TF
    from Hydro_Analysis.Models.Radiation import RadFunctions as RF
except ImportError:
    from Atmos_Thermo import Thermo_Fun as TF
    from Radiation import RadFunctions as RF
    
# -----------
# Functions
# -----------
def Calc_Model_PenmanMonteith(Ta,P,A,v10,Alt,Lat,J,HR,Tp):
    '''
    DESCRIPTION:
    
        This function calculates the evaporation with the Penman-Monteith 
        equation.
    _________________________________________________________________________
    INPUT:
        :param Ta: A float or ndarray, ambient temperature.
        :param P: A float or ndarray, Atmospheric Pressure in kPa. 
        :param A: A float, Thank cross sectional area in km^2.
        :param v10: A float or ndarray, Wind velocity at 10 m height in m/s.
        :param Alt: A float, Water body atitude.
        :param Lat: A float, Latitude.
        :param J: A int, Julian day of the year.
        :param HR: A float or ndarray, Relative Humidity.
    _________________________________________________________________________
    OUTPUT:
        :return E: A float, eccentricity correction factor.

    '''
    # -----------
    # Constants
    # -----------
    SBParam = 4.903*10**(-9) # MJ/m^2K^4d
    # -------------
    # Parameters
    # -------------
    # Heath Vaporization
    Lambda = TF.HeatVaporization(Ta)
    # Psychometric Constant
    gamma = (P*cp)(/(0.622*Lambda))
    # Aerodynamic resistance
    fu = (5/A)**(0.05)*(3.80+1.57*v10)
    ra = (da*cp)/(Gamma*fu/86400)
    # Net Radiation
    KET = RF.ExRad2(Lat,J)
    Kclear = KET*(0.75+(2*10**(-5)*Alt))
    Kratio = Ki/Kclear
    if Kratio > 0.9:
        Cf = 2*(1-Kratio)
    elif Kratio <= 0.9:
        Cf = 1.1-Kratio
    Li = SBParam*(Cf+(1-Cf)*(1-(0.261*np.exp(-7.77*10**(-4)*Ta**2))))*(Ta+273.15)**4
    Lo = 0.97*SBParam*(Tp+273.15)**4
    Q = KET*(1-alpha)+Li-Lo
    # Adittional Temperatures
    ea = (HR)*np.exp((17.27*Ta)/(Ta+237.3)) # Ambient Vapor Partial Pressure (kPa)
    td = (116.9+(237.3*np.log(ea)))/(16.78-np.log(ea)) # Dew temperature (°C)
    Tn = (0.066*Ta+(4098*ea/(Ts+237.3)**2)*Td)/(0.066+(4098*ea/(Ts+237.3)**2)*Td) # Wet Bulb Temperature (°C)
    # Change in stored heat
    N = di*cpi*Z*(Tp-Tp0)
    # Saturation vapor curve slope
    pv = 0.6108*np.exp(17.27*Tp/(Tp+237.3))
    ds = (4098*pv)/(Tp*237.3)**2

    E = (1/Lambda)*((ds*(Q-N)+(86400*da*cp*(es-ea)/ra))/(ds+gamma))

    return E
