# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 05/10/2017
#______________________________________________________________________________
#______________________________________________________________________________
#
# DESCRIPTION:
#   This Functions allows to make several calculus relation with radiation
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
    
# -----------
# Functions
# -----------
def EccCorAl(J,Method=0):
    '''
    DESCRIPTION:
    
        This function calculates the eccentricity correction factor, 
        formulas in Almorox(2004).
    _________________________________________________________________________
    OUTPUT:
        :return E0: A float, eccentricity correction factor.

    '''
    if Method == 0:
        Gamma = 2*np.pi*(J-1)/365
        E0 = 1.00011 + 0.034221* np.cos(Gamma)+0.00128*np.sin(Gamma)\
            +0.000719*np.cos(2*Gamma)+0.000077*np.sin(2*Gamma)
    return E0

def DeclFacAl(J,Method=0):
    '''
    DESCRIPTION:
    
        This function calculates the solar declination 
        formulas in Almorox(2004).
    _________________________________________________________________________
    OUTPUT:
        :return Inc: A float, solar declination in radians.
    '''
    if Method == 0:
        Gamma = 2*np.pi*(J-1)/365
        Inc = (180/np.pi)*(0.006918-0.399912*np.cos(Gamma)+0.070257*np.sin(Gamma)
                -0.006758*np.cos(2*Gamma)+0.000907*np.sin(2*Gamma)
                -0.002697*np.cos(3*Gamma)+0.00148*np.sin(3*Gamma))

    return np.deg2rad(Inc)

def WsAl(Lat,J,Method=0):
    '''
    DESCRIPTION:
    
        This function calculates the geometric mean sunrise hour angle
        formulas in Almorox(2004). 
    _________________________________________________________________________
    OUTPUT:
        :return Ws: A float, Sunrise hour angle in radians.

    '''
    LatR = np.deg2rad(Lat)
    Inc = DeclFacAl(J,Method=Method)
    if Method == 0:
        Ws = np.arccos((-np.sin(LatR)*np.sin(Inc))/(np.cos(LatR)*np.cos(Inc)))

    return Ws

def ExRad2(Lat,J):
    '''
    DESCRIPTION:
    
        This function calculates the daily extraterrestial solar radiation, 
        the equations are detailed by Almorox et al. (2013).
    _________________________________________________________________________

    INPUT:
        :param Lat: a float, Latitude degrees.
        :param J:   an int, Day of the year.
    _________________________________________________________________________
    
    OUTPUT:
        :return H0: A float, Daily extraterrestial radiation in MJ/m^2/day.
    '''
    # We pass the latitude to radians
    LatR = np.deg2rad(Lat)
    # Eccentricity correction factor of the Earths Orbit
    E0 = EccCorAl(J)
    # Solar declination angle
    Inc = DeclFacAl(J)
    # Sunset hour angle
    Ws = WsAl(Lat,J)
    # Extraterrestial Radiation
    H0 = (1/np.pi)*118.108*E0*(np.cos(LatR)*np.cos(Inc)*np.sin(Ws)+(np.pi/180)\
        *np.sin(LatR)*np.sin(Inc)*Ws)

    return H0

def DaylenH(Lat,J):
    '''
    DESCRIPTION:
    
        This function calculates the daylength in hours, the equations are 
        detailed by Almorox et al. (2004).
    _________________________________________________________________________

    INPUT:
        :param Lat: a float, Latitude degrees.
        :param J:   an int, Day of the year.
    _________________________________________________________________________
    
    OUTPUT:
        :return N: A float, Daylength in hours.
    '''
    # We pass the latitude to radians
    LatR = np.deg2rad(Lat)
    # Eccentricity correction factor of the Earths Orbit
    E0 = EccCorAl(J)
    # Solar declination angle
    Inc = DeclFacAl(J)
    # Sunset hour angle
    Ws = WsAl(Lat,J); Ws = np.rad2deg(Ws)
    # Daylength in hours
    N = Ws/7.5

    return N

def ExRad(Lat,J):
    '''
    DESCRIPTION:
    
        This function calculates the daily extraterrestial solar radiation, 
        the equations are detailed by Allen et al. (1998).
    _________________________________________________________________________

    INPUT:
        :param Lat: a float, Latitude degrees.
        :param J:   an int, Day of the year.
    _________________________________________________________________________
    
    OUTPUT:
        :return R0: A float, Daily extraterrestial radiation.
    '''
    # We pass the latitude to radians
    LatR = Lat*(np.pi/180)
    # Relative distance between the sun and the earth.
    dr = 1+0.033*np.cos((2*np.pi/365)*J) 
    # Solar declination angle
    Inc = 0.4093*np.sin((2*np.pi/365)*J-1.39)
    # Sunset hour angle
    Ws = np.arccos(-np.tan(LatR)*np.tan(Inc))
    # Extraterrestial Radiation
    R0 = 37.6*dr*((Ws*np.sin(LatR)*np.sin(Inc))+(np.cos(LatR)*np.cos(Inc)*np.sin(Ws)))

    return R0

def EqLi(Method,Tmax,Tmin,Lat,J):
    '''
        DESCRIPTION:
    
    This function calculates the radiation from temperature.
    _________________________________________________________________________

        INPUT:
    + Method: Equation.
    + Tmax: Daily maximum temperature.
    + Tmin: Daily minimum temperature.
    _________________________________________________________________________
    
        OUTPUT:
    - R: Daily radiation.
    '''
    # Constants
    a = [0,0,0,[0.043,-0.04],0.221]
    b = [0,0,0,-0.072,-0.282]

    R0 = self.ExRad(Lat,J)

    if Method == 4:
        R = R0*(b[3] + a[3][0]*Tmax+a[3][1]*Tmin)
    elif Method == 5:
        R = R0*(b[4]+a[4]*(Tmax-Tmin)**(0.5))

    return R

def EqAlmorox(Method,Tmax,Tmin,Lat,J,Z=0,Tmin2=0,dTM=0,FlagRea=False):
    '''
        DESCRIPTION:
    
    This function calculates the radiation from temperature and other variables
    using the models described in Almorox et al. (2013).

    Constants Kr, A and C must be changed depending on the region of study,
    for more information visit the article  Almorox et al. (2013).
    _________________________________________________________________________

        INPUT:
    + Method: Equation.
    + Tmax: Daily maximum temperature.
    + Tmin: Daily minimum temperature.
    + Z: Altitude, needed for Method 2.
    + Tmin: Daily minimum temperature of the next day, necessary for Method 4.
    + dTM: Monthly average temperature delta, for Method 4.
    _________________________________________________________________________
    
        OUTPUT:
    - Hc: Daily radiation.
    '''

    H0 = self.ExRad2(Lat,J)

    if Method == 1:
        Kr = 0.1463 # For interior regions
        Hc = H0 * (Kr*(Tmax-Tmin)**(1/2))
    elif Method == 2:
        P = self.PHeq(Z)
        A = 0.1486 # For interior regions
        Hc = H0 * (A * (P/1013)**(1/2)*(Tmax-Tmin)**(1/2)) 
    elif Method == 4:
        if FlagRea:
            A = 1.5
            C = 1.8
        else:
            A = 0.85
            C = 1.2
        try:
            q = len(Tmin2)
            qq = sum(sum(~np.isnan(Tmin2)))
            if qq == 0:
                dT = Tmax-Tmin
            else:
                dT = Tmax-0.5*(Tmin+Tmin2)
            x = np.where(dT < 0)
            if len(x) != 0:
                dT = Tmax-Tmin
        except TypeError:
            if np.isnan(Tmin2):
                dT = Tmax-Tmin
            else:
                dT = Tmax-0.5*(Tmin+Tmin2)
            if dT < 0:
                dT = Tmax-Tmin
        B = 0.036*np.exp(-0.154*dTM)
        Hc = H0 * A * (1-np.exp(-B*dT**C))
        # if np.isnan(Hc):
        #   print('Tmax',Tmax)
        #   print('Tmin',Tmin)
        #   print('Tmin2',Tmin2)
        #   print('dT',dT)
        #   print('dTM',dTM)
        #   print('B',B)
        #   print('Hc',Hc)
        #   print('----')
            

    return Hc

def AngstromPrescottEq(H0,N,n,flagRad=False,a=None,b=None):
    '''
    DESCRIPTION:
    
        This function contains the Angstrom-Prescott model to calculate the
        regresion from the relation of radiation and the relation of the 
        day length.
    _________________________________________________________________________

    INPUT:
        :param H0: a ndArray, Extraterrestial Radiation in MJ/m^2/day.
        :param H:  a ndArray, Surface Radiation in MJ/m^2/day.
        :param N:  a ndArray, Day length in hours.
        :param n:  a ndArray, sunshine duration in hours.
    _________________________________________________________________________
    
    OUTPUT:
        :return Eq: Equation or result of data.
    '''
    if flagRad:
        return (a+(b*(n/N)))*H0
    else:
        return c+d*(n/N)

