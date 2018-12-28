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
# System
import sys
import os
import glob as gl
import re
import warnings

# ------------------
# Personal Modules
# ------------------
from EMSD.Data_Man import Data_Man as DM
# ------------------------
# Funciones
# ------------------------

def ClipSeries(DatesP,DataP,DatesS,DataS):
    '''
    _________________________________________________________________________
    
    DESCRIPTION:
        
        This functions clips data from 2 different series.
    _________________________________________________________________________

    INPUT:
        :param DatesP:  A ndArray, vector with Dates of first series.
        :param DataP:   A ndArray, vector with values of first series.
        :param DatesS:  A ndArray, vector with Dates of second series.
        :param DataS:   A ndArray, vector with values of second series.
    _________________________________________________________________________
    OUTPUT:
        :return Series: A ndarray, dictionary with the field data.
    '''
    # ----------------
    # Error Managment
    # ----------------
    assert isinstance(DatesP[0],datetime) or isinstance(DatesP[0],date)
    assert isinstance(DatesS[0],datetime) or isinstance(DatesS[0],date)

    Dif = DatesP[1]-DatesP[0]
    if Dif.days == 0 and Dif.seconds == 3600:
        TimeScale = 'Horarios'
    elif Dif.days == 1 and Dif.seconds == 0:
        TimeScale = 'Diarios'
    # Se completa la información
    YiP = DatesP[0].year
    YfP = DatesP[-1].year
    YiS = DatesS[0].year
    YfS = DatesS[-2].year

    if YiP >= YiS:
        Yi = YiP
    else:
        Yi = YiS
    if YfP >= YfS:
        Yf = YfS
    else:
        Yf = YfP

    if YfS < YiP or YfP < YiS:
        raise Exception('Series have no common time')

    if TimeScale == 'Horarios':
        SeriesP = DM.CompDC(DatesP,DataP,datetime(Yi,1,1,0,0),
                datetime(Yf,12,31,23,59),dtm=timedelta(0,3600)) 
        SeriesS = DM.CompDC(DatesS,DataS,datetime(Yi,1,1,0,0),
                datetime(Yf,12,31,23,59),dtm=timedelta(0,3600)) 
    if TimeScale == 'Diarios':
        SeriesP = DM.CompDC(DatesP,DataP,date(Yi,1,1),date(Yf,12,31),
                dtm=timedelta(1)) 
        SeriesS = DM.CompDC(DatesS,DataS,date(Yi,1,1),
                date(Yf,12,31),dtm=timedelta(1,0))
    return SeriesP, SeriesS
