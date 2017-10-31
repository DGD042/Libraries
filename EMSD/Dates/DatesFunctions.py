# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 29/10/2017
#______________________________________________________________________________
#______________________________________________________________________________
# ------------------------
# Importing Modules
# ------------------------ 
# Manipulate Data
import numpy as np
from datetime import date, datetime, timedelta
# System
import sys
import os
import glob as gl
import re
import warnings

# ------------------------
# Parameteres
# ------------------------
Date_Formats = ['%Y/%m/%d','%Y-%m-%d','%Y%m%d',\
                     '%d/%m/%Y','%d-%m-%Y','%d%m%Y',\
                     '%m/%d/%Y','%m-%d-%Y','%m%d%Y',\
                     '%Y/%d/%m','%Y-%d-%m''%Y%d%m',\
                     '%y/%m/%d','%y-%m-%d','%y%m%d',\
                     '%d/%m/%y','%d-%m-%y','%d%m%y',\
                     '%m/%d/%y','%m-%d-%y','%m%d%y',\
                     '%y/%d/%m','%y-%d-%m''%y%d%m']
Hour = [' %H%M','-%H%M',' %H:%M','-%H:%M','_%H%M']
DateTime_Formats = [i + j for i in Date_Formats for j in Hour]

Hour = [' %I%M %p','-%I%M %p',' %I:%M %p','-%I:%M %p','_%I%M %p']
DateTimeAMPM_Formats = [i + j for i in Date_Formats for j in Hour]

# ------------------------
# Functions
# ------------------------

def Dates_str2datetime(Dates,Date_Format=None,flagQuick=False):
    '''
    DESCRIPTION:

        From a string type vector the function converts the dates to 
        python date or datetime data from a given format, if no date
        format is specified the function looks up for a fitting format.
    _______________________________________________________________________

    INPUT:
        :param Dates:       A lisr or ndArray, String date vector that 
                            needs to be changed to date or datetime 
                            vector.
        :param Date_Format: A str, Format of the dates given, it must 
                            be given in datetime string format like 
                            %Y/%m/%d %H%M.
    _______________________________________________________________________
    
    OUTPUT:
        :return DatesP: A ndArray, Python date or datetime format vector.
    '''
    # ----------------
    # Error managment
    # ----------------
    if not(isinstance(Dates[0], str)):
        raise TypeError('Dates should be string')
    # -------------------------
    # Temporal verification
    # -------------------------
    lenDates = len(Dates[0])
    flagHour = False
    if lenDates > 10:
        flagHour = True
    # -------------------------
    # Date_Format Verification
    # -------------------------
    if Date_Format == None:
        if flagHour:
            Date_Formats = DateTime_Formats
        else:
            Date_Formats = Date_Formats
    else:
        Date_Formats = [Date_Format]
    # -------------------------
    # Transformation
    # -------------------------
    for iF,F in enumerate(Date_Formats):
        try:
            if flagQuick:
                if flagHour:
                    DatesP2 = np.array([datetime(int(i[:4]),int(i[5:7]),int(i[8:10]),int(i[11:13]),int(i[13:15])) for i in Dates])
                else:
                    DatesP2 = np.array([datetime(int(i[:4]),int(i[5:7]),int(i[8:10]),0,0) for i in Dates])
            else:   
                DatesP2 = np.array([datetime.strptime(i,F) for i in Dates])
            break
        except:
            if iF == len(Date_Formats)-1:
                Er = utl.ShowError('Dates_str2datetime','DatesUtil','Bad date format, change format')
                raise TypeError
            else:
                continue
    # -------------------------
    # Changing Dates
    # -------------------------
    if flagHour == False:
        DatesP = np.array([i.date() for i in DatesP2])
    else:
        DatesP = DatesP2

    return DatesP

def Dates_datetime2str(DatesP,Date_Format=None):
    '''
    DESCRIPTION:

        This function takes a Python date or datetime vector and return a
        string data vector.
    _______________________________________________________________________

    INPUT:
        :param DatesP:       A list or ndArray, Python date or datetime 
                             vector.
        :param Dates_Format: A list or ndArray, Format of the dates given, 
                             it must be given in datetime string format 
                             like %Y/%m/%d %H%M.
    _______________________________________________________________________
    
    OUTPUT:
        :return Dates: A ndArray, Dates string vector.
    '''
    # ----------------
    # Error managment
    # ----------------
    if not(isinstance(DatesP[0], datetime)) and not(isinstance(DatesP[0], date)):
        raise TypeError('Dates should be string')

    # -------------------------
    # Date_Format Verification
    # -------------------------
    if Date_Format == None:
        if isinstance(DatesP[0],datetime):
            Date_Formats = DateTime_Formats
        else:
            Date_Formats = Date_Formats
    else:
        Date_Formats = [Date_Format]
    # -------------------------
    # Changing Dates
    # -------------------------
    Dates = np.array([i.strftime(Date_Formats[0]) for i in DatesP])
    return Dates

def Dates_ampm224h(Dates,Hours=None,Date_Format=None):
    '''
    DESCRIPTION:

        Converts dates from AM/PM to 24 hour dates.         
    _______________________________________________________________________

    INPUT:
        + Dates: Vector with Dates in string format.
        + DateE: Vector with Hours in string format, defaulted to None if
                 the Dates vector has the dates.
    _______________________________________________________________________
    
    OUTPUT:
    
        - DatesP: Complete Python datetime vector in 24 hour format.
    '''

    if Hours != None:
        Hours2 = []
        if Hours[0][-1] == 'a' or Hours[0][-1] == 'p':
            for H in Hours:
                Hours2.append(H+'m')
        else:
            Hours2 = Hours
        DatesC = np.array([i+' '+Hours2[ii] for ii,i in enumerate(Dates)])
    else:
        DatesC = []
        if Dates[0][-1] == 'a' or Dates[0][-1] == 'p':
            for D in Dates:
                DatesC.append(D+'m')
        else:
            DatesC =Dates
        DatesC = np.array(DatesC)
    # -------------------------
    # Date_Format Verification
    # -------------------------
    if Date_Format == None:
        Date_Formats = DateTimeAMPM_Formats
    else:
        Date_Formats = [Date_Format]

    # -------------------------
    # Transformation
    # -------------------------
    for iF,F in enumerate(Date_Formats):
        try:
            DatesP = np.array([datetime.strptime(i,F) for i in DatesC])
            break
        except:
            DatesP = None
            if iF == len(Date_Formats)-1:
                Er = utl.ShowError('Dates_ampm224h','DatesUtil','Bad date format, change format')
                return Er
            else:
                continue

    return DatesP

def Dates_Comp(DateI,DateE,dtm=timedelta(1)):
    '''
    DESCRIPTION:

        This function creates a date or datetime vector from an initial 
        date to a ending date.          
    _______________________________________________________________________

    INPUT:
        :param DateI: A datetime value, Initial Date in date or datetime.
        :param DateE: A datetime value, Ending Date in date or datetime.
        :param dtm:   A timedelta, Time delta.
    _______________________________________________________________________
    
    OUTPUT:
    
        :return DatesC: A ndarray, Complete date or datetime vector.
    '''
    # ---------------------
    # Constants
    # ---------------------
    flagH = False # flag for the hours
    flagM = False # flag for the minutes
    # ---------------------
    # Error managment
    # ---------------------
    if isinstance(dtm,timedelta) == False:
        Er = utl.ShowError('Dates_Comp','DatesUtil','Bad dtm format')
        raise Er

    if isinstance(DateI,datetime) and dtm.seconds == 0:
        Er = utl.ShowError('Dates_Comp','DatesUtil','Bad time delta given')
        raise Er

    if isinstance(DateI,date) == False or isinstance(DateE,date) == False:
        Er = utl.ShowError('Dates_Comp','DatesUtil','Bad DateI and DateE format')
        raise Er

    if isinstance(DateI,datetime) and isinstance(DateE,datetime) == False:
        Er = utl.ShowError('Dates_Comp','DatesUtil','Bad DateI and DateE format')
        raise Er

    if isinstance(DateI,datetime) == False and isinstance(DateE,datetime):
        Er = utl.ShowError('Dates_Comp','DatesUtil','Bad DateI and DateE format')
        raise Er
    # ---------------------
    # Generate series
    # ---------------------     
    if isinstance(DateI,datetime):
        flagH = True

    yeari = DateI.year
    yearf = DateE.year
    monthi = DateI.month
    monthf = DateE.month

    DatesC = [DateI]
    Date = DateI
    for Y in range(yeari,yearf+1):
        for m in range(1,13):
            Fi = date(Y,m,1)
            if m == 12:
                Ff = date(Y+1,1,1)
            else:
                Ff = date(Y,m+1,1)
            Difd = (Ff-Fi).days
            for d in range(Difd):
                if flagH:
                    Dif = dtm.seconds
                    if Dif < 3600:
                        flagM = True
                    if int(Dif/60/60) == 0:
                        dtt = 1
                    else:
                        dtt = int(Dif/60/60)
                    for h in range(0,24,dtt):
                        if flagM:
                            # DifM = Dif/60
                            for M in range(0,60,int(Dif/60)):
                                if Date <= DateI or Date > DateE+dtm:
                                    Date += dtm
                                else:
                                    DatesC.append(Date)
                                    Date += dtm
                        else:
                            if Date <= DateI or Date > DateE+dtm:
                                Date += dtm
                            else:
                                DatesC.append(Date)
                                Date += dtm
                else:
                    if Date <= DateI or Date >= DateE+dtm:
                        Date += dtm
                    else:
                        DatesC.append(Date)
                        Date += dtm
    DatesC = np.array(DatesC)

    return DatesC







