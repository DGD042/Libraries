# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 19/05/2017
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
import xlrd # Para poder abrir archivos de Excel
import xlsxwriter as xlsxwl
# System
import sys
import os
import glob as gl
import re
import warnings
# ---------------------------
# Importing personal files
# ---------------------------
try:
    from EMSD.Dates import DatesFunctions as DUtil
except ImportError:
    import DatesFunctions as DUtil

# ------------------------
# Class
# ------------------------

class DatesC(object):
    '''
    ____________________________________________________________________________
    
    CLASS DESCRIPTION:
        
        This class takes different date formats and converts them into 
        Python datetime format, making it easier to work with timeseries 
        data.
        
    
        This class is of free use and can be modify, if you have some 
        problem please contact the programmer to the following e-mails:
    
        - danielgondu@gmail.com 
        - dagonzalezdu@unal.edu.co
        - daniel.gonzalez17@eia.edu.co
    
        --------------------------------------
         How to use the library
        --------------------------------------

    ____________________________________________________________________________

    '''

    def __init__(self,Dates,Date_Format=None,flagQuick=False):
        '''
        DESCRIPTION:
            This class uses data from dates and converts them into
            datetime or string data, having all the data packed in
            one object.
        '''
        # -----------------
        # Error managment
        # -----------------
        assert isinstance(Dates[0],str) or isinstance(Dates[0],datetime) or isinstance(Dates[0],date)

        if isinstance(Dates[0],str):
            self.datetime = DUtil.Dates_str2datetime(Dates,Date_Format=Date_Format,flagQuick=flagQuick)
            self.str = DUtil.Dates_datetime2str(self.datetime,Date_Format=None)

        if isinstance(Dates[0],datetime) or isinstance(Dates[0],date):
            self.datetime = Dates
            self.str = DUtil.Dates_datetime2str(Dates)

        return





