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
try:
    from pathlib import Path
except ImportError:
    from unipath import Path

# ------------------
# Personal Modules
# ------------------
sys.path.append(os.getcwd())
try:
    from EMSD.Dates import DatesFunctions as DUtil
    from EMSD.Dates.DatesC import DatesC 
except ImportError:
    from Dates import DatesFunctions as DUtil
    from Dates.DatesC import DatesC 

# ------------------------
# Functions
# ------------------------

def test_str2datetime():
    Date = DUtil.Dates_str2datetime(['2012-02-01 0100'],flagQuick=True)
    assert isinstance(Date[0],datetime) or isinstance(Date[0],date)

def test_datetime2str():
    Date = DUtil.Dates_datetime2str([datetime(2012,1,1,0,0)])
    assert isinstance(Date[0],str)

def test_DatesCstr2datetime():
    Dates = DatesC(['2012-01-01 0100'])
    assert isinstance(Dates.str[0],str) and (isinstance(Dates.datetime[0],datetime) and isinstance(Dates.datetime[0],date))

def test_DatesCdatetime2str():
    Dates = DatesC([datetime(2012,1,1,1,0)])
    assert isinstance(Dates.str[0],str) and (isinstance(Dates.datetime[0],datetime) and isinstance(Dates.datetime[0],date))

