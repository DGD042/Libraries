# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#						Coded by Daniel González Duque
#						    Last revised 22/05/2017
#______________________________________________________________________________
#______________________________________________________________________________
# ------------------------
# Importing Modules
# ------------------------ 
# Manipulate Data
import numpy as np
import scipy.io as sio
from datetime import date, datetime, timedelta
import plotly
from plotly.graph_objs import Scatter, Layout
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

# ------------------
# Personal Modules
# ------------------
# Importing Modules
from Utilities import Utilities
from DatesUtil import DatesUtil
# Aliases
utl = Utilities()
DUtil = DatesUtil()