# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 17/11/2017
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
import xlrd
import xlsxwriter as xlsxwl
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


def EDnetCDFFile(File,VarDict=None,VarRangeDict=None,time='time'):
    '''
    DESCRIPTION:

        With this function the information of an netCDF type file can 
        be extracted.
    _______________________________________________________________________

    INPUT:
        :param File:         A str, File that would be extracted including 
                             the path.
        :param VarDict:      A dict, List of variables that would be 
                             extracted from the netCDF file. Defaulted 
                             to None.
        :param VarRangeDict: A dict, Range of data that would be extracted 
                             per variable. It is defaulted to None if all 
                             the Range wants to be extracted.
                             It must be a list with two values for each 
                             variable.
    _______________________________________________________________________
    
    OUTPUT:
        :return Data: A dict, Extracted Data Dictionary.    
    '''
    # Importing netCDF libary
    try: 
        import netCDF4 as nc
    except ImportError:
        utl.ShowError('EDNCFile','EDSM','netCDF4 not installed, please install the library to continue')

    # Open File
    dataset = nc.Dataset(File)

    if VarDict == None:
        Data = dataset.variables
    else:
        Data = dict()
        for iVar, Var in enumerate(VarDict):
            try:
                P = dataset.variables[Var]
            except KeyError:
                utl.ShowError('EDNCFile','EDSM','Key %s not in the nc file.' %Var)
            if VarRangeDict == None:
                if Var == time:
                    Data[Var] = nc.num2date(dataset.variables[Var][:],dataset.variables[Var].units,dataset.variables[Var].calendar)
                else:
                    Data[Var] = dataset.variables[Var][:]
            else:
                a = dataset.variables[Var] # Variable
                dimensions = a.dimensions # Keys of dimensions
                LenD = len(dimensions) # Number of dimensions
                totalshape = a.shape # Shape of the matrix

                Range = dict()
                for iVarR,VarR in enumerate(dimensions):
                    try:
                        Range[VarR] = VarRangeDict[VarR]
                    except:
                        Range[VarR] = [0,dataset.variables[VarR].shape[0]]

                if LenD == 1:
                    if Var == time:
                        Data[Var] = nc.num2date(dataset.variables[Var][:],dataset.variables[Var].units,dataset.variables[Var].calendar)[slice(Range[dimensions[0]][0],Range[dimensions[0]][1])]
                    else:
                        Data[Var] = dataset.variables[Var][slice(Range[dimensions[0]][0],Range[dimensions[0]][1])]
                elif LenD == 2:
                    Data[Var] = dataset.variables[Var][slice(Range[dimensions[0]][0],Range[dimensions[0]][1]),:]
                    Data[Var] = dataset.variables[Var][:,slice(Range[dimensions[1]][0],Range[dimensions[1]][1])]
                elif LenD == 3:
                    Data[Var] = dataset.variables[Var][slice(Range[dimensions[0]][0],Range[dimensions[0]][1]),:,:]
                    Data[Var] = dataset.variables[Var][:,slice(Range[dimensions[1]][0],Range[dimensions[1]][1]),:]
                    Data[Var] = dataset.variables[Var][:,:,slice(Range[dimensions[2]][0],Range[dimensions[2]][1])]
                elif LenD == 4:
                    Data[Var] = dataset.variables[Var][slice(Range[dimensions[0]][0],Range[dimensions[0]][1]),:,:,:]
                    Data[Var] = dataset.variables[Var][:,slice(Range[dimensions[1]][0],Range[dimensions[1]][1]),:,:]
                    Data[Var] = dataset.variables[Var][:,:,slice(Range[dimensions[2]][0],Range[dimensions[2]][1]),:]
                    Data[Var] = dataset.variables[Var][:,:,:,slice(Range[dimensions[3]][0],Range[dimensions[3]][1])]
                elif LenD == 5:
                    Data[Var] = dataset.variables[Var][slice(Range[dimensions[0]][0],Range[dimensions[0]][1]),:,:,:,:]
                    Data[Var] = dataset.variables[Var][:,slice(Range[dimensions[1]][0],Range[dimensions[1]][1]),:,:,:]
                    Data[Var] = dataset.variables[Var][:,:,slice(Range[dimensions[2]][0],Range[dimensions[2]][1]),:,:]
                    Data[Var] = dataset.variables[Var][:,:,:,slice(Range[dimensions[3]][0],Range[dimensions[3]][1]),:]
                    Data[Var] = dataset.variables[Var][:,:,:,:,slice(Range[dimensions[4]][0],Range[dimensions[4]][1])]
        dataset.close()

    return Data

