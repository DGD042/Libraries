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
# Open and saving data
import csv
import xlrd
import xlsxwriter as xlsxwl
import json
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
from Utilities import DatesUtil as DUtil; DUtil=DUtil()

# ------------------------
# Class
# ------------------------


# Writting Data
def WriteCSVSeries(Dates,Values,Headers,deli=None,flagHeader=True,Pathout='',Name='',NaNdata=''):
    '''
    DESCRIPTION:
    
        This function aims to save data in a csv file from.
    _______________________________________________________________________

    INPUT:
        :param Dates: A list, Dates of the series.
        :param Values: A Dict or list, Value dictionary.
        :param deli: a str, delimeter of the data.
        :param Headers: A list, Headers of the data, Labels of the Dates and Values.
        :param flagHeaders: a Boolean, if headers are needed.
        :param Pathout: a str, Saving directory.
        :param Name: a str, File Name with extension.
        :param NaNdata: a str, Manage the NaN type data.
    _______________________________________________________________________
    
    OUTPUT:
        This function return a .csv file.
        
    '''
    # Create the directory
    utl.CrFolder(Pathout)

    if deli == None:
        deli = self.deli
    try:
        f = open(Pathout+Name+'.csv','w',encoding='utf-8')
    except:
        f = open(Pathout+Name+'.csv','w')
    
    for iL,L in enumerate(Dates):
        # Headers
        if iL == 0 and flagHeader:
            wr = deli.join(Headers)+'\n'
            f.write(wr)
        f.write(L+deli)
        # Change nan values to empty
        Line = []
        for Head in Headers[1:]:
            Line.append(str(Values[Head][iL]))

        for iL,L in enumerate(Line):
            if L == 'nan':
                Line[iL] = NaNdata
            elif L == '':
                Line[iL] = NaNdata

        # Line = np.array(Line).astype('>U4')
        # Line[Line == 'nan'] = NaNdata
        # Line[Line == ''] = NaNdata
        
        wr = deli.join(Line)+'\n'
        f.write(wr)

    f.close()

    return

def WriteFile(Labels,Values,deli=None,Headers=None,flagHeader=False,Pathout='',Name='',NaNdata=''):
    '''
    DESCRIPTION:
    
        This function aims to save data in a csv file from.
    _______________________________________________________________________

    INPUT:
        :param Labels: A list, String left labels, this acts as the keys.
        :param Values: A dict, Value dictionary.
        :param deli: A str, delimeter of the data.
        :param Headers: A list, Headers of the data, defaulted to None.
        :param flagHeaders: A boolean, if headers are needed.
        :param Pathout: a str, Saving directory.
        :param Name: A str, File Name with extension.
        :param NaNdata: A str, Manage the NaN type data.
    _______________________________________________________________________
    
    OUTPUT:
        
    '''
    # Create the directory
    utl.CrFolder(Pathout)

    if deli == None:
        deli = self.deli
    try:
        f = open(Pathout+Name,'w',encoding='utf-8')
    except:
        f = open(Pathout+Name,'w')
    
    for iL,L in enumerate(Labels):
        # Headers
        if iL == 0 and flagHeader:
            wr = deli.join(Headers)+'\r\n'
            f.write(wr)
        f.write(L+deli)
        # Change nan values to empty
        wr1 = np.array(Values[L]).astype(str)
        wr1[wr1=='nan'] = NaNdata
        wr = deli.join(wr1)+'\r\n'
        f.write(wr)

    f.close()

    return

def Writemat(Dates,Data,savingkeys,datekeys=None,datakeys=None,pathout='',Names='Data'):
    '''
    DESCRIPTION:
    
        This function aims to save data in a mat file.
    _______________________________________________________________________

    INPUT:
        :param Dates: A list, String date dictionary or arrays.
        :param Values: A dict or list, Value dictionary or arrays.
        :param savingkeys: A list, New keys for savng files, first have to
                           be the dates.
        :param datekeys: A list, dates directory keys organized.
        :param datakeys: A list, data directory keys organized.
        :param Pathout: A str, Saving directory.
        :param Name: A str, File Name without extension.
    _______________________________________________________________________
    
    OUTPUT:
        Save a mat file.
    '''
    flagDatesDict = False
    flagDataDict = False
    if isinstance(Dates,dict):
        flagDatesDict = True
        keyDates = datekeys
    if isinstance(Data,dict):
        flagDataDict = True
        keyData = datakeys

    SaveData = dict()
    key = 0
    if flagDatesDict:
        for keyD in keyDates:
                SaveData[savingkeys[key]] = Dates[keyD]
                key += 1
    else:
        SaveData[savingkeys[key]] = Dates
        key += 1
    
    if flagDataDict:
        for keyD in keyData:
                SaveData[savingkeys[key]] = Data[keyD]
                key += 1
    else:
        SaveData[savingkeys[key]] = Data
        key += 1

    # Saving data
    nameout = pathout+Names+'.mat'
    sio.savemat(nameout,SaveData)
    return

def St_Document(Pathout='',Name='Stations_Info',St_Info_Dict=None,Data_Flags=None):
    '''
    DESCRIPTION:

        This function saves the station information in an Excel (.xlsx)
        worksheet.
    _______________________________________________________________________

    INPUT:
        + Pathout: Saving directory.
        + Name: File Name.
        + St_Info_Dict: Dictionary with the information of the stations.
                        It must have the following information:
                        CODE: Station code.
                        NAME: Station name.
                        ELEVATION: Station Elevation.
                        LATITUDE: Station latitud.
                        LONGITUDE: Station Longitude.
        + Data_Flags: Possible flags that the data has defaulted to None.
    _______________________________________________________________________
    
    OUTPUT:
    
        Return a document.
    '''
    if St_Info_Dict == None and self.St_Info == None:
        Er = utl.ShowError('St_Document','EDSM','No station data added')
        return
    elif St_Info_Dict == None:
        St_Info_Dict = self.St_Info     

    keys = ['CODE','NAME','ELEVATION','LATITUDE','LONGITUDE']


    St_Info_Dict['LATITUDE'] = list(St_Info_Dict['LATITUDE'])
    St_Info_Dict['LONGITUDE'] = list(St_Info_Dict['LONGITUDE'])

    for i in range(len(St_Info_Dict['LATITUDE'])):
        if St_Info_Dict['LATITUDE'][i][-1] == 'N':
            St_Info_Dict['LATITUDE'][i] = float(St_Info_Dict['LATITUDE'][i][:5])
        elif St_Info_Dict['LATITUDE'][i][-1] == 'S':
            St_Info_Dict['LATITUDE'][i] = float('-'+St_Info_Dict['LATITUDE'][i][:5])

    for i in range(len(St_Info_Dict['LONGITUDE'])):
        if St_Info_Dict['LONGITUDE'][i][-1] == 'E':
            St_Info_Dict['LONGITUDE'][i] = float(St_Info_Dict['LONGITUDE'][i][:5])
        elif St_Info_Dict['LONGITUDE'][i][-1] == 'W':
            St_Info_Dict['LONGITUDE'][i] = float('-'+St_Info_Dict['LONGITUDE'][i][:5])

    Nameout = Pathout+Name+'.xlsx'
    W = xlsxwl.Workbook(Nameout)
    # Stations Sheet
    WS = W.add_worksheet('STATIONS')
    # Cell formats
    Title = W.add_format({'bold': True,'align': 'center','valign': 'vcenter'\
        ,'font_name':'Arial','font_size':11,'top':1,'bottom':1,'right':1\
        ,'left':1})
    Data_Format = W.add_format({'bold': False,'align': 'left','valign': 'vcenter'\
        ,'font_name':'Arial','font_size':11,'top':1,'bottom':1,'right':1\
        ,'left':1})
    # Column Formats
    WS.set_column(1, 3, 20.0)
    WS.set_column(1, 3, 20.0)
    WS.set_column(1, 4, 15.0)
    WS.set_column(1, 5, 15.0)
    # Titles
    WS.write(1,1,'CODE',Title)
    WS.write(1,2,'NAME',Title)
    WS.write(1,3,'ELEVATION',Title)
    WS.write(1,4,'LATITUDE (N)',Title)
    WS.write(1,5,'LONGITUDE (E)',Title)

    Col = 1
    for key in keys:
        Row = 2
        for dat in St_Info_Dict[key]:
            WS.write(Row,Col,dat,Data_Format)
            Row += 1
        Col += 1

    if Data_Flags == None and self.FlagM == None:
        W.close()           
        return
    elif Data_Flags == None:
        Data_Flags = self.FlagM 
        WS = W.add_worksheet('FLAGS')
        WS.write(1,1,'FLAGS',Title)
        WS.set_column(1, 1, 20.0)
        Row = 2
        for flag in Data_Flags:
            WS.write(Row,1,flag,Data_Format)
            Row += 1
    W.close()
    return

def SaveDictJson(Data,Pathout='',Name='Dictionary'):
    '''
    DESCRIPTION:
    
        This function aims to save data in a csv file from.
    _______________________________________________________________________

    INPUT:
        :param Data: A dict, Dictionary with the data.
        :param Pathout: a str, Saving directory.
        :param Name: A str, File Name with extension.
    _______________________________________________________________________
    
    OUTPUT:
        Save a Json file with the dictionary.
        
    '''
    # Create folder
    utl.CrFolder(Pathout)

    with open(Pathout+Name+'.json','w') as f:
        json.dump(Data,f)

    return


