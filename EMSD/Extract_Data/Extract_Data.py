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
# Funciones
# ------------------------
def LoopDataDaily_IDEAM(DateP,Value,Flags,xRow_St,xCol_St,Years,Lines,Sum2=2):
    '''
    DESCRIPTION:
    
        With this function we extract the values in an IDEAM document type.
    _________________________________________________________________________

    INPUT:
        + DateP: Empty list or with previous date values.
        + Value: Empty list or with previous values.
        + Flags: Empty list or with previous flags.
        + xRow_St: Row where the data begins.
        + xCol_St: Column where data begins.
        + Year: Year that wants to be extracted.
        + Lines: readlines variable.
    _________________________________________________________________________
    
    OUTPUT:
        - DateP: Extracted dates.
        - Value: Extracted data.
    '''
    for j in range(1,13):
        DateSt = date(int(Years),j,1)
        if j == 12:
            DateEnd = date(int(Years)+1,1,1)
        else:
            DateEnd = date(int(Years),j+1,1)
        Days = (DateEnd-DateSt).days
        
        xR = xRow_St
        for i in range(1,Days+1):
            DateP.append(date(int(Years),j,i))
            try:
                Value.append(float(Lines[xR][xCol_St:xCol_St+5]))
            except ValueError:
                Value.append(np.nan)
            except IndexError:
                Value.append(np.nan)

            try:
                Flags.append(Lines[xR][xCol_St+5+1:xCol_St+5+2])
            except IndexError:
                Flags.append(' ')
            if Flags[-1] == '' or Flags[-1] == ' ' or Flags[-1] == '\r':
                Flags[-1] = np.nan

            xR += Sum2
        xCol_St += 9

    return DateP,Value,Flags


# ------------------------
# Class
# ------------------------


# Information Extraction
def EDTXT(File,deli=',',colStr=(0,),colData=(1,),row_skip=1,flagHeader=True,
        rowH=0,row_end=0,str_NaN=None,num_NaN=None,dtypeData=float):
    '''
    DESCRIPTION:

        This function extract data series from a plain text file or a csv
        type file.
    _______________________________________________________________________

    INPUT:
        :param File:       A str, File that needs to be open with 
                                  extention.
        :param deli:       A str, Delimiter of the data. Defaulted to ','.
        :param colStr:     A tuple, tuple sequence with the columns where
                                    the string data is. Defaulted to (0).
        :param colData:    A tuple, tuple sequence with the columns where 
                                    the floar data is.  Defaulted to (1).
        :param row_skip:   An int, begining row. Defaulted to 0.
        :param flagHeader: A boolean, flag to get the header of the 
                                      information.
        :param rowH:       A str, Header row.
        :param row_end:    A str, Ending row if nedded, defaulted to None.
        :param str_NaN:    A str, NaN string for data. Defaulted to None.
        :param num_NaN:    A str, NaN number for data. Defaulted to None.
        :param dtypeData:  A str, data type. Defaulted to float.
    _______________________________________________________________________
    OUTPUT:
        Dates and values are given in dictionaries.
    '''
    # ----------------
    # Error Managment
    # ----------------
    if not(isinstance(colStr,tuple)) and not(isinstance(colStr,list)) and colStr != None:
        Er = utl.ShowError('EDTXT','EMSD.Extract_Data',
                'colStr not in tuple or list')
        raise TypeError
    elif not(isinstance(colStr,tuple)) and colStr != None:
        colStr = tuple(colStr)
    if not(isinstance(colData,tuple)) and not(isinstance(colData,list)) and colData != None:
        Er = utl.ShowError('EDTXT','EMSD.Extract_Data',
                'colData not in tuple or list')
        raise TypeError
    elif not(isinstance(colData,tuple)) and colStr != None:
        colData = tuple(colData)

    # Verify values
    if num_NaN == None:
        flagnumNaN = False
    else:
        flagnumNaN = True
        if isinstance(num_NaN,str) == False:
            num_NaN = float(num_NaN)
    if str_NaN == None:
        flagstrNaN = False
    else:
        flagstrNaN = True

    # -------------------
    # Extracting Values
    # -------------------
    # Headers
    if flagHeader:
        if colStr == None and colData == None:
            Headers = np.genfromtxt(File,dtype=str,
                    skip_header=rowH,delimiter=deli,max_rows=1)
        elif colStr != None and colData == None:
            Headers = np.genfromtxt(File,dtype=str,usecols=colStr,
                    skip_header=rowH,delimiter=deli,max_rows=1)
        elif colData != None and colStr == None:
            Headers = np.genfromtxt(File,dtype=str,usecols=colData,
                    skip_header=rowH,delimiter=deli,max_rows=1)
        else:
            Headers = np.genfromtxt(File,dtype=str,usecols=colStr+colData,
                    skip_header=rowH,delimiter=deli,max_rows=1)
    else:
        rowH = 0
        if colStr == None and colData == None:
            Headers = np.genfromtxt(File,dtype=str,
                    skip_header=rowH,delimiter=deli,max_rows=1)
        elif colStr == None:
            Headers = np.genfromtxt(File,dtype=str,usecols=colStr,
                    skip_header=rowH,delimiter=deli,max_rows=1)
        elif colData == None:
            Headers = np.genfromtxt(File,dtype=str,usecols=colData,
                    skip_header=rowH,delimiter=deli,max_rows=1)
        else:
            Headers = np.genfromtxt(File,dtype=str,usecols=colStr+colData,
                    skip_header=rowH,delimiter=deli,max_rows=1)
        Headers = np.arange(0,len(Headers))

    # R = {'Headers':Headers}
    R = dict()

    # String Data
    if colStr != None:
        DataStr = np.genfromtxt(File,dtype=str,usecols=colStr,delimiter=deli,
                skip_header=row_skip,skip_footer=row_end,unpack=True)
        if flagstrNaN:
            DataStr[DataStr == str_NaN] = 'nan'
        if len(colStr) == 1:
            R[Headers[0]] = DataStr
        elif len(colStr) > 1:
            for icol,col in enumerate(colStr):
                R[Headers[icol]] = DataStr[icol]

    if colData != None:
        Data = np.genfromtxt(File,dtype=dtypeData,usecols=colData,delimiter=deli,
                skip_header=row_skip,skip_footer=row_end,unpack=True)
        if flagnumNaN:
            Data[Data== num_NaN] = np.nan

        if len(colData) == 1:
            if colStr == None:
                R[Headers[icol]] = Data
            else:
                R[Headers[len(colStr)+1]] = Data
        elif len(colData) > 1:
            for icol,col in enumerate(colData):
                if colStr == None:
                    R[Headers[icol]] = Data[icol]
                else:
                    R[Headers[len(colStr)+icol]] = Data[icol]

    elif colData == None and colStr == None:
        Data = np.genfromtxt(File,dtype=dtypeData,delimiter=deli,
                skip_header=row_skip,skip_footer=row_end,unpack=True)
        if flagnumNaN:
            Data[Data== num_NaN] = np.nan
        for icol,col in enumerate(Headers):
            R[Headers[icol]] = Data[icol]
            
    return R

def EDExcel(File=None,sheet=None,colDates=(0,),colData=(1,),row_skip=1,flagHeader=True,row_end=None,num_NaN=None):
    '''
    DESCRIPTION:

        This function extracts time series data from a sheet in an Excel
        file.
    _______________________________________________________________________

    INPUT:
        + File: File that needs to be open.
        + sheet: Index (number) or name (string) of the sheet.
        + colDates: tuple sequence with the columns where the dates or 
                    string data is. Defaulted to (0). if None it extracts
                    all the data as the original format dictionary.
        + colData: tuple sequence with the columns where the data is.
                   Defaulted to (1). If None it extracts all the data
                   as a float data.
        + row_skip: begining row of the data. Defaulted to 1
        + flagHeader: flag to get the header of the information.
        + row_end: Ending row if nedded, defaulted to None.
        + num_NaN

    _______________________________________________________________________
    
    OUTPUT:
        Dates and values are given in dictionaries.
    '''
    # ----------------
    # Error Managment
    # ----------------
    # Verify values
    if File == None:
        Er = utl.ShowError('EDExcel','EDSM','No file was added')
        return None, None, None
    if sheet == None:
        Er = utl.ShowError('EDExcel','EDSM','No sheet was added')
        return None, None, None
    if num_NaN == None:
        flagnumNaN = False
    else:
        flagnumNaN = True
        if isinstance(num_NaN,str):
            Er = utl.ShowError('EDExcel','EDSM','num_NaN must be a number')
            return None, None, None
        num_NaN = float(num_NaN)
    # ----------------
    # Data extraction
    # ----------------
    flagName = False
    flagcolDates = False
    flagcolData = False
    if isinstance(sheet,str):
        flagName = True
    if colDates == None:
        flagcolDates = False
    if colData == None:
        flagcolData = False

    # Open Excel File using xldr
    B = xlrd.open_workbook(File)
    # Se carga la página en donde se encuentra el información
    if flagName:
        S = B.sheet_by_name(sheet)  
    else:
        S = B.sheet_by_index(sheet)

    # Verify the number of columns and rows

    ncol = S.ncols
    nrow = S.nrows
    if max(colDates) > ncol-1 or max(colData) > ncol-1:
        Er = utl.ShowError('EDExcel','EDSM','column exceed dimension of the sheet')
    if row_end != None:
        if row_end > nrow-1:
            row_end = nrow-1
    else:
        row_end = nrow-1

    # Header Exctraction
    if flagHeader:
        Header = S.row_values(row_skip-1)
    Header = list(np.array(Header)[list(colDates)])+ list(np.array(Header)[list(colData)])
    # Extracting time
    Dates = dict()
    Data = dict()
    for iCDate,CDate in enumerate(colDates): # Revisar Fechas
        Dates1 = S.col_values(CDate,start_rowx=row_skip,end_rowx=row_end)
        try:
            a = datetime(*xlrd.xldate_as_tuple(Dates1[0], B.datemode))
            dif1 = xlrd.xldate_as_tuple(Dates1[1], B.datemode)[3]-xlrd.xldate_as_tuple(Dates1[1], B.datemode)[3]
            dif2 = xlrd.xldate_as_tuple(Dates1[1], B.datemode)[4]-xlrd.xldate_as_tuple(Dates1[1], B.datemode)[4]
            try:
                dif3 = xlrd.xldate_as_tuple(Dates1[1], B.datemode)[5]-xlrd.xldate_as_tuple(Dates1[1], B.datemode)[5]
            except IndexError:
                dif3 = 0
            if dif1 == 0 and dif2 == 0 and dif3 == 0:
                Dates[CDate] = np.array([datetime(*xlrd.xldate_as_tuple(i, B.datemode)) for i in Dates1])
            else:
                Dates[CDate] = np.array([datetime(*xlrd.xldate_as_tuple(i, B.datemode)).date() for i in Dates1])
        except:
            Data[CDate] = np.array(Dates1)
    # Exctracting Data
    for CData in colData: # Revisar Fechas
        Data1 = S.col_values(CData,start_rowx=row_skip,end_rowx=row_end)
        Data2 = []
        # Verify data to become NaN
        for dat in Data1:
            try:
                Data2.append(float(dat))
            except ValueError:
                Data2.append(np.nan)
        Data2 = np.array(Data2)
        if flagnumNaN:
            x = np.where(Data2 == num_NaN)
            Date2[x] = np.nan
        Data[CData] = Data2

    return  Dates, Data, Header

def EDDAT_NCDCCOOP(Tot,Stations=None,Header=False,row_skip=0):
    '''
    DESCRIPTION:
    
        Con esta función se pretende extraer la información de los archivos 
        con extensión .dat de la base de datos de estaciones de precipitación
        cada 15 minutos generadas por el NCDC_COOP.
    _______________________________________________________________________

    INPUT:
        + Tot: Es la ruta completa del archivo que se va a abrir.
        + sheet: Número de la hoja del documento de excel
        + Header: Se pregunta si el archivo tiene encabezado.
        + Stations: por defecto None, este parámetro permite preguntar si se van 
                    a extraer estaciones específicas.
    _______________________________________________________________________
    
    OUTPUT:
        - Fecha: Diccionario con todas las fechas separadas por estaciones.
        - V1: Diccionario con todos los datos separados por estaciones.
    '''

    # Se organizan las fechas
    Months = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6\
        ,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}

    Year = int(Tot[-4-4:-4])
    Month = Months[Tot[-4-4-3:-4-4]]
    Fi = date(Year,Month,1)
    if Month == 12:
        Ff = date(Year+1,1,1)
    else:
        Ff = date(Year,Month+1,1)
    
    Fii = datetime(Year,Month,1,0,15)
    Fff = datetime(Year,Month,1,0,30)
    Dif = Fff-Fii
    FechaN = [Fii]
    for i in range((Ff-Fi).days):
        for j in range(96):
            FechaN.append(FechaN[-1]+Dif)
    FechaN.pop()
    FechaC = np.array([i.strftime('%Y/%m/%d-%H%M') for i in FechaN])
    FechaN = np.array(FechaN)
    # Datos iniciales 
    PrecC = np.zeros(len(FechaN))
    
    # Se extraen los datos
    delidat = (3,6,2,4,2,4,2,2,2,3,4)
    St_Info = np.genfromtxt(Tot,dtype=str,unpack=True,delimiter=delidat,usecols=(0,1,2,3,4,5,6,8,9,10))
    StationCode = St_Info[1]
    HTorHI = St_Info[4]
    QPCPorQGAG = St_Info[3]
    # Data Counts
    Data_Counts = St_Info[-2]

    # Indicatives
    Indi = {'a':'START','A':'END',",":'BEGIN MONTH','{':'DELETED BEGIN'\
        ,'}':'DELETED END','[':'MISSING BEGIN',']':'MISSING END'\
        ,'g':'BEGIN DAY','b':'BLANK','I':'INCOMPLETE','P':'DAILY TOTAL'\
        ,'X':'CAUTION','Z':'MELTING','R':'TIME PROBLEMS','Q':'HOURLY DATA'\
        ,'q':'EXCLUDE DATA'}

    FechaD = np.array([St_Info[5,i]+'/'+St_Info[6,i]+'/'+St_Info[7,i] for i in range(len(St_Info[0]))])
    # Data
    f = open(Tot,'r',encoding='utf-8')
    Data = f.readlines()

    Station = []
    Fecha = []
    Prec = []
    Hour = []
    I = []
    x = (QPCPorQGAG == 'QPCP')
    StationCode = StationCode[x]
    Data_Counts = Data_Counts[x]
    FechaD = FechaD[x]
    HTorHI = HTorHI[x]
    for irow,row in enumerate(np.array(Data)[x]):

        Station.append(StationCode[irow])
        Fecha.append(FechaD[irow])
        Count = int(Data_Counts[irow])
        Hour.append(row[30:34])
        Prec.append(float(row[35:40]))
        I.append(row[40])
        # print('Prec=',Prec[-1])
        # print('HTorHI=',HTorHI[irow])
        # print('I=',I[-1])
        if Prec[-1] == 99999.0 or Prec[-1] == 9999. or I[-1] == 'I':
            Prec[-1] = 9999.
        elif HTorHI[irow] == 'HI':
            Prec[-1] = round(Prec[-1] /1000 * 25.4,1)
        elif HTorHI[irow] == 'HT':
            if Prec[-1] >= 100:
                Prec[-1] = round(Prec[-1] /1000 * 25.4,1)
            else:
                Prec[-1] = round(Prec[-1] /100 * 25.4,1)
            # Prec[-1] = Prec[-1]
        if I[-1] == 'P' or Hour[-1] == '2500':
            Hour.pop()
            Prec.pop()
            I.pop()
            Station.pop()
            Fecha.pop()
        iCol = 42
        if Count != 0:
            for iDat in range(1,Count):

                Station.append(StationCode[irow])
                Fecha.append(FechaD[irow])
                Hour.append(row[iCol:iCol+4])
                Prec.append(float(row[iCol+5:iCol+5+5]))
                I.append(row[iCol+5+5])
                # print('Prec=',Prec[-1])
                # print('HTorHI=',HTorHI[irow])
                # print('I=',I[-1])
                # print(HTorHI[irow])
                if Prec[-1] == 99999.0 or Prec[-1] == 9999. or I[-1] == 'I':
                    Prec[-1] = 9999.
                elif HTorHI[irow] == 'HI':
                    Prec[-1] = round(Prec[-1] / 1000 * 25.4,1)
                elif HTorHI[irow] == 'HT':
                    if Prec[-1] >= 100:
                        Prec[-1] = round(Prec[-1] /1000 * 25.4,1)
                    else:
                        Prec[-1] = round(Prec[-1] /100 * 25.4,1)
                    # Prec[-1] = Prec[-1]
                if I[-1] == 'P' or Hour[-1] == '2500':
                    Hour.pop()
                    Prec.pop()
                    I.pop()
                    Station.pop()
                    Fecha.pop()
                iCol += 5+5+2
    Station = np.array(Station)
    Fecha = np.array(Fecha)
    HH = np.array([i[:2] for i in Hour])
    MM = np.array([i[2:] for i in Hour])
    Fecha_Hour = np.array([Fecha[i]+'-'+Hour[i] for i  in range(len(Fecha))])
    FechaH = []
    for i in Fecha_Hour:
        if i[-4:] == '2400':
            i = i[:-4]+'2345'
            FechaH.append(datetime.strptime(i,'%Y/%m/%d-%H%M')+Dif)
        else:
            FechaH.append(datetime.strptime(i,'%Y/%m/%d-%H%M'))
    FechaH = np.array(FechaH)

    Hour = np.array(Hour)
    Prec = np.array(Prec)
    I = np.array(I)

    if isinstance(Stations,list) or isinstance(Stations,(np.ndarray,np.generic)):
        StationCodeU = Stations
    else:
        if Stations == None:
            StationCodeU = np.unique(StationCode)
        else:
            StationCodeU = Stations

    Fecha_Dict = dict()
    Prec_Dict = dict()
    for St in StationCodeU:
        x = np.where(Station == St)
        if len(x) == 0:
            PrecC = np.empty(len(FechaN))*np.nan
        else:
            InStations = I[x]
            # print('x =',x)
            # print('St = ',St)
            # Se completa la información
            xx = FechaN.searchsorted(FechaH[x])
            PrecC[xx] = Prec[x]


            # Se llenan los datos faltantes
            x = np.where(PrecC == 9999.)
            PrecC[x] = np.nan
            x = np.where(PrecC == 99999.0)
            PrecC[x] = np.nan

            I_St = np.where(InStations == '{')[0]
            I_End = np.where(InStations == '}')[0]
            if len(I_St) != 0:
                for i in range(len(I_St)):
                    FechaEl_St = FechaH[I_St[i]]
                    FechaEl_End = FechaH[I_End[i]]
                    xFi = np.where(FechaN == FechaEl_St)[0][0]
                    xFf = np.where(FechaN == FechaEl_End)[0][0]
                    PrecC[xFi:xFf+1] = np.nan
            I_St = np.where(InStations == '[')[0]
            I_End = np.where(InStations == ']')[0]
            if len(I_St) != 0:
                for i in range(len(I_St)):
                    FechaEl_St = FechaH[I_St[i]]
                    FechaEl_End = FechaH[I_End[i]]
                    xFi = np.where(FechaN == FechaEl_St)[0][0]
                    xFf = np.where(FechaN == FechaEl_End)[0][0]
                    PrecC[xFi:xFf+1] = np.nan
        Fecha_Dict[St] = FechaC
        Prec_Dict[St] = PrecC


    return StationCodeU,Fecha_Dict,Prec_Dict

def EDIDEAM(File=None):
    '''
    DESCRIPTION:

        With this function the information of an IDEAM type file can 
        be extracted.
    _______________________________________________________________________

    INPUT:
        + File: File that would be extracted including the path.
                Default value is None because it works with Open_Data.
    _______________________________________________________________________
    
    OUTPUT:
        
    '''
    if File == None:
        Er = utl.ShowError('EDIDEAM','EDSM','No file was added')
        return None, None, None, None, None
    # Match Variables
    Station_Compile = re.compile('ESTACION')
    Lat_Compile = re.compile('LATITUD')
    Lon_Compile = re.compile('LONGITUD')
    Elv_Compile = re.compile('ELEVACION')
    Year_Compile = re.compile(' ANO ')
    Var_Temp_Compile = re.compile('TEMPERATURA')
    Var_Prec_Compile = re.compile('PRECIPITACION')
    Var_BS_Compile = re.compile('BRILLO SOLAR')
    Med_Temp_Compile = re.compile('VALORES MEDIOS')
    Min_Temp_Compile = re.compile('VALORES MINIMOS')
    Max_Temp_Compile = re.compile('VALORES MAXIMOS')

    Start_Compile = re.compile('ENERO')

    # Open the file
    try:
        f = open(File,'r',encoding='utf-8')
        Lines = np.array(f.readlines())
        Sum1 = 6
        Sum2 = 2
    except TypeError:
        f = open(File,'r')
        Lines1 = f.readlines()
        Lines = np.array([i.encode('UTF-8') for i in Lines1])
        Sum1 = 3
        Sum2 = 1

    if platform.system() == 'Windows':
        Sum1 = 3
        Sum2 = 1
    
    # Variables para la estación
    Station_Match = []
    Row_Station = []
    Station_Code = []
    Station_Name = []
    Lat_Match = []
    Lats = []
    Lon_Match = []
    Lons = []
    Elv_Match = []
    Elvs = []
    # Variables para el año
    Year_Match = []
    Row_Year = []
    Years = []
    # Estaciones con Temperatura
    Var_Temp_Match = []
    Row_Var_Temp = []
    # Estaciones con Precipitación
    Var_Prec_Match = []
    Row_Var_Prec = []
    # Estaciones con Brillo Solar
    Var_BS_Match = []
    Row_Var_BS = []
    # Valores Medios
    Med_Temp_Match = []
    Row_Med_Temp = []
    # Inicio de datos
    Start_Match = []
    Row_Start = []

    keys = []
    Stations_Code = []
    Stations_Name = []
    Value_Dict = dict()
    DateP_Dict = dict()
    Flags_Dict = dict()
    x = 0
    for irow,row in enumerate(Lines):
        # Stations
        if re.search(Station_Compile,row) != None:
            Row_Station.append(irow)
            Station_Match.append(re.search(Station_Compile,row))
            Station_Code.append(row[Station_Match[-1].end()+3:Station_Match[-1].end()+3+8])
            Station_Name.append(row[Station_Match[-1].end()+3+8+1:-1])
        # Latitude
        if re.search(Lat_Compile,row) != None:
            Lat_Match.append(re.search(Lat_Compile,row))
            Lats.append(row[Lat_Match[-1].end()+4:Lat_Match[-1].end()+4+6])
        # Longitude
        if re.search(Lon_Compile,row) != None:
            Lon_Match.append(re.search(Lon_Compile,row))
            Lons.append(row[Lon_Match[-1].end()+3:Lon_Match[-1].end()+3+6])
        # Elevation
        if re.search(Elv_Compile,row) != None:
            Elv_Match.append(re.search(Elv_Compile,row))
            Elvs.append(row[Elv_Match[-1].end()+2:Elv_Match[-1].end()+2+4])
        # Years
        if re.search(Year_Compile,row) != None:
            Row_Year.append(irow)
            Year_Match.append(re.search(Year_Compile,row))
            Years.append(row[Year_Match[-1].end()+1:Year_Match[-1].end()+1+4])
        # Temperature
        if re.search(Var_Temp_Compile,row) != None:
            Row_Var_Temp.append(irow)
            Var_Temp_Match.append('TEMPERATURA')
            if re.search(Med_Temp_Compile,row) != None:
                Row_Med_Temp.append(irow)
                Med_Temp_Match.append('MEDIO')
            elif re.search(Min_Temp_Compile,row) != None:
                Med_Temp_Match.append('MINIMO')
            elif re.search(Max_Temp_Compile,row) != None:
                Med_Temp_Match.append('MAXIMO')
            else:
                Med_Temp_Match.append(0)
        # Precipitation
        if re.search(Var_Prec_Compile,row) != None:
            Row_Var_Prec.append(irow)
            Var_Temp_Match.append('PRECIPITACION')
            Med_Temp_Match.append('TOTALES')

        # Brillo Solar
        if re.search(Var_BS_Compile,row) != None:
            Row_Var_BS.append(irow)
            Var_Temp_Match.append('BRILLO SOLAR')
            Med_Temp_Match.append('TOTALES') 
        # Data
        if re.search(Start_Compile,row) != None:
            Row_Start.append(irow)
            Start_Match.append(re.search(Start_Compile,row))

            xRow_St = Row_Start[-1] + Sum1
            xCol_St = Start_Match[-1].start()
            if Row_Start[-1] <= 20:
                DateP = []
                Value = []
                Flags = []
                DateP,Value,Flags = LoopDataDaily_IDEAM(DateP,Value,Flags,xRow_St,xCol_St,Years[-1],Lines,Sum2)
            elif (Station_Code[-1] == Station_Code[-2]) and (Med_Temp_Match[-1] == Med_Temp_Match[-2]) and (Var_Temp_Match[-1] == Var_Temp_Match[-2]):
                DateP,Value,Flags = LoopDataDaily_IDEAM(DateP,Value,Flags,xRow_St,xCol_St,Years[-1],Lines,Sum2)
            else:
                keys.append(Station_Code[-2]+'_'+Var_Temp_Match[-2]+'_'+Med_Temp_Match[-2])
                Stations_Name.append(Station_Name[-2])
                Stations_Code.append(Station_Code[-2])
                DateP_Dict[Station_Code[-2]+'_'+Var_Temp_Match[-2]+'_'+Med_Temp_Match[-2]]=\
                    DateP
                Value_Dict[Station_Code[-2]+'_'+Var_Temp_Match[-2]+'_'+Med_Temp_Match[-2]]=\
                    Value
                Flags_Dict[Station_Code[-2]+'_'+Var_Temp_Match[-2]+'_'+Med_Temp_Match[-2]]=\
                    Flags
                DateP = []
                Value = []
                Flags = []
                DateP,Value,Flags = LoopDataDaily_IDEAM(DateP,Value,Flags,xRow_St,xCol_St,Years[-1],Lines,Sum2)

        if irow == len(Lines)-1:
            keys.append(Station_Code[-1]+'_'+Var_Temp_Match[-1]+'_'+Med_Temp_Match[-1])
            DateP_Dict[Station_Code[-1]+'_'+Var_Temp_Match[-1]+'_'+Med_Temp_Match[-1]]=\
                DateP
            Value_Dict[Station_Code[-1]+'_'+Var_Temp_Match[-1]+'_'+Med_Temp_Match[-1]]=\
                Value
            Flags_Dict[Station_Code[-1]+'_'+Var_Temp_Match[-1]+'_'+Med_Temp_Match[-1]]=\
                Flags


    # Station information
    Stations_Code_Un,xU = np.unique(np.array(Station_Code),return_index=True)
    Stations_Name_Un = np.array(Station_Name)[xU]
    Lat_Un = np.array(Lats)[xU]
    Lon_Un = np.array(Lons)[xU]
    Elv_Un = np.array(Elvs)[xU]

    Stations_Information = dict()
    Stations_Information['CODE'] = Stations_Code_Un
    Stations_Information['NAME'] = Stations_Name_Un
    Stations_Information['LATITUDE'] = [i[:2]+'.'+i[2:] for i in Lat_Un]
    Stations_Information['LONGITUDE'] = [i[:2]+'.'+i[2:] for i in Lon_Un]
    Stations_Information['ELEVATION'] = Elv_Un

    # Search for the Flags meaning
    Flag_Meaning_Dict = dict()

    keys = list(Flags_Dict)

    for ikey, key in enumerate(keys):
        if ikey == 0:
            Flag_Un = np.unique(np.array(Flags_Dict[key]))
        else:
            Flag_Un = np.hstack((Flag_Un,np.unique(np.array(Flags_Dict[key]))))

    Flag_Un2 = np.unique(Flag_Un)
    
    # Removing the nan values
    x = np.where(Flag_Un2 == 'nan')
    Flag_Un2 = np.delete(Flag_Un2,x)

    # Verify the flags meanining
    Flag_Meaning = []
    for flag in Flag_Un2:
        Flag_Compile = re.compile(flag+' :')
        
        for irow,row in enumerate(Lines):
            # Stations
            if re.search(Flag_Compile,row) != None:
                Flag_Match = re.search(Flag_Compile,row)
                if Sum1 == 6:
                    Flag_Meaning.append(row[Flag_Match.start():-1])
                else:
                    Flag_Meaning.append(row[Flag_Match.start():-3])
                break

    return DateP_Dict, Value_Dict, Flags_Dict, Flag_Meaning, Stations_Information

def EDWundergrounds(File):
    '''
    DESCRIPTION:

        This function extract information from the Wundergrounds type
        file.
    _______________________________________________________________________

    INPUT:
        + File: File that is going to be extracted.
    _______________________________________________________________________
    
    OUTPUT:
    '''
    # -----------------------
    # Extraction Parameters
    # -----------------------
    self.LabelsWund = ['Time','TemperatureC','DewpointC','PressurehPa','WindDirection',
            'WindDirectionDegrees','WindSpeedKMH','WindSpeedGustKMH','Humidity',
            'HourlyPrecipMM','Conditions','Clouds','dailyrainMM','SoftwareType','DateUTC']

    self.DataTypes = {'Time':str,'TemperatureC':float,'DewpointC':float,'PressurehPa':float,
            'WindDirection':str,'WindDirectionDegrees':float,'WindSpeedKMH':float,
            'WindSpeedGustKMH':float, 'Humidity':float,'HourlyPrecipMM':float,
            'Conditions':str,'Clouds':str,'dailyrainMM':float,'SoftwareType':str,'DateUTC':str}
    # Variables 
    Data = dict()
    for lab in self.LabelsWund:
        Data[Lab] = [-9999]

    # -----------------------
    # Data Extraction
    # -----------------------
    # Headers
    Headers = np.genfromtxt(File,skip_header=0,dtype=str,delimiter=',',max_rows=1)

def EDnetCDFFile(File,VarDict=None,VarRangeDict=None):
    '''
    DESCRIPTION:

        With this function the information of an IDEAM type file can 
        be extracted.
    _______________________________________________________________________

    INPUT:
        + File: File that would be extracted including the path.
        + VarDict: List of variables that would be extracted from the 
                   netCDF file. Defaulted to None.
        + VarRangeDict: Range of data that would be extracted per 
                        variable. It is defaulted to None if all the 
                        Range wants to be extracted.
                        It must be a list with two values for each variable.
    _______________________________________________________________________
    
    OUTPUT:
        - Data: Extracted Data Dictionary.    
    '''
    # Importing netCDF libary
    try: 
        import netCDF4 as nc
    except:
        Er = utl.ShowError('EDNCFile','EDSM','netCDF4 not installed, please install the library to continue')
        raise Er

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
                Er = utl.ShowError('EDNCFile','EDSM','Key %s not in the nc file.' %Var)
                raise Er
            if VarRangeDict == None:
                if Var == 'time':
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
                    if Var == 'time':
                        Data[Var] = nc.num2date(dataset.variables[Var][:],dataset.variables[Var].units,dataset.variables[Var].calendar)[slice(Range[dimensions[0]][0],Range[dimensions[0]][1])]
                    else:
                        Data[Var] = dataset.variables[Var][slice(Range[dimensions[0]][0],Range[dimensions[0]][1])]
                elif LenD == 2:
                    Data[Var] = dataset.variables[Var][slice(Range[dimensions[0]][0],Range[dimensions[0]][1])][slice(Range[dimensions[1]][0],Range[dimensions[1]][1])]
                elif LenD == 3:
                    Data[Var] = dataset.variables[Var][slice(Range[dimensions[0]][0],Range[dimensions[0]][1])][slice(Range[dimensions[1]][0],Range[dimensions[1]][1])][slice(Range[dimensions[2]][0],Range[dimensions[2]][1])]
                elif LenD == 4:
                    Data[Var] = dataset.variables[Var][slice(Range[dimensions[0]][0],Range[dimensions[0]][1])][slice(Range[dimensions[1]][0],Range[dimensions[1]][1])][slice(Range[dimensions[2]][0],Range[dimensions[2]][1])][slice(Range[dimensions[3]][0],Range[dimensions[3]][1])]
                elif LenD == 5:
                    Data[Var] = dataset.variables[Var][slice(Range[dimensions[0]][0],Range[dimensions[0]][1])][slice(Range[dimensions[1]][0],Range[dimensions[1]][1])][slice(Range[dimensions[2]][0],Range[dimensions[2]][1])][slice(Range[dimensions[3]][0],Range[dimensions[3]][1])][slice(Range[dimensions[4]][0],Range[dimensions[4]][1])]


        dataset.close()

    return Data

def EDmat(File):
    '''
    DESCRIPTION:

        This function extract the information of a mat file.
    _______________________________________________________________________

    INPUT:
        :param File: A str, File that would be extracted including the 
                     path.
    _______________________________________________________________________
    
    OUTPUT:
        :return Data: A dict, Extracted Data Dictionary.    
    '''
    f = sio.loadmat(File)

    Keys = list(f)
    # Extracción info
    Data = dict()
    for key in Keys:
        try:
            _shape = f[key].shape
            lenshape = len(f[key].shape)
            if lenshape == 2 and _shape[0] == 1:
                Data[key] = f[key][0]
            else:
                Data[key] = f[key]
        except AttributeError:
            continue

    return Data
