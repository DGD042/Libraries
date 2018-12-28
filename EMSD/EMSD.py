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

try:
    from EMSD.Extract_Data import Extract_Data as ExD
    from EMSD.Data_Man import Data_Man as DMan
    from EMSD.Functions import Gen_Functions as GFun
except ImportError:
    from Extract_Data import Extract_Data as ExD
    from Data_Man import Data_Man as DMan
    from Functions import Gen_Functions as GFun

# ------------------------
# Class
# ------------------------

class EMSD(object):
    '''
    ____________________________________________________________________________
    
    CLASS DESCRIPTION:
        
        This class is for Extracting, Manipulating and Saving Data 
        (EMSD). Functions in this class aims to manipulate times series 
        information with a preference for climate data, however you can 
        extract data from any file.
    
        This class is of free use and can be modify, if you have some 
        problem please contact the programmer to the following e-mails:
    
        - danielgondu@gmail.com 
        - dagonzalezdu@unal.edu.co
        - daniel.gonzalez17@eia.edu.co
    
        --------------------------------------
         How to use the library
        --------------------------------------
        
        You can use any function of the class separatley but using the 
        function Open_Data first would allow you to have a better 
        control of the full class.
    ____________________________________________________________________________

    '''

    def __init__(self):
        # Define Variables 
        self.File = None
        self.deli = ','
        self.St_Info = None
        self.FlagM = None
        self.Dates = None
        self.Values = None
        self.dtm = None
        # self.Date_Formats = ['%Y/%m/%d','%Y-%m-%d','%Y%m%d',\
        #                      '%d/%m/%Y','%d-%m-%Y','%d%m%Y',\
        #                      '%m/%d/%Y','%m-%d-%Y','%m%d%Y',\
        #                      '%Y/%d/%m','%Y-%d-%m''%Y%d%m',\
        #                      '%y/%m/%d','%y-%m-%d','%y%m%d',\
        #                      '%d/%m/%y','%d-%m-%y','%d%m%y',\
        #                      '%m/%d/%y','%m-%d-%y','%m%d%y',\
        #                      '%y/%d/%m','%y-%d-%m''%y%d%m']
        # Hour = [' %H%M','-%H%M',' %H:%M','-%H:%M','_%H%M']
        # self.DateTime_Formats = [i + j for i in self.Date_Formats for j in Hour]
        self.DateLab = ['DateH','DateD','DateM']
        self.DateNLab = ['DateHN','DateDN','DateMN']
        self.VLab = ['H','maxH','minH','D','maxD','minD','M','maxM','minM','NF','NNF']
        return

    def Open_Data(self,File,flagCompelete=True,DataBase={'DataBaseType':'txt','deli':',','colStr':(0,),
        'colData':(1,), 'row_skip':1,'flagHeader':True,'rowH':0,'row_end':0,'str_NaN':None,
        'num_NaN':None,'dtypeData':float}):
        '''
        DESCRIPTION:
    
            This is the principal function of the class, this function 
            takes the given information of the data and opens the 
            different files needed to be open.
        _______________________________________________________________________

        INPUT:
            :param File:         A str, File that needs to be open.
            :param flagComplete: A str, flag to know if the data is completed.
            :param DataBase:     A dict, a dictionary with all the data to
                                         apply the data extraction.
            This parameter has to have all the information of the document that
            wants to be extracted, if it does not have it i t assumes it's the
            defaulted of the function. For more information consult the usage 
            of the EMSD package.
        _______________________________________________________________________
        
        OUTPUT:
            :return Data: a dict, dictionary with all the data.
        
        '''
        # ------------
        # Parameters
        # ------------
        self.File = File # Files
        # Complete information
        self.Flags = dict()
        self.Values = dict()
        self.Dates = dict()
        self.DatesN = dict()
        self.DatesO = dict()

        if DataBase['DataBaseType'].lower() == 'txt' or DataBase['DataBaseType'].lower() == 'csv':
            # Verify parameters
            ParamsLab = ['deli','colStr','colData','row_skip','flagHeader','rowH','row_end','str_NaN',
                'num_NaN','dtypeData']
            Params = {'deli':',','colStr':(0,),
                'colData':(1,), 'row_skip':1,'flagHeader':True,'rowH':1,'row_end':0,'str_NaN':None,
                'num_NaN':None,'dtypeData':float}
            ParamsFunc = dict()
            for iPar,Par in enumerate(ParamsLab):
                try:
                    ParamsFunc[Par] = DataBase[Par]
                except KeyError:
                    ParamsFunc[Par] = Params[Par]
            # Data Open
            R = ExD.EDTXT(File,deli=ParamsFunc['deli'],colStr=ParamsFunc['colStr'],
                    colData=ParamsFunc['colData'],row_skip=ParamsFunc['row_skip'],
                    flagHeader=ParamsFunc['flagHeader'],rowH=ParamsFunc['rowH'],
                    row_end=ParamsFunc['row_end'],str_NaN=ParamsFunc['str_NaN'],
                    num_NaN=ParamsFunc['num_NaN'],dtypeData=ParamsFunc['dtypeData'])
            self.Data = R
            return R


        # Verify for irregular data extraction
        elif DataBase['DataBaseType'].lower() == 'ideam':
            DatesP, Values, Flags,self.FlagM, self.St_Info = self.EDIDEAM(File=self.File)
            keys = list(Values)
            keysDates = list(DatesP)
            if flagCompelete:
                for ikey,key in enumerate(keys):
                    # print(key)
                    DatesStr = [i.strftime(DUtil.Date_Formats[0]) for i in DatesP[key]]
                    try:
                        R = DMan.CompD(DatesStr,Values[key])
                        self.Dates[key]= R['DatesC']
                        self.DatesN[key] = R['DatesN']
                        self.Values[key] = R['VC']
                        R = DMan.CompD(DatesStr,Flags[key])
                        self.Flags[key] = R['VC']
                    except:
                        self.Dates[key]= DatesStr
                        self.DatesN[key]= DatesP
                        self.Values[key] = Values[key]
                        print('Problema en '+key)
        elif File[-3:] == 'xls' or File[-4:] == 'xlsx':
            # Excel data extraction
            DatesP, Values, self.Header = self.EDExcel(File=File,sheet=sheet,colDates=colDates,colData=colData,row_skip=row_skip,flagHeader=Header,row_end=row_end)
            self.DatesP = DatesP
            keys = list(Values)
            keysDates = list(DatesP)
            # Hour and minute verification
            flagHor = False
            flagMin = False
            if isinstance(DatesP[keysDates[0]][0],datetime):
                flagHor = True
                HH = [i.strftime('%H') for i in DatesP[keysDates[0]]]
                dif1 = DatesP[keysDates[0]][1]-DatesP[keysDates[0]][0]
                if dif1.seconds < 3600:
                    flagMin = True
                    MM = [i.strftime('%M') for i in DatesP[keysDates[0]]]
            # Complement data
            DatesStr = [i.strftime(self.Date_Formats[0]) for i in DatesP[keysDates[0]]]
            if flagHor and flagMin:
                DatesStrComp = [DatesStr[i]+' '+HH[i]+MM[i] for i in range(len(DatesStr))]
            elif flagHor:
                DatesStrComp = [DatesStr[i]+' '+HH[i]+'00' for i in range(len(DatesStr))]
            else:
                DatesStrComp = [DatesStr[i] for i in range(len(DatesStr))]

            if flagCompelete:
                for ikey,key in enumerate(keys):
                    if flagHor and flagMin:
                        self.Dates,self.Values[key],self.DatesN,self.DatesO = DMan.CompD(DatesStr,Values[key],flagHor,HH,flagMin,MM)
                    elif flagHor:
                        self.Dates,self.Values[key],self.DatesN,self.DatesO = DMan.CompD(DatesStr,Values[key],flagHor,HH)
                    else:
                        self.Dates,self.Values[key],self.DatesN,self.DatesO = DMan.CompD(DatesStr,Values[key])
                # dtm verification
                Dif = self.DatesN[1]-self.DatesN[0]
                if Dif.days == 0:
                    self.dtm = Dif.seconds/60
                else:
                    self.dtm = Dif.days
            else:
                self.Dates = DatesStrComp
                self.Values = Values
                self.DatesN = DatesP[keysDates[0]]
        elif File[-3:] == 'mat':
            self.f = sio.loadmat(Tot)

        return

    # Information Extraction
    def EDExcel(self,File=None,sheet=None,colDates=(0,),colData=(1,),row_skip=1,flagHeader=True,row_end=None,num_NaN=None):
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

    def ED(self,Tot,flagD=True,sheet=0,Header=True,ni=0,n=1,deli=';',rrr=2):
        '''
        DESCRIPTION:
        
            Con esta función se pretende extraer la información de un archivo, por 
            defecto siempre extrae la primera columna y luego la segunda, si es
            necesario extraer otra columna adicional se debe incluir un vector en n.

            Máximo se extraerán 4 columnas y los datos que se extraen deben estar en
            flias y las variables en columnas.
        _______________________________________________________________________

        INPUT:
            + Tot: Es la ruta completa del archivo que se va a abrir.
            + flagD: Para ver que tipo de archivo se abrirá.
                    True: Para archivos de Excel.
                    False: Para el resto.
            + sheet: Número de la hoja del documento de excel
            + Header: Se pregunta si el archivo tiene encabezado.
            + ni: es la columna inicial en donde se tomarán los datos de tiempo.
            + n: es la columna o columnas que se van a extraer.
            + deli: Delimitador que separa las variables en caso que no sean de Excel.
            + rrr: Fila en donde comienzan los valores
        _______________________________________________________________________
        
        OUTPUT:
            - Tiempo: Es la primera variable que se extrae del documento en formato txt.
            - Hora: Valores de hora, solo si len(ni) > 1.
            - V1: Variable 1 que se extrae como float
            - V2: Variable 2 que se extrae como float, depende de la cantidad de columnas.
            - V3: Variable 3 que se extrae como float, depende de la cantidad de columnas.
            - V4: Variable 4 que se extrae como float, depende de la cantidad de columnas.
        
        '''
        # Se mira cuantas columnas se va a extraer como m
        try:
            mi = len(ni)
        except:
            mi = 1 
        try:
            m = len(n)
        except:
            m = 1 

        # Contador de filas
        if Header == True:
            rr = rrr-1
            Head0 = ["" for k in range(1)] # Variable para los encabezados
            Head1 = ["" for k in range(1)] # Variable para los encabezados
            Head2 = ["" for k in range(1)] # Variable para los encabezados
            Head3 = ["" for k in range(1)] # Variable para los encabezados
            Head4 = ["" for k in range(1)] # Variable para los encabezados
        # else:
        #   rr = 0
        #   rrr = 0

        # Se inicializa la variable de tiempo
        if mi >= 1 and mi <= 2:
            Tiempo = ["" for k in range(1)]
            Hora = ["" for k in range(1)]
        else:
            utl.ShowError('ED','EMSD','No se pueden incluir más de dos columans con texto.')

        if flagD == True:

            # Se inicializan las variables que se van a extraer
            if mi == 1:
                Tiempo = []
            elif mi==2:
                Tiempo = []
                Hora = []

            if m >= 1 and m<=4:
                V1 = []
                V2 = []
                V3 = []
                V4 = []
            else:
                utl.ShowError('ED','EMSD',\
                    'No se pueden extraer tantas variables, por favor reduzca el número de variables')

            # --------------------------------------
            # Se abre el archivo de Excel
            # --------------------------------------
            book = xlrd.open_workbook(Tot)
            # Se carga la página en donde se encuentra el información
            S = book.sheet_by_index(sheet)

            # Se extren los encabezados
            if Header == True:
                if mi == 1:
                    Head0 = S.cell(rrr-1,ni).value
                else:
                    Head0 = S.cell(rrr-1,ni[0]).value # Se toman los encabezados
                    Head01 = S.cell(rrr-1,ni[1]).value  # Se toman los encabezados
                if m == 1:
                    Head1 = S.cell(rrr-1,n).value
                else:
                    Head1 = S.cell(rrr-1,n[0]).value
                    try:
                        Head2 = S.cell(rrr-1,n[1]).value
                    except:
                        Head2 = np.nan
                    try:
                        Head3 = S.cell(rrr-1,n[2]).value
                    except:
                        Head3 = np.nan
                    try:
                        Head4 = S.cell(rrr-1,n[3]).value
                    except:
                        Head4 = np.nan


            # Se genera un ciclo para extraer toda la información
            k = 1
            xx = rrr # Contador de las filas
            while k == 1:
                if mi == 1:
                    if m == 1:
                        try:
                            # Se verifica que se esté sacando una variable correcta
                            a = S.cell(xx,ni).value # Tiempo, debería estar completo
                            if a != '':
                                # Se carga el Tiempo
                                Tiempo.append(S.cell(xx,ni).value)
                                # Se carga la variable
                                V1.append(S.cell(xx,n).value)
                                if V1[-1] == '':
                                    V1[-1] = np.nan
                        except IndexError:
                            k = 2
                        xx += 1

            if Header == True:
            # Se guarda la información
                if mi == 1:
                    if m == 1:
                        Head = [Head0,Head1]
                        return Tiempo, V1, Head
            

        elif flagD == False:

            # Se inicializan las variables dependiendo del número de filas 
            # que se tengan

            if m>= 1 and m<=4:
                V1 = []
                V2 = []
                V3 = []
                V4 = []
            else:
                utl.ShowError('ED','EMSD',\
                    'No se pueden extraer tantas variables, por favor reduzca el número de variables')

            if isinstance(ni,list):
                nii = tuple(ni)
            else:
                ni = [ni]
                nii = (ni)

            if isinstance(ni,list):
                nn = tuple(n)
            else:
                n = [n]
                nn = (n)

            # Se extraen los encabezados
            if Header == True:
                Headni = np.char.decode(np.genfromtxt(Tot,delimiter=deli,usecols=nii,skip_header=rrr-1,dtype='S20',max_rows=1))
                if mi == 1:
                    Head0 = Headni
                else:
                    Head0 = Headni[0]
                    Head01 = Headni[1]
                
                Headn = np.char.decode(np.genfromtxt(Tot,delimiter=deli,usecols=nn,skip_header=rrr-1,dtype='S20',max_rows=1))
                if m == 1:
                    Head1 = Headn # Se toman los encabezados
                else:
                    Head1 = Headn[0] # Se toman los encabezados
                    try:
                        Head2 = Headn[1] # Se toman los encabezados
                    except:
                        Head2 = float('nan')
                    try:
                        Head3 = Headn[2] # Se toman los encabezados
                    except:
                        Head3 = float('nan')
                    try:
                        Head4 = Headn[3] # Se toman los encabezados
                    except:
                        Head4 = float('nan')
            
            # Se extraen los datos string
            if mi == 1:
                Tiempo = np.char.decode(np.genfromtxt(Tot,delimiter=deli,usecols=nii,skip_header=rrr,dtype='S20'))
            else:
                Fecha = np.char.decode(np.genfromtxt(Tot,delimiter=deli,usecols=nii,skip_header=rrr,dtype='S20'))
                Tiempo = Fecha[:,0]
                Hora = Fecha[:,1]
            # Se extraen las variables
            # print(rrr)
            Data = np.genfromtxt(Tot,delimiter=deli,usecols=nn,skip_header=rrr,dtype=float)
            if m == 1:
                V1 = Data
            else:
                try:
                    V1 = Data[:,0]
                except:
                    V1 = float('nan')
                try:
                    V2 = Data[:,1]
                except:
                    V2 = float('nan')
                try:
                    V3 = Data[:,2]
                except:
                    V3 = float('nan')
                try:
                    V4 = Data[:,3]
                except:
                    V4 = float('nan')


            # Se extrae la información

            # if m >= 1 and m<=4:
            #   V1 = ["" for k in range(1)]
            #   V2 = ["" for k in range(1)]
            #   V3 = ["" for k in range(1)]
            #   V4 = ["" for k in range(1)]
            # else:
            #   utl.ShowError('ED','EMSD',\
            #       'No se pueden extraer tantas variables, por favor reduzca el número de variables')

            # # Se abre el archivo que se llamó
            # f = open(Tot)

            # # Se inicializan las variables
            # ff = csv.reader(f, dialect='excel', delimiter= deli)
            # # ff = csv.reader(f, delimiter= deli)

            # r = 0 # Contador de filas
            # # Ciclo para todas las filas
            # for row in ff:
            #   if Header == True:
            #       if (rr == rrr-1 and r == rrr-1):
            #           if mi ==1:
            #               Head0 = row[ni] # Se toman los encabezados
            #           else:
            #               Head0 = row[ni[0]] # Se toman los encabezados
            #               Head01 = row[ni[1]] # Se toman los encabezados
            #           if m == 1:
            #               Head1 = row[n] # Se toman los encabezados
            #           else:
            #               Head1 = row[n[0]] # Se toman los encabezados
            #               try:
            #                   Head2 = row[n[1]] # Se toman los encabezados
            #               except:
            #                   Head2 = float('nan')
            #               try:
            #                   Head3 = row[n[2]] # Se toman los encabezados
            #               except:
            #                   Head3 = float('nan')
            #               try:
            #                   Head4 = row[n[3]] # Se toman los encabezados
            #               except:
            #                   Head4 = float('nan')

            #   if r == rrr:
            #       if mi ==1:
            #           Tiempo[0] = row[ni]
            #       else:
            #           Tiempo[0] = row[ni[0]]
            #           Hora[0] = row[ni[1]]

            #       if m == 1:
            #           try:
            #               V1[0] = float(row[n])
            #           except:
            #               V1[0] = float('nan')
            #       else:
            #           try:
            #               V1[0] = float(row[n[0]])
            #           except:
            #               V1[0] = float('nan')
            #           try:
            #               V2[0] = float(row[n[1]])
            #           except:
            #               V2[0] = float('nan')
            #           try:
            #               V3[0] = float(row[n[2]])
            #           except:
            #               V3[0] = float('nan')
            #           try:
            #               V4[0] = float(row[n[3]])
            #           except:
            #               V4[0] = float('nan')
                
            #   elif r > rr:
                    
            #       if mi ==1:
            #           Tiempo.append(row[ni])
            #       else:
            #           Tiempo.append(row[ni[0]])
            #           Hora.append(row[ni[1]])

            #       if m == 1:
            #           try:
                            
            #               V1.append(float(row[n]))
            #           except:
            #               V1.append(float('nan'))
            #       else:
            #           try:
            #               V1.append(float(row[n[0]]))
            #           except:
            #               V1.append(float('nan'))
            #           try:
            #               V2.append(float(row[n[1]]))
            #           except:
            #               V2.append(float('nan'))
            #           try:
            #               V3.append(float(row[n[2]]))
            #           except:
            #               V3.append(float('nan'))
            #           try:
            #               V4.append(float(row[n[3]]))
            #           except:
            #               V4.append(float('nan'))
            #   r += 1
            # f.close()
            if Header == True:
                if mi == 1:
                    if m == 1: 
                        Head = [Head0,Head1]
                        return Tiempo, V1, Head
                    elif m == 2:
                        Head = [Head0,Head1,Head2]
                        return Tiempo, V1, V2, Head
                    elif m == 3:
                        Head = [Head0,Head1,Head2,Head3]
                        return Tiempo, V1, V2, V3, Head
                    elif m == 4:
                        Head = [Head0,Head1,Head2,Head3,Head4]
                        return Tiempo, V1, V2, V3, V4, Head
                elif mi == 2:
                    if m == 1: 
                        Head = [Head0,Head01,Head1]
                        return Tiempo, Hora, V1, Head
                    elif m == 2:
                        Head = [Head0,Head01,Head1,Head2]
                        return Tiempo, Hora, V1, V2, Head
                    elif m == 3:
                        Head = [Head0,Head01,Head1,Head2,Head3]
                        return Tiempo, Hora, V1, V2, V3, Head
                    elif m == 4:
                        Head = [Head0,Head01,Head1,Head2,Head3,Head4]
                        return Tiempo, Hora, V1, V2, V3, V4, Head
            else:
                if mi == 1:
                    if m == 1: 
                        return Tiempo, V1
                    elif m == 2:
                        return Tiempo, V1, V2
                    elif m == 3:
                        return Tiempo, V1, V2, V3
                    elif m == 4:
                        return Tiempo, V1, V2, V3, V4
                elif mi == 2:
                    if m == 1: 
                        return Tiempo, Hora, V1
                    elif m == 2:
                        return Tiempo, Hora, V1, V2
                    elif m == 3:
                        return Tiempo, Hora, V1, V2, V3
                    elif m == 4:
                        return Tiempo, Hora, V1, V2, V3, V4

    def EDDAT_NCDCCOOP(self,Tot,Stations=None,Header=False,row_skip=0):
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

    def EDIDEAM(self,File=None):
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
                    DateP,Value,Flags = GFun.LoopDataDaily_IDEAM(DateP,Value,Flags,xRow_St,xCol_St,Years[-1],Lines,Sum2)
                elif (Station_Code[-1] == Station_Code[-2]) and (Med_Temp_Match[-1] == Med_Temp_Match[-2]) and (Var_Temp_Match[-1] == Var_Temp_Match[-2]):
                    DateP,Value,Flags = GFun.LoopDataDaily_IDEAM(DateP,Value,Flags,xRow_St,xCol_St,Years[-1],Lines,Sum2)
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
                    DateP,Value,Flags = GFun.LoopDataDaily_IDEAM(DateP,Value,Flags,xRow_St,xCol_St,Years[-1],Lines,Sum2)

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

    def EDWundergrounds(self,File):
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

    def EDnetCDFFile(self,File,VarDict=None,VarRangeDict=None):
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

    # Data Managment
    def CompDC(self,Dates,V,DateI,DateE,dtm=None):
        '''
        DESCRIPTION:
        
            This function completes or cut data from specific dates.
        _____________________________________________________________________

        INPUT:
            + Dates: Data date, it must be a string like this 'Year/month/day' 
                    the separator '/' could be change with any character. 
                    It must be a string vector or a date or datetime vector.
            + VC: Variable. 
            + DateI: Initial Date in date or datetime format.
            + DateE: Final Date in date or datetime format.
            + dtm: Time delta for the full data, if None it would use the
                   timedelta from the 2 values of the original data
        _____________________________________________________________________
        
        OUTPUT:
            - DatesC: Comlete date string vector.
            - V1C: Filled data values.
            - DatesN: Complete date Python datetime vector.
        
        '''
        
        V = np.array(V)
        Dates = np.array(Dates)
        # ---------------------
        # Error managment
        # ---------------------

        if isinstance(Dates[0],str) == False and isinstance(Dates[0],date) == False and isinstance(Dates[0],datetime) == False:
            Er = utl.ShowError('CompD','EDSM','Bad format in dates')
            raise Er
        if len(Dates) != len(V):
            Er = utl.ShowError('CompD','EDSM','Date and V are different length')
            raise Er
        if dtm != None and isinstance(dtm,timedelta) == False:
            Er = utl.ShowError('CompD','EDSM','Bad dtm format')
            raise Er

        # ---------------------
        # Dates Calculations
        # ---------------------
        # Original Dates
        if isinstance(Dates[0],str):
            DatesO = DUtil.Dates_str2datetime(Dates)
        else:
            DatesO = Dates

        if dtm == None:
            dtm = DatesO[1]-DatesO[0]
        DatesN = DUtil.Dates_Comp(DateI,DateE,dtm=dtm)
        
        # -------------------------------------------
        # Data Extraction
        # -------------------------------------------

        # Filled data
        if isinstance(V[0],str):
            VC = (np.empty(len(DatesN))*np.nan).astype(str)
        else:
            VC = (np.empty(len(DatesN))*np.nan)
        
        DatesN = np.array(DatesN)
        DatesO = np.array(DatesO)
        for iF,F in enumerate(DatesO):
            x = np.where(DatesN == F)[0]
            if len(x) == 1:
                VC[x] = V[iF]
            elif len(x) > 1:
                VC[x[0]] = V[iF]
        
        DatesC = DUtil.Dates_datetime2str(DatesN)

        Results = {'DatesC':DatesC,'DatesN':DatesN,'VC':VC}
        return Results

    def Ca_E(self,FechaC,V1C,dt=24,escala=1,op='mean',flagMa=False,flagDF=False):
        '''
        DESCRIPTION:
        
            Con esta función se pretende cambiar de escala temporal los datos,
            agregándolos a diferentes escalas temporales, se deben insertar series
            completas de tiempo.

            Los datos faltantes deben estar como NaN.
        _______________________________________________________________________

        INPUT:
            + FechaC: Fecha de los datos organizada como 'año/mes/dia - HHMM' 
                      los '/' pueden ser substituidos por cualquier caracter. 
                      Debe ser un vector string y debe tener años enteros.
            + V1C: Variable que se desea cambiar de escala temporal. 
            + dt: Delta de tiempo para realizar la agregación, depende de la naturaleza
                  de los datos.
                  Si se necesitan datos mensuales, el valor del dt debe ser 1.
            + escala: Escala a la cual se quieren pasar los datos:
                    0: de minutal o horario.
                    1: a diario.
                    2: a mensual, es necesario llevarlo primero a escala diaria.
            + op: Es la operación que se va a llevar a cabo para por ahora solo responde a:
                  'mean': para obtener el promedio.
                  'sum': para obtener la suma.
            + flagMa: Para ver si se quieren los valores máximos y mínimos.
                    True: Para obtenerlos.
                    False: Para no calcularos.
            + flagDF: Para ver si se quieren los datos faltantes por mes, solo funciona
                      en los datos que se dan diarios.
                    True: Para calcularlos.
                    False: Para no calcularos.
        _______________________________________________________________________
        
        OUTPUT:
            - FechaEs: Nuevas fechas escaladas.
            - FechaNN: Nuevas fechas escaladas como vector fechas. 
            - VE: Variable escalada.
            - VEMax: Vector de máximos.
            - VEMin: Vector de mínimos.
        '''
        # Se desactivan los warnings en este codigo para que corra más rápido, los
        # warnings que se generaban eran por tener realizar promedios de datos NaN
        # no porque el código tenga un problema en los cálculos.
        warnings.filterwarnings('ignore')

        if escala > 2:
            utl.ShowError('EMSD','Ca_E','Todavía no se han programado estas escalas')

        # -------------------------------------------
        # Inicialización de variables
        # -------------------------------------------
        # Se inicializan las variables que se utilizarán
        FechaNN = ["" for k in range(1)]
        FechaEs = ["" for k in range(1)]
        VE = ["" for k in range(1)]
        VEMax = ["" for k in range(1)]
        VEMin = ["" for k in range(1)]

        NF = [] # Porcentaje de datos faltantes
        NNF = [] # Porcentaje de datos no faltantes
        rr = 0

        # -------------------------------------------
        # Vector de fechas
        # -------------------------------------------

        # Se toman los años
        yeari = int(FechaC[0][0:4]) # Año inicial
        yearf = int(FechaC[len(FechaC)-1][0:4]) # Año final
        Sep = FechaC[0][4] # Separador de la Fecha


        # Los años se toman para generar el output de FechasEs
        if escala == 0 or escala == 1: # Para datos horarios o diarios
            for result in perdelta(date(int(yeari), 1, 1), date(int(yearf)+1, 1, 1), timedelta(days=1)):
                FR = result.strftime('%Y'+Sep+'%m'+Sep+'%d') # Fecha
                if escala == 0:
                    for i in range(0,24):
                        if rr == 0:
                            FechaNN[0] = result
                            if i < 10:
                                FechaEs[rr] = FR + '-0' +str(i)+'00'
                            else:
                                FechaEs[rr] = FR + '-' +str(i)+'00'
                        else:
                            FechaNN.append(result)
                            if i < 10:
                                FechaEs.append(FR + '-0' +str(i)+'00')
                            else:
                                FechaEs.append(FR + '-' +str(i)+'00')

                        rr += 1 # Se suman las filas
                elif escala == 1:
                    if rr == 0:
                        FechaNN[0] = result
                        FechaEs[rr] = FR
                    else:
                        FechaNN.append(result)
                        FechaEs.append(FR)
                    rr += 1
        if escala == 2:
            x = 0
            for i in range(int(yeari),int(yearf)+1):
                for j in range(1,13):
                    if i == int(yeari) and j == 1:
                        FechaNN[0] = date(i,j,1)
                        FechaEs[0] = FechaNN[0].strftime('%Y'+Sep+'%m')
                    else:
                        FechaNN.append(date(i,j,1))
                        FechaEs.append(FechaNN[x].strftime('%Y'+Sep+'%m'))
                    x += 1
        # -------------------------------------------
        # Cálculo del escalamiento
        # -------------------------------------------
        dtt = 0 # Contador de la diferencia
        if op == 'mean':
            if escala == 0 or escala == 1: 
                # Ciclo para realizar el agregamiento de los datos
                for i in range(0,len(V1C),dt): 
                    dtt = dtt + dt # Se aumenta el número de filas en el contador
                    if i == 0:
                        q = np.isnan(V1C[i:dtt])
                        qq = sum(q)
                        if qq > dt/2:
                            VE[0] = float('nan')
                            if flagMa == True:
                                VEMax[0] = float('nan')
                                VEMin[0] = float('nan')
                        else:
                            try:
                                VE[0] = float(np.nanmean(V1C[i:dtt]))
                            except ValueError:
                                VE[0] = float('nan')
                            if flagMa == True:
                                try:
                                    VEMax[0] = float(np.nanmax(V1C[i:dtt]))
                                except ValueError:
                                    VEMax[0] = float('nan')
                                try:
                                    VEMin[0] = float(np.nanmin(V1C[i:dtt]))
                                except ValueError:
                                    VEMin[0] = float('nan')
                    else:
                        q = np.isnan(V1C[i:dtt])
                        qq = sum(q)
                        if qq > dt/2:
                            VE.append(float('nan'))
                            if flagMa == True:
                                VEMax.append(float('nan'))
                                VEMin.append(float('nan'))
                        else:
                            try:
                                VE.append(float(np.nanmean(V1C[i:dtt])))
                            except ValueError:
                                VE.append(float('nan'))
                            if flagMa == True:
                                try:
                                    VEMax.append(float(np.nanmax(V1C[i:dtt])))
                                except ValueError:
                                    VEMax.append(float('nan'))
                                try:
                                    VEMin.append(float(np.nanmin(V1C[i:dtt])))
                                except ValueError:
                                    VEMin.append(float('nan'))
            elif escala == 2: # promedio mensual
                d = 0
                for i in range(int(yeari),int(yearf)+1):
                    for j in range(1,13):
                        Fi = date(i,j,1)
                        if j == 12:
                            Ff = date(i+1,1,1)
                        else:
                            Ff = date(i,j+1,1)
                        DF = Ff-Fi
                        dtt = dtt + DF.days # Delta de días
                        if i == int(yeari) and j == 1:
                            q = np.isnan(V1C[d:dtt])
                            qq = sum(q)
                            NF.append(qq/len(V1C[d:dtt]))
                            NNF.append(1-NF[-1])    
                            if qq > DF.days/2:
                                VE[0] = float('nan')
                                if flagMa == True:
                                    VEMax[0] = float('nan')
                                    VEMin[0] = float('nan')
                            else:
                                try:
                                    VE[0] = float(np.nanmean(V1C[d:dtt]))
                                except ValueError:
                                    VE[0] = float('nan')
                                if flagMa == True:
                                    try:
                                        VEMax[0] = float(np.nanmax(V1C[d:dtt]))
                                    except ValueError:
                                        VEMax[0] = float('nan')
                                    try:
                                        VEMin[0] = float(np.nanmin(V1C[d:dtt]))
                                    except ValueError:
                                        VEMin[0] = float('nan')
                        else:
                            q = np.isnan(V1C[d:dtt])
                            qq = sum(q)
                            NF.append(qq/len(V1C[d:dtt]))
                            NNF.append(1-NF[-1])
                            if qq > DF.days/2:
                                VE.append(float('nan'))
                                if flagMa == True:
                                    VEMax.append(float('nan'))
                                    VEMin.append(float('nan'))
                            else:
                                try:
                                    VE.append(float(np.nanmean(V1C[d:dtt])))
                                except ValueError:
                                    VE.append(float('nan'))
                                if flagMa == True:
                                    try:
                                        VEMax.append(float(np.nanmax(V1C[d:dtt])))
                                    except ValueError:
                                        VEMax.append(float('nan'))
                                    try:
                                        VEMin.append(float(np.nanmin(V1C[d:dtt])))
                                    except ValueError:
                                        VEMin.append(float('nan'))
                        d = dtt


        elif op == 'sum':
            if escala == 0 or escala == 1: 
                # Ciclo para realizar el agregamiento de los datos
                for i in range(0,len(V1C),dt): 
                    dtt = dtt + dt # Se aumenta el número de filas en el contador
                    if i == 0:
                        q = np.isnan(V1C[i:dtt])
                        qq = sum(q)
                        
                        if qq > dt/2:
                            VE[0] = float('nan')
                            if flagMa == True:
                                VEMax[0] = float('nan')
                                VEMin[0] = float('nan')
                        else:
                            try:
                                VE[0] = float(np.nansum(V1C[i:dtt]))
                            except ValueError:
                                VE[0] = float('nan')
                            if flagMa == True:
                                try:
                                    VEMax[0] = float(np.nanmax(V1C[i:dtt]))
                                except ValueError:
                                    VEMax[0] = float('nan')
                                try:
                                    VEMin[0] = float(np.nanmin(V1C[i:dtt]))
                                except ValueError:
                                    VEMin[0] = float('nan')
                    else:
                        q = np.isnan(V1C[i:dtt])
                        qq = sum(q)
                        if qq > dt/2:
                            VE.append(float('nan'))
                            if flagMa == True:
                                VEMax.append(float('nan'))
                                VEMin.append(float('nan'))
                        else:
                            try:
                                VE.append(float(np.nansum(V1C[i:dtt])))
                            except ValueError:
                                VE.append(float('nan'))
                            if flagMa == True:
                                try:
                                    VEMax.append(float(np.nanmax(V1C[i:dtt])))
                                except ValueError:
                                    VEMax.append(float('nan'))
                                try:
                                    VEMin.append(float(np.nanmin(V1C[i:dtt])))
                                except ValueError:
                                    VEMin.append(float('nan'))
            elif escala == 2: # Agregamiento mensual
                d = 0
                for i in range(int(yeari),int(yearf)+1):
                    for j in range(1,13):
                        Fi = date(i,j,1)
                        if j == 12:
                            Ff = date(i+1,1,1)
                        else:
                            Ff = date(i,j+1,1)
                        DF = Ff-Fi
                        dtt = dtt + DF.days # Delta de días
                        if i == int(yeari) and j == 1:
                            q = np.isnan(V1C[d:dtt])
                            qq = sum(q)
                            NF.append(qq/len(V1C[d:dtt]))
                            NNF.append(1-NF[-1])    
                            if qq > DF.days/2:
                                VE[0] = float('nan')
                                if flagMa == True:
                                    VEMax[0] = float('nan')
                                    VEMin[0] = float('nan')
                            else:
                                try:
                                    VE[0] = float(np.nansum(V1C[d:dtt]))
                                except ValueError:
                                    VE[0] = float('nan')
                                if flagMa == True:
                                    try:
                                        VEMax[0] = float(np.nanmax(V1C[d:dtt]))
                                    except ValueError:
                                        VEMax[0] = float('nan')
                                    try:
                                        VEMin[0] = float(np.nanmin(V1C[d:dtt]))
                                    except ValueError:
                                        VEMin[0] = float('nan')
                        else:
                            q = np.isnan(V1C[d:dtt])
                            qq = sum(q)
                            NF.append(qq/len(V1C[d:dtt]))
                            NNF.append(1-NF[-1])
                            if qq > DF.days/2:
                                VE.append(float('nan'))
                                if flagMa == True:
                                    VEMax.append(float('nan'))
                                    VEMin.append(float('nan'))
                            else:
                                try:
                                    VE.append(float(np.nansum(V1C[d:dtt])))
                                except ValueError:
                                    VE.append(float('nan'))
                                if flagMa == True:
                                    try:
                                        VEMax.append(float(np.nanmax(V1C[d:dtt])))
                                    except ValueError:
                                        VEMax.append(float('nan'))
                                    try:
                                        VEMin.append(float(np.nanmin(V1C[d:dtt])))
                                    except ValueError:
                                        VEMin.append(float('nan'))
                        d = dtt

        # -------------------------------------------
        # Se dan los resultados
        # -------------------------------------------
        if flagMa == True:
            if  flagDF:
                return FechaEs, FechaNN, VE, VEMax, VEMin, NF,NNF
            else:
                return FechaEs, FechaNN, VE, VEMax, VEMin
        elif flagMa == False:
            if flagDF:
                return FechaEs, FechaNN, VE,NF,NNF
            else:
                return FechaEs, FechaNN, VE

    def MIA(self,FechasC,Fechas,Data):
        '''
        DESCRIPTION:
        
            Con esta función se pretende encontrar la cantidad de datos faltantes de
            una serie.
        _______________________________________________________________________

        INPUT:
            + FechaC: Fecha inicial y final de la serie original.
            + Fechas: Vector de fechas completas de las series.
            + Data: Vector de fechas completo
        _______________________________________________________________________
        
        OUTPUT:
            N: Vector con los datos NaN.
            FechaNaN: Fechas en donde se encuentran los valores NaN
        '''
        # Se toman los datos 
        Ai = Fechas.index(FechasC[0])
        Af = Fechas.index(FechasC[1])

        DD = Data[Ai:Af+1]

        q = np.isnan(DD) # Cantidad de datos faltantes
        qq = ~np.isnan(DD) # Cantidad de datos no faltantes
        FechaNaN = [Fechas[k] for k in q] # Fechas en donde se encuentran los NaN

        # Se sacan los porcentajes de los datos faltantes
        NN = sum(q)/len(DD)
        NNN = sum(qq)/len(DD)

        N = [NN,NNN]

        return N, FechaNaN

    # Writting Data
    def WriteD(self,path,fieldnames,data,deli=','):
        '''
        DESCRIPTION:
        
            Esta función fue extraída de la página web: 
            https://dzone.com/articles/python-101-reading-and-writing
            Y los códigos fueron creados por Micheal Driscoll

            Con esta función se pretende guardar los datos en un .csv.
        _______________________________________________________________________

        INPUT:
            + path: Ruta con nombre del archivo.
            + fieldnames: Headers del archivo.
            + data: Lista con todos los datos.
        _______________________________________________________________________
        
        OUTPUT:
            Esta función arroja un archivo .csv con todos los datos.
        '''
        with open(path, "w") as out_file:
            writer = csv.DictWriter(out_file, delimiter=deli, fieldnames=fieldnames)

            writer.writeheader()
            for row in data:
                writer.writerow(row)

    def WriteCSVSeries(self,Dates,Values,Headers,deli=None,flagHeader=True,Pathout='',Name='',NaNdata=''):
        '''
        DESCRIPTION:
        
            This function aims to save data in a csv file from.
        _______________________________________________________________________

        INPUT:
            + Dates: Dates of the series.
            + Values: Value dictionary.
            + deli: delimeter of the data.
            + Headers: Headers of the data, Labels of the Dates and Values.
            + flagHeaders: if headers are needed.
            + Pathout: Saving directory.
            + Name: File Name with extension.
            + NaNdata: Manage the NaN type data.
        _______________________________________________________________________
        
        OUTPUT:
            
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

    def WriteFile(self,Labels,Values,deli=None,Headers=None,flagHeader=False,Pathout='',Name='',NaNdata=''):
        '''
        DESCRIPTION:
        
            This function aims to save data in a csv file from.
        _______________________________________________________________________

        INPUT:
            + Labels: String left labels, this acts as the keys.
            + Values: Value dictionary.
            + deli: delimeter of the data.
            + Headers: Headers of the data, defaulted to None.
            + flagHeaders: if headers are needed.
            + Pathout: Saving directory.
            + Name: File Name with extension.
            + NaNdata: Manage the NaN type data.
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

    def Writemat(self,Dates,Data,savingkeys,datekeys=None,datakeys=None,pathout='',Names='Data'):
        '''
        DESCRIPTION:
        
            This function aims to save data in a mat file.
        _______________________________________________________________________

        INPUT:
            + Dates: String date dictionary or arrays.
            + Values: Value dictionary or arrays.
            + savingkeys: New keys for savng files, first have to
                          be the dates.
            + datekeys: dates directory keys organized.
            + datakeys: data directory keys organized.
            + Pathout: Saving directory.
            + Name: File Name without extension.
        _______________________________________________________________________
        
        OUTPUT:
            Save a mat file.
        '''
        # Create Folder
        utl.CrFolder(pathout)
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

    def St_Document(self,Pathout='',Name='Stations_Info',St_Info_Dict=None,Data_Flags=None):
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

    # Other Functions
    def GetValues(self):
        '''
        DESCRIPTION:
    
            This function 
        _______________________________________________________________________

        INPUT:
        _______________________________________________________________________
        
        OUTPUT:
            - Dates: Dictionary with dates.
            - Values: Dictionary with values.
            - keys: dictionary keys.
        '''
        if self.File[-3:] == 'mat':
            return self.f
        else:
            return self.Dates,self.DatesN, self.Values
            
    def Ca_EC(self,Date=None,V1=None,op='mean',key=None,dtm=None,op2=None,op3=None,flagHour=True,flagDaily=True):
        '''
        DESCRIPTION:

            This function takes the data in the given format a passes it to 
            hourly, daily and mothly data with the specified operation. 
        _______________________________________________________________________

        INPUT:
            + Date: Data date, it must be a string like this 'Year/month/day' 
                    the separator '/' could be change with any character. 
                    It must be a string vector.
            + V1: Variable that needs to be changed.
            + op: Operation for the temporal aggregation, it only responds to:
                  'mean': temporal mean.
                  'sum': temporal sumatory.
            + key: key of the dictionary, only needed if V1 is None.
        '''

        DateLab = ['DateH','DateD','DateM']
        DateNLab = ['DateHN','DateDN','DateMN']
        VLab = ['H','maxH','minH','D','maxD','minD','M','maxM','minM','NF','NNF']
        DatesC = dict()
        DatesNC = dict()
        VC = dict()

        # Value verification
        if Date is None and self.Dates is None:
            Er = utl.ShowError('Ca_EC','EDSM','No dates was added')
            return None
        elif Date is None:
            Date = self.Dates
        if V1 is None and self.Values is None:
            Er = utl.ShowError('Ca_EC','EDSM','No data was added')
            return None
        elif V1 is None:
            if key is None:
                Er = utl.ShowError('Ca_EC','EDSM','key value is needed')
                return None
            else:
                V1 = self.Values[key]
        if dtm is None:
            dtm = self.dtm

        # Temporal verification
        lenDates = len(Date[0])
        if flagHour:
            D_Formats = DUtil.DateTime_Formats
            dt = int(60/dtm)
            if dt != 60:
                # Hourly
                DateH, DateHN, VH,VmaxH,VminH = DMan.Ca_E(Date,V1,dt,0,op=op,flagMa=True,flagDF=False,flagNaN=True)
                DatesC[DateLab[0]] = DateH
                DatesNC[DateNLab[0]] = DateHN
                VC[VLab[0]] = VH
                VC[VLab[1]] = VmaxH
                VC[VLab[2]] = VminH
        else:
            D_Formats = DUtil.Date_Formats
            VH = V1
        # Daily data
        if flagDaily:
            if op2 == None:
                op2 = op
            if flagHour:
                dt = 24
                DateD, DateDN, VD,VmaxD,VminD = DMan.Ca_E(DateH,VH,dt,1,op=op2,flagMa=True,flagDF=False)
            else:
                DateD, DateDN, VD,VmaxD,VminD = DMan.Ca_E(Date,V1,dtm,1,op=op2,flagMa=True,flagDF=False)
            DatesC[DateLab[1]] = DateD
            DatesNC[DateNLab[1]] = DateDN
            VC[VLab[3]] = VD
            VC[VLab[4]] = VmaxD
            VC[VLab[5]] = VminD
        else:
            VC[VLab[3]] = V1
            VD = V1
            DateD = Date


        if op3 == None:
            op3 = op

        DateM, DateMN, VM,VmaxM,VminM,VNF,VNNF = DMan.Ca_E(DateD,VD,1,2,op=op3,flagMa=True,flagDF=True)
        DatesC[DateLab[2]] = DateM
        DatesNC[DateNLab[2]] = DateMN
        VC[VLab[6]] = VM
        VC[VLab[7]] = VmaxM
        VC[VLab[8]] = VminM
        VC[VLab[9]] = VNF
        VC[VLab[10]] = VNNF

        return DatesC, DatesNC, VC



