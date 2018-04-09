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
# Functions
# ------------------------

'''

    
    Este paquete contiene funciones que perimite, cargar la información
    de la información primaria y secundaria de la base de datos 
    meteorológica para la generación de análisis.
_________________________________________________________________________

'''

# Carga de información
def LoadStationsInfo(endingmatR='_CBF_MR'):
    '''
    DESCRIPTION:
    
        Función pra cargar las rutas de los diferentes archivos, este se 
        basa en los archivos en donde se tiene la base de datos de cada
        estación ubicados en Tesis_MscR/03_Data/, cada archivo debe tener 
        una hoja llamada Inf_Data que describe la hoja que se tomará para
        extraer los datos de las estaciones.

        Cada hoja debe tener una organización específica, mirar un ejemplo 
        para tener la organización determinada.

        Esta función cargará las rutas a los archivos .mat que se tienen
        dentro de la carpeta Est_Information/ y los resultados de la 
        carpeta Tesis_Msc/03_Data/, cada ruta de archivos contiene
        una manera diferente de llamar la información y tiene una forma
        diferente de llamar las variables que contiene, esto se debe tener
        en cuenta a la hora de crear los archivos con las diferentes bases de
        datos.

        En este caso se utilizarán los parámetros utilizados para crear
        las bases de datos utilizados por el programador.

    _________________________________________________________________________

    INPUT:
        :param endingmatR: A str, Terminación del archivo .mat para los datos
                                 filtrados.
    _________________________________________________________________________
    
    OUTPUT:

        :return Arch:         A dict, Diccionario con las listas de las 
                                      rutas a los archivos de las 
                                      diferentes fuentes de información 
                                      original y resultante.
        :return StationsInfo: A dict, Diccionario con la información de las 
                                      estaciones.
    '''
    # ---------------------
    # Parámetros
    # ---------------------
    # Rutas
    PathDataOriginal = 'Est_Information/'
    PathDataResults = 'Tesis_MscR/03_Data/'
    PathDataImp = 'Tesis_MscR/03_Data/'
    # Las listas se organizan como [Manizales,Medellin,Amazonas,USA]
    # Rutas originales de los datos
    DataImpNames = ['Data_Imp','Data_Imp_Med','Data_Imp_Amazon',
            'Data_Imp_Wunder_USA','Data_Imp_Wunder_USA','Data_Imp_Wunder_USA',
            'Data_Imp_Wunder_USA','Data_Imp_Wunder_USA']
    Original_DataBases = ['IDEA_CDIAC/02_mat/Manizales/',
            'Medellin_SIATA/02_mat/',
            'Amazon_LBA/ATTO/02_mat/',
            'Wunder/02_mat/',
            'Wunder/02_mat/',
            'Wunder/02_mat/',
            'Wunder/02_mat/',
            'Wunder/02_mat/',
            ]
    Filt_DataBases = ['02_Col_Data/01_Manizales/01_mat/01_CFilt/',
            '02_Col_Data/02_Medellin/01_mat/01_CFilt/',
            '03_Amazonas/01_mat/01_CFilt/',
            '01_USA/01_mat/01_CFilt/',
            '01_USA/01_mat/01_CFilt/',
            '01_USA/01_mat/01_CFilt/',
            '01_USA/01_mat/01_CFilt/',
            '01_USA/01_mat/01_CFilt/',
            ]

    # ---------------------
    # Original
    # ---------------------
    # Parametros
    DataBasesP = ['Manizales','Medellin','Amazonas','Wunder','WunderCentro',
            'WunderSur','WunderEste','WunderOeste']
    TimeScale = ['Original','Horarios','Diarios','CFilt','Paths']
    StationInfo = ['ID','Name','Name_Arch','Altura','Latitud','Longitud']
    StationInfoType = {'ID':str,'Name':str,'Name_Arch':str,'Altura':float,
            'Latitud':float,'Longitud':float}

    Arch = dict()
    StationsInfo = dict()
    for DB in DataBasesP:
        Arch[DB] = dict()
        for TS in TimeScale:
            Arch[DB][TS] = []
        StationsInfo[DB] = dict()
        for SI in StationInfo:
            StationsInfo[DB][SI] = []

    # Se buscan los archivos de importación
    ArchDataImp = [PathDataImp+DataImpNames[0]+'.xlsx',
            PathDataImp+DataImpNames[1]+'.xlsx',
            PathDataImp+DataImpNames[2]+'.xlsx',
            PathDataImp+DataImpNames[3]+'.xlsx',
            PathDataImp+DataImpNames[3]+'.xlsx',
            PathDataImp+DataImpNames[3]+'.xlsx',
            PathDataImp+DataImpNames[3]+'.xlsx',
            PathDataImp+DataImpNames[3]+'.xlsx',
            ]

    Sheet = ['Center','South','East','West']

    # Se cargan los archivos
    for iar,ar in enumerate(ArchDataImp):
        # Se carga la información de las estaciones
        book = xlrd.open_workbook(ar)
        if iar >= 4:
            S = book.sheet_by_name(Sheet[iar-4])
        else:
            SS = book.sheet_by_name('inf_Data')
            # Datos de las estaciones
            Hoja = int(SS.cell(2,2).value)
            S = book.sheet_by_name(str(Hoja))
        NEst = int(S.cell(1,0).value)

        x = 3 # Filas
        # Ciclo para tomar los datos
        for N in range(NEst):
            y = [0,1,2,5,9,10]
            for iSI,SI in enumerate(StationInfo):
                if SI == 'ID':
                    try:
                        StationsInfo[DataBasesP[iar]][SI].append(str(int(S.cell(x,y[iSI]).value)))
                    except ValueError:
                        StationsInfo[DataBasesP[iar]][SI].append(str(S.cell(x,y[iSI]).value))
                    except IndexError:
                        break
                else:
                    try:
                        StationsInfo[DataBasesP[iar]][SI].append(StationInfoType[SI]
                                (S.cell(x,y[iSI]).value))
                    except ValueError:
                        StationsInfo[DataBasesP[iar]][SI].append(np.nan)
            x += 1

        # Se carga la información
        for iSt,St in enumerate(StationsInfo[DataBasesP[iar]]['Name_Arch']):
            # Original
            Arch[DataBasesP[iar]]['Original'].append(PathDataOriginal
                +Original_DataBases[iar]+St+'.mat')
            # Horarios
            Arch[DataBasesP[iar]]['Horarios'].append(PathDataOriginal
                +Original_DataBases[iar]+'01_Horario/'+St+'_THT.mat')
            # Diarios
            Arch[DataBasesP[iar]]['Diarios'].append(PathDataOriginal
                +Original_DataBases[iar]+'02_Diario/'+St+'_TdiasM.mat')
            # CFilt
            Arch[DataBasesP[iar]]['CFilt'].append(PathDataResults
                +Filt_DataBases[iar]+St+endingmatR+'.mat')


        Arch[DataBasesP[iar]]['Paths'].append(PathDataOriginal
                +Original_DataBases[iar])
        Arch[DataBasesP[iar]]['Paths'].append(PathDataOriginal
                +Original_DataBases[iar]+'02_Diario/')
        Arch[DataBasesP[iar]]['Paths'].append(PathDataOriginal
                +Original_DataBases[iar]+'01_Horario/')
        Arch[DataBasesP[iar]]['Paths'].append(PathDataResults+Filt_DataBases[iar])
    return StationsInfo,Arch

def LoadData(ArchP,DataBase,irow=0,TimeScale='Diarios'):
    '''
    DESCRIPTION:
        
        Función para cargar la información de cada archivo de las estaciones
    _________________________________________________________________________

    INPUT:
        :param DataBase:  A str, Base de datos que se va a extraer.
        :param irow:      An int, Número del archivo que se va a extraer.
        :param TimeScale: A str, Escala temporal que se abrirá.
    _________________________________________________________________________
    
    OUTPUT:
        
        :return Met:  A list, Lista con toda la metadata del
                              archivo.
        :return Data: A dict, Diccionario con toda la información del
                              archivo.
    '''

    Arch = ArchP[DataBase][TimeScale]

    f = sio.loadmat(Arch[irow])
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

def LoadDataVerify(Arch,VerifData,irow=0):
    '''
    DESCRIPTION:
        
        Función para cargar la información de cada archivo de las estaciones
        a partir de un set de datos a verificar.
    _________________________________________________________________________

    INPUT:
        :param Arch:     A list, Rutas de archivos que se abrirán.
        :param VerifData: A list, Nombre de las variables que se extraerán.
        :param irow:      An int, Número del archivo que se va a extraer.
    _________________________________________________________________________
    
    OUTPUT:
        
        :return Data:  A dict, Diccionario con toda la información del
                               archivo.
        :return flag: A dict, Diccionario con booleanos, denotando si se 
                               se encontró o no la información.
    '''
    # Variables
    flag = dict()
    Data = dict()

    f = sio.loadmat(Arch[irow])
    Keys = list(f)
    # Extracción info
    for key in VerifData:
        try:
            # Se verifica la existencia de la variable 
            a = Keys.index(key)
            flag[key] = True
        except ValueError:
            flag[key] = False
            continue
        try:
            _shape = f[key].shape
            lenshape = len(f[key].shape)
            if lenshape == 2 and _shape[0] == 1:
                Data[key] = f[key][0]
            else:
                Data[key] = f[key]
        except AttributeError:
            continue

    return Data,flag


