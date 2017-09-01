# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 04/03/2016
#______________________________________________________________________________
#______________________________________________________________________________


# ----------------------------
# Se importan las librerias
# ----------------------------
# Manejo de datos
import numpy as np
# Importar datos
import csv
import xlrd
import xlsxwriter as xlsxwl
import scipy.io as sio
from scipy import stats as st
# Graficos 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates
import matplotlib.mlab as mlab
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
import matplotlib.image as mpimg
from matplotlib import animation
# Sistemas
import os
import glob as gl
import subprocess
import sys
import warnings
# Se importan los paquetes para manejo de fechas
from datetime import date, datetime, timedelta
import time

# ----------------------------
# Librerías personales
# ----------------------------
from Utilities import Utilities as utl
from Utilities import DatesUtil as DUtil
from AnET import CorrSt as cr
from AnET import CFitting as CF
from Hydro_Analysis import Hydro_Plotter as HyPL
from Hydro_Analysis import Hydro_Analysis as HA
from Hydro_Analysis import Thermo_An as TA
from AnET import AnET as anet
from EMSD import EMSD
from BPumpL.BPumpL import BPumpL as BP

class Scatter_Gen(object): 
    '''
    DESCRIPTION:

        Clase par abrir los documentos que se necesitan para hacer los diferentes
        estudios de los diagramas de dispersión.
    '''

    def __init__(self,PathDataImp='',PathData='',PathData2='',DataImp='Data_Imp',endingmat='',TipoD='IDEA',PathImg=''):
        '''
        DESCRIPTION:

            Clase para abrir los documentos que se necesitan para hacer los diferentes
            estudios de los diagramas de dispersión.
        _________________________________________________________________________

        INPUT:
            + DataImp: Hoja de Excel en donde se encuentra la información.
        _________________________________________________________________________

        OUTPUT:
        '''

        self.PathImg = PathImg
        # ----------------------------
        # Constantes
        # ----------------------------
        # Información para gráficos
        LabelV = ['Precipitación','Temperatura','Humedad Relativa','Presión','Humedad Especifica']
        LabelVU = ['Precipitación [mm]','Temperatura [°C]','Hum. Rel. [%]','Presión [hPa]','Hum. Espec. [kg/kg]']
        
        self.mmHg2hPa = 1.3332239

        # Projections
        self.epsgWGS84 = 4326
        self.epsgMAGNA = 3116

        # Diccionarios con las diferentes variables
        self.Variables = ['PrecC','TC','HRC','PresC','qC']
        self.Variables_N = dict()
        self.Variables2 = dict()
        self.Variables3 = dict()
        self.Variables4 = dict()
        self.VariablesF = dict()
        self.VariablesComp = dict()
        self.LabelV = dict()
        self.LabelVU = dict()
        for iv,v in enumerate(self.Variables):
            self.Variables_N[v] = v
            self.Variables2[v] = v[:-1]+'2'
            self.Variables3[v] = v[:-1]+'3'
            self.Variables4[v] = v[:-1]+'4'
            self.VariablesF[v] = v[:-1]+'_F'
            self.VariablesComp[v] = v[:-1]+'Comp'
            self.LabelV[v] = LabelV[iv]
            self.LabelVU[v] = LabelVU[iv]
        # Información para gráficos
        LabelV = ['Precipitación','Temperatura','Humedad Relativa','Presión','Humedad Especifica']
        LabelVU = ['Precipitación [mm]','Temperatura [°C]','Hum. Rel. [%]','Presión [hPa]','Hum. Espec. [kg/kg]']
        self.DatesDoc = ['FechaEv','FechaC']
        # Flags
        self.flag = dict()
        self.flag['Variables1'] = False
        self.flag['Variables2'] = False
        self.flag['Variables3'] = False
        self.flag['VariablesF'] = False
        self.flag['VariablesComp'] = False
        # ------------------
        # Archivos a abrir
        # ------------------

        # Se carga la información de las estaciones
        Tot = PathDataImp + DataImp + '.xlsx'
        book = xlrd.open_workbook(Tot) # Se carga el archivo
        SS = book.sheet_by_index(0) # Datos para trabajar
        # Datos de las estaciones
        Hoja = int(SS.cell(2,2).value)
        S = book.sheet_by_index(Hoja) # Datos para trabajar
        NEst = int(S.cell(1,0).value) # Número de estaciones y sensores que se van a extraer

        # Se inicializan las variables que se tomarán del documento
        x = 3 # Contador
        self.ID = []
        Names = [] # Nombre de las estaciones para las gráficas
        NamesArch = [] # Nombre de los archivos
        Tipo = [] # Tipo de datos - Se utiliza para encontrar la localización
        ET = [] # Tipo de información 0: Horaria, 1: Diaria
        self.ZT = [] # Alturas de las estaciones

        # Ciclo para tomar los datos
        for i in range(NEst):
            self.ID.append(str(int(S.cell(x,0).value)))
            Names.append(S.cell(x,1).value)
            NamesArch.append(S.cell(x,2).value)
            Tipo.append(S.cell(x,3).value)
            ET.append(int(S.cell(x,4).value))
            try:
                self.ZT.append(int(S.cell(x,5).value))
            except:
                self.ZT.append(float('nan'))
            x += 1

        # En esta sección se extraerá la información 
        self.Arch = []
        self.Names = []
        self.NamesArch = []
        self.Arch2 = []

        x = 0
        for i in range(NEst): # Ciclo para las estaciones
            if Tipo[i] == TipoD:
                if ET[i] == -1:
                    # Se extrae la información horaria
                    Tot = PathData + NamesArch[i] + endingmat +'.mat'
                    self.Arch.append(Tot)
                    self.Names.append(Names[i])
                    self.NamesArch.append(NamesArch[i])
                    self.Arch2.append(PathData2+self.NamesArch[i]+'.mat')
            x += 1

        self.ID = np.array(self.ID)
        self.Names = np.array(self.Names)
        return

    def LoadData(self,irow=0):
        '''
        DESCRIPTION:
        
            Función para cargar los datos.
        _________________________________________________________________________

        INPUT:
            + irow: Fila en donde se encuentran los datos que se quieren cargar.
        _________________________________________________________________________
        
        OUTPUT:
            - var: Variables que se tienen en los datos.
            - FechaEv: Fechas de cada evento.
            - PresC: Compuestos de Presión.
            - PrecC: Compuestos de Precipitación
            - TC: Compuestos de Temperatura.
            - HRC: Compuestos de Humedad Relativa
            - qC: Compuestos de Humedad Específica
        '''
        self.irow = irow

        # Se carga el archivo.
        self.f = sio.loadmat(self.Arch[irow])

        # Se verifica la existencia de todas las variables
        try: 
            self.f['PrecC']
            self.PrecC = True
            self.Middle = int(self.f['PrecC'].shape[1]/2)
        except KeyError:
            self.PrecC = False
        try: 
            self.f['PresC']
            self.PresC = True
            self.Middle = int(self.f['PresC'].shape[1]/2)
        except KeyError:
            self.PresC = False
        try: 
            self.f['TC']
            self.TC = True
            self.Middle = int(self.f['TC'].shape[1]/2)
        except KeyError:
            self.TC = False
        try: 
            self.f['HRC']
            self.HRC = True
        except KeyError:
            self.HRC = False
        try: 
            self.f['qC']
            self.qC = True
        except KeyError:
            self.qC = False
        try: 
            self.f['FechaEv']
            self.f['FechaEvP'] = np.empty(self.f['FechaEv'].shape)
            for i in range(len(self.f['FechaEv'])):
                if i == 0:
                    self.f['FechaEvP'] = DUtil.Dates_str2datetime(self.f['FechaEv'][i],Date_Format='%Y/%m/%d-%H%M',flagQuick=True)
                else:
                    self.f['FechaEvP'] = np.vstack((self.f['FechaEvP'],DUtil.Dates_str2datetime(self.f['FechaEv'][i],Date_Format='%Y/%m/%d-%H%M',flagQuick=True)))
            self.FechaEv = True
        except KeyError:
            self.FechaEv = False
        try: 
            self.f['Prec'][0]
            self.Prec = True
        except KeyError:
            self.Prec = False
        try: 
            self.f['Pres_F'][0]
            self.Pres_F = True
        except KeyError:
            self.Pres_F = False
        try: 
            self.f['T_F'][0]
            self.T_F = True
        except KeyError:
            self.T_F = False
        try: 
            self.f['HR_F'][0]
            self.HR_F = True
        except KeyError:
            self.HR_F = False
        try: 
            self.f['q_F'][0]
            self.q_F = True
        except KeyError:
            self.q_F = False
        try: 
            self.f['FechaC']
            self.FechaC = True
            self.f['FechaCP'] = DUtil.Dates_str2datetime(self.f['FechaC'],Date_Format='%Y/%m/%d-%H%M',flagQuick=True)
            # Delta de tiempo
            self.dtm = str(int(self.f['FechaC'][1][-2:]))
            if self.dtm == '0':
                self.dtm = str(int(self.f['FechaC'][1][-2-2:-2]))
                if self.dtm == '0':
                    print('Revisar el delta de tiempo')
        except KeyError:
            try:
                f = sio.loadmat(self.Arch2[irow])
                self.f['FechaC'] = f['FechaC']
                self.FechaC = True
                self.f['FechaCP'] = DUtil.Dates_str2datetime(self.f['FechaC'],Date_Format='%Y/%m/%d-%H%M',flagQuick=True)
                # Delta de tiempo
                self.dtm = str(int(self.f['FechaC'][1][-2:]))
                if self.dtm == '0':
                    self.dtm = str(int(self.f['FechaC'][1][-2-2:-2]))
                    if self.dtm == '0':
                        print('Revisar el delta de tiempo')
            except KeyError:
                self.FechaC = False

        self.var = self.f.keys()
        self.VerifyData()
        return

    def VerifyData(self):
        '''
            DESCRIPTION:
        
        Función para verificar los datos que se tienen
        _________________________________________________________________________

            INPUT:
        _________________________________________________________________________
        
            OUTPUT:
        '''
        if not(self.PrecC) or not(self.TC) or not(self.HRC) or not(self.qC) \
            or not(self.PresC) or not(self.Prec) or not(self.Pres_F) \
            or not(self.HR_F) or not(self.T_F) or not(self.q_F) \
            or not(self.FechaEv) or not(self.FechaC):
            print('No se tienen las siguientes variables:')
            if not(self.PrecC):
                print(' -PrecC')
            if not(self.PresC):
                print(' -PresC')
            if not(self.TC):
                print(' -TC')
            if not(self.HRC):
                print(' -HRC')
            if not(self.qC):
                print(' -qC')
            if not(self.Prec):
                print(' -Prec')
            if not(self.Pres_F):
                print(' -Pres_F')
            if not(self.T_F):
                print(' -T_F')
            if not(self.HR_F):
                print(' -HR_F')
            if not(self.q_F):
                print(' -q_F')
            if not(self.FechaEv):
                print(' -FechaEv')
            if not(self.FechaC):
                print(' -FechaC')
        return

    def DP(self,dt=None):
        '''
        DESCRIPTION:
        
            Función para calcular valores de cada evento de precipitación.
        _________________________________________________________________________

        INPUT:
        
        _________________________________________________________________________
        
        OUTPUT:
            Se generan diferentes variables o se cambian las actuales.
        '''
        if dt == None:
            dt = int(self.dtm)

        R = HA.PrecCount(self.f['PrecC'],self.f['FechaEv'],dt=dt,M=self.Middle)
        self.f.update(R)
        self.var = self.f.keys()
        return

    def EventsPrecDetection(self):
        '''
            DESCRIPTION:

        Función para detectar si llovió en el evento encontrado en el diagrama
        de compuestos.
        _________________________________________________________________________

            INPUT:
        _________________________________________________________________________

            OUTPUT:
        '''

        self.EvDYes = dict()
        self.EvDNo = dict()
        xP1 = 0
        xP2 = 0
        xT1 = 0
        xT2 = 0

        if self.dtm == '1':
           MaxPrec = 0.2 
        else:
           MaxPrec = 0.5 

        if self.PrecC:
            if self.PresC and self.TC:
                # Se intenta primero con presión
                for i in range(len(self.f['PrecC_Pres'])):
                    if np.nanmax(self.f['PrecC_Pres'][i][self.Middle:self.Middle+(60/int(self.dtm)*2)+1]) >= MaxPrec:
                        if xP1 == 0:
                            self.EvDYes['PrecC_Pres'] = self.f['PrecC_Pres'][i]
                            self.EvDYes['TC_Pres'] = self.f['TC_Pres'][i]
                            self.EvDYes['PresC_Pres'] = self.f['PresC_Pres'][i]
                            self.EvDYes['HRC_Pres'] = self.f['HRC_Pres'][i]
                            self.EvDYes['qC_Pres'] = self.f['qC_Pres'][i]
                            self.EvDYes['FechaEv_Pres'] = self.f['FechaEv_Pres'][i]
                            xP1 += 1
                        else:
                            self.EvDYes['PrecC_Pres']   = np.vstack((self.EvDYes['PrecC_Pres'],self.f['PrecC_Pres'][i]))
                            self.EvDYes['TC_Pres']      = np.vstack((self.EvDYes['TC_Pres'],self.f['TC_Pres'][i]))
                            self.EvDYes['PresC_Pres']   = np.vstack((self.EvDYes['PresC_Pres'],self.f['PresC_Pres'][i]))
                            self.EvDYes['HRC_Pres']     = np.vstack((self.EvDYes['HRC_Pres'],self.f['HRC_Pres'][i]))
                            self.EvDYes['qC_Pres']      = np.vstack((self.EvDYes['qC_Pres'],self.f['qC_Pres'][i]))
                            self.EvDYes['FechaEv_Pres']      = np.vstack((self.EvDYes['FechaEv_Pres'],self.f['FechaEv_Pres'][i]))
                    else:
                        if xP2 == 0:
                            self.EvDNo['PrecC_Pres'] = self.f['PrecC_Pres'][i]
                            self.EvDNo['TC_Pres'] = self.f['TC_Pres'][i]
                            self.EvDNo['PresC_Pres'] = self.f['PresC_Pres'][i]
                            self.EvDNo['HRC_Pres'] = self.f['HRC_Pres'][i]
                            self.EvDNo['qC_Pres'] = self.f['qC_Pres'][i]
                            self.EvDNo['FechaEv_Pres'] = self.f['FechaEv_Pres'][i]
                            xP2 += 1
                        else:
                            self.EvDNo['PrecC_Pres']   = np.vstack((self.EvDNo['PrecC_Pres'],self.f['PrecC_Pres'][i]))
                            self.EvDNo['TC_Pres']      = np.vstack((self.EvDNo['TC_Pres'],self.f['TC_Pres'][i]))
                            self.EvDNo['PresC_Pres']   = np.vstack((self.EvDNo['PresC_Pres'],self.f['PresC_Pres'][i]))
                            self.EvDNo['HRC_Pres']     = np.vstack((self.EvDNo['HRC_Pres'],self.f['HRC_Pres'][i]))
                            self.EvDNo['qC_Pres']      = np.vstack((self.EvDNo['qC_Pres'],self.f['qC_Pres'][i]))
                            self.EvDNo['FechaEv_Pres']      = np.vstack((self.EvDNo['FechaEv_Pres'],self.f['FechaEv_Pres'][i]))

                # Se intenta primero con presión
                for i in range(len(self.f['PrecC_Temp'])):
                    if np.nanmax(self.f['PrecC_Temp'][i][self.Middle:self.Middle+(60/int(self.dtm)*2)+1]) >= MaxPrec:
                        if xT1 == 0:
                            self.EvDYes['PrecC_Temp'] = self.f['PrecC_Temp'][i]
                            self.EvDYes['TC_Temp'] = self.f['TC_Temp'][i]
                            self.EvDYes['PresC_Temp'] = self.f['PresC_Temp'][i]
                            self.EvDYes['HRC_Temp'] = self.f['HRC_Temp'][i]
                            self.EvDYes['qC_Temp'] = self.f['qC_Temp'][i]
                            self.EvDYes['FechaEv_Temp'] = self.f['FechaEv_Temp'][i]
                            xT1 += 1
                        else:
                            self.EvDYes['PrecC_Temp']   = np.vstack((self.EvDYes['PrecC_Temp'],self.f['PrecC_Temp'][i]))
                            self.EvDYes['TC_Temp']      = np.vstack((self.EvDYes['TC_Temp'],self.f['TC_Temp'][i]))
                            self.EvDYes['PresC_Temp']   = np.vstack((self.EvDYes['PresC_Temp'],self.f['PresC_Temp'][i]))
                            self.EvDYes['HRC_Temp']     = np.vstack((self.EvDYes['HRC_Temp'],self.f['HRC_Temp'][i]))
                            self.EvDYes['qC_Temp']      = np.vstack((self.EvDYes['qC_Temp'],self.f['qC_Temp'][i]))
                            self.EvDYes['FechaEv_Temp']      = np.vstack((self.EvDYes['FechaEv_Temp'],self.f['FechaEv_Temp'][i]))
                    else:
                        if xT2 == 0:
                            self.EvDNo['PrecC_Temp'] = self.f['PrecC_Temp'][i]
                            self.EvDNo['TC_Temp'] = self.f['TC_Temp'][i]
                            self.EvDNo['PresC_Temp'] = self.f['PresC_Temp'][i]
                            self.EvDNo['HRC_Temp'] = self.f['HRC_Temp'][i]
                            self.EvDNo['qC_Temp'] = self.f['qC_Temp'][i]
                            self.EvDNo['FechaEv_Temp'] = self.f['FechaEv_Temp'][i]
                            xT2 += 1
                        else:
                            self.EvDNo['PrecC_Temp']   = np.vstack((self.EvDNo['PrecC_Temp'],self.f['PrecC_Temp'][i]))
                            self.EvDNo['TC_Temp']      = np.vstack((self.EvDNo['TC_Temp'],self.f['TC_Temp'][i]))
                            self.EvDNo['PresC_Temp']   = np.vstack((self.EvDNo['PresC_Temp'],self.f['PresC_Temp'][i]))
                            self.EvDNo['HRC_Temp']     = np.vstack((self.EvDNo['HRC_Temp'],self.f['HRC_Temp'][i]))
                            self.EvDNo['qC_Temp']      = np.vstack((self.EvDNo['qC_Temp'],self.f['qC_Temp'][i]))
                            self.EvDNo['FechaEv_Temp']      = np.vstack((self.EvDNo['FechaEv_Temp'],self.f['FechaEv_Temp'][i]))
        return

    def EventsGraphSeries(self,ImgFolder='Manizales/Events/'):
        '''
            DESCRIPTION:

        Función para graficar los eventos
        _________________________________________________________________________

            INPUT:
        + ImgFolder: Carpeta donde se guardarán los datos.
        _________________________________________________________________________

            OUTPUT:
        Se generan los gráficos
        '''
        self.ImgFolder = ImgFolder

        # Diagramas de compuestos promedios
        if self.PrecC:
            if self.PresC and self.TC:
                PrecH,PrecBin,PresH,PresBin,TH,TBin = BP.graphEv(self.f['PrecC'],self.f['PresC'],self.f['TC'],'Precipitación','Presión Barométrica','Temperatura','Prec','Pres','Temp','Precipitación [mm]','Presión [hPa]','Temperatura [°C]','b-','k-','r-','b','k','r',self.irow,self.NamesArch[self.irow],self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,TH,TBin = BP.graphEv(self.f['PrecC_Pres'],self.f['PresC_Pres'],self.f['TC_Pres'],'Precipitación','Presión Barométrica','Temperatura','Prec','Pres','Temp','Precipitación [mm]','Presión [hPa]','Temperatura [°C]','b-','k-','r-','b','k','r',self.irow,self.NamesArch[self.irow]+'_Pres',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,TH,TBin = BP.graphEv(self.f['PrecC_Temp'],self.f['PresC_Temp'],self.f['TC_Temp'],'Precipitación','Presión Barométrica','Temperatura','Prec','Pres','Temp','Precipitación [mm]','Presión [hPa]','Temperatura [°C]','b-','k-','r-','b','k','r',self.irow,self.NamesArch[self.irow]+'_Temp',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.PresC and self.HRC:
                PrecH,PrecBin,PresH,PresBin,HRH,TBin = BP.graphEv(self.f['PrecC'],self.f['PresC'],self.f['HRC'],'Precipitación','Presión Barométrica','Humedad Relativa','Prec','Pres','HR','Precipitación [mm]','Presión [hPa]','Humedad Relativa [%]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow],self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,HRH,TBin = BP.graphEv(self.f['PrecC_Pres'],self.f['PresC_Pres'],self.f['HRC_Pres'],'Precipitación','Presión Barométrica','Humedad Relativa','Prec','Pres','HR','Precipitación [mm]','Presión [hPa]','Humedad Relativa [%]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Pres',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,HRH,TBin = BP.graphEv(self.f['PrecC_Temp'],self.f['PresC_Temp'],self.f['HRC_Temp'],'Precipitación','Presión Barométrica','Humedad Relativa','Prec','Pres','HR','Precipitación [mm]','Presión [hPa]','Humedad Relativa [%]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Temp',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.PresC and self.qC:
                PrecH,PrecBin,PresH,PresBin,qH,TBin = BP.graphEv(self.f['PrecC'],self.f['PresC'],self.f['qC'],'Precipitación','Presión Barométrica','Humedad Específica','Prec','Pres','q','Precipitación [mm]','Presión [hPa]','Humedad Específica [kg/kg]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow],self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,qH,TBin = BP.graphEv(self.f['PrecC_Pres'],self.f['PresC_Pres'],self.f['qC_Pres'],'Precipitación','Presión Barométrica','Humedad Específica','Prec','Pres','q','Precipitación [mm]','Presión [hPa]','Humedad Específica [kg/kg]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Pres',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,qH,TBin = BP.graphEv(self.f['PrecC_Temp'],self.f['PresC_Temp'],self.f['qC_Temp'],'Precipitación','Presión Barométrica','Humedad Específica','Prec','Pres','q','Precipitación [mm]','Presión [hPa]','Humedad Específica [kg/kg]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Temp',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.PresC and self.qC:
                PrecH,PrecBin,TH,TBin,HRH,TBin = BP.graphEv(self.f['PrecC'],self.f['TC'],self.f['qC'],'Precipitación','Temperatura','Humedad Específica','Prec','Temp','q','Precipitación [mm]','Temperatura [°C]','Humedad Específica [kg/kg]','b-','r-','g-','b','r','g',self.irow,self.NamesArch[self.irow],self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,TH,TBin,HRH,TBin = BP.graphEv(self.f['PrecC_Pres'],self.f['TC_Pres'],self.f['qC_Pres'],'Precipitación','Temperatura','Humedad Específica','Prec','Temp','q','Precipitación [mm]','Temperatura [°C]','Humedad Específica [kg/kg]','b-','r-','g-','b','r','g',self.irow,self.NamesArch[self.irow]+'_Pres',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,TH,TBin,HRH,TBin = BP.graphEv(self.f['PrecC_Temp'],self.f['TC_Temp'],self.f['qC_Temp'],'Precipitación','Temperatura','Humedad Específica','Prec','Temp','q','Precipitación [mm]','Temperatura [°C]','Humedad Específica [kg/kg]','b-','r-','g-','b','r','g',self.irow,self.NamesArch[self.irow]+'_Temp',self.PathImg+self.ImgFolder,DTT=self.dtm)
        else:
            print('No se tiene información de precipitación para realizar los diagramas')
        return

    def EventsGraphSeriesEvDYes(self,ImgFolder='Manizales/Events/'):
        '''
            DESCRIPTION:

        Función para graficar los eventos
        _________________________________________________________________________

            INPUT:
        + ImgFolder: Carpeta donde se guardarán los datos.
        _________________________________________________________________________

            OUTPUT:
        Se generan los gráficos
        '''
        self.ImgFolder = ImgFolder

        # Diagramas de compuestos promedios
        if self.PrecC:
            if self.PresC and self.TC:
                PrecH,PrecBin,PresH,PresBin,TH,TBin = BP.graphEv(self.EvDYes['PrecC_Pres'],self.EvDYes['PresC_Pres'],self.EvDYes['TC_Pres'],'Precipitación','Presión Barométrica','Temperatura','Prec','Pres','Temp','Precipitación [mm]','Presión [hPa]','Temperatura [°C]','b-','k-','r-','b','k','r',self.irow,self.NamesArch[self.irow]+'_Pres_Yes',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,TH,TBin = BP.graphEv(self.EvDYes['PrecC_Temp'],self.EvDYes['PresC_Temp'],self.EvDYes['TC_Temp'],'Precipitación','Presión Barométrica','Temperatura','Prec','Pres','Temp','Precipitación [mm]','Presión [hPa]','Temperatura [°C]','b-','k-','r-','b','k','r',self.irow,self.NamesArch[self.irow]+'_Temp_Yes',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.PresC and self.HRC:
                PrecH,PrecBin,PresH,PresBin,HRH,TBin = BP.graphEv(self.EvDYes['PrecC_Pres'],self.EvDYes['PresC_Pres'],self.EvDYes['HRC_Pres'],'Precipitación','Presión Barométrica','Humedad Relativa','Prec','Pres','HR','Precipitación [mm]','Presión [hPa]','Humedad Relativa [%]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Pres_Yes',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,HRH,TBin = BP.graphEv(self.EvDYes['PrecC_Temp'],self.EvDYes['PresC_Temp'],self.EvDYes['HRC_Temp'],'Precipitación','Presión Barométrica','Humedad Relativa','Prec','Pres','HR','Precipitación [mm]','Presión [hPa]','Humedad Relativa [%]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Temp_Yes',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.PresC and self.qC:
                PrecH,PrecBin,PresH,PresBin,qH,TBin = BP.graphEv(self.EvDYes['PrecC_Pres'],self.EvDYes['PresC_Pres'],self.EvDYes['qC_Pres'],'Precipitación','Presión Barométrica','Humedad Específica','Prec','Pres','q','Precipitación [mm]','Presión [hPa]','Humedad Específica [kg/kg]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Pres_Yes',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,qH,TBin = BP.graphEv(self.EvDYes['PrecC_Temp'],self.EvDYes['PresC_Temp'],self.EvDYes['qC_Temp'],'Precipitación','Presión Barométrica','Humedad Específica','Prec','Pres','q','Precipitación [mm]','Presión [hPa]','Humedad Específica [kg/kg]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Temp_Yes',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.PresC and self.qC:
                PrecH,PrecBin,TH,TBin,HRH,TBin = BP.graphEv(self.EvDYes['PrecC_Pres'],self.EvDYes['TC_Pres'],self.EvDYes['qC_Pres'],'Precipitación','Temperatura','Humedad Específica','Prec','Temp','q','Precipitación [mm]','Temperatura [°C]','Humedad Específica [kg/kg]','b-','r-','g-','b','r','g',self.irow,self.NamesArch[self.irow]+'_Pres_Yes',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,TH,TBin,HRH,TBin = BP.graphEv(self.EvDYes['PrecC_Temp'],self.EvDYes['TC_Temp'],self.EvDYes['qC_Temp'],'Precipitación','Temperatura','Humedad Específica','Prec','Temp','q','Precipitación [mm]','Temperatura [°C]','Humedad Específica [kg/kg]','b-','r-','g-','b','r','g',self.irow,self.NamesArch[self.irow]+'_Temp_Yes',self.PathImg+self.ImgFolder,DTT=self.dtm)
        else:
            print('No se tiene información de precipitación para realizar los diagramas')
        return

    def EventsGraphSeriesEvDNo(self,ImgFolder='Manizales/Events/'):
        '''
            DESCRIPTION:

        Función para graficar los eventos
        _________________________________________________________________________

            INPUT:
        + ImgFolder: Carpeta donde se guardarán los datos.
        _________________________________________________________________________

            OUTPUT:
        Se generan los gráficos
        '''
        self.ImgFolder = ImgFolder

        # Diagramas de compuestos promedios
        if self.PrecC:
            if self.PresC and self.TC:
                PrecH,PrecBin,PresH,PresBin,TH,TBin = BP.graphEv(self.EvDNo['PrecC_Pres'],self.EvDNo['PresC_Pres'],self.EvDNo['TC_Pres'],'Precipitación','Presión Barométrica','Temperatura','Prec','Pres','Temp','Precipitación [mm]','Presión [hPa]','Temperatura [°C]','b-','k-','r-','b','k','r',self.irow,self.NamesArch[self.irow]+'_Pres_No',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,TH,TBin = BP.graphEv(self.EvDNo['PrecC_Temp'],self.EvDNo['PresC_Temp'],self.EvDNo['TC_Temp'],'Precipitación','Presión Barométrica','Temperatura','Prec','Pres','Temp','Precipitación [mm]','Presión [hPa]','Temperatura [°C]','b-','k-','r-','b','k','r',self.irow,self.NamesArch[self.irow]+'_Temp_No',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.PresC and self.HRC:
                PrecH,PrecBin,PresH,PresBin,HRH,TBin = BP.graphEv(self.EvDNo['PrecC_Pres'],self.EvDNo['PresC_Pres'],self.EvDNo['HRC_Pres'],'Precipitación','Presión Barométrica','Humedad Relativa','Prec','Pres','HR','Precipitación [mm]','Presión [hPa]','Humedad Relativa [%]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Pres_No',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,HRH,TBin = BP.graphEv(self.EvDNo['PrecC_Temp'],self.EvDNo['PresC_Temp'],self.EvDNo['HRC_Temp'],'Precipitación','Presión Barométrica','Humedad Relativa','Prec','Pres','HR','Precipitación [mm]','Presión [hPa]','Humedad Relativa [%]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Temp_No',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.PresC and self.qC:
                PrecH,PrecBin,PresH,PresBin,qH,TBin = BP.graphEv(self.EvDNo['PrecC_Pres'],self.EvDNo['PresC_Pres'],self.EvDNo['qC_Pres'],'Precipitación','Presión Barométrica','Humedad Específica','Prec','Pres','q','Precipitación [mm]','Presión [hPa]','Humedad Específica [kg/kg]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Pres_No',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,qH,TBin = BP.graphEv(self.EvDNo['PrecC_Temp'],self.EvDNo['PresC_Temp'],self.EvDNo['qC_Temp'],'Precipitación','Presión Barométrica','Humedad Específica','Prec','Pres','q','Precipitación [mm]','Presión [hPa]','Humedad Específica [kg/kg]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Temp_No',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.PresC and self.qC:
                PrecH,PrecBin,TH,TBin,HRH,TBin = BP.graphEv(self.EvDNo['PrecC_Pres'],self.EvDNo['TC_Pres'],self.EvDNo['qC_Pres'],'Precipitación','Temperatura','Humedad Específica','Prec','Temp','q','Precipitación [mm]','Temperatura [°C]','Humedad Específica [kg/kg]','b-','r-','g-','b','r','g',self.irow,self.NamesArch[self.irow]+'_Pres_No',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,TH,TBin,HRH,TBin = BP.graphEv(self.EvDNo['PrecC_Temp'],self.EvDNo['TC_Temp'],self.EvDNo['qC_Temp'],'Precipitación','Temperatura','Humedad Específica','Prec','Temp','q','Precipitación [mm]','Temperatura [°C]','Humedad Específica [kg/kg]','b-','r-','g-','b','r','g',self.irow,self.NamesArch[self.irow]+'_Temp_No',self.PathImg+self.ImgFolder,DTT=self.dtm)
        else:
            print('No se tiene información de precipitación para realizar los diagramas')
        return

    def PRoTVvDP(self,Mar=0.8,MarT=0.8,FechaEvst_Aft=0,FechaEvend_Aft=0,flagAf=False,flagEv_Pres=False,flagEv_T=False,flagIng=False,ImgFolder_Scatter='/Manizales/Scatter/',Specific_Folder='Events_3'):
        '''
            DESCRIPTION:

        Función para obtener los valores de Duración, tasas y cambios de presión
        y de temperatura.
        _________________________________________________________________________

            INPUT:
        + Mar: Valor del cambio de presión mínimo para calcular las tasas antes
               del evento de precipitación.
        + FechaEvst_Aft: Fecha de comienzo del evento, en el mismo formato que FechaEv.
        + FechaEvend_Aft: Fecha de finalización del evento, en el mismo formato que FechaEv.
        + flagAf: Bandera para ver si se incluye un treshold durante el evento.
        + flagEV: Bandera para graficar los eventos.
        + flagIng: Bander para saber si se lleva a inglés los ejes.
        + ImgFolder_Scatter: Ruta donde se guardará el documento.
        _________________________________________________________________________

            OUTPUT:
        Se generan diferentes variables o se cambian las actuales.
        '''

        self.ImgFolder_Scatter = ImgFolder_Scatter
        ImgFolder_Scatter_Specific_Pres = ImgFolder_Scatter+'Pres/'+Specific_Folder+'/'
        ImgFolder_Scatter_Specific_Temp = ImgFolder_Scatter+'Temp/'+Specific_Folder+'/'
        if self.PrecC:
            if self.PresC:
                self.f['DurPrec'], self.f['PresRateA'], self.f['PresRateB'], self.f['DurPresA'], self.f['DurPresB'], self.f['PresChangeA'], \
                    self.f['PresChangeB'], self.f['TotalPrec'], self.f['MaxPrec'], self.f['FechaEvst'] = \
                    BP.PRvDP_C(self.f['PrecC'],self.f['PresC'],self.f['FechaEv'],FechaEvst_Aft,FechaEvend_Aft,Mar,flagAf,int(self.dtm),self.Middle,flagEv_Pres,PathImg+ImgFolder_Scatter_Specific_Pres,self.NamesArch[self.irow],flagIng)
            else:
                print('No se tiene información de presión')
            if self.TC:
                self.f['DurPrecT'], self.f['TempRateA'], self.f['TempRateB'], self.f['TempChangeA'], self.f['TempChangeB'] \
                    = BP.TvDP_C(self.f['PrecC'],self.f['TC'],self.f['FechaEv'],FechaEvst_Aft,FechaEvend_Aft,MarT,flagAf,int(self.dtm),self.Middle,flagEv_T,PathImg+ImgFolder_Scatter_Specific_Temp,self.NamesArch[self.irow],flagIng)
            else:
                print('No se tiene información de temperatura')
        else:
            print('No se tiene información de precipitación para generar los conteos')
        self.var = self.f.keys()
        return

    def VC_VR_DP(self,DatesEv,Prec,Pres,MP=None,MPres=None,flagEv_Pres=False,flagEv_T=False,flagIng=False,ImgFolder_Scatter='/Manizales/Scatter/',Specific_Folder='Events_3'):
        '''
        DESCRIPTION:

            Función para obtener los valores de Duración, tasas y cambios de presión
            y de temperatura.
        _________________________________________________________________________

        INPUT:
            + DatesEv: Fechas de los diagramas de compuestos.
            + Prec: Compuestos de precipitación.
            + Pres: Compuestos de presión.
            + MP: Valor donde comienza a buscar para los datos de
                  precipitación.
            + MPres: Valor medio donde comienza a buscar presión.
            + flagEV: Bandera para graficar los eventos.
            + flagIng: Bander para saber si se lleva a inglés los ejes.
            + ImgFolder_Scatter: Ruta donde se guardará el documento.
        _________________________________________________________________________

            OUTPUT:
        Se generan diferentes variables o se cambian las actuales.
        '''

        # Información para hacer el gráfico
        self.ImgFolder_Scatter = ImgFolder_Scatter
        ImgFolder_Scatter_Specific_Pres = ImgFolder_Scatter+'Pres/'+Specific_Folder+'/'
        ImgFolder_Scatter_Specific_Temp = ImgFolder_Scatter+'Temp/'+Specific_Folder+'/'

        # Se incluyen los valores de la clase
        dt = int(self.dtm)
        if MP == None:
            MP = list()
            for iC in range(len(Prec)):
                MP.append(np.where(Prec[iC][self.Middle:]==
                    np.nanmax(Prec[iC][self.Middle:]))[0][0]+self.Middle)
        if MPres == None:
            MPres = self.Middle

        # Se calcula la duración de la tormenta
        if self.PrecC:
            self.Res_Pres = HA.PrecCount(Prec,DatesEv,dt=dt,M=MP)

        if self.PresC:
            Results = BP.C_Rates_Changes(Pres,dt=dt,M=MPres,MaxMin='min')

        self.Res_Pres.update(Results)

        return
