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
from Utilities import DatesUtil as DUtil; DUtil=DUtil()
from Utilities import Data_Man as DM
from AnET import CFitting as CF; CF=CF()
from Hydro_Analysis import Hydro_Plotter as HyPl;HyPl=HyPl()
from Hydro_Analysis.Meteo import MeteoFunctions as HyMF
from BPumpL.BPumpL import BPumpL as BP; BP=BP()
from BPumpL.Data_Man import Load_Data as LD

class Scatter_Gen(object): 
    '''
    DESCRIPTION:

        Clase par abrir los documentos que se necesitan para hacer los diferentes
        estudios de los diagramas de dispersión.
    '''

    def __init__(self,DataBase='Manizales',endingmat='',PathImg='Tesis_MscR/02_Docs/01_Tesis_Doc/Kap3/Img/'):
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
        self.Variables = ['PrecC','TC','HRC','PresC','qC',
                'PrecC_Pres','TC_Pres','HRC_Pres','PresC_Pres','qC_Pres',
                'PrecC_Temp','TC_Temp','HRC_Temp','PresC_Temp','qC_Temp']
        # Información para gráficos
        LabelV = ['Precipitación','Temperatura','Humedad Relativa','Presión','Humedad Especifica']
        LabelVU = ['Precipitación [mm]','Temperatura [°C]','Hum. Rel. [%]','Presión [hPa]','Hum. Espec. [kg/kg]']
        self.DatesDoc = ['FechaEv','FechaC']
        # ------------------
        # Archivos a abrir
        # ------------------
        StInfo,Arch = LD.LoadStationsInfo(endingmatR=endingmat)
        self.StInfo = StInfo

        self.Arch = Arch[DataBase]['CFilt']
        self.Arch2 = Arch[DataBase]['Original']
        self.Names = np.array(StInfo[DataBase]['Name'])
        self.NamesArch = StInfo[DataBase]['Name_Arch'] 
        self.ID = np.array(StInfo[DataBase]['ID'])
        self.Latitude = np.array(StInfo[DataBase]['Latitud'])
        self.Longitude = np.array(StInfo[DataBase]['Longitud'])

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
        Variables = ['FechaEv','FechaC']+self.Variables+['FechaEv_Pres','FechaEv_Temp']
        self.f,self.flag = LD.LoadDataVerify(self.Arch,VerifData=Variables,irow=irow)
        if not(self.flag['FechaEv']):
            raise utl.ShowError('LoadData','BPumpL.Scatter_Gen','No se encuentran fechas de los evento, revisar archivo')

        # Se organizan las fechas
        self.f['FechaEvP'] = np.empty(self.f['FechaEv'].shape)
        for i in range(len(self.f['FechaEv'])):
            if i == 0:
                self.f['FechaEvP'] = DUtil.Dates_str2datetime(self.f['FechaEv'][i],Date_Format='%Y/%m/%d-%H%M',flagQuick=True)
            else:
                self.f['FechaEvP'] = np.vstack((self.f['FechaEvP'],DUtil.Dates_str2datetime(self.f['FechaEv'][i],Date_Format='%Y/%m/%d-%H%M',flagQuick=True)))

        if self.flag['FechaC']:
            self.f['FechaC']
            self.f['FechaCP'] = DUtil.Dates_str2datetime(self.f['FechaC'],Date_Format='%Y/%m/%d-%H%M',flagQuick=True)
            # Delta de tiempo
            self.dtm = str(int(self.f['FechaC'][1][-2:]))
            if self.dtm == '0':
                self.dtm = str(int(self.f['FechaC'][1][-2-2:-2]))
                if self.dtm == '0':
                    print('Revisar el delta de tiempo')
        else:
            self.flag['FechaC'] = True
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

        self.Middle=int(self.f['PrecC'].shape[1]/2)
        self.var = Variables + ['FechaEvP','FechaCP']
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

        if self.flag['PrecC']:
            if self.flag['PresC'] and self.flag['TC']:
                # Presión
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

                # Temperatura
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
        if self.flag['PrecC']:
            if self.flag['PresC'] and self.flag['TC']:
                PrecH,PrecBin,PresH,PresBin,TH,TBin = BP.graphEv(self.f['PrecC'],self.f['PresC'],self.f['TC'],'Precipitación','Presión Barométrica','Temperatura','Prec','Pres','Temp','Precipitación [mm]','Presión [hPa]','Temperatura [°C]','b-','k-','r-','b','k','r',self.irow,self.NamesArch[self.irow],self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,TH,TBin = BP.graphEv(self.f['PrecC_Pres'],self.f['PresC_Pres'],self.f['TC_Pres'],'Precipitación','Presión Barométrica','Temperatura','Prec','Pres','Temp','Precipitación [mm]','Presión [hPa]','Temperatura [°C]','b-','k-','r-','b','k','r',self.irow,self.NamesArch[self.irow]+'_Pres',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,TH,TBin = BP.graphEv(self.f['PrecC_Temp'],self.f['PresC_Temp'],self.f['TC_Temp'],'Precipitación','Presión Barométrica','Temperatura','Prec','Pres','Temp','Precipitación [mm]','Presión [hPa]','Temperatura [°C]','b-','k-','r-','b','k','r',self.irow,self.NamesArch[self.irow]+'_Temp',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.flag['PresC'] and self.flag['HRC']:
                PrecH,PrecBin,PresH,PresBin,HRH,TBin = BP.graphEv(self.f['PrecC'],self.f['PresC'],self.f['HRC'],'Precipitación','Presión Barométrica','Humedad Relativa','Prec','Pres','HR','Precipitación [mm]','Presión [hPa]','Humedad Relativa [%]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow],self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,HRH,TBin = BP.graphEv(self.f['PrecC_Pres'],self.f['PresC_Pres'],self.f['HRC_Pres'],'Precipitación','Presión Barométrica','Humedad Relativa','Prec','Pres','HR','Precipitación [mm]','Presión [hPa]','Humedad Relativa [%]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Pres',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,HRH,TBin = BP.graphEv(self.f['PrecC_Temp'],self.f['PresC_Temp'],self.f['HRC_Temp'],'Precipitación','Presión Barométrica','Humedad Relativa','Prec','Pres','HR','Precipitación [mm]','Presión [hPa]','Humedad Relativa [%]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Temp',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.flag['PresC'] and self.flag['qC']:
                PrecH,PrecBin,PresH,PresBin,qH,TBin = BP.graphEv(self.f['PrecC'],self.f['PresC'],self.f['qC'],'Precipitación','Presión Barométrica','Humedad Específica','Prec','Pres','q','Precipitación [mm]','Presión [hPa]','Humedad Específica [kg/kg]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow],self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,qH,TBin = BP.graphEv(self.f['PrecC_Pres'],self.f['PresC_Pres'],self.f['qC_Pres'],'Precipitación','Presión Barométrica','Humedad Específica','Prec','Pres','q','Precipitación [mm]','Presión [hPa]','Humedad Específica [kg/kg]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Pres',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,qH,TBin = BP.graphEv(self.f['PrecC_Temp'],self.f['PresC_Temp'],self.f['qC_Temp'],'Precipitación','Presión Barométrica','Humedad Específica','Prec','Pres','q','Precipitación [mm]','Presión [hPa]','Humedad Específica [kg/kg]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Temp',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.flag['TC'] and self.flag['qC']:
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
        if self.flag['PrecC']:
            if self.flag['PresC'] and self.flag['TC']:
                PrecH,PrecBin,PresH,PresBin,TH,TBin = BP.graphEv(self.EvDYes['PrecC_Pres'],self.EvDYes['PresC_Pres'],self.EvDYes['TC_Pres'],'Precipitación','Presión Barométrica','Temperatura','Prec','Pres','Temp','Precipitación [mm]','Presión [hPa]','Temperatura [°C]','b-','k-','r-','b','k','r',self.irow,self.NamesArch[self.irow]+'_Pres_Yes',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,TH,TBin = BP.graphEv(self.EvDYes['PrecC_Temp'],self.EvDYes['PresC_Temp'],self.EvDYes['TC_Temp'],'Precipitación','Presión Barométrica','Temperatura','Prec','Pres','Temp','Precipitación [mm]','Presión [hPa]','Temperatura [°C]','b-','k-','r-','b','k','r',self.irow,self.NamesArch[self.irow]+'_Temp_Yes',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.flag['PresC'] and self.flag['HRC']:
                PrecH,PrecBin,PresH,PresBin,HRH,TBin = BP.graphEv(self.EvDYes['PrecC_Pres'],self.EvDYes['PresC_Pres'],self.EvDYes['HRC_Pres'],'Precipitación','Presión Barométrica','Humedad Relativa','Prec','Pres','HR','Precipitación [mm]','Presión [hPa]','Humedad Relativa [%]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Pres_Yes',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,HRH,TBin = BP.graphEv(self.EvDYes['PrecC_Temp'],self.EvDYes['PresC_Temp'],self.EvDYes['HRC_Temp'],'Precipitación','Presión Barométrica','Humedad Relativa','Prec','Pres','HR','Precipitación [mm]','Presión [hPa]','Humedad Relativa [%]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Temp_Yes',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.flag['PresC'] and self.flag['qC']:
                PrecH,PrecBin,PresH,PresBin,qH,TBin = BP.graphEv(self.EvDYes['PrecC_Pres'],self.EvDYes['PresC_Pres'],self.EvDYes['qC_Pres'],'Precipitación','Presión Barométrica','Humedad Específica','Prec','Pres','q','Precipitación [mm]','Presión [hPa]','Humedad Específica [kg/kg]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Pres_Yes',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,qH,TBin = BP.graphEv(self.EvDYes['PrecC_Temp'],self.EvDYes['PresC_Temp'],self.EvDYes['qC_Temp'],'Precipitación','Presión Barométrica','Humedad Específica','Prec','Pres','q','Precipitación [mm]','Presión [hPa]','Humedad Específica [kg/kg]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Temp_Yes',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.flag['TC'] and self.flag['qC']:
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
        if self.flag['PrecC']:
            if self.flag['PresC'] and self.flag['TC']:
                PrecH,PrecBin,PresH,PresBin,TH,TBin = BP.graphEv(self.EvDNo['PrecC_Pres'],self.EvDNo['PresC_Pres'],self.EvDNo['TC_Pres'],'Precipitación','Presión Barométrica','Temperatura','Prec','Pres','Temp','Precipitación [mm]','Presión [hPa]','Temperatura [°C]','b-','k-','r-','b','k','r',self.irow,self.NamesArch[self.irow]+'_Pres_No',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,TH,TBin = BP.graphEv(self.EvDNo['PrecC_Temp'],self.EvDNo['PresC_Temp'],self.EvDNo['TC_Temp'],'Precipitación','Presión Barométrica','Temperatura','Prec','Pres','Temp','Precipitación [mm]','Presión [hPa]','Temperatura [°C]','b-','k-','r-','b','k','r',self.irow,self.NamesArch[self.irow]+'_Temp_No',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.flag['PresC'] and self.flag['HRC']:
                PrecH,PrecBin,PresH,PresBin,HRH,TBin = BP.graphEv(self.EvDNo['PrecC_Pres'],self.EvDNo['PresC_Pres'],self.EvDNo['HRC_Pres'],'Precipitación','Presión Barométrica','Humedad Relativa','Prec','Pres','HR','Precipitación [mm]','Presión [hPa]','Humedad Relativa [%]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Pres_No',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,HRH,TBin = BP.graphEv(self.EvDNo['PrecC_Temp'],self.EvDNo['PresC_Temp'],self.EvDNo['HRC_Temp'],'Precipitación','Presión Barométrica','Humedad Relativa','Prec','Pres','HR','Precipitación [mm]','Presión [hPa]','Humedad Relativa [%]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Temp_No',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.flag['PresC'] and self.flag['qC']:
                PrecH,PrecBin,PresH,PresBin,qH,TBin = BP.graphEv(self.EvDNo['PrecC_Pres'],self.EvDNo['PresC_Pres'],self.EvDNo['qC_Pres'],'Precipitación','Presión Barométrica','Humedad Específica','Prec','Pres','q','Precipitación [mm]','Presión [hPa]','Humedad Específica [kg/kg]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Pres_No',self.PathImg+self.ImgFolder,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,qH,TBin = BP.graphEv(self.EvDNo['PrecC_Temp'],self.EvDNo['PresC_Temp'],self.EvDNo['qC_Temp'],'Precipitación','Presión Barométrica','Humedad Específica','Prec','Pres','q','Precipitación [mm]','Presión [hPa]','Humedad Específica [kg/kg]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Temp_No',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.flag['PresC'] and self.flag['qC']:
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
        if self.flag['PrecC']:
            self.Res_Pres = HyMF.PrecCount(Prec,DatesEv,dt=dt,M=MP)

        if self.flag['PresC']:
            Results = BP.C_Rates_Changes(Pres,dt=dt,M=MPres,MaxMin='min')

        self.Res_Pres.update(Results)
        Data = {'Prec':Prec,'Pres':Pres}
        self.Res_Pres['VminPos'] = np.ones(self.Res_Pres['DurPrec'].shape)*self.Middle
        BP.EventsScatter(DatesEv,Data,self.Res_Pres,
                PathImg=self.PathImg+ImgFolder_Scatter_Specific_Pres,
                Name=self.Names[self.irow],flagIng=False)

        return