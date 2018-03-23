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
        self.mmHg2hPa = 1.3332239

        # Projections
        self.epsgWGS84 = 4326
        self.epsgMAGNA = 3116

        # Diccionarios con las diferentes variables
        self.Variables = ['Prec','T_F','HR_F','Pres_F','q_F','W_F',
                'PrecC','TC','HRC','PresC','qC','WC',
                'PrecC_Pres','TC_Pres','HRC_Pres','PresC_Pres','qC_Pres','WC_Pres'
                'PrecC_Temp','TC_Temp','HRC_Temp','PresC_Temp','qC_Temp','WC_Temp']
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

    def EventsGenPrec(self,Var='PresC',oper=np.nanmin,Comp='<',ValComp=-0.3):
        '''
        DESCRIPTION:

            Función para detectar los eventos que tuvieron cambios importantes
            de alguna variable atmosférica con respecto al inicio de la 
            precipitación.
        _________________________________________________________________________

        INPUT:
            :param Var:     A str, string de la variable de cambio.
            :param oper:    A function, función para encontrar el valor.
            :param Comp:    A str, método de comparación que se evaluará.
            :param ValComp: An float, número que se evaluará.
        _________________________________________________________________________

        OUTPUT:
        '''
        # -------------
        # Parameters
        # -------------
        Comp = DM.Oper_Det(Comp)
        dt = int(self.dtm)
        # Se calculan los parámetros de precipitación
        self.PrecCount = HyMF.PrecCount(self.f['PrecC'],self.f['FechaEv'],dt=dt,M=self.Middle)
        # Se obtienen los meses 
        self.Months = np.array([i.month for i in self.PrecCount['DatesEvst']])
        DatesEvP = self.f['FechaEvP']
        # Eventos con cambios
        EvYes = []
        MEvYes = []
        # Eventos sin cambios
        EvNo = []
        MEvNo = []
        # Posición del cambio
        PosYes = []
        PosNo = []
        # ----
        # Horas atras
        TBef = 3
        # minutos hacia adelante
        TAft = 15
        for iC in range(len(self.f['PresC'])):
            # Se encuentra el inicio del evento
            xEv = np.where(DatesEvP[iC] == self.PrecCount['DatesEvst'][iC])[0][0]
            Bef = xEv-int(60/dt*TBef)
            Aft = xEv+int(60/dt*(TAft/60)) 
            q = sum(~np.isnan(self.f[Var][iC][Bef:Aft]))
            if Bef <= 10 or q <= 0.7*len(self.f[Var][iC][Bef:Aft]):
                EvNo.append(iC)
                MEvNo.append(self.Months[iC])
                PosNo.append(np.nan)
                continue
                
            # Se encuentra el mínimo de presión
            Min = oper(self.f[Var][iC][Bef:Aft])
            xMin1 = np.where(self.f[Var][iC][:Aft]==Min)[0][-1]
            if Comp(Min,ValComp):
                EvYes.append(iC)
                PosYes.append(xMin1)
                MEvYes.append(self.Months[iC])
            else:
                EvNo.append(iC)
                PosNo.append(xMin1)
                MEvNo.append(self.Months[iC])

        EvTot = np.array(EvYes+EvNo)
        MEvTot = np.array(MEvYes+MEvNo)
        PosTot = np.array(PosYes+PosNo)
        EvYes = np.array(EvYes)
        MEvYes = np.array(MEvYes)
        EvNo =  np.array(EvNo)
        MEvNo =  np.array(MEvNo)
        PosYes = np.array(PosYes)
        PosNo =  np.array(PosNo)

        Results = {'EvTot':EvTot,'MEvTot':MEvTot,
                'PosTot':PosTot,
                'EvYes':EvYes,'MEvYes':MEvYes,
                'EvNo':EvNo,'MEvNo':MEvNo,
                'PosYes':PosYes,'PosNo':PosNo}

        return Results

    def EventsPrecPresDetection(self):
        '''
        DESCRIPTION:

            Función para detectar los eventos que tuvieron caidas importantes
            de presión antes del evento de precipitación.
        _________________________________________________________________________

        INPUT:
        _________________________________________________________________________

        OUTPUT:
        '''
        dt = int(self.dtm)
        self.PrecCount = HyMF.PrecCount(self.f['PrecC'],self.f['FechaEv'],dt=dt,M=self.Middle)

        self.Months = np.array([i.month for i in self.PrecCount['DatesEvst']])
        DatesEvP = self.f['FechaEvP']
        self.MPresYes = list()
        self.MPresNo = list()
        # Eventos con caidas de presión antes
        self.EvYes = []
        self.MEvYes = []
        # Eventos sin caidas de presión antes
        self.EvNo = []
        self.MEvNo = []
        # ----
        # Horas atras
        TBef = 3
        # minutos hacia adelante
        TAft = 15
        for iC in range(len(self.f['PresC'])):
            # Se encuentra el inicio del evento
            xEv = np.where(DatesEvP[iC] == self.PrecCount['DatesEvst'][iC])[0][0]
            Bef = xEv-int(60/dt*TBef)
            Aft = xEv+int(60/dt*(TAft/60)) 
            if Bef <= 10:
                self.EvNo.append(iC)
                self.MEvNo.append(self.Months[iC])
                continue
                
            # Se encuentra el mínimo de presión
            Min = np.nanmin(self.f['PresC'][iC][Bef:Aft])
            xMin1 = np.where(self.f['PresC'][iC][:Aft]==Min)[0][-1]
            if Min < -0.3:
                self.EvYes.append(iC)
                self.MPresYes.append(xMin1)
                self.MEvYes.append(self.Months[iC])
            else:
                self.EvNo.append(iC)
                self.MPresNo.append(xMin1)
                self.MEvNo.append(self.Months[iC])

        self.EvYes = np.array(self.EvYes)
        self.MEvYes = np.array(self.MEvYes)
        self.EvNo =  np.array(self.EvNo)
        self.MEvNo =  np.array(self.MEvNo)

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
                NoAn = 0
                for i in range(len(self.f['PrecC_Pres'])):
                    if sum(np.isnan(self.f['PrecC_Pres'][i][self.Middle:self.Middle+(60/int(self.dtm)*1)+1])) < 0.30*(60/int(self.dtm)*1):
                        if np.nanmax(self.f['PrecC_Pres'][i][self.Middle:self.Middle+(60/int(self.dtm)*1)+1]) >= MaxPrec and np.nanmax(self.f['PrecC_Pres'][i][self.Middle-(60/int(self.dtm)*1)+1:self.Middle]) <= MaxPrec:
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
                    else:
                        NoAn += 1

                self.Statistics = {'Perc_Pres_Yes':len(self.EvDYes['PrecC_Pres'])/(len(self.f['PrecC_Pres'])-NoAn),
                        'Perc_Pres_No':len(self.EvDNo['PrecC_Pres'])/(len(self.f['PrecC_Pres'])-NoAn)}

                # Temperatura
                # for i in range(len(self.f['PrecC_Temp'])):
                #     if np.nanmax(self.f['PrecC_Temp'][i][self.Middle:self.Middle+(60/int(self.dtm)*1)+1]) >= MaxPrec and np.nanmax(self.f['PrecC_Temp'][i][self.Middle-(60/int(self.dtm)*1)+1:self.Middle]) <= MaxPrec:
                #         if xT1 == 0:
                #             self.EvDYes['PrecC_Temp'] = self.f['PrecC_Temp'][i]
                #             self.EvDYes['TC_Temp'] = self.f['TC_Temp'][i]
                #             self.EvDYes['PresC_Temp'] = self.f['PresC_Temp'][i]
                #             self.EvDYes['HRC_Temp'] = self.f['HRC_Temp'][i]
                #             self.EvDYes['qC_Temp'] = self.f['qC_Temp'][i]
                #             self.EvDYes['FechaEv_Temp'] = self.f['FechaEv_Temp'][i]
                #             xT1 += 1
                #         else:
                #             self.EvDYes['PrecC_Temp']   = np.vstack((self.EvDYes['PrecC_Temp'],self.f['PrecC_Temp'][i]))
                #             self.EvDYes['TC_Temp']      = np.vstack((self.EvDYes['TC_Temp'],self.f['TC_Temp'][i]))
                #             self.EvDYes['PresC_Temp']   = np.vstack((self.EvDYes['PresC_Temp'],self.f['PresC_Temp'][i]))
                #             self.EvDYes['HRC_Temp']     = np.vstack((self.EvDYes['HRC_Temp'],self.f['HRC_Temp'][i]))
                #             self.EvDYes['qC_Temp']      = np.vstack((self.EvDYes['qC_Temp'],self.f['qC_Temp'][i]))
                #             self.EvDYes['FechaEv_Temp']      = np.vstack((self.EvDYes['FechaEv_Temp'],self.f['FechaEv_Temp'][i]))
                #     else:
                #         if xT2 == 0:
                #             self.EvDNo['PrecC_Temp'] = self.f['PrecC_Temp'][i]
                #             self.EvDNo['TC_Temp'] = self.f['TC_Temp'][i]
                #             self.EvDNo['PresC_Temp'] = self.f['PresC_Temp'][i]
                #             self.EvDNo['HRC_Temp'] = self.f['HRC_Temp'][i]
                #             self.EvDNo['qC_Temp'] = self.f['qC_Temp'][i]
                #             self.EvDNo['FechaEv_Temp'] = self.f['FechaEv_Temp'][i]
                #             xT2 += 1
                #         else:
                #             self.EvDNo['PrecC_Temp']   = np.vstack((self.EvDNo['PrecC_Temp'],self.f['PrecC_Temp'][i]))
                #             self.EvDNo['TC_Temp']      = np.vstack((self.EvDNo['TC_Temp'],self.f['TC_Temp'][i]))
                #             self.EvDNo['PresC_Temp']   = np.vstack((self.EvDNo['PresC_Temp'],self.f['PresC_Temp'][i]))
                #             self.EvDNo['HRC_Temp']     = np.vstack((self.EvDNo['HRC_Temp'],self.f['HRC_Temp'][i]))
                #             self.EvDNo['qC_Temp']      = np.vstack((self.EvDNo['qC_Temp'],self.f['qC_Temp'][i]))
                #             self.EvDNo['FechaEv_Temp']      = np.vstack((self.EvDNo['FechaEv_Temp'],self.f['FechaEv_Temp'][i]))

                # self.Statistics.update({'Perc_Temp_Yes':len(self.EvDYes['PrecC_Temp'])/len(self.f['PrecC_Temp']),
                #         'Perc_Temp_No':len(self.EvDNo['PrecC_Temp'])/len(self.f['PrecC_Temp'])})
        return

    def EventsSeriesGenGraph(self,R=None,
            ImgFolder='',Evmax=1,EvType='Tot',
            flags={'TC':False,'PresC':False,'HRC':False,'qC':False,'WC':False},
            flagAver=False,flagBig=False,DataV=None,DataKeyV=['DatesEvst','DatesEvend'],
            vm = {'vmax':[None],'vmin':[0]},
            GraphInfoV={'color':['-.b','-.g'],'label':['Inicio del Evento','Fin del Evento']}):
        '''
        DESCRIPTION:
            Con esta función se pretenden graficar los diferentes eventos de 
            precipitación por separado.

            Es necesario haber corrido la función EventsPrecPresDetection antes.
        _________________________________________________________________________

        INPUT:
            + ImgFolder: Carpeta donde se guardarán los datos.
        _________________________________________________________________________

        OUTPUT:
            Se generan los gráficos

        '''
        self.ImgFolder = ImgFolder


        Labels = ['PresC','TC','HRC','qC','WC']
        UnitsDict = {'TC':'Temperatura [°C]','PresC':'Presión [hPa]','HRC':'Humedad Relativa [%]',
                'qC':'Humedad Específica [g/kg]','WC':'Relación de Mezcla [g/kg]'}
        ColorDict = {'TC':'r','PresC':'k','HRC':'g',
                'qC':'g','WC':'m'}
        LabelDict = {'TC':'Temperatura','PresC':'Presión','HRC':'Humedad Relativa',
                'qC':'Humedad Específica','WC':'Relación de Mezcla de V.A.'}


        DataKeys = ['PrecC']
        Units = ['Precipitación [mm]']
        Color = ['b']
        Label = ['Precipitación']
        Vars = 'Prec'
        for iLab,Lab in enumerate(Labels):
            if flags[Lab]:
                DataKeys.append(Lab)
                Units.append(UnitsDict[Lab])
                Color.append(ColorDict[Lab])
                Label.append(LabelDict[Lab])
                Vars += '_'+Lab[:-1]
        Vars += '_Events'


        if isinstance(R,dict):
            EvTot = dict()
            if EvType == 'Tot':
                EvTot = self.f
            elif EvType == 'Yes':
                for iLab,Lab in enumerate(Labels+['PrecC','FechaEv','FechaEvP']):
                    EvTot[Lab] = self.f[Lab][R['EvYes']]
            elif EvType == 'No':
                for iLab,Lab in enumerate(Labels+['PrecC','FechaEv','FechaEvP']):
                    EvTot[Lab] = self.f[Lab][R['EvNo']]
        elif isinstance(R,np.ndarray):
            EvTot = dict()
            for iLab,Lab in enumerate(Labels+['PrecC','FechaEv','FechaEvP']):
                EvTot[Lab] = self.f[Lab][R]
        else:
            EvTot = dict()
            if EvType == 'Tot':
                EvTot = self.f
            elif EvType == 'Yes':
                for iLab,Lab in enumerate(Labels+['PrecC']):
                    EvTot[Lab] = self.f[Lab][self.EvYes]
            elif EvType == 'No':
                for iLab,Lab in enumerate(Labels+['PrecC']):
                    EvTot[Lab] = self.f[Lab][self.EvNo]

        if flagAver:
            # Se grafican los eventos en promedio
            dt = int(self.dtm)
            Data = dict()
            Data['PrecC'] = np.nanmean(EvTot['PrecC'],axis=0)
            for iLab,Lab in enumerate(Labels):
                if flags[Lab]:
                    Data[Lab] = np.nanmean(EvTot[Lab],axis=0)
            BP.EventsSeriesGen(EvTot['FechaEv'][0],Data,self.PrecCount,
                    DataKeyV=DataKeyV,DataKey=DataKeys,
                    PathImg=self.PathImg+ImgFolder+'Series/'+EvType+'/'+Vars+'/',
                    Name=self.Names[self.irow],NameArch=self.NamesArch[self.irow],
                    GraphInfo={'ylabel':Units,'color':Color,
                        'label':Label},
                    GraphInfoV={'color':['-.b','-.g'],
                        'label':['Inicio del Evento','Fin del Evento']},
                    flagBig=flagBig,vm=vm,Ev=0,
                    flagV=False,flagAverage=True,dt=dt)
        else:
            # Se grafican los eventos
            for iEv in range(len(EvTot['PrecC'])):
                if iEv <= Evmax:
                    Data = dict()
                    DataVer = dict()
                    Data['PrecC'] = EvTot['PrecC'][iEv]
                    for iLab,Lab in enumerate(Labels):
                        if flags[Lab]:
                            Data[Lab] = EvTot[Lab][iEv]
                    for iLab,Lab in enumerate(DataKeyV):
                        DataVer[Lab] = DataV[Lab][iEv]
                    BP.EventsSeriesGen(EvTot['FechaEv'][iEv],Data,DataVer,
                            DataKeyV=DataKeyV,DataKey=DataKeys,
                            PathImg=self.PathImg+ImgFolder+'Series/'+EvType+'/'+Vars+'/',
                            Name=self.Names[self.irow],NameArch=self.NamesArch[self.irow],
                            GraphInfo={'ylabel':Units,'color':Color,
                                'label':Label},
                            GraphInfoV=GraphInfoV,flagV=False,
                            flagBig=flagBig,vm={'vmax':[None],'vmin':[0]},Ev=iEv,
                            Date=DUtil.Dates_datetime2str([self.PrecCount['DatesEvst'][iEv]])[0])

    def EventsSeriesGraphAEv(self,Dates,ImgFolder='',
            flags={'T_F':False,'Pres_F':False,'HR_F':False,'q_F':False,'W_F':False},
            flagBig=False,vm={'vmax':[1.45,1.15,3.7,21.2],'vmin':[0,-1.1,-4.9,-17.1]}):
        '''
        DESCRIPTION:
            Con esta función se pretenden graficar los diferentes eventos de 
            precipitación por separado.

            Es necesario haber corrido la función EventsPrecPresDetection antes.
        _________________________________________________________________________

        INPUT:
            + Dates: Fecha inicial y final del evento.
            + ImgFolder: Carpeta donde se guardarán los datos.
        _________________________________________________________________________

        OUTPUT:
            Se generan los gráficos

        '''
        self.ImgFolder = ImgFolder
        if not(isinstance(Dates[0],datetime)):
            Dates = DUtil.Dates_str2datetime(Dates)

        Labels = ['Pres_F','T_F','HR_F','q_F','W_F']
        UnitsDict = {'T_F':'Temperatura [°C]','Pres_F':'Presión [hPa]','HR_F':'Humedad Relativa [%]',
                'q_F':'Humedad Específica [g/kg]','W_F':'Tasa de Mezcla [g/kg]'}
        ColorDict = {'T_F':'r','Pres_F':'k','HR_F':'g',
                'q_F':'g','W_F':'m'}
        LabelDict = {'T_F':'Temperatura','Pres_F':'Presión','HR_F':'Humedad Relativa',
                'q_F':'Humedad Específica','W_F':'Tasa de Mezcla de V.A.'}


        DataKeys = ['Prec']
        Units = ['Precipitación [mm]']
        Color = ['b']
        Label = ['Precipitación']
        Vars = 'Prec'
        for iLab,Lab in enumerate(Labels):
            if flags[Lab]:
                DataKeys.append(Lab)
                Units.append(UnitsDict[Lab])
                Color.append(ColorDict[Lab])
                Label.append(LabelDict[Lab])
                Vars += '_'+Lab
        Vars += '_Events'

        # Se encuentra el evento
        FechaCP = self.f['FechaCP']
        xDatei = np.where(FechaCP==Dates[0])[0][0]
        xDatef = np.where(FechaCP==Dates[1])[0][0]
        EvTot = dict()
        for iE,E in enumerate(DataKeys):
            FechaCPC = FechaCP[xDatei:xDatef+1]
            EvTot[E] = self.f[E][xDatei:xDatef+1]
            print(E,'vmax',np.nanmax(EvTot[E]))
            print(E,'vmin',np.nanmin(EvTot[E]))

        BP.EventsSeriesGen(FechaCPC,EvTot,0,
                DataKeyV=['DatesEvst','DatesEvend'],DataKey=DataKeys,
                PathImg=self.PathImg+ImgFolder+'EventComp/'+'/'+Vars+'/',
                Name=self.Names[self.irow],NameArch=self.NamesArch[self.irow],
                GraphInfo={'ylabel':Units,'color':Color,
                    'label':Label},
                GraphInfoV={'color':['-.b','-.g'],
                    'label':['Inicio del Evento','Fin del Evento']},
                flagBig=flagBig,vm=vm,Ev=0,flagV=False,Date=Dates[0].strftime('%Y/%m/%d'),
                flagEvent=True)

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
                PrecH,PrecBin,PresH,PresBin,TH,TBin = BP.graphEv(self.f['PrecC'],self.f['PresC'],self.f['TC'],'Precipitación','Presión Barométrica','Temperatura','Prec','Pres','Temp','Precipitación [mm]','Presión [hPa]','Temperatura [°C]','b-','k-','r-','b','k','r',self.irow,self.NamesArch[self.irow],self.PathImg+self.ImgFolder+'Original/' ,DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,TH,TBin = BP.graphEv(self.f['PrecC_Pres'],self.f['PresC_Pres'],self.f['TC_Pres'],'Precipitación','Presión Barométrica','Temperatura','Prec','Pres','Temp','Precipitación [mm]','Presión [hPa]','Temperatura [°C]','b-','k-','r-','b','k','r',self.irow,self.NamesArch[self.irow]+'_Pres',self.PathImg+self.ImgFolder,DTT=self.dtm)
                # PrecH,PrecBin,PresH,PresBin,TH,TBin = BP.graphEv(self.f['PrecC_Temp'],self.f['PresC_Temp'],self.f['TC_Temp'],'Precipitación','Presión Barométrica','Temperatura','Prec','Pres','Temp','Precipitación [mm]','Presión [hPa]','Temperatura [°C]','b-','k-','r-','b','k','r',self.irow,self.NamesArch[self.irow]+'_Temp',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.flag['PresC'] and self.flag['HRC']:
                PrecH,PrecBin,PresH,PresBin,HRH,TBin = BP.graphEv(self.f['PrecC'],self.f['PresC'],self.f['HRC'],'Precipitación','Presión Barométrica','Humedad Relativa','Prec','Pres','HR','Precipitación [mm]','Presión [hPa]','Humedad Relativa [%]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow],self.PathImg+self.ImgFolder+'Original/',DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,HRH,TBin = BP.graphEv(self.f['PrecC_Pres'],self.f['PresC_Pres'],self.f['HRC_Pres'],'Precipitación','Presión Barométrica','Humedad Relativa','Prec','Pres','HR','Precipitación [mm]','Presión [hPa]','Humedad Relativa [%]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Pres',self.PathImg+self.ImgFolder,DTT=self.dtm)
                # PrecH,PrecBin,PresH,PresBin,HRH,TBin = BP.graphEv(self.f['PrecC_Temp'],self.f['PresC_Temp'],self.f['HRC_Temp'],'Precipitación','Presión Barométrica','Humedad Relativa','Prec','Pres','HR','Precipitación [mm]','Presión [hPa]','Humedad Relativa [%]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Temp',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.flag['PresC'] and self.flag['qC']:
                PrecH,PrecBin,PresH,PresBin,qH,TBin = BP.graphEv(self.f['PrecC'],self.f['PresC'],self.f['qC'],'Precipitación','Presión Barométrica','Humedad Específica','Prec','Pres','q','Precipitación [mm]','Presión [hPa]','Humedad Específica [kg/kg]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow],self.PathImg+self.ImgFolder+'Original/',DTT=self.dtm)
                PrecH,PrecBin,PresH,PresBin,qH,TBin = BP.graphEv(self.f['PrecC_Pres'],self.f['PresC_Pres'],self.f['qC_Pres'],'Precipitación','Presión Barométrica','Humedad Específica','Prec','Pres','q','Precipitación [mm]','Presión [hPa]','Humedad Específica [kg/kg]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Pres',self.PathImg+self.ImgFolder,DTT=self.dtm)
                # PrecH,PrecBin,PresH,PresBin,qH,TBin = BP.graphEv(self.f['PrecC_Temp'],self.f['PresC_Temp'],self.f['qC_Temp'],'Precipitación','Presión Barométrica','Humedad Específica','Prec','Pres','q','Precipitación [mm]','Presión [hPa]','Humedad Específica [kg/kg]','b-','k-','g-','b','k','g',self.irow,self.NamesArch[self.irow]+'_Temp',self.PathImg+self.ImgFolder,DTT=self.dtm)
            if self.flag['TC'] and self.flag['qC']:
                PrecH,PrecBin,TH,TBin,HRH,TBin = BP.graphEv(self.f['PrecC'],self.f['TC'],self.f['qC'],'Precipitación','Temperatura','Humedad Específica','Prec','Temp','q','Precipitación [mm]','Temperatura [°C]','Humedad Específica [kg/kg]','b-','r-','g-','b','r','g',self.irow,self.NamesArch[self.irow],self.PathImg+self.ImgFolder+'Original/',DTT=self.dtm)
                PrecH,PrecBin,TH,TBin,HRH,TBin = BP.graphEv(self.f['PrecC_Pres'],self.f['TC_Pres'],self.f['qC_Pres'],'Precipitación','Temperatura','Humedad Específica','Prec','Temp','q','Precipitación [mm]','Temperatura [°C]','Humedad Específica [kg/kg]','b-','r-','g-','b','r','g',self.irow,self.NamesArch[self.irow]+'_Pres',self.PathImg+self.ImgFolder,DTT=self.dtm)
                # PrecH,PrecBin,TH,TBin,HRH,TBin = BP.graphEv(self.f['PrecC_Temp'],self.f['TC_Temp'],self.f['qC_Temp'],'Precipitación','Temperatura','Humedad Específica','Prec','Temp','q','Precipitación [mm]','Temperatura [°C]','Humedad Específica [kg/kg]','b-','r-','g-','b','r','g',self.irow,self.NamesArch[self.irow]+'_Temp',self.PathImg+self.ImgFolder,DTT=self.dtm)
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

    def Calc_Changes(self,DatesEv,Prec,VC,Pos=None,MaxMin='min',Var='Pres',EvType='Tot',MPP=None):
        '''
        DESCRIPTION:

            Función para obtener los valores de Duración, tasas y cambios de 
            cualquier variable con los datos dados. 

            Requiere que se corra el método de EventsGenPrec antes de utilizar
            este método ya que requiere unas cuentas adicionales.
        _________________________________________________________________________

        INPUT:
            + Prec: Compuestos de precipitación.
            + VC: Compuestos de la variable
            + Pos: Posición inicial del cambio en la variable, se relaciona
                   a los datos de comienzo del evento de precipitación.
            + MinMax: Valor que se encuentra en el centro de los datos.
            + EvType: Key para el diccionario de resultados.
        _________________________________________________________________________

        OUTPUT:
            Se generan diferentes variables o se cambian las actuales.
        '''
        # ---------------
        # Parameters
        # ---------------
        # Se incluyen los valores de la clase
        dt = int(self.dtm)
        MP = self.Middle # Minimo de precipitación
        try:
            self.Res_Prec[Var] = dict()
        except AttributeError:
            self.Res_Prec = dict()
            self.Res_Prec[Var] = dict()
        # -----------------
        # Calculos
        # -----------------
        # Se calcula la duración de la tormenta
        self.Res_Prec[Var][EvType] = HyMF.PrecCount(Prec,DatesEv,dt=dt,M=MP)
        # Datos de las variables de cambio
        Results = BP.C_Rates_Changes(VC,dt=dt,MP=Pos,MaxMin=MaxMin)
        self.Res_Prec[Var][EvType].update(Results)
        self.Res_Prec[Var][EvType]['PosI'] = Pos
        # -----------------
        # Posiciones
        # -----------------
        # print(self.Res_Prec[Var][EvType]['PosB'])
        self.Res_Prec[Var][EvType]['DatesMidVC'] = []
        self.Res_Prec[Var][EvType]['DatesBVC']   = []
        self.Res_Prec[Var][EvType]['DatesAVC']   = []
        for i in range(len(DatesEv)):
            DatesEvP = DUtil.Dates_str2datetime(DatesEv[i])
            self.Res_Prec[Var][EvType]['DatesMidVC'].append(DatesEvP[MPP[i]])
            if not(np.isnan(self.Res_Prec[Var][EvType]['PosB'][i])):
                self.Res_Prec[Var][EvType]['DatesBVC'].append(DatesEvP[int(self.Res_Prec[Var][EvType]['PosB'][i])])
            else:
                self.Res_Prec[Var][EvType]['DatesBVC'].append(np.nan)
            if not(np.isnan(self.Res_Prec[Var][EvType]['PosA'][i])):
                self.Res_Prec[Var][EvType]['DatesAVC'].append(DatesEvP[int(self.Res_Prec[Var][EvType]['PosA'][i])])
            else:
                self.Res_Prec[Var][EvType]['DatesAVC'].append(np.nan)

        return

    def GraphChanges(self,Var='Pres',EvType='Tot',Var2=None,flagIng=False,ImgFolder_Scatter='/Manizales/Scatter/',Specific_Folder='Events_3',H=''):
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

        # -----------------
        # Parameters
        # -----------------
        # Información para hacer el gráfico
        self.ImgFolder_Scatter = ImgFolder_Scatter
        ImgFolder_Scatter_Specific_Pres = ImgFolder_Scatter+Var+'/'+EvType+'/'+Specific_Folder+'/'

        # Datos para los gráficos
        Variables = {'DurPrec': 'Duración del Evento [h]',
                'IntPrec': 'Intensidad del Evento [mm/h]',
                'MaxPrec': 'Máximo de Precipitación [mm]',
                'TotalPrec': 'Total de Precipitación [mm]',
                'IntPrecMax': 'Intensidad Máxima del Evento [mm/h]',
                'Pindex': 'Relación de Intensidades',
                'TasaPrec': 'Tasa de Cambio de Precipitación [mm/h]'}
        VarVC = {'Pres' : {'VRateB': 'Tasa de Cambio de Presión Antes [hPa/h]',
                'VRateA':'Tasa de Cambio de Presión Durante [hPa/h]',
                'VChangeB':'Cambio de Presión Antes [hPa]',
                'VChangeA':'Cambio de Presión Durante [hPa]'},
                'Temp' : {'VRateB': 'Tasa de Cambio de Temperatura Antes [°C/h]',
                'VRateA':'Tasa de Cambio de Temperatura Durante [°C/h]',
                'VChangeB':'Cambio de Temperatura Antes [°C]',
                'VChangeA':'Cambio de Temperatura Durante [°C]'},
                'HR' : {'VRateB': 'Tasa de Cambio de Hum. Rel. Antes [°C/h]',
                'VRateA':'Tasa de Cambio de Hum. Rel. Durante [°C/h]',
                'VChangeB':'Cambio de Hum. Rel. Antes [°C]',
                'VChangeA':'Cambio de Hum. Rel. Durante [°C]'},
                }

        Abre = {'DurPrec': 'DP',
                'IntPrec': 'IP',
                'MaxPrec': 'MP',
                'TotalPrec': 'TP',
                'IntPrecMax':'IPM',
                'Pindex':'Pi','TasaPrec':'RPr'}
        AbreVC = {'Pres' : {'VRateB':'RPB',
                    'VRateA':'RPA',
                    'VChangeB':'CPB',
                    'VChangeA':'CPA'},
                   'Temp' : {'VRateB':'TTB',
                    'VRateA':'RTA',
                    'VChangeB':'CTB',
                    'VChangeA':'CTA'},
                   'HR' : {'VRateB':'PHRB',
                    'VRateA':'RHRA',
                    'VChangeB':'CHRB',
                    'VChangeA':'CHRA'},
                   }
        VPrec = ['DurPrec','IntPrec','MaxPrec','TotalPrec',
                'IntPrecMax','Pindex','TasaPrec']
        VCC = ['VRateB','VRateA','VChangeB','VChangeA']

        # --------------
        # Data
        # --------------
        R1 = self.Res_Prec[Var][EvType]
        if Var2 is None:
            V1 = VPrec
            V2 = VCC
            R2 = self.Res_Prec[Var][EvType]
            VarL1 = Variables
            VarL2 = VarVC[Var]
            AbreL1 = Abre
            AbreL2 = AbreVC[Var]
        else:
            V1 = VCC
            V2 = VCC
            R2 = self.Res_Prec[Var2][EvType]
            VarL1 = VarVC[Var]
            VarL2 = VarVC[Var2]
            AbreL1 = AbreVC[Var]
            AbreL2 = AbreVC[Var2]

        for Vi1 in V1:
            for Vi2 in V2:
                HyPl.SPvDPPotGen(R1[Vi1],R2[Vi2],
                    Fit='',Title='Diagrama de Dispersión',
                    xLabel=VarL1[Vi1],
                    yLabel=VarL2[Vi2],
                    Name=self.NamesArch[self.irow]+'_'+AbreL1[Vi1]+'v'+AbreL2[Vi2],
                    PathImg=self.PathImg+ImgFolder_Scatter+Var+'/'+EvType+'/'+'Adjusted/'+H+'/',
                    FlagA=True,FlagAn=False)

        return

    def VC_VR_DP(self,DatesEv,Prec,Pres,MP=None,MPres=None,flagEv_Pres=False,LimitEvP=1000,flagEv_T=False,flagIng=False,ImgFolder_Scatter='/Manizales/Scatter/',Specific_Folder='Events_3'):
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
                MP.append(np.where(Prec[iC][self.Middle:self.Middle+int(60/int(self.dtm)*1)+1]==
                    np.nanmax(Prec[iC][self.Middle:self.Middle+int(60/int(self.dtm)*1)+1]))[0][0]+self.Middle)
        if MPres == None:
            MPres = self.Middle

        # Se calcula la duración de la tormenta
        if self.flag['PrecC']:
            self.Res_Pres = HyMF.PrecCount(Prec,DatesEv,dt=dt,M=MP)

        if self.flag['PresC']:
            Results = BP.C_Rates_Changes(Pres,dt=dt,MP=MPres,MaxMin='min')

        self.Res_Pres.update(Results)
        Data = {'Prec':Prec,'Pres':Pres}
        self.Res_Pres['VminPos'] = np.ones(self.Res_Pres['DurPrec'].shape)*self.Middle


        # Datos para los gráficos
        Variables = {'DurPrec': 'Duración del Evento [h]',
                'IntPrec': 'Intensidad del Evento [mm/h]',
                'MaxPrec': 'Máximo de Precipitación [mm]',
                'TotalPrec': 'Total de Precipitación [mm]',
                'IntPrecMax': 'Intensidad Máxima del Evento [mm/h]',
                'Pindex': 'Relación de Intensidades',
                'TasaPrec': 'Tasa de Cambio de Precipitación [mm/h]',
                'VRateB': 'Tasa de Cambio de Presión Antes [hPa/h]',
                'VRateA':'Tasa de Cambio de Presión Durante [hPa/h]',
                'VChangeB':'Cambio de Presión Antes [hPa]',
                'VChangeA':'Cambio de Presión Durante [hPa]'}

        Abre = {'DurPrec': 'DP',
                'IntPrec': 'IP',
                'MaxPrec': 'MP',
                'TotalPrec': 'TP',
                'IntPrecMax':'IPM',
                'Pindex':'Pi',
                'TasaPrec':'RPr',
                'VRateB': 'RPB',
                'VRateA':'RPA',
                'VChangeB':'CPB',
                'VChangeA':'CPA'}


        V1 = ['DurPrec','IntPrec','MaxPrec','TotalPrec','IntPrecMax','Pindex','TasaPrec']
        V2 = ['VRateB','VRateA','VChangeB','VChangeA']
        for Vi1 in V1:
            for Vi2 in V2:
                HyPl.SPvDPPotGen(self.Res_Pres[Vi1],self.Res_Pres[Vi2],
                        Fit='',Title='Diagrama de Dispersión',
                        xLabel=Variables[Vi1],
                        yLabel=Variables[Vi2],
                        Name=self.NamesArch[self.irow]+'_'+Abre[Vi1]+'v'+Abre[Vi2],
                        PathImg=self.PathImg+ImgFolder_Scatter+'Pres/'+'Adjusted/',
                        FlagA=True,FlagAn=False)


        if flagEv_Pres:
            BP.EventsScatter(DatesEv,Data,self.Res_Pres,
                    PathImg=self.PathImg+ImgFolder_Scatter_Specific_Pres,
                    Name=self.Names[self.irow]+'_DPvRP',flagIng=False,LimitEv=LimitEvP)

            BP.EventsScatter(DatesEv,Data,self.Res_Pres,
                    PathImg=self.PathImg+ImgFolder_Scatter_Specific_Pres,
                    Name=self.Names[self.irow]+'DP_CP',flagIng=False,
                    LimitEv=LimitEvP,Fit ='',
                    LabelsScatter=['DurPrec','VChangeB','VChangeA'],
                    Scatter_Info=['Cambios de Presión Antes del Evento',
                    'Duration [h]','Cambio de Presión [hPa]',
                    'Cambios en Presión Atmosférica Durante el Evento'])
        return

    def VC_DP_VR(self,DatesEv,Prec,Pres,DatesEvP=None,MP=None,MPres=None,flagEv_Pres=False,LimitEvP=1000,flagEv_T=False,flagIng=False,ImgFolder_Scatter='/Manizales/Scatter/',Specific_Folder='Events_3'):
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
        ImgFolder_Scatter_Specific_Pres = ImgFolder_Scatter+'Pres_Prec/'+Specific_Folder+'/'
        ImgFolder_Scatter_Specific_Temp = ImgFolder_Scatter+'Temp/'+Specific_Folder+'/'

        # Se incluyen los valores de la clase
        dt = int(self.dtm)
        if MP == None:
            MP = self.Middle
        # Se calcula la duración de la tormenta
        if self.flag['PrecC']:
            self.Res_Prec = HyMF.PrecCount(Prec,DatesEv,dt=dt,M=MP)

        if DatesEvP == None:
            DatesEvP = self.f['FechaEvP']
        if MPres == None:
            MPres = list()
            for iC in range(len(Pres)):
                xEv = np.where(DatesEvP[iC] == self.Res_Prec['DatesEvst'][iC])[0][0]
                Bef = xEv-int(60/dt*3)
                Aft = xEv+int(60/dt*(15/60)) 
                Min = np.nanmin(Pres[iC][Bef:Aft])
                xMin1 = np.where(Pres[iC][:Aft]==Min)[0][-1]
                MPres.append(xMin1)


        if self.flag['PresC']:
            Results = BP.C_Rates_Changes(Pres,dt=dt,MP=MPres,MaxMin='min')

        self.Res_Prec.update(Results)
        Data = {'Prec':Prec,'Pres':Pres}
        self.Res_Prec['VminPos'] = np.array(MPres)


        # Datos para los gráficos
        Variables = {'DurPrec': 'Duración del Evento [h]',
                'IntPrec': 'Intensidad del Evento [mm/h]',
                'MaxPrec': 'Máximo de Precipitación [mm]',
                'TotalPrec': 'Total de Precipitación [mm]',
                'IntPrecMax': 'Intensidad Máxima del Evento [mm/h]',
                'Pindex': 'Relación de Intensidades',
                'TasaPrec': 'Tasa de Cambio de Precipitación [mm/h]',
                'VRateB': 'Tasa de Cambio de Presión Antes [hPa/h]',
                'VRateA':'Tasa de Cambio de Presión Durante [hPa/h]',
                'VChangeB':'Cambio de Presión Antes [hPa]',
                'VChangeA':'Cambio de Presión Durante [hPa]'}

        Abre = {'DurPrec': 'DP',
                'IntPrec': 'IP',
                'MaxPrec': 'MP',
                'TotalPrec': 'TP',
                'IntPrecMax':'IPM',
                'Pindex':'Pi',
                'TasaPrec':'RPr',
                'VRateB': 'RPB',
                'VRateA':'RPA',
                'VChangeB':'CPB',
                'VChangeA':'CPA'}


        V1 = ['DurPrec','IntPrec','MaxPrec','TotalPrec','IntPrecMax','Pindex','TasaPrec']
        V2 = ['VRateB','VRateA','VChangeB','VChangeA']
        for Vi1 in V1:
            for Vi2 in V2:
                HyPl.SPvDPPotGen(self.Res_Prec[Vi1],self.Res_Prec[Vi2],
                        Fit='',Title='Diagrama de Dispersión',
                        xLabel=Variables[Vi1],
                        yLabel=Variables[Vi2],
                        Name=self.NamesArch[self.irow]+'_'+Abre[Vi1]+'v'+Abre[Vi2],
                        PathImg=self.PathImg+ImgFolder_Scatter+'Pres_Prec/'+'Adjusted/',
                        FlagA=True,FlagAn=False)

        if flagEv_Pres:
            BP.EventsScatter(DatesEv,Data,self.Res_Prec,
                    PathImg=self.PathImg+ImgFolder_Scatter_Specific_Pres,
                    Fit ='',
                    Name=self.NamesArch[self.irow]+'_DPvRP',
                    flagIng=False,LimitEv=LimitEvP)

            # BP.EventsScatter(DatesEv,Data,self.Res_Prec,
            #         PathImg=self.PathImg+ImgFolder_Scatter_Specific_Pres,
            #         Name=self.NamesArch[self.irow]+'DP_CP',flagIng=False,
            #         LimitEv=LimitEvP,Fit ='',
            #         LabelsScatter=['DurPrec','VChangeB','VChangeA'],
            #         Scatter_Info=['Cambios de Presión Antes del Evento',
            #         'Duration [h]','Cambio de Presión [hPa]',
            #         'Cambios en Presión Atmosférica Durante el Evento'])
        return

    def HistGraph(self,ImgFolder_Scatter='/Manizales/Scatter/',Specific_Folder='Events_3'):
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
        ImgFolder_Scatter_Specific_Pres = ImgFolder_Scatter+'Pres_Prec/'+Specific_Folder+'/'
        ImgFolder_Scatter_Specific_Temp = ImgFolder_Scatter+'Temp/'+Specific_Folder+'/'

        # Se incluyen los valores de la clase
        dt = int(self.dtm)

        # Datos para los gráficos
        Variables = {'DurPrec': 'Duración del Evento [h]',
                'IntPrec': 'Intensidad del Evento [mm/h]',
                'MaxPrec': 'Máximo de Precipitación [mm]',
                'TotalPrec': 'Total de Precipitación [mm]',
                'IntPrecMax': 'Intensidad Máxima del Evento [mm/h]',
                'Pindex': 'Relación de Intensidades',
                'TasaPrec': 'Tasa de Cambio de Precipitación [mm/h]',
                'VRateB': 'Tasa de Cambio de Presión Antes [hPa/h]',
                'VRateA':'Tasa de Cambio de Presión Durante [hPa/h]',
                'VChangeB':'Cambio de Presión Antes [hPa]',
                'VChangeA':'Cambio de Presión Durante [hPa]',
                'Hour': 'Hora del Evento [LT]'}

        Abre = {'DurPrec': 'DP',
                'IntPrec': 'IP',
                'MaxPrec': 'MP',
                'TotalPrec': 'TP',
                'IntPrecMax':'IPM',
                'Pindex':'Pi',
                'TasaPrec':'RPr',
                'VRateB': 'RPB',
                'VRateA':'RPA',
                'VChangeB':'CPB',
                'VChangeA':'CPA',
                'Hour': 'HEv'}

        self.PrecCount['Hour'] = np.array([i.hour for i in self.PrecCount['DatesEvst']])

        V1 = ['Hour','DurPrec','IntPrec','MaxPrec','TotalPrec','IntPrecMax','Pindex',
                'TasaPrec']
        V2 = ['VRateB','VRateA','VChangeB','VChangeA']

        Bins2d=12
        for iV1,Vi1 in enumerate(V1):
            if iV1 > 0:
                continue
            if Vi1 == 'Hour':
                FlagHour = True
                flagEst = False
                Bins=np.arange(0,24)
            else:
                Bins=12
                FlagHour = False
                flagEst = False
            for iV2,Vi2 in enumerate(V1[iV1:]):
                if Vi1 != Vi2:
                    if Vi2 == 'IntPrec':
                        Bins2 = np.arange(0,45,5)
                    else:
                        Bins2=12
                    # Total
                    HyPl.Histogram2d(self.PrecCount[Vi1],self.PrecCount[Vi2],
                            [Bins,Bins2],Title='Frecuencias en 2 Dimensiones',
                            Var1=Variables[Vi1],Var2=Variables[Vi2],
                            Name=self.NamesArch[self.irow]+'_'+Abre[Vi1]+'_'+Abre[Vi2]+'_EvT',
                            PathImg=self.PathImg+ImgFolder_Scatter+'Histograms2d/',
                            M=True,
                            FlagHour=False,FlagBig=True) 
                    # EvYes
                    HyPl.Histogram2d(self.PrecCount[Vi1][self.EvYes],
                            self.PrecCount[Vi2][self.EvYes],
                            [Bins,Bins2],Title='Frecuencias en 2 Dimensiones',
                            Var1=Variables[Vi1],Var2=Variables[Vi2],
                            Name=self.NamesArch[self.irow]+'_'+Abre[Vi1]+'_'+Abre[Vi2]+'_EvYes',
                            PathImg=self.PathImg+ImgFolder_Scatter+'Histograms2d/',
                            M=True,
                            FlagHour=False,FlagBig=True) 
                    # EvNo
                    HyPl.Histogram2d(self.PrecCount[Vi1][self.EvNo],
                            self.PrecCount[Vi2][self.EvNo],
                            [Bins,Bins2],Title='Frecuencias en 2 Dimensiones',
                            Var1=Variables[Vi1],Var2=Variables[Vi2],
                            Name=self.NamesArch[self.irow]+'_'+Abre[Vi1]+'_'+Abre[Vi2]+'_EvNo',
                            PathImg=self.PathImg+ImgFolder_Scatter+'Histograms2d/',
                            M=True,
                            FlagHour=False,FlagBig=True) 
            # Total
            HyPl.HistogramNP(self.PrecCount[Vi1],Bins,Title=' Frecuencias',
                    Var=Variables[Vi1],
                    Name=self.NamesArch[self.irow]+'_'+Abre[Vi1]+'_EvT',
                    PathImg=self.PathImg+ImgFolder_Scatter+'Histograms/',
                    M='porcen',FEn=False,Left=True,FlagHour=FlagHour,flagEst=flagEst)
            # EvYes
            HyPl.HistogramNP(self.PrecCount[Vi1][self.EvYes],
                    Bins,Title='Frecuencias',
                    Var=Variables[Vi1],
                    Name=self.NamesArch[self.irow]+'_'+Abre[Vi1]+'_EvYes',
                    PathImg=self.PathImg+ImgFolder_Scatter+'Histograms/',
                    M='porcen',FEn=False,Left=True,FlagHour=FlagHour,flagEst=flagEst)
            # EvNo
            HyPl.HistogramNP(self.PrecCount[Vi1][self.EvNo],
                    Bins,Title='Frecuencias',
                    Var=Variables[Vi1],
                    Name=self.NamesArch[self.irow]+'_'+Abre[Vi1]+'_EvNo',
                    PathImg=self.PathImg+ImgFolder_Scatter+'Histograms/',
                    M='porcen',FEn=False,Left=True,FlagHour=FlagHour,flagEst=flagEst)
            
    def HistGraphSel(self,Var1,Var='Pres',Name=None,EndImg='T',EndFold='',ImgFolder_Scatter='/Manizales/Scatter/',Specific_Folder='Events_3'):
        '''
        DESCRIPTION:

            Función para graficar el histograma de frecuencias de la 
            información que uno adicione.
        _________________________________________________________________________

        INPUT:
            + Var1: Variable 1
            + ImgFolder_Scatter: Ruta donde se guardará el documento.
            + Specific_Folder: Ruta donde se guardará el documento.
        _________________________________________________________________________

            OUTPUT:
        Se generan diferentes variables o se cambian las actuales.
        '''

        # Información para hacer el gráfico
        self.ImgFolder_Scatter = ImgFolder_Scatter

        # Se incluyen los valores de la clase
        dt = int(self.dtm)

        if Name is None:
            Name = self.NamesArch[self.irow]

        if Var == 'Pres':
            Var2 = 'Presión'
            Uni = 'hPa'
            Ind = 'P'
        elif Var == 'T' or Var == 'Temp':
            Var2 = 'Temperatura'
            Uni = '°C'
            Ind = 'T'
        elif Var == 'HR':
            Var2 = 'Hum. Rel.'
            Uni = '%'
            Ind = 'HR'


        # Datos para los gráficos
        Variables = {'DurPrec': 'Duración del Evento [h]',
                'IntPrec': 'Intensidad del Evento [mm/h]',
                'MaxPrec': 'Máximo de Precipitación [mm]',
                'TotalPrec': 'Total de Precipitación [mm]',
                'IntPrecMax': 'Intensidad Máxima del Evento [mm/h]',
                'Pindex': 'Relación de Intensidades',
                'TasaPrec': 'Tasa de Cambio de Precipitación [mm/h]',
                'VRateB': 'Tasa de Cambio de {} Antes [{}/h]'.format(Var2,Uni),
                'VRateA':'Tasa de Cambio de {} Durante [{}/h]'.format(Var2,Uni),
                'VChangeB':'Cambio de {} Antes [{}]'.format(Var2,Uni),
                'VChangeA':'Cambio de {} Durante [{}]'.format(Var2,Uni),
                'Hour': 'Hora del Evento [LT]'}

        Abre = {'DurPrec': 'DP',
                'IntPrec': 'IP',
                'MaxPrec': 'MP',
                'TotalPrec': 'TP',
                'IntPrecMax':'IPM',
                'Pindex':'Pi',
                'TasaPrec':'R{}r'.format(Ind),
                'VRateB': 'R{}B'.format(Ind),
                'VRateA':'R{}A'.format(Ind),
                'VChangeB':'C{}B'.format(Ind),
                'VChangeA':'C{}A'.format(Ind),
                'Hour': 'HEv'}

        V1 = ['Hour','DurPrec','IntPrec','MaxPrec','TotalPrec','IntPrecMax','Pindex',
                'TasaPrec']
        V2 = ['VRateB','VRateA','VChangeB','VChangeA']

        Bins2d=12
        for iV1,Vi1 in enumerate(V1):
            if Vi1 == 'Hour':
                continue
            if Vi1 == 'Hour':
                FlagHour = True
                flagEst = False
                Bins=np.arange(0,24)
                vmax = None
            else:
                FlagHour = False
                flagEst = False
                Bins=12
                vmax = None
            # Total
            HyPl.HistogramNP(Var1[Vi1],Bins,Title=' Frecuencias',
                    Var=Variables[Vi1],
                    Name=Name+'_'+Abre[Vi1]+'_Ev'+EndImg,
                    PathImg=self.PathImg+ImgFolder_Scatter+'Histograms'+EndFold+'/',
                    M='porcen',FEn=False,Left=True,FlagHour=FlagHour,flagEst=flagEst,
                    FlagBig=True,vmax=vmax)

        for iV1,Vi1 in enumerate(V2):
            if Vi1 == 'Hour':
                FlagHour = True
                flagEst = False
                Bins=np.arange(0,24)
                vmax = None
            else:
                FlagHour = False
                flagEst = False
                Bins=12
                vmax = None
            # Total
            HyPl.HistogramNP(Var1[Vi1],Bins,Title=' Frecuencias',
                    Var=Variables[Vi1],
                    Name=Name+'_'+Abre[Vi1]+'_Ev'+EndImg,
                    PathImg=self.PathImg+ImgFolder_Scatter+'Histograms'+EndFold+'/',
                    M='porcen',FEn=False,Left=True,FlagHour=FlagHour,flagEst=flagEst,
                    FlagBig=True,vmax=vmax)
