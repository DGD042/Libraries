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
from Utilities import Data_Man as DM
from AnET import CFitting as CF; CF=CF()
from Hydro_Analysis import Hydro_Plotter as HyPl;HyPl=HyPl()
from Hydro_Analysis.Meteo import MeteoFunctions as HyMF
from BPumpL.BPumpL import BPumpL as BP; BP=BP()
from BPumpL.Dates.DatesC import DatesC

class Detect_Events(object): 
    '''
    DESCRIPTION:

        Class that detects precipitation events from filtered series of
        Pressure, Temperature and Relative Humidity
    _________________________________________________________________________

    INPUT:
        :param DateA: A ndarray, Array with Dates, can be in text or datetime
                      format.
        :param Pres: A ndarray, Array with the Pressure series.
        :param T: A ndarray, Array with the Temperature series.
        :param HR: A ndarray, Array with the Relative Humidity series.
        :param dt: A int, integer with the time delta of the series in minutes.
        :param Limits: A dict, Dictionary with the limit values for search.
                       It has the following values: 'Pres','T','HR' and
                       inside each one has the following:
                       'VRateB': minium limit for the Rate of the variable.
                       'MV': Minimum value for the anomalie of the variable.
    _________________________________________________________________________

    OUTPUT:
    '''

    def __init__(self,DateA,Pres,T,HR,Extract=[5*60,5*60],Limits={'Pres':{'VRateB':0.5,'MV':-0.2},
                'T':{'VRateB':1.7,'MV':0.2},
                'HR':{'VRateB':10.4,'MV':-0.2}}):
        '''
        '''
        # ---------------
        # Error Managment
        # ---------------
        assert isinstance(DateA,np.ndarray)
        assert isinstance(Pres,np.ndarray)
        assert isinstance(T,np.ndarray)
        assert isinstance(HR,np.ndarray)
        # ---------------
        # Attributes
        # ---------------
        Var = ['Pres','T','HR']
        self.Series = dict()
        self.Series['Dates'] = DatesC(DateA)
        self.Series['Pres'] = Pres
        self.Series['T'] = T
        self.Series['HR'] = HR
        self.Limits = Limits
        self.dt = int((self.Series['Dates'].datetime[1] - self.Series['Dates'].datetime[0]).seconds/60)
        self.Extract = Extract
        self.Erase = 2*60/self.dt

        # Created Attributes
        self.SeriesC = dict() # Extracted Data
        self.MT = [] # Location from the series
        self.M = dict() # Location from 
        self.Changes = dict()
        for V in Var:
            self.M[V] = []
            self.Changes[V] = dict()


        self.Oper = {'max':np.nanmax,'min':np.nanmin}
        return

    def FindValue(self,Var='Pres',Oper='min',Comp='<',M=None,limits=[-5,5]):
        '''
        DESCRIPTION:
            Method that find the minimum or maximum value in the series
            and calculates the changes.
        _________________________________________________________________________
        
        INPUT:
         :param Var: An str, str of the dictionary of the Attribute Series. 
         :param VarC: An str, str of the dictionary of the Attribute Series when
                      comparation. 
         :param Oper: A str, Function that would search the value ('max','min').
         :param Extract: A list, List with the limits of the extracted values 
                         in minutes.
         :param Comp: A str, string with the comparation that would be made.
         :param M: A int or None, Integer with the position of the maximum of 
                   minimum, if None it searches in the whole series.
         :param limits: A list or None, List with the limits of search from the 
                        value of M.
        _________________________________________________________________________
        
        OUTPUT:
        '''
        # Constants
        dt = self.dt
        Extract = self.Extract 
        # Determine Operation
        O = self.Oper[Oper.lower()]
        Comp = DM.Oper_Det(Comp)
        flagRange = True

        if M is None:
            flagRange = False


        if flagRange:
            MVal = O(self.Series[Var][int(limits[0]/dt)+M:M+int(limits[1]/dt)])
            # Validation of the data
            if Comp(MVal,self.Limits[Var]['MV']):
                # Se extrae el evento
                Event = self.Series[Var][-int(Extract[0]/dt)+M:int(Extract[1]/dt)+M].copy()
                if -int(self.Extract[0])+M < 0:
                    self.Series[Var][0:int(self.Erase)+M] = np.nan 
                    return -2
                elif int(self.Extract[1])+M > len(self.Series[Var]):
                    self.Series[Var][-int(self.Erase)+M:len(self.Series[Var])] = np.nan 
                    return -2
                else:
                    self.Series[Var][-int(self.Erase)+M:int(self.Erase)+M] = np.nan 
                q = sum(~np.isnan(Event))
                if q < 0.9*len(Event):
                    return -2
                MP = np.where(Event == MVal)[0][0]
                # Se calculan los cambios
                Changes = BP.C_Rates_Changes(Event,dt=dt,MP=MP,MaxMin=Oper)
                if Changes['VRateB'] < self.Limits[Var]['VRateB'] or np.isnan(Changes['VRateB']):
                    return -2

                try:
                    VarC = list(Changes)
                    for V in VarC:
                        a = len(self.Changes[Var][V])
                        self.Changes[Var][V] = np.hstack((self.Changes[Var][V],Changes[V]))
                    self.SeriesC[Var+'C'] = np.vstack((self.SeriesC[Var+'C'],Event))
                except KeyError:
                    self.Changes[Var] = Changes.copy()
                    self.SeriesC[Var+'C'] = Event.copy()
                self.M[Var].append(MP)
            else:
                return -2
        else:
            # Finding the value
            MVal = O(self.Series[Var])
            M = np.where(self.Series[Var] == MVal)[0][0]
            # Validation of the data
            if Comp(MVal,self.Limits[Var]['MV']):
                # Se extrae el evento
                Event = self.Series[Var][-int(Extract[0]/dt)+M:int(Extract[1]/dt)+M].copy()
                if -int(self.Extract[0])+M < 0:
                    self.Series[Var][0:int(self.Erase)+M] = np.nan 
                    return -2
                elif int(self.Extract[1])+M > len(self.Series[Var]):
                    self.Series[Var][-int(self.Erase)+M:len(self.Series[Var])] = np.nan 
                    return -2
                else:
                    self.Series[Var][-int(self.Erase)+M:int(self.Erase)+M] = np.nan 
                q = sum(~np.isnan(Event))
                if q < 0.9*len(Event):
                    return -2
                MP = np.where(Event == MVal)[0][0]
                # Se calculan los cambios
                Changes = BP.C_Rates_Changes(Event,dt=dt,MP=MP,MaxMin=Oper)
                if Changes['VRateB'] < self.Limits[Var]['VRateB'] or np.isnan(Changes['VRateB']):
                    return -2

                Dates = self.Series['Dates'].datetime[-int(Extract[0]/dt)+M:int(Extract[1]/dt)+M]
                try:
                    VarC = list(Changes)
                    for V in VarC:
                        a = len(self.Changes[Var][V])
                        self.Changes[Var][V] = np.hstack((self.Changes[Var][V],Changes[V]))
                    self.SeriesC['DatesP'] = np.vstack((self.SeriesC['DatesP'],Dates))
                    self.SeriesC['Dates'] = np.vstack((self.SeriesC['Dates'],self.Series['Dates'].str[-int(Extract[0]/dt)+M:int(Extract[1]/dt)+M]))
                    self.SeriesC[Var+'C'] = np.vstack((self.SeriesC[Var+'C'],Event))
                except KeyError:
                    self.Changes[Var] = Changes.copy()
                    self.SeriesC['DatesP'] = self.Series['Dates'].datetime[-int(Extract[0]/dt)+M:int(Extract[1]/dt)+M]
                    self.SeriesC['Dates'] = self.Series['Dates'].str[-int(Extract[0]/dt)+M:int(Extract[1]/dt)+M]
                    self.SeriesC[Var+'C'] = Event.copy()
                self.MT.append(M)
                self.M[Var].append(MP)
            else:
                return -1
        return 1

    def GraphValidation(self,Prec,Name='',NameArch='',PathImg='',
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
        # Constants
        self.PathImg = PathImg
        dt = self.dt
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


        self.EvSi = 0
        self.EvNo = 0
        EvTot = self.SeriesC
        for iM,M in enumerate(self.MT):
            if iM == 0:
                EvTot['PrecC'] = Prec[-int(self.Extract[0]/dt)+M:M+int(self.Extract[1]/dt)]
            else:
                EvTot['PrecC'] = np.vstack((EvTot['PrecC'],Prec[-int(self.Extract[0]/dt)+M:M+int(self.Extract[1]/dt)]))

            if np.nanmax(Prec[M:M+int(2*60/dt)]) > 0:
                self.EvSi += 1
            else:
                self.EvNo += 1



        if flagAver:
            # Se grafican los eventos en promedio
            dt = int(self.dt)
            Data = dict()
            Data['PrecC'] = np.nanmean(EvTot['PrecC'],axis=0)
            for iLab,Lab in enumerate(Labels):
                if flags[Lab]:
                    Data[Lab] = np.nanmean(EvTot[Lab],axis=0)
            BP.EventsSeriesGen(EvTot['DatesP'][0],Data,self.PrecCount,
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
                    Data['PrecC'] = EvTot['PrecC'][iEv]
                    for iLab,Lab in enumerate(Labels):
                        if flags[Lab]:
                            Data[Lab] = EvTot[Lab][iEv]
                    if not(DataV is None):
                        DataVer = dict()
                        for iLab,Lab in enumerate(DataKeyV):
                            DataVer[Lab] = DataV[Lab][iEv]
                    else:
                        DataVer = None
                    BP.EventsSeriesGen(EvTot['DatesP'][iEv],Data,DataVer,
                            DataKeyV=DataKeyV,DataKey=DataKeys,
                            PathImg=self.PathImg+ImgFolder+'Series/'+EvType+'/'+Vars+'/',
                            Name=Name,NameArch=NameArch,
                            GraphInfo={'ylabel':Units,'color':Color,
                                'label':Label},
                            GraphInfoV=GraphInfoV,flagV=False,
                            flagBig=flagBig,vm={'vmax':[None],'vmin':[0]},Ev=iEv,
                            Date=EvTot['Dates'][iEv][0])
        return 









