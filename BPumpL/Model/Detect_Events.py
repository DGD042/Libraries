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
import BPumpL as BPL

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

    def __init__(self,ID,DateA,Pres,T,HR,Extract=[5*60,5*60],Limits={'Pres':{'VRateB':0.5,'MV':-0.2},
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
        self.IDA = ID
        self.Limits = Limits
        self.Extract = Extract

        # Created Attributes
        self.SeriesC = dict() # Extracted Data
        self.MT = [] # Location from the series
        self.M = dict() # Location from 
        self.Changes = dict()
        for V in Var:
            self.M[V] = []
            self.Changes[V] = dict()
        self.dt = int((self.Series['Dates'].datetime[1] - self.Series['Dates'].datetime[0]).seconds/60)
        self.Erase = 2*60/self.dt
        self.PrecT = dict()
        LabelVal = ['DurPrec','TotalPrec','IntPrec']
        LabelValV = ['Max','Min','Mean','Sum']
        Oper = {'Max':np.nanmax,'Min':np.nanmin,'Mean':np.nanmean,'Sum':np.nansum}
        Values = dict()
        # Iniziaticed Values
        for L1 in LabelValV:
            self.PrecT[L1] = dict()
            for L2 in LabelVal:
                Values[L2] = []
                self.PrecT[L1][L2] = []

        # Indicativos para la extracción de eventos
        self.x = 0

        self.EvSiT = 0
        self.EvT = 0


        self.Oper = {'max':np.nanmax,'min':np.nanmin}
        return

    def __FindValue(self,Var='Pres',Oper='min',Comp='<',M=None,limits=[-5,5]):
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

    def __ValidateEvent(self):
        '''
        DESCRIPTION:
            Method that finds an event in the series
        _________________________________________________________________________
        
        INPUT:
        _________________________________________________________________________
        
        OUTPUT:
        '''
        x = self.x

        IndPres = self.__FindValue(Var='Pres',Oper='min',Comp='<',
            M=None,limits=[-5,5])
        # Validación
        if IndPres == -2:
            return -2
        if IndPres == -1:
            return -1
        IndTemp = self.__FindValue(Var='T',Oper='max',Comp='>',
            M=self.MT[-1],limits=[-2*60,2*60])
        # Validación
        if IndTemp == -2 or IndTemp == -1:
            if x == 0:
                del(self.SeriesC['PresC'])
                del(self.SeriesC['DatesP'])
                del(self.SeriesC['Dates'])
                self.MT.pop()
                del(self.Changes['Pres'])
            else:
                self.SeriesC['PresC'] = np.delete(self.SeriesC['PresC'],x,0)
                self.SeriesC['DatesP'] = np.delete(self.SeriesC['DatesP'],x,0)
                self.SeriesC['Dates'] = np.delete(self.SeriesC['Dates'],x,0)
                self.MT.pop()
                VarC = list(self.Changes['Pres'])
                for V in VarC:
                    self.Changes['Pres'][V] = np.delete(self.Changes['Pres'][V],x,0)
            if IndTemp == -1:
                return -1
            elif IndTemp == -2:
                return -2

        IndHR = self.__FindValue(Var='HR',Oper='min',Comp='<',
            M=self.MT[-1],limits=[-2*60,2*60])
        # Validación
        if IndHR == -2 or IndHR == -1:
            if x == 0:
                del(self.SeriesC['PresC'])
                del(self.SeriesC['TC'])
                del(self.SeriesC['DatesP'])
                del(self.SeriesC['Dates'])
                self.MT.pop()
                del(self.Changes['Pres'])
                del(self.Changes['T'])
            else:
                self.SeriesC['PresC'] = np.delete(self.SeriesC['PresC'],x,0)
                self.SeriesC['TC'] = np.delete(self.SeriesC['TC'],x,0)
                self.SeriesC['DatesP'] = np.delete(self.SeriesC['DatesP'],x,0)
                self.SeriesC['Dates'] = np.delete(self.SeriesC['Dates'],x,0)
                self.MT.pop()
                VarC = list(self.Changes['Pres'])
                for V in VarC:
                    self.Changes['Pres'][V] = np.delete(self.Changes['Pres'][V],x,0)
                    self.Changes['T'][V] = np.delete(self.Changes['T'][V],x,0)
            if IndHR == -1:
                return -1
            elif IndHR == -2:
                return -2 
        if IndPres == 1 and IndTemp == 1 and IndHR == 1:
            x += 1

        self.x = x
        return 1

    def DetEvent(self,EvNum=None):
        '''
        DESCRIPTION:
            Method that finds several events in the series.
        _________________________________________________________________________
        
        INPUT:
            :param EvNum: An int, Number of events that wants to be extracted.
        _________________________________________________________________________
        
        OUTPUT:
        '''
        if EvNum is None:
            x = 1000000
        else:
            x = EvNum
        while self.x != x:
            Ind = self.__ValidateEvent()
            if Ind == -1:
                break
        return

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

    def LoadStations(self,DataBase='Medellin',endingmat='',Ev=0):
        '''
        DESCRIPTION:
            Method that load station information for comparation, need to run
            DtEvents first.
        _________________________________________________________________________
        
        INPUT:
            :param DataBase: A str with the data for comparation.
            :param Ev: An Int, Number of the event that wants to be extracted.
        _________________________________________________________________________
        
        OUTPUT:
        '''
        # ----------------
        # Load Information
        # ----------------
        self.SC = BPL.Scatter_Gen(DataBase=DataBase,
                endingmat=endingmat,PathImg='')
        # ---------------------
        # Station information
        # ---------------------
        Labels = ['ID','Name','Latitud','Longitud']
        self.ID = self.SC.ID
        self.St_Info = {}
        for iSt,St in enumerate(self.ID):
            self.St_Info[St] = {}
            for Lab in Labels:
                self.St_Info[St][Lab] = self.SC.StInfo[DataBase][Lab][iSt]
            self.St_Info[St]['CodesNames'] = self.SC.StInfo[DataBase]['ID'][iSt]+ ' ' + self.SC.StInfo[DataBase]['Name'][iSt]
        self.Ev=Ev

        # ----------------
        # Constants
        # ----------------
        # Dates
        DateI = self.SeriesC['DatesP'][Ev][0]
        DateE = self.SeriesC['DatesP'][Ev][-1]
        lenArch = len(self.SC.Arch)
        lenData = int((DateE-DateI).seconds/60)
        self.DataSt = {}
        Labels = ['FechaC','FechaCP','Prec','Pres_F','T_F','HR_F','W_F','q_F']
        self.vmax = {}
        self.vmin = {}
        for Lab in Labels[2:]:
            self.vmax[Lab] = []
            self.vmin[Lab] = []

        # ----------------
        # Extract Data
        # ----------------
        for iar in range(lenArch):
            self.DataSt[self.ID[iar]] = {}
            self.SC.LoadData(irow=iar)
            xi = np.where(self.SC.f['FechaCP'] == DateI)[0]
            xf = np.where(self.SC.f['FechaCP'] == DateE)[0]
            if len(xi) == 0 or len(xf) == 0:
                for Lab in Labels:
                    self.DataSt[self.ID[iar]][Lab] = np.array(lenData)
                continue
            for iLab,Lab in enumerate(Labels):
                self.DataSt[self.ID[iar]][Lab] = self.SC.f[Lab][xi:xf+1]
                if iLab > 1:
                    self.vmax[Lab].append(np.nanmax(self.SC.f[Lab][xi:xf+1]))
                    self.vmin[Lab].append(np.nanmin(self.SC.f[Lab][xi:xf+1]))

        self.vmax2 = self.vmax.copy()
        self.vmin2 = self.vmin.copy()
        for Lab in Labels[2:]:
            self.vmax[Lab] = np.nanmax(self.vmax[Lab])+0.1
            if Lab == 'Prec':
                self.vmin[Lab] = np.nanmin(self.vmin[Lab])
            else:
                self.vmin[Lab] = np.nanmin(self.vmin[Lab])-0.1


        return

    def GraphStSeries(self,Var,NameArch='',PathImg=''):
        '''
        DESCRIPTION:
            Method that load station information for comparation, need to run
            LoadStations first.
        _________________________________________________________________________
        
        INPUT:
            :param Var: A str, Variable to be compared.
            :param Name: A str, Name of the graph.
            :param NameArch: A str, Name of the file.
            :param PathImg: A str, Path to save the file.

        _________________________________________________________________________
        
        OUTPUT:
        '''
        VarL = {'Pres_F':'Presión'}
        VarLL = {'Pres_F':'Presión [hPa]'}


        LabelVal = ['DurPrec','TotalPrec','IntPrec']
        LabelValV = ['Max','Min','Mean','Sum']
        Oper = {'Max':np.nanmax,'Min':np.nanmin,'Mean':np.nanmean,'Sum':np.nansum}
        Values = dict()
        # Iniziaticed Values
        for L1 in LabelVal:
            Values[L1] = []

        # Folder Creation
        fH= 20
        fV = fH * (2/3)

        fig = plt.figure(figsize=DM.cm2inch(fH,fV))
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': 'Arial'\
            ,'xtick.labelsize': 15,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': 15,'ytick.major.size': 12,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})

        x = 0
        xx = 0
        for iID,ID in enumerate(self.ID):
            if xx == 0:
                self.EvT += 1
                xx = 1
            PP = int(len(self.DataSt[ID]['Prec'])/2)
            if np.nanmax(self.DataSt[ID]['Prec'][PP:PP+int(2*60/self.dt)]) > 0.01:
                if x == 0:
                    self.EvSiT += 1
                    x = 1
                # Precipitation event parameters
                PMax = np.nanmax(self.DataSt[ID]['Prec'][PP:PP+int(2*60/self.dt)])
                M = np.where(self.DataSt[ID]['Prec'][PP:PP+int(2*60/self.dt)] == PMax)[0][0]+PP
                self.PrecCount = HyMF.PrecCount(self.DataSt[ID]['Prec'],self.DataSt[ID]['FechaC'],dt=self.dt,M=M)

                # Label of the parameters
                Dur = round(self.PrecCount['DurPrec'],3)
                TotPrec = round(self.PrecCount['TotalPrec'],3)
                if np.isnan(Dur):
                    LabelT = self.St_Info[ID]['CodesNames'] + ' (No hay datos suficientes)'
                else:
                    Datest = self.PrecCount['DatesEvst'][0].strftime('%H:%M')
                    LabelT = self.St_Info[ID]['CodesNames'] + ' ({}, {} h, {} mm)'.format(Datest,Dur,TotPrec)
                    for L1 in LabelVal:
                        Values[L1].append(self.PrecCount[L1])

                plt.plot(self.DataSt[ID]['FechaCP'],self.DataSt[ID][Var],'--',label=LabelT)
            else:
                plt.plot(self.DataSt[ID]['FechaCP'],self.DataSt[ID][Var],'-',label=self.St_Info[ID]['CodesNames'])

    

        ax = plt.gca()
        yTL = ax.yaxis.get_ticklocs() 
        Ver = self.DataSt[ID]['FechaCP'][int(len(self.DataSt[ID]['FechaCP'])/2)]
        plt.plot([Ver,Ver],[yTL[0],yTL[-1]],'k--')
        plt.ylabel(VarLL[Var])
        for tick in plt.gca().get_xticklabels():
            tick.set_rotation(45)
        # plt.title(VarL[Var])
        lgd = plt.legend(bbox_to_anchor=(-0.15, 1.02, 1.15, .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize=8.0)
        # plt.legend(loc=1)
        plt.grid()

        yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # Minor tick value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().yaxis.set_minor_locator(minorLocatory)

        utl.CrFolder(PathImg+'/')
        plt.savefig(PathImg+'/'+Var[:-2]+'_'+NameArch+'_'+str(self.Ev)+'.png' ,
                format='png',dpi=200,bbox_extra_artists=(lgd,),bbox_inches='tight')
        plt.close('all')

        if len(Values[L1])> 0:
            for L1 in LabelValV:
                for L2 in LabelVal:
                    self.PrecT[L1][L2].append(Oper[L1](np.array(Values[L2])))

        return













