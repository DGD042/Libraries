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
from Utilities import DatesUtil as DUtil;DUtil=DUtil()
from Utilities import Data_Man as DM
from AnET import CorrSt as cr;cr=cr()
from AnET import CFitting as CF; CF=CF()
from Hydro_Analysis import Hydro_Plotter as HyPl;HyPl=HyPl()
from Hydro_Analysis import Hydro_Analysis as HA; 
from Hydro_Analysis.Models.Atmos_Thermo import Thermo_Fun as TA
from AnET import AnET as anet;anet=anet()
from EMSD import EMSD;EMSD=EMSD()
from EMSD.Data_Man import Data_Man as DMan
from BPumpL.BPumpL import BPumpL as BP;BP=BP()
from BPumpL.Data_Man import Load_Data as LD

class Proc(object): 
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
        # ----------------------------
        # Constantes
        # ----------------------------
        # Información para gráficos
        LabelV = ['Precipitación','Temperatura','Humedad Relativa','Presión','Humedad Especifica']
        LabelVU = ['Precipitación [mm]','Temperatura [°C]','Hum. Rel. [%]','Presión [hPa]','Hum. Espec. [kg/kg]']
        self.DataBase = DataBase
        
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
            self.VariablesComp[v+'_Pres'] = v[:-1]+'Comp_Pres'
            self.VariablesComp[v+'_Temp'] = v[:-1]+'Comp_Temp'
            self.LabelV[v] = LabelV[iv]
            self.LabelVU[v] = LabelVU[iv]
        # Información para gráficos
        LabelV = ['Precipitación','Temperatura','Humedad Relativa','Presión','Humedad Especifica']
        LabelVU = ['Precipitación [mm]','Temperatura [°C]','Hum. Rel. [%]','Presión [hPa]','Hum. Espec. [kg/kg]']
        self.DatesDoc = ['FechaEv','FechaEv_Pres','FechaEv_Temp','FechaC']
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
        StInfo,Arch = LD.LoadStationsInfo(endingmatR=endingmat)
        self.StInfo = StInfo

        self.Arch = Arch[DataBase]['Original']
        self.ArchT = Arch
        self.ArchD = Arch[DataBase]['Diarios']
        self.Paths = Arch[DataBase]['Paths']
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
        # print('\nSe carga la estación: ',self.Names[irow])

        # Se carga el archivo.
        self.f = sio.loadmat(self.Arch[irow])

        # Se verifica la existencia de todas las variables
        try: 
            self.f['PrecC'] = self.f['PrecC'][0]
            self.flag['PrecC'] = True
        except KeyError:
            self.flag['PrecC'] = False
        try: 
            self.f['PresC'] = self.f['PresC'][0]
            self.flag['PresC'] = True
        except KeyError:
            try:
                self.f['PresC'] = self.f['PresBC'][0]
                self.flag['PresC'] = True
            except KeyError:
                self.flag['PresC'] = False
        try: 
            self.f['TC'] = self.f['TC'][0]
            self.flag['TC'] = True
        except KeyError:
            try: 
                self.f['TC'] = self.f['TempC'][0]
                self.flag['TC'] = True
            except KeyError:
                self.flag['TC'] = False
        try:
            self.f['HRC'] = self.f['HRC'][0]
            self.flag['HRC'] = True
        except KeyError:
            self.flag['HRC'] = False
        try: 
            self.f['FechaC']
            self.flag['FechaC'] = True
            # last_time = time.time()
            self.f['FechaCP'] = DUtil.Dates_str2datetime(self.f['FechaC'],Date_Format='%Y/%m/%d-%H%M',flagQuick=True)
            if isinstance(self.f['FechaCP'],int):
                self.f['FechaCP'] = DUtil.Dates_str2datetime(self.f['FechaC'],Date_Format='%Y-%m-%d-%H%M',flagQuick=True)
            if self.DataBase == 'Amazonas':
                self.f['FechaCP'] = self.f['FechaCP']-timedelta(0,4*60*60)
                self.f['FechaC'] = DUtil.Dates_datetime2str(self.f['FechaCP'])
            # print('Loop took {} seconds'.format(time.time()-last_time))
            # Delta de tiempo
            self.dtm = str(int(self.f['FechaC'][1][-2:]))
            if self.dtm == '0':
                self.dtm = str(int(self.f['FechaC'][1][--2:-2]))
                if self.dtm == '0':
                    print('Revisar el delta de tiempo')
        except KeyError:
            self.flag['FechaC'] = False

        self.f['qC'] = TA.qeq(self.f['PresC'],self.f['HRC'],self.f['TC'])*1000
        self.flag['qC'] = True

        # except KeyError:
        #     self.flag['qC'] = False
        
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
        x = 0
        for v in self.Variables:
            if not(self.flag[v]):
                if x == 0:
                    print('No se tiene las siguientes variables')
                print('-'+v)
                x += 1
        return
    
    def InterData(self):
        '''
        DESCRIPTION:

            Función para interpolar los datos NaN que se puedan.
        '''
        print(' Se interpolan los datos NaN')
        # Se interpolan los datos que se pueden interpolar
        for iv,v in enumerate(self.Variables):
            if self.flag[v]:
                self.f[self.Variables2[v]] = BP.NaNEl(self.f[v])

        self.var = list(self.f)
        self.flag['Variables2'] = True
        return
    
    def MeanRemove(self):
        '''
        DESCRIPTION:

            Función para remover la media total de los datos.
        '''
        print(' Se remueven los datos promedios')
        Var = self.Variables
        
        # Se interpolan los datos que se pueden interpolar
        for iv,v in enumerate(Var):
            if self.flag[self.Variables[iv]]:
                if v == 'PrecC':
                    if self.flag['Variables2']:
                        self.f[self.Variables2[v]] = self.f[self.Variables2[v]]
                    else:
                        self.f[v] = self.f[v]
                else:
                    if self.flag['Variables2']:
                        self.f[self.Variables2[v]] = self.f[self.Variables2[v]]-np.nanmean(self.f[self.Variables2[v]])
                    else:
                        self.f[v] = self.f[v]-np.nanmean(self.f[v])

        self.var = list(self.f)
        return
    
    def ButterFilt(self,lowcut,highcut,order=2,flagG=False,xTi=0,xTf=24*60/5,PathImg=''):
        '''
        DESCRIPTION:

            Función para aplicar el fitro Butterworth.
        '''
        print(' Se realiza el filtro Butterworth')
        Var = self.Variables
            
        for iv,v in enumerate(Var):
            if self.flag[v]:
                if self.flag['Variables2']:
                    qn = ~np.isnan(self.f[self.Variables2[v]])
                    self.f[self.Variables3[v]] = self.f[self.Variables2[v]][qn]
                    if v != 'PrecC':
                        fs = len(self.f[self.Variables3[v]])
                        b,a = anet.ButterworthFiler([lowcut,highcut],order,fs,btype='bandpass',flagG=flagG,worN=fs,PathImg=PathImg,Name='Filt_Function')
                        self.f[self.Variables4[v]] = anet.Filt_ButterworthApp(self.f[self.Variables3[v]],[lowcut,highcut],order,fs,btype='bandpass')
                        # b,a = anet.ButterworthFiler(highcut,order,fs,btype='lowpass',flagG=flagG,worN=fs,PathImg=PathImg,Name='Filt_Function')
                        # self.f[self.Variables4[v]] = anet.Filt_ButterworthApp(self.f[self.Variables3[v]],highcut,order,fs,btype='lowpass')
                        self.f[self.VariablesF[v]] = np.empty(len(self.f[self.Variables2[v]]))*np.nan
                        self.f[self.VariablesF[v]][qn] = self.f[self.Variables4[v]]
                        if flagG:
                            Fol = '(%0.f_%.0f) Hours/' %(highcut*int(self.dtm)/60,lowcut*int(self.dtm)/60)
                            HyPl.FilComp(self.f['FechaCP'],self.f[self.Variables2[v]],self.f[self.VariablesF[v]],xTi,xTf,PathImg=PathImg+'Comp/'+Fol,Name=self.Names[self.irow],VarU=self.LabelVU[v],Var=v[:-1],Filt='')
        self.flag['VariablesF'] = True
        self.var = list(self.f)
        return

    def AnomaliasSeries(self):
        '''
        DESCRIPTION:

            Función para calcular las anomalías mensuales.
        '''
        print(' Se obtienen las anomalias de los datos')
        # Deltas de tiempo
        h12 = int(12*(60/int(self.dtm)))
        h24 = int(24*(60/int(self.dtm)))
        
        if self.flag['VariablesF']:
            Var2 = self.VariablesF
        elif self.flag['Variables2']:
            Var2 = self.Variables2
        else:
            Var2 = self.Variables_N
        Var = self.Variables
        for iv,v in enumerate(Var):
            if self.flag[v]:
                if v != 'PrecC':
                    # Se promedia cada 12 horas
                    self.f[Var2[v]] = anet.AnomGen(self.f[Var2[v]],h12)
                    # Se promedia cada 24 horas
                    self.f[Var2[v]] = anet.AnomGen(self.f[Var2[v]],h24)
        return

    def EventSeparation(self,Ci=60,Cf=60,m=0.2,M=100,mP=-1,MP=-0.5,mT=0.5,MT=1):
        '''
        DESCRIPTION:

            Función para separar los eventos.
        '''
        print(' Se generan los diferentes diagramas de compuestos')
        if self.flag['VariablesF']:
            Var2 = self.VariablesF
            Var2['PrecC'] = 'PrecC'
        elif self.flag['Variables2']:
            Var2 = self.Variables2
        else:
            Var2 = self.Variables_N

        Var = self.Variables
        Variable = dict()
        VariablePres = dict()
        VariableTemp = dict()
        for iv,v in enumerate(Var[1:]):
            if self.flag[v]:
                PrecC, Variable[v], FechaEv = BP.ExEv(self.f[Var2['PrecC']],self.f[Var2[v]],self.f['FechaC'],Ci=Ci,Cf=Cf,m=m,M=M,dt=int(self.dtm))
        print(PrecC.shape)

        for iv,v in enumerate(Var):
            if v != 'PresC':
                if self.flag[v]:
                    PresC, VariablePres[v], FechaEvPres = BP.ExEvGen(self.f[Var2['PresC']],self.f[Var2[v]],self.f['FechaC'],Ci=Ci,Cf=Cf,m=mP,M=MP,MaxMin='min')

        for iv,v in enumerate(Var):
            if v != 'TC':
                if self.flag[v]:
                    TC, VariableTemp[v], FechaEvTemp = BP.ExEvGen(self.f[Var2['TC']],self.f[Var2[v]],self.f['FechaC'],Ci=Ci,Cf=Cf,m=mT,M=MT,MaxMin='max')
        xx = 0
        FechaEv2=[]
        # Se realiza una limpeza de los datos adicional
        for i in range(len(PrecC)):
            q = ~np.isnan(PrecC[i])
            N = np.sum(q)
            q2 = ~np.isnan(Variable['PresC'][i])
            N2 = np.sum(q2)
            q3 = ~np.isnan(Variable['TC'][i])
            N3 = np.sum(q3)
            a = len(q)*0.70
            if N >= a and N2 >= a and N3 >= a:
                if xx == 0:
                    PrecC2 = PrecC[i]
                    PresC2 = Variable['PresC'][i]
                    TC2 = Variable['TC'][i]
                    HRC2 = Variable['HRC'][i]
                    qC2 = Variable['qC'][i]
                    xx += 1
                else:
                    PrecC2 = np.vstack((PrecC2,PrecC[i]))
                    PresC2 = np.vstack((PresC2,Variable['PresC'][i]))
                    TC2 = np.vstack((TC2,Variable['TC'][i]))
                    HRC2 = np.vstack((HRC2,Variable['HRC'][i]))
                    qC2 = np.vstack((qC2,Variable['qC'][i]))
                FechaEv2.append(FechaEv[i])
                xx += 1
        self.f[self.VariablesComp['PrecC']] = PrecC2
        self.f[self.VariablesComp['PresC']] = PresC2
        self.f[self.VariablesComp['TC']] = TC2
        self.f[self.VariablesComp['HRC']] = HRC2
        self.f[self.VariablesComp['qC']] = qC2
        self.f['FechaEv'] = FechaEv2

        xx = 0
        FechaEv2=[]
        # Se realiza una limpeza de los datos adicional de presión
        for i in range(len(PresC)):
            q = ~np.isnan(PresC[i])
            N = np.sum(q)
            q2 = ~np.isnan(VariablePres['PrecC'][i])
            N2 = np.sum(q2)
            q3 = ~np.isnan(VariablePres['TC'][i])
            N3 = np.sum(q3)
            a = len(q)*0.70
            if N >= a and N2 >= a and N3 >= a:
                if xx == 0:
                    PresC2 = PresC[i]
                    PrecC2 = VariablePres['PrecC'][i]
                    TC2 = VariablePres['TC'][i]
                    HRC2 = VariablePres['HRC'][i]
                    qC2 = VariablePres['qC'][i]
                    xx += 1
                else:
                    PresC2 = np.vstack((PresC2,PresC[i]))
                    PrecC2 = np.vstack((PrecC2,VariablePres['PrecC'][i]))
                    TC2 = np.vstack((TC2,VariablePres['TC'][i]))
                    HRC2 = np.vstack((HRC2,VariablePres['HRC'][i]))
                    qC2 = np.vstack((qC2,VariablePres['qC'][i]))
                FechaEv2.append(FechaEvPres[i])
                xx += 1
        self.f[self.VariablesComp['PrecC_Pres']] = PrecC2
        self.f[self.VariablesComp['PresC_Pres']] = PresC2
        self.f[self.VariablesComp['TC_Pres']] = TC2
        self.f[self.VariablesComp['HRC_Pres']] = HRC2
        self.f[self.VariablesComp['qC_Pres']] = qC2
        self.f['FechaEv_Pres'] = FechaEv2


        xx = 0
        FechaEv2=[]
        # Se realiza una limpeza de los datos adicional de presión
        for i in range(len(TC)):
            q = ~np.isnan(TC[i])
            N = np.sum(q)
            q2 = ~np.isnan(VariableTemp['PresC'][i])
            N2 = np.sum(q2)
            q3 = ~np.isnan(VariableTemp['PrecC'][i])
            N3 = np.sum(q3)
            a = len(q)*0.70
            if N >= a and N2 >= a and N3 >= a:
                if xx == 0:
                    TC2 = TC[i]
                    PrecC2 = VariableTemp['PrecC'][i]
                    PresC2 = VariableTemp['PresC'][i]
                    HRC2 = VariableTemp['HRC'][i]
                    qC2 = VariableTemp['qC'][i]
                    xx += 1
                else:
                    TC2 = np.vstack((TC2 ,TC[i]))
                    PrecC2 = np.vstack((PrecC2,VariableTemp['PrecC'][i]))
                    PresC2 = np.vstack((PresC2,VariableTemp['PresC'][i]))
                    HRC2 = np.vstack((HRC2,VariableTemp['HRC'][i]))
                    qC2 = np.vstack((qC2,VariableTemp['qC'][i]))
                FechaEv2.append(FechaEvTemp[i])
                xx += 1
        self.f[self.VariablesComp['PrecC_Temp']] = PrecC2
        self.f[self.VariablesComp['PresC_Temp']] = PresC2
        self.f[self.VariablesComp['TC_Temp']] = TC2
        self.f[self.VariablesComp['HRC_Temp']] = HRC2
        self.f[self.VariablesComp['qC_Temp']] = qC2
        self.f['FechaEv_Temp'] = FechaEv2
        self.flag['VariablesComp'] = True

        return

    def SaveInf(self,pathout='',endingmat=''):
        '''
        DESCRIPTION:

            Función para guardar la información.
        '''
        
        print(' Se guarda la información')
        if self.flag['VariablesF']:
            Var2 = self.VariablesF
        elif self.flag['Variables2']:
            Var2 = self.Variables2
        else:
            Var2 = self.Variables_N
        # Diagrama de compuestos
        if not(self.flag['VariablesComp']):
            print(' No se ha realizado los diagramas de compuestos, ¡Revisar!')
            return
        else:
            VarComp =  self.VariablesComp
        
        Var = self.Variables
        Dates = dict()
        Dates['FechaC'] = self.f['FechaC']
        Dates['FechaEv'] = self.f['FechaEv']
        Dates['FechaEv_Pres'] = self.f['FechaEv_Pres']
        Dates['FechaEv_Temp'] = self.f['FechaEv_Temp']
        savingkeys = ['FechaEv','FechaEv_Pres','FechaEv_Temp','FechaC']
        Data = dict()
        # Datos completos
        for iv,v in enumerate(Var):
            if self.flag[v]:
                if v == 'PrecC':
                    savingkeys.append(v[:-1])
                    Data[v[:-1]] = self.f[Var2[v]]
                else:
                    savingkeys.append(v[:-1]+'_F')
                    Data[v[:-1]+'_F'] = self.f[Var2[v]]
        # Datos Compuestos
        for iv,v in enumerate(Var):
            if self.flag[v]:
                savingkeys.append(v[:-1]+'C')
                Data[v[:-1]+'C'] = self.f[VarComp[v]]
                savingkeys.append(v[:-1]+'C_Pres')
                Data[v[:-1]+'C_Pres'] = self.f[VarComp[v+'_Pres']]
                savingkeys.append(v[:-1]+'C_Temp')
                Data[v[:-1]+'C_Temp'] = self.f[VarComp[v+'_Temp']]
        EMSD.Writemat(Dates,Data,savingkeys,datekeys=self.DatesDoc,datakeys=savingkeys[4:],pathout=pathout,Names=self.NamesArch[self.irow]+endingmat)
        return
    
    def GraphEvents(self):
        '''
        DESCRIPTION:

            Gráfica de los eventos.
        '''
        PathImg = 'Tesis_MscR/02_Docs/01_Tesis_Doc/Kap2/Img/'+self.DataBase+'/Series/'
        Name = self.NamesArch[self.irow]+'_Series'
        Var = ['Precipitación','Temperatura','Humedad Relativa','Presión']

        BP.GraphIDEA(self.f['FechaCP'],self.f['PrecC'],
                self.f['TC'],self.f['HRC'],self.f['PresC'],Var,PathImg,0,V=0,Name=Name)
        return

    def Cycle_An(self):
        '''
        DESCRIPTION:

            Gráfica los diferentes ciclos climatológicos y meteorológicos.
        '''
        PathImg = 'Tesis_MscR/02_Docs/01_Tesis_Doc/Kap2/Img/'+self.DataBase+'/Hidro_A/'
        Name = self.NamesArch[self.irow]
        Var = ['Precipitación','Temperatura','Humedad Relativa','Presión']
        Variables = ['Prec','T','HR','Pres']

        # Se carga la información Horaria
        DataH = LD.LoadData(self.ArchT,self.DataBase,irow=self.irow,TimeScale='Horarios')
        self.DataH = DataH
        if self.DataBase == 'Manizales':
            FechaC = DataH['FechaEs']
        elif self.DataBase == 'Medellin':
            FechaC = DataH['FechasC']
        elif self.DataBase == 'Amazonas' or self.DataBase == 'Wunder':
            if self.irow == 1 and self.DataBase == 'Amazonas':
                FechaC = DataH['FechasC']
            else:
                FechaC = DataH['FechaC']

        if self.DataBase == 'Amazonas':
            FechaCP = DUtil.Dates_str2datetime(FechaC,Date_Format='%Y/%m/%d-%H%M',flagQuick=True)
            FechaCP = FechaCP - timedelta(0,4*60*60)
            for Lab in Variables:
                VC = DMan.CompD(FechaCP,DataH[Lab+'H'],dtm=timedelta(0,60*60))
                DataH[Lab+'H'] = VC['VC']
                if Lab == 'PrecC':
                    FechaC = VC['DatesC']
        # Se carga la información Diaria
        DataD = LD.LoadData(self.ArchT,self.DataBase,irow=self.irow,TimeScale='Diarios')
        self.DataD = DataD
        if self.DataBase == 'Manizales':
            FechaCD = DataD['FechaEs']
            FechaCD = np.array([i[:4]+'/'+i[5:] for i in FechaCD])
        elif self.DataBase == 'Medellin' or self.DataBase == 'Amazonas':
            FechaCD = DataD['FechaEs']
        elif self.DataBase == 'Wunder':
            FechaCD = DataD['FechaM']
        self.FechaCD = FechaCD

        # Se carga la información
        CH = dict()
        CM = dict()
        for Lab in Variables:
            # Horario
            try:
                HASR = HA(DateH=FechaC,VarH=DataH[Lab+'H'])
            except KeyError:
                HASR = HA(DateH=FechaC,VarH=DataH[Lab+'CH'])
            CH[Lab] = HASR.CiclD(FlagG=False)

            # Diario
            try:
                HASR = HA(DateM=FechaCD,VarM=DataD[Lab+'M'])
            except KeyError:
                HASR = HA(DateM=FechaCD,VarM=DataD[Lab+'CM'])
            CM[Lab] = HASR.CiclA(FlagG=False)

        self.CH = CH
        self.CM = CM

        # Se grafica el ciclo diurno
        BP.Graph_CiclD(CH['Prec'],CH['T'],CH['HR'],CH['Pres'],Var,PathImg,Name=Name)
        BP.Graph_CiclA(CM['Prec'],CM['T'],CM['HR'],CM['Pres'],Var,PathImg,Name=Name)
        return
