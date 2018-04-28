# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 04/10/2017
#______________________________________________________________________________
#______________________________________________________________________________

# ------------------------
# Importing Modules
# ------------------------ 
# Manipulate Data
import numpy as np
from datetime import date, datetime, timedelta
# Open and saving data
import csv
import xlrd # Para poder abrir archivos de Excel
import xlsxwriter as xlsxwl
# System
import sys
import os
import glob as gl
import re
import warnings

# ----------------------
# Personal Libraries
# ----------------------
from Utilities import Utilities as utl
from EMSD import EMSD
from EMSD.Data_Man import Data_Man as DMan
from Utilities import DatesUtil as DUtil; DUtil = DUtil()
from Hydro_Analysis import Hydro_Plotter as HyPl; HyPl = HyPl()

# --------------------------------------------------
# Class
# --------------------------------------------------

class Wunder(object):
    '''
    DESCRIPTION:
        This Class opens and agregates the information
        from Wunderground stations, the information should be
        located in a Folder. The name of the station should be
        the name of the folder and inside the folder the 
        information would be separated in different files with 
        dates.
    _____________________________________________________________
    INPUT:
        :param PathData: A str, Path to the station information.
        :param Stations: A list, List with stations.
        :param PathImg:  A str, Path to the images.
        :param verbose:  A boolean, flag to print information.

    '''
    def __init__(self,PathData,Stations=None):
        '''
        DESCRIPTION:
            Class Constructor.
        '''
        # -------------------------
        # Error Managment
        # -------------------------
        if not(isinstance(Stations,list)) and not(isinstance(Stations,str)) and Stations != None:
            r = utl.ShowError('OpenWundergrounds','__init__','Erroneus type for Station information')
            raise TypeError
        # -------------------------
        # Parameters
        # -------------------------
        self.PathData = PathData
       
        self.deli = ',' # Delimeter
        # Important Variables 
        # String Variables
        self.LabelsStr = np.array(['Time'])
        self.LabelsData = np.array(['TemperatureC','DewpointC','PressurehPa',
            'WindSpeedKMH','WindDirectionDegrees','HourlyPrecipMM',
            'SolarRadiationWatts/m^2','Humidity'])

        self.LabDataOper = {'TemperatureC':'mean','DewpointC':'mean',
                'PressurehPa':'mean','WindSpeedKMH':'mean',
                'WindDirectionDegrees':'mean','HourlyPrecipMM':'sum',
                'SolarRadiationWatts/m^2':'mean','Humidity':'mean'}

        self.LabDataOper1 = {'TemperatureC':'mean','DewpointC':'mean',
                'PressurehPa':'mean','WindSpeedKMH':'mean',
                'WindDirectionDegrees':'mean','HourlyPrecipMM':'sum',
                'SolarRadiationWatts/m^2':'mean','Humidity':'mean'}

        self.LabDataOper2 = {'TemperatureC':'mean','DewpointC':'mean',
                'PressurehPa':'mean','WindSpeedKMH':'mean',
                'WindDirectionDegrees':'mean','HourlyPrecipMM':'sum',
                'SolarRadiationWatts/m^2':'sum','Humidity':'mean'}

        self.LabDataOper3 = {'TemperatureC':'mean','DewpointC':'mean',
                'PressurehPa':'mean','WindSpeedKMH':'mean',
                'WindDirectionDegrees':'mean','HourlyPrecipMM':'sum',
                'SolarRadiationWatts/m^2':'mean','Humidity':'mean'}

        self.LabDataSave = {'TemperatureC':'TC','DewpointC':'Td',
                'PressurehPa':'PresC','WindSpeedKMH':'WSC',
                'WindDirectionDegrees':'WDC','HourlyPrecipMM':'PrecC',
                'SolarRadiationWatts/m^2':'RSC','Humidity':'HRC'}

        self.ElimOver = {'TC':None,'Td':None,
                'PresC':None,'WSC':None,
                'WDC':None,'PrecC':None,
                'RSC':None,'HRC':None}

        self.ElimLow = {'TC':None,'Td':None,
                'PresC':None,'WSC':None,
                'WDC':None,'PrecC':None,
                'RSC':None,'HRC':None}
        self.LabelsWithUnits = {'TC':'Temperatura [°C]','Td':'Punto de Rocio [°C]',
                        'PresC':'Presión [hPa]','WSC':'Vel. Viento [m/s]',
                        'WDC':'Dirección del Viento [Grados]','PrecC':'Precipitación [mm]',
                        'RSC':r'Radiación Solar [W/m$^2$]','HRC':'Hum. Rel. [%]'}

        self.LabelsNoUnits = {'TC':'Temperatura','Td':'Punto de Rocio',
                        'PresC':'Presión','WSC':'Vel. Viento',
                        'WDC':'Dirección del Viento','PrecC':'Precipitación',
                        'RSC':r'Radiación Solar','HRC':'Hum. Rel.'}

        self.LabelsColors = {'TC':'r','Td':'r',
                        'PresC':'k','WSC':'k',
                        'WDC':'k','PrecC':'b',
                        'RSC':'y','HRC':'g'}
        # -------------------------
        # Get Stations
        # -------------------------
        if Stations == None:
            Stations = utl.GetFolders(PathData)
        elif isinstance(Stations,str):
            Stations = [Stations]

        # Stations
        self.Stations = Stations
        self.Arch = dict()        
        for St in Stations:
            self.Arch[St] = gl.glob(PathData+St+'/*.txt')
            if len(self.Arch[St]) == 0:
                print('In Station',St,'No data was found, review station')

        return

    def LoadData(self,Station=None,flagComplete=True,dt=5):
        '''
        DESCRIPTION:
            This function loads the data of a station and compiles 
            it in a dictionary.
        ___________________________________________________________________
        INPUT:
            :param Station:      A str, List with the stations that would
                                    be extracted.
            :param flagComplete: A boolean, flag to determine if
                                            completes the data
        '''
        self.flagComplete = flagComplete
        self.Station = Station
        # -------------------------
        # Error Managment
        # -------------------------
        if not(isinstance(Station,list)) and Station != None and not(isinstance(Station,str)):
            Er = utl.ShowError('OpenWundergrounds','LoadData','Erroneus type for parameter Station')
            raise TypeError
        # -------------------------
        # Stations
        # -------------------------
        if Station == None:
            Station = self.Stations[0]
        elif isinstance(Station,list):
            Station = Station[0]
        # -------------------------
        # Parameters
        # -------------------------
        DataBase = {'DataBaseType':'txt','deli':self.deli,
                'colStr':None,'colData':None,'row_skip':1,'flagHeader':True,
                'rowH':0,'row_end':0,'str_NaN':'','num_NaN':None,
                'dtypeData':float} 
        LabelsH = ['H','maxH','minH']
        LabelsD = ['D','NF', 'NNF', 'maxD', 'M', 'minM', 'maxM', 'minD']

        self.LabelsH = LabelsH
        self.LabelsD = LabelsD

        # -------------------------
        # Verify Headers
        # -------------------------
        # Headers
        Headers = np.genfromtxt(self.Arch[Station][0],dtype=str,skip_header=0,delimiter=self.deli,max_rows=1)
        # Verify colStr data
        colStr = []
        LabStrNo = []
        LabStrYes = []
        for lab in self.LabelsStr:
            x = np.where(Headers == lab)[0]
            if len(x) > 0:
                colStr.append(x[0])
                LabStrYes.append(lab)
            else:
                LabStrNo.append(lab)
        DataBase['colStr'] = tuple(colStr)
        # Verify colData data
        colData = []
        LabDataNo = []
        LabDataYes = []
        for lab in self.LabelsData:
            x = np.where(Headers == lab)[0]
            if len(x) > 0:
                colData.append(x[0])
                LabDataYes.append(lab)
            else:
                LabDataNo.append(lab)
        self.LabStrYes = LabDataYes
        DataBase['colData'] = tuple(colData)

        # -------------------------
        # Extract information
        # -------------------------
        EM = EMSD()

        for iar,ar in enumerate(self.Arch[Station]):
            try:
                R = EM.Open_Data(ar,DataBase=DataBase)
            except ValueError:
                print('Error document data:',ar)
                continue
            if iar == 0:
                Data = R
            else:
                for iLab,Lab in enumerate(LabStrYes):
                    Data[Lab] = np.hstack((Data[Lab],R[Lab]))
                for iLab,Lab in enumerate(LabDataYes):
                    try:
                        Data[Lab] = np.hstack((Data[Lab],R[Lab]))
                    except KeyError:
                        Data[Lab] = np.hstack((Data[Lab],np.empty(R[LabDataYes[0]].shape)*np.nan))
                
        # for Lab in LabStrNo:
        #     R[Lab] = np.array(['nan' for i in len(R['Time'])])
        # for Lab in LabDataNo:
        #     R[Lab] = np.array([np.nan for i in len(R['Time'])])

        # DatesS = [i[:16] for i in Data['Time']]
        DatesS = Data['Time']
        Dates = DUtil.Dates_str2datetime(DatesS,Date_Format='%Y-%m-%d %H:%M:%S')
        Data.pop('Time',None)

        # -------------------------
        # Data Completion
        # -------------------------
        LabelsHmat = []
        LabelsDmat = []
        # Data in years
        DataC = dict()
        DataCC = dict()
        # Se llenan los datos
        DataH = dict() # Datos Horarios
        DataD = dict() # Datos Diarios
        self.DatesC = dict()
        self.DatesD = dict() # Fechas diarias
        for iLab,Lab in enumerate(LabDataYes):
            VC = DMan.CompD(Dates,Data[Lab],dtm=timedelta(0,60))
            # Precipitation corrected
            if Lab == 'HourlyPrecipMM':
                VC['VC'] = VC['VC']*5/60
            if iLab == 0:
                DatesC = VC['DatesC']
            DataC[self.LabDataSave[Lab]] = VC['VC']
            # Se pasa la información a cada 5 minutos
            DatesCC,DatesCN,DataCC[self.LabDataSave[Lab]] = DMan.Ca_E(DatesC,
                    DataC[self.LabDataSave[Lab]],dt=dt,escala=-1,
                    op=self.LabDataOper[Lab],flagNaN=False)
            # Data Eliminations
            if self.ElimOver[self.LabDataSave[Lab]] != None:
                DataCC[self.LabDataSave[Lab]][DataCC[self.LabDataSave[Lab]] > self.ElimOver[self.LabDataSave[Lab]]] = np.nan
            if self.ElimLow[self.LabDataSave[Lab]] != None:
                DataCC[self.LabDataSave[Lab]][DataCC[self.LabDataSave[Lab]] < self.ElimLow[self.LabDataSave[Lab]]] = np.nan

            # Se convierten los datos
            DatesC2, DatesNC2, VC2 = EM.Ca_EC(Date=DatesCC,V1=DataCC[self.LabDataSave[Lab]],
                    op=self.LabDataOper1[Lab],
                    key=None,dtm=dt,op2=self.LabDataOper2[Lab],op3=self.LabDataOper3[Lab])

            for LabH in LabelsH:
                DataH[self.LabDataSave[Lab]+LabH] = VC2[LabH]
                LabelsHmat.append(self.LabDataSave[Lab]+LabH)
            for LabD in LabelsD:
                DataD[self.LabDataSave[Lab]+LabD] = VC2[LabD]
                LabelsDmat.append(self.LabDataSave[Lab]+LabD)
            
        self.DataH = DataH
        self.DataD = DataD
        self.DatesH = DatesC2['DateH']
        self.DatesD['DatesD'] = DatesC2['DateD']
        self.DatesD['DatesM'] = DatesC2['DateM']
        self.DataCC = DataCC
        self.DatesCC = DatesCC
        self.DatesCN = DatesCN
        self.DatesNC2 = DatesNC2['DateMN']
        self.LabelsHmat = LabelsHmat
        self.LabelsDmat = LabelsDmat
        return 

    def AddElim(self,Elim):
        '''
        DESCRIPTION:
            This function loads the data of a station and compiles 
            it in a dictionary.
        _____________________________________________________________
        INPUT:
            :param Elim: A Dict, dictionary with the values that would
                                 be eliminated over and lower.
                                 Ex: Elim = {'ElimOver':{'RSC':3000},
                                 'ElimLow:{'TC':-1}}
        '''
        FlagKeyOver = True
        FlagKeyLow = True
        try:
            ElimLabOver = list(Elim['ElimOver'])
        except KeyError:
            FlagKeyOver = False
        except TypeError:
            FlagKeyOver = False

        try:
            ElimLabLow = list(Elim['ElimLow'])
        except KeyError:
            FlagKeyLow = False
        except TypeError:
            FlagKeyLow = False

        if not(FlagKeyOver) and not(FlagKeyLow):
            r = utl.ShowError('AddElim','OpenWundergrounds','No elimination was added, review the Elim parameter')
            raise KeyError

        if FlagKeyOver:
            for Lab in ElimLabOver:
                self.ElimOver[Lab] = Elim['ElimOver'][Lab]

        if FlagKeyLow:
            for Lab in ElimLabLow:
                self.ElimLow[Lab] = Elim['ElimLow'][Lab]

    def GraphData(self,PathImg=''):
        '''
        DESCRIPTION:
            This function graphs all the variables
        _____________________________________________________________
        INPUT:
            :param PathImg: A str, path to save the images.
        '''
        Labels = [self.LabDataSave[i] for i in self.LabStrYes]

        for iLab,Lab in enumerate(Labels):
            HyPl.DalyS(self.DatesCN,self.DataCC[Lab],self.LabelsWithUnits[Lab],
                    Lab,True,self.LabelsNoUnits[Lab],PathImg=PathImg+self.Station+'/Series/',
                    color=self.LabelsColors[Lab])
            HyPl.NaNMGr(self.DatesNC2,self.DataD[Lab+'NF'],self.DataD[Lab+'NNF'],
                    Lab,True,self.LabelsNoUnits[Lab],Lab,PathImg+self.Station+'/NaN_Data'+'/')

    def SaveData(self,Pathout=''):
        '''
        DESCRIPTION:
            This function saves the data in a .mat file
        _____________________________________________________________
        INPUT:
            :param Pathout: A str, path to save the data.
        '''
        Labels = [self.LabDataSave[i] for i in self.LabStrYes]
        EM = EMSD()
        # -------
        # mat
        # -------
        # original Information
        savingkeys = np.array(['FechaC']+list(Labels))
        EM.Writemat(self.DatesCC,self.DataCC,savingkeys,
                datekeys=None,datakeys=Labels,
                pathout=Pathout,Names=self.Station)

        # Hourly Information 
        savingkeys = np.array(['FechaC']+list(self.LabelsHmat))
        EM.Writemat(self.DatesH,self.DataH,savingkeys,datekeys=None,
                datakeys=self.LabelsHmat,
                pathout=Pathout+'01_Horario/',Names=self.Station+'_THT')

        # Monthly and Daily information
        savingkeys = np.array(['FechaC','FechaM']+list(self.LabelsDmat))
        EM.Writemat(self.DatesD,self.DataD,savingkeys,
                datekeys=['DatesD','DatesM'],datakeys=self.LabelsDmat,
                pathout=Pathout+'02_Diario/',Names=self.Station+'_TdiasM')
