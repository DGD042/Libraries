# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 31/10/2017
#______________________________________________________________________________
#______________________________________________________________________________

# --------------------
# Importing Packages
# --------------------
# Data managment
import numpy as np
import scipy.io as sio 
from scipy import stats as st 
# Documents
import xlsxwriter as xlsxwl
from xlsxwriter.utility import xl_range_abs
# Graphics
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates 
import matplotlib.mlab as mlab
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from mpl_toolkits.mplot3d import Axes3D
# Time
from datetime import date, datetime, timedelta
import time
# Sistem
import os
import sys
import warnings

# ------------------
# Personal Modules
# ------------------
# Importing Modules
from Utilities import Utilities as utl
from Utilities import DatesUtil as DUtil; DUtil=DUtil()
from Utilities import Data_Man as DM
from AnET.CFitting import CFitting as CF;
from EMSD.Dates.DatesC import DatesC
from Hydro_Analysis.Models.Radiation import RadFunctions as RadM



class Model_AngstromPrescott(object):
    '''
    DESCRIPTION:

        This class transforms the information of sunshine duration
        to radiation using the Angostrom-Prescott model described in
        Almorox, Benito, Hontoria (2004) "Estimation of monthly
        Angstrom-Prescott equation coefficient from mesured daily
        data in Todelo, Spain".

        The model uses the equation:

        H/H0 = a+b(n/N),

        where H is the global radiation, H0 is global extraterrestial
        radiation, n is the sunshine duration and N is the maximum
        daily sunshine duration. a and b are parameters that can be 
        adjusted having the information of Sunshine duration and 
        Radiation.

    _________________________________________________________________
    INPUT:
        :param Lat:        A float, Latitude of the region.
        :param Dates:      A list or ndArray, Vector with datetime 
                           or string dates.
        :param SD:         A list or ndArray, 
                           Vector with Sunshine duration data in 
                           hours.
        :param Rad:        A list or ndArray, Vector with Radiation 
                           data in MJ/m^2/day.
        :param Parameters: A dict, Dictionary with parameters of 
                           the equation needed to adjust new series.
                           a and b are the parameters of the 
                           equation.
    _________________________________________________________________

    '''

    def __init__(self,Lat,Dates,SD,Rad=None,Parameters=None):
        '''
        DESCRIPTION:

            This class transforms the information of sunshine duration
            to radiation using the Angostrom-Prescott model described in
            Almorox, Benito, Hontoria (2004) "Estimation of monthly
            Angstrom-Prescott equation coefficient from mesured daily
            data in Todelo, Spain".

            The model uses the equation:

            H/H0 = a+b(n/N),

            where H is the global radiation, H0 is global extraterrestial
            radiation, n is the sunshine duration and N is the maximum
            daily sunshine duration. a and b are parameters that can be 
            adjusted having the information of Sunshine duration and 
            Radiation.

        _________________________________________________________________
        INPUT:
            :param Lat:        A float, Latitude of the region.
            :param Dates:      A list or ndArray, Vector with datetime 
                               or string dates.
            :param SD:         A list or ndArray, 
                               Vector with Sunshine duration data in 
                               hours.
            :param Rad:        A list or ndArray, Vector with Radiation 
                               data in MJ/m^2/day.
            :param Parameters: A dict, Dictionary with parameters of 
                               the equation needed to adjust new series.
                               a and b are the parameters of the 
                               equation.
        _________________________________________________________________

        '''
        # ------------------
        # Error Managment
        # ------------------
        try:
            assert isinstance(Dates[0],date) or isinstance(Dates[0],str)
        except AssertionError:
            self.Error('Model_AngstromPrescott','__init__','Wrong date format')
        try:
            assert len(Dates) == len(SD)
        except AssertionError:
            self.Error('Model_AngstromPrescott','__init__','Data and Dates must have the same lenght')
        try:
            assert not(Rad is None) or not(Parameters is None)
        except AssertionError:
            self.Error('Model_AngstromPrescott','__init__','Rad and Parameters are None, at least one has to have data')
        # ------------------
        # Parameters
        # ------------------
        # Dates
        Dates = DatesC(Dates)
        # Calculate Julian Day
        self.Julian_Day = np.array([i.timetuple().tm_yday for i in Dates.datetime])
        self.Dates = Dates
        # Latitude
        self.Lat = Lat
        # Parameters
        self.Parameters = Parameters
        # ---------------------------
        # Calculations
        # ---------------------------
        self.Data = dict()
        self.Data['H0'],self.Data['N'] = self.CalcRadN()
        self.Data['H'] = Rad
        self.Data['n'] = SD
        if not(Rad is None):
            self.RelN = self.Data['n']/self.Data['N']
            self.RelRad = self.Data['H']/self.Data['H0']

        return

    def CalcRadN(self,Lat=None,J=None):
        '''
        DESCRIPTION:

            This method calculates the extraterrestial radiation in the
            hole year and the day lenght in hours.
        _________________________________________________________________
        INPUT:
            :param Lat:        A float, Latitude of the region.
            :param J:          An int, Julian Day.
        _________________________________________________________________
        OUTPUT:
            :return H0: Extraterrestial radiation in a definde latitude.
            :return N: Day length in hours.

        '''
        if Lat is None:
            Lat = self.Lat
        if J is None:
            J = self.Julian_Day

        H0 = np.empty(len(J))
        N = np.empty(len(J))
        H0 = RadM.ExRad2(Lat,J)
        N = RadM.DaylenH(Lat,J)

        return H0,N

    def ClearData(self,model=1):
        '''
        DESCRIPTION:

            This method clear the data using the different methods.
        _________________________________________________________________
        INPUT: 
            :param method: method to implement: 1 Using the STD.
                                                2 Using the polynomial fit.
        '''
        if model == 1:
            MN = np.nanmean(self.RelN)
            STDN = np.nanstd(self.RelN)
            NN = len(self.RelN)
            print(STDN)
            STDN = STDN/np.sqrt(NN)
            print(NN)
            print(STDN)
            print(MN)
            MRad = np.nanmean(self.RelRad)
            STDRad = np.nanstd(self.RelRad)
            NRad = len(self.RelRad)
            STDRad = STDRad/np.sqrt(NRad)
            M = 10
            self.RelN[(self.RelN > MN+M*STDN) | (self.RelN < MN-M*STDN)] = np.nan
            self.RelRad[(self.RelRad > MRad+M*STDRad) | (self.RelRad < MRad-M*STDRad)] = np.nan
        elif model == 2:
            self.AdjustModel(model=2)
            FitC = self.R
            VC = FitC['Function'](self.RelN, *FitC['Coef'])
            DatOver = VC+FitC['EErr']
            DatLower = VC-FitC['EErr']
            self.RelRad[(self.RelRad > DatOver) | (self.RelRad < DatLower)] = np.nan
        return
        
    def AdjustModel(self,model=1):
        '''
        DESCRIPTION:

            This method Adjusts the model to calculate the parameters
        _________________________________________________________________
        INPUT: 
            :param model: model to implement: 1 is the linear.
                                              2 is polinomical.
        '''
        if self.Data['H'] is None:
            self.Error('AdjustModel','Model_AngstromPrescott','No Radiation data was given, cannot make the the fitting')
        CF2 = CF()
        if model == 1:
            fun = RadM.AngstromPrescottEq2
            CF2.addfun(fun,key='anpreq',funeq=r'$H/H0 = %.4f + %.4f(n/N)$')
            self.R = CF2.FF(self.RelN,self.RelRad,F='AnPrEq')
        elif model == 2:
            fun = RadM.AngstromPrescottEq3
            CF2.addfun(fun,key='anpreq',funeq=r'$H/H0 = %.4f(n/N) + %.4f(n/N)^2 + %.4f(n/N)^3 + %.4f$')
            self.R = CF2.FF(self.RelN,self.RelRad,F='AnPrEq')
        self.Parameters = self.R['Coef']
        return

    def GraphAdj(self,Name='Image',PathImg='',flagConInt=False):
        '''
        DESCRIPTION:

            This method makes the scatter plot of the given data.
        _________________________________________________________________
        INPUT:
            :param Name:    A str, Name of the document.
            :param PathImg: A str, path to save the Image.

        '''
        V1 = self.RelN
        V2 = self.RelRad
        FitC = self.R
        # Se realiza el ajuste
        x = np.linspace(np.nanmin(self.RelN),np.nanmax(self.RelN),100)
        VC = FitC['Function'](x, *FitC['Coef'])

        # Tamaño de la Figura
        fH = 20 # Largo de la Figura
        fV = fH*(2/3) # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)

        # Se genera la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': 'Arial'
            ,'xtick.labelsize': 15,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': 16,'ytick.major.size': 12,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})
        plt.tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='on',direction='out')
        plt.tick_params(axis='y',which='both',left='on',right='off',\
            labelleft='on')
        plt.tick_params(axis='y',which='major',direction='inout') 
        plt.grid()
        # Precipitación
        plt.scatter(V1,V2,color='dodgerblue',alpha=0.7)
        # plt.title('',fontsize=16)
        plt.ylabel('Índice de claridad '+r'($H/H0$)',fontsize=16)
        plt.xlabel('Fracción de Brillo Solar '+r'($n/N$)',fontsize=16)
        # Axes
        ax = plt.gca()
        xTL = ax.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
        plt.xlim([0,np.nanmax(V1)+2*MxL])

        xTL = ax.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
        minorLocatorx = MultipleLocator(MxL)
        yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
        MyL = np.abs(np.abs(yTL[1])-np.abs(yTL[0]))/5 # minorLocatory value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().xaxis.set_minor_locator(minorLocatorx)
        plt.gca().yaxis.set_minor_locator(minorLocatory)

        # Se incluye el ajuste
        Label = FitC['FunctionEq']+'\n'+r'$R^2=%.3f$'
        plt.plot(x,VC,'k--',label=Label %tuple(list(FitC['Coef'])+[FitC['R2']]))
        plt.legend(loc=1,fontsize=10,framealpha=0.6)
        if flagConInt:
            for i in range(2):
                Coef = []
                for j in range(len(FitC['Coef'])):
                    Coef.append(FitC['ConInt'][j][i])
                VC = FitC['Function'](x, *Coef)
                plt.plot(x,VC,'r--')


        plt.tight_layout()
        Nameout = PathImg+Name
        plt.savefig(Nameout+'.png',format='png',dpi=200)
        plt.close('all')
        return
        
    def ExtractData(self,Name='Doc',Pathout=''):
        '''
        DESCRIPTION:

            This method extracts the data to an Excel File with several
            sheets.
        _________________________________________________________________

        '''
        if self.Data['H'] is None:
            self.Error('Model_AngstromPrescott','ExtractData','Cannot extract information if Rad is not added')
        # Parameters
        Vars = ['Dates','H','H0','n','N']
        # Create Folder
        utl.CrFolder(Pathout)
        # Create document
        B = xlsxwl.Workbook(Pathout+Name+'.xlsx')
        # Add Data Sheet
        S = B.add_worksheet('Data')

        # Headers
        S.write(0,0,'Dates')
        S.write(0,1,'Julian Day')
        S.write(0,2,'H')
        S.write(0,3,'H0')
        S.write(0,4,'n')
        S.write(0,5,'N')
        S.write(0,6,'RelH')
        S.write(0,7,'RelN')

        # Fill data
        for iD,D in enumerate(self.Dates.str):
            # Dates
            S.write(iD+1,0,D)
            # Julian Day
            S.write(iD+1,1,self.Julian_Day[iD])
            x = 2
            for var in Vars[1:]:
                if np.isnan(self.Data[var][iD]):
                    D = ''
                else:
                    D = self.Data[var][iD]
                S.write(iD+1,x,D)
                x += 1
            # RelH 
            if np.isnan(self.RelRad[iD]):
                D = ''
            else:
                D = self.RelRad[iD]
            S.write(iD+1,6,D)
            # RelN 
            if np.isnan(self.RelN[iD]):
                D = ''
            else:
                D = self.RelN[iD]
            S.write(iD+1,7,D)

        # Add Graph Sheet 
        S = B.add_worksheet('Graphs')
        # ---------------
        # Create graphs
        # ---------------
        # Ranges of the data
        VarCol = {'Dates':0,'H':2,'H0':3,'n':4,'N':5}
        VarsRel = ['RelH','RelN']
        VarsRelCol = {'RelH':6,'RelN':7}
        Ranges = dict()
        Names = dict()
        RangesRel = dict()
        for ivar,var in enumerate(Vars):
            Ranges[var] = xl_range_abs(1,VarCol[var],iD+1,VarCol[var])
            Names[var] = xl_range_abs(0,VarCol[var],0,VarCol[var])

        for ivar,var in enumerate(VarsRelCol):
            RangesRel[var] = xl_range_abs(1,VarsRelCol[var],iD+1,VarsRelCol[var])

        x = 1
        xx = 1
        for ichart in range(2):
            chart = B.add_chart({'type':'line'})
            chart.add_series({
                'name':"=Data!"+Names[Vars[x]],
                'categories':"=Data!"+Ranges['Dates'],
                'values':"=Data!"+Ranges[Vars[x]]
                })
            chart.add_series({
                'name':"=Data!"+Names[Vars[x+1]],
                'categories':"=Data!"+Ranges['Dates'],
                'values':"=Data!"+Ranges[Vars[x+1]]
                })
            S.insert_chart(xx,1,chart,{'x_offset':0,'y_offset':0,'x_scale':2.,'y_scale':1.0})
            x += 2
            xx += 15
        # Scatter Linear
        chart = B.add_chart({'type':'scatter'})
        chart.add_series({
            'name':"Relación",
            'categories':"=Data!"+RangesRel['RelN'],
            'values':"=Data!"+RangesRel['RelH'],
            'trendline':{'type':'linear',
                'display_equation':True,
                'display_r_squared':True}
            })
        chart.set_title({'name':'Relación entre índices lineal'})
        chart.set_x_axis({'name':'Fracción de Brillo Solar (n/N)'})
        chart.set_y_axis({'name':'Índice de claridad (H/H0)'})
        chart.set_style(1)
        chart.set_legend({'none':True})
        S.insert_chart(32,1,chart,{'x_offset':0,'y_offset':0,'x_scale':1.5,'y_scale':1.5})

        # Scatter Polynomial
        chart = B.add_chart({'type':'scatter'})
        chart.add_series({
            'name':"Relación",
            'categories':"=Data!"+RangesRel['RelN'],
            'values':"=Data!"+RangesRel['RelH'],
            'trendline':{'type':'polynomial',
                'display_equation':True,
                'display_r_squared':True,
                'order':3}
            })
        chart.set_title({'name':'Relación entre índices Polinomica'})
        chart.set_x_axis({'name':'Fracción de Brillo Solar (n/N)'})
        chart.set_y_axis({'name':'Índice de claridad (H/H0)'})
        chart.set_style(1)
        chart.set_legend({'none':True})
        S.insert_chart(32+25,1,chart,{'x_offset':0,'y_offset':0,'x_scale':1.5,'y_scale':1.5})
        B.close()

    def Error(self,Cl,method,msg):
        raise Exception('ERROR: in class <'+Cl+'> in method <'+method+'>\n'+msg)

    def __str__(self):
        L1 = '\nAngostrom-Prescott Model'
        if self.Data['H'] is None:
            L2 = '\nRadiation Data: NO'
            L2 += '\n->Cannot calculate parameters without this information'
        else:
            L2 = '\nRadiation Data: OK'
        if self.Parameters is None:
            L3 = '\nNo parameters added or calculated yet'
        else:
            L3 = '\nParameters: '+str(self.Parameters)
            Label = '\nEquation: '+self.R['FunctionEq'][1:-1]+'\n\t'+r'  R^2 = %.3f'
            return L1+L2+L3+Label %tuple(list(self.R['Coef'])+[self.R['R2']])
        return L1+L2+L3

