# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 11/03/2016
#______________________________________________________________________________
#______________________________________________________________________________

'''


 CLASS DESCRIPTION:
   This class have different routines for hydrological analysis. 

   This class do not use Pandas in any function, it uses directories and save
   several images in different folders. It is important to include the path 
   to save the images.
   
______________________________________________________________________________
'''
# ------------------------
# Importing Modules
# ------------------------ 
# Manipulate Data
import numpy as np
import scipy.io as sio
from scipy import stats as st
from datetime import date, datetime, timedelta
import time
# Graph
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates
import matplotlib.mlab as mlab
# System
import sys
import os
import glob as gl
import re
import operator
import warnings

# ------------------
# Personal Modules
# ------------------
# Importing Modules
from Utilities import Utilities as utl
from Utilities import Data_Man as DM

try:
    from Hydro_Analysis.Hydro_Plotter import Hydro_Plotter as HyPl; HyPl=HyPl()
    from Hydro_Analysis.Gen_Functions.Functions import *
    from Hydro_Analysis.Meteo import Cycles as MCy
    from Hydro_Analysis.Climate import Cycles as CCy
    from Hydro_Analysis.Dates.DatesC import DatesC
except ImportError:
    from Hydro_Plotter import Hydro_Plotter as HyPl; HyPl=HyPl()
    from Gen_Functions.Functions import *
    from Meteo import Cycles as MCy
    from Climate import Cycles as CCy
    from Dates.DatesC import DatesC

class Hydro_Analysis(object):
    '''
    ____________________________________________________________________________
    
    CLASS DESCRIPTION:
        
        This class makes different hydrological analysis, such as:
        - Diurnal Cycle
        - Diurnal-Annual Cycle
        - Annual Cycle
    _________________________________________________________________________

    INPUT:
        :param DateH:    A ndarray or list, Vector with hourly dates.
        :param VarH:     A ndarray or list, Vector with houtly or less that 
                                            hour data.
        :param DateM:    A ndarray or list, Vector with monthly dates.
        :param VarM:     A ndarray or list, Vector with monthly data.
        :param DTH:      A int, Determine the number of data that 
                                takes to complete a day it is 
                                defaulted to 24.
        :param PathImg:  A str, Ruta para guardar las imágenes.
        :param Info:     A list, Información para realizar los gráficos,
                                 la lista debe incluir:
                                 [<Nombre_Archivo>,<Nombre Estación>,
                                 <Variable a medir>,<Variable con unidades>,
                                 <Color de la línea>]
    _________________________________________________________________________

    '''

    def __init__(self,DateH=None,VarH=None,DateM=None,VarM=None,DTH=24,PathImg='',Info=['Image','Nombre','Precipitación','Precipitación [mm]','b']):
        '''
        '''
        # Parameters
        self.operations = DM.operations
        if not(DateH is None):
            DatesH = DatesC(DateH)
            if isinstance(DateH[0],str) or isinstance(DateH[0],datetime):
                self.DateH = DatesH.datetime
            else:
                self.DateH = None
        else:
            self.DateH = None
        if not(DateM is None):
            DatesM = DatesC(DateM)
            self.DateM = DatesM.datetime
        else:
            self.DateM = None
        if not(VarH is None):
            self.VarH = np.array(VarH)
        else:
            self.VarH = None
        if not(VarM is None):
            self.VarM = np.array(VarM)
        else:
            self.VarM = None

        self.DTH = DTH
        # Parameters for calculus
        self.flagZeros = False

        # Parameters for Graphs
        self.FlagG = True
        self.PathImg = PathImg
        self.NameA = Info[0]
        self.Name = Info[1]
        self.VarLL = Info[2]
        self.VarL = Info[3]
        self.color = Info[4]

        return

    def CiclD(self,Var=None,Years=None,Dates=None,DTH=None,flagZeros=None,FlagG=None,PathImg=None,NameA=None,VarL=None,VarLL=None,C=None,Name=None):
        '''
        DESCRIPTION:
        
            This function calculates the diurnal cycle of a variable from the 
            hourly data of a time series, it would obtain the monthly diurnal 
            cycle from all the different months and the total.

            Graphs can be added if the user wants to.
        _____________________________________________________________________

        INPUT:

            :param Var:       A list or ndarray, Variable that need to be 
                                                 treated.
            :param Years:     A list or ndarray, Vector with the begining year and the ending year.
            :param Dates:     A list or ndarray, Vector with the dates in string 
                                                 format yyyy/mm/dd HHMM.
            :param DTH:       A int, Determine the number of data that 
                                     takes to complete a day it is 
                                     defaulted to 24.
            :param flagZeros: A boolean, flag to know if the zeros are taking
                                         into account.
            :param FalgG:     A boolean, Flag for Graph indicator
            :param PathImg:   A str, Path to save the images.
            :param NameA:     A str, Name of the Image.
            :param VarL:      A str, Variable Label with units.
            :param VarLL:     A str, Name of the variable
            :param C:         A str, Line Color.
            :param Name:      A str, Name of the Station.
        _____________________________________________________________________
        
        OUTPUT:

            This code outputs 2 to 4 graphs, 1 for the diurnal cycle 
            in all the data and other for the diurnal cycle for per month.

            :return Resutls: A dict, The dictionary has:
                :return MonthsM:  A dict, Dictionary with the Monthly data.
                :return MonthsMM: A dict, Dictionary with the results 
                                          per month.
                :return MonthsME: A dict, Dictionary with the mean errors 
                                          per month.
                :return CiDT:     A ndarray, Mean complete diurnal cycle.
                :return ErrT:     A ndarray, Mean Errors.

        '''
        # Graph warnings ignored
        warnings.filterwarnings('ignore')

        # ----------------
        # Error Managmet
        # ----------------
        if Var is None and self.VarH is None:
            raise utl.ShowError('CiclD','Hydro_Analysis','No Data added')
        if Years == None and self.DateH is None:
            raise utl.ShowError('CiclD','Hydro_Analysis','No Dates added')
        if Years is None and Dates is None:
            Dates = self.DateH
        if Var is None:
            Var = self.VarH
        if DTH is None:
            DTH = self.DTH
        if FlagG is None:
            FlagG = self.FlagG
        if PathImg is None:
            PathImg = self.PathImg
        if NameA is None:
            NameA = self.NameA
        if Name is None:
            Name = self.Name
        if VarLL is None:
            VarLL = self.VarLL
        if VarL is None:
            VarL = self.VarL
        if C is None:
            C = self.color

        # Calculus
        Results = MCy.CiclD(Var=Var,Years=Years,Dates=Dates,DTH=DTH,flagZeros=flagZeros) 
        CiDT = Results['CiDT']
        ErrT = Results['ErrT']
        MonthsMM = Results['MonthsMM']
        MonthsME = Results['MonthsME']
        MaxV = []
        MinV = []
        for i in range(1,13):
            MaxV.append(np.nanmax(MonthsMM[i])+np.nanmax(MonthsME[i]*1.2))
            MinV.append(np.nanmin(MonthsMM[i])-np.abs(np.nanmin(MonthsME[i]*1.2)))


        if FlagG:
            # Se realizan los labels para las horas
            if DTH == 24:
                HH = np.arange(0,24)
                HH2 = np.arange(0,24)
                HHL = np.arange(0,24)
            elif DTH == 48:
                HH = np.arange(0,48)
                HH2 = np.arange(0,48,2)
                HHL = np.arange(0,24)
            HyPl.DalyCycle(HH,CiDT,ErrT,VarL,VarLL,Name,NameA,PathImg,color=C,label=VarLL,lw=1.5)
            # Figura normal
            Mes = ['Enero','Febrero','Marzo','Abril','Mayo','Junio',\
                'Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'];
            # Tamaño de la Figura
            fH=30 # Largo de la Figura
            fV = fH*(2/3) # Ancho de la Figura
            # Se crea la carpeta para guardar la imágen
            utl.CrFolder(PathImg)
            # Se realiza la curva para todos los meses
            plt.rcParams.update({'font.size': 10,'font.family': 'sans-serif'\
                ,'font.sans-serif': 'Arial'\
                ,'xtick.labelsize': 10,'xtick.major.size': 6,'xtick.minor.size': 4\
                ,'xtick.major.width': 1,'xtick.minor.width': 1\
                ,'ytick.labelsize': 10,'ytick.major.size': 12,'ytick.minor.size': 4\
                ,'ytick.major.width': 1,'ytick.minor.width': 1\
                ,'axes.linewidth':1\
                ,'grid.alpha':0.1,'grid.linestyle':'-'})
            fig, axs = plt.subplots(4,3, figsize=DM.cm2inch(fH,fV))
            axs = axs.ravel() # Para hacer un loop con los subplots

            for ii,i in enumerate(range(1,13)):
                axs[ii].tick_params(axis='x',which='both',bottom='on',top='off',\
                    labelbottom='on',direction='in')
                axs[ii].tick_params(axis='x',which='major',direction='inout')
                axs[ii].tick_params(axis='y',which='both',left='on',right='off',\
                    labelleft='on')
                axs[ii].tick_params(axis='y',which='major',direction='inout') 
                axs[ii].grid()

                axs[ii].errorbar(HH,MonthsMM[i],yerr=MonthsME[i],fmt='-',color=C,label=VarLL)
                axs[ii].set_title(Mes[ii],fontsize=15)
                axs[ii].set_xlim([0,23])
                axs[ii].set_ylim([np.nanmin(MinV),np.nanmax(MaxV)])
                axs[ii].xaxis.set_ticks(HH2)
                axs[ii].set_xticklabels(HHL,rotation=90)

                if ii == 0 or ii == 3 or ii == 6 or ii == 9:
                    axs[ii].set_ylabel(VarL,fontsize=13)
                if ii >=9:
                    axs[ii].set_xlabel('Horas',fontsize=13)
                if ii == 2:
                    axs[ii].legend(loc='best')

            plt.tight_layout()
            plt.savefig(PathImg + 'CMErr_' + NameA +'.png',format='png',dpi=300 )
            plt.close('all')

        return Results

    def CiclDPer(self,Var,Fecha,PV90=90,PV10=10,FlagG=True,PathImg='',NameA='',VarL='',VarLL='',C='k',Name='',flagTri=False,flagTriP=False,PathImgTri='',DTH=24):
        '''
        DESCRIPTION:
        
            Con esta función se pretende realizar el ciclo diurno de una variable a
            partir de los datos horarios de una serie de datos, se obtendrá el ciclo
            diurno para todos los datos totales con los percentiles que se elijan, 
            la media y la mediana.

            Además pueden obtenerse las gráficas si así se desea.
        _______________________________________________________________________

        INPUT:
            + Var: Variable que se desea tratar.
            + Fecha: Variable con el año inicial y el año final de los datos.
            + PV90: Valor que se quiere tomar para el percentil superrior.
            + PV10: Valor que se quiere tomar para el percentil inferior.
            + FalgG: Activador para desarrollar las gráfcias.
            + PathImg: Ruta para guardar las imágenes.
            + NameA: Nombre de la estación de las imágenes
            + VarL: Label de la variable.
            + VarLL: Nombre de la variable
            + C: Color de la línea.
            + Name: Nombre de la estación que se está analizando.
            + flagTri: Activador de variables trimestrales.
            + flagTriP: Activador para gráfica trimestral.
            + PathImgTri: Ruta para las imágenes trimestrales.
            + DTH: Cuantos datos se dan para completar un día, se asume que son 24 
                   datos porque podrían estar en horas
        _______________________________________________________________________
        
        OUTPUT:
            Este código saca 2 gráficas, una por cada mes y el promedio de todos los
            meses.
            - MesesMM: Variable directorio con los resultados por mes
        
        '''

        # Se obtienen todos los datos en una matriz
        VarM = np.reshape(Var,(-1,24))      

        # Se calculan los momentos necesarios
        zM = np.nanmean(VarM,axis=0)
        zMed = np.nanmedian(VarM,axis=0)
        P90 = [np.nanpercentile(VarM[:,i],PV90) for i in range(VarM.shape[1])]
        P10 = [np.nanpercentile(VarM[:,i],PV10) for i in range(VarM.shape[1])]

        if FlagG:
            x = np.arange(0,24,24/DTH)
            HH = x
            HyPl.DalyCyclePer(HH,zM,zMed,P10,P90,PV90,PV10,VarL,VarLL,Name,NameA,PathImg)
            

        return zM,zMed,P90,P10

    def CiclDPvP(self,Var1,Var2,Fecha,DTH=24):
        '''
            DESCRIPTION:
        
        Con esta función se pretende realizar el ciclo diurno de una variable
        que depende de otra variable.

        Además pueden obtenerse las gráficas si así se desea.
        _______________________________________________________________________

            INPUT:
        + Var1: Variable que se desea tratar debe ser precipitación.
        + Var2: Variable que se desea tratar.
        + Fecha: Variable con la Fecha inicial y la fecha final de los datos.
        + DTH: Cuantos datos se dan para completar un día, se asume que son 24 
               datos porque podrían estar en horas
        _______________________________________________________________________
        
            OUTPUT:
        Este código saca 2 gráficas, una por cada mes y el promedio de todos los
        meses.
        - MesesMM: Variable directorio con los resultados por mes
        
        '''
        # Se quitan los warnings de las gráficas
        warnings.filterwarnings('ignore')

        # Se inicializan las variables
        MesesM1 = dict() # Meses
        MesesMM1 = dict() # Meses
        MesesMMM1 = dict() # Meses
        MesesMD1 = dict() # Meses
        MesesME1 = dict()

        MesesM2 = dict() # Meses
        MesesMM2 = dict() # Meses
        MesesMMM2 = dict() # Meses
        MesesMD2 = dict() # Meses
        MesesME2 = dict()

        VarM1 = np.reshape(Var1[:],(-1,DTH))
        VarM2 = np.reshape(Var2[:],(-1,DTH))
        
        Yi = int(Fecha[0])
        Yf = int(Fecha[1])
        d = 0
        dtt = 0
        x = 0
        # Se calculan los ciclos diurnos de todos los meses
        for i in range(Yi,Yf+1): # ciclo para los años
            for j in range(1,13): # ciclo para los meses
                Fi = date(i,j,1)
                if j == 12:
                    Ff = date(i+1,1,1)
                else:
                    Ff = date(i,j+1,1)
                DF = Ff-Fi
                dtt = dtt + DF.days # Delta de días

                if x == 0:
                    # Se saca el promedio de los datos
                    MesesM1[j] = VarM1[d:dtt,:]
                    MesesM2[j] = VarM2[d:dtt,:]
                    
                else:
                    MesesM1[j] = np.vstack((MesesM1[j],VarM1[d:dtt,:]))
                    MesesM2[j] = np.vstack((MesesM2[j],VarM2[d:dtt,:]))

                d = dtt
            x += 1
        
        TM1 = []
        TM2 = []
        # Se eliminan todos los ceros de las variables.
        for i in range(1,13):
            q = np.where(MesesM1[i] <= 0.1)
            MesesM1[i][q] = float('nan')
            MesesM2[i][q] = float('nan')

        for i in range(1,13): # Meses
            MesesMM1[i] = np.nanmean(MesesM1[i],axis=0)
            MesesMD1[i] = np.nanstd(MesesM1[i],axis=0)
            MesesME1[i] = [k/np.sqrt(len(MesesM1[i])) for k in MesesMD1[i]]

            MesesMM2[i] = np.nanmean(MesesM2[i],axis=0)
            MesesMD2[i] = np.nanstd(MesesM2[i],axis=0)
            MesesME2[i] = [k/np.sqrt(len(MesesM2[i])) for k in MesesMD2[i]]

            TM1.append(np.sum(MesesMM1[i]))
            MesesMMM1[i] = [(k/TM1[i-1])*100 for k in MesesMM1[i]]          

        
        q = np.where(VarM1 <= 0.1)
        VarM1[q] = np.nan
        VarM2[q] = np.nan

        # Se calcula el ciclo diurno para todos los datos
        CiDT1 = np.nanmean(VarM1, axis=0)
        DesT1 = np.nanstd(VarM1, axis=0)
        # Se calcula el ciclo diurno para todos los datos
        CiDT2 = np.nanmean(VarM2, axis=0)
        DesT2 = np.nanstd(VarM2, axis=0)

        VarMNT1 = []
        VarMNT2 = []
        # Se calcula el número total de datos
        for i in range(len(VarM1[0])):
            VarMNT1.append(sum(~np.isnan(VarM1[:,i])))
            VarMNT2.append(sum(~np.isnan(VarM2[:,i])))

        ErrT1 = [k/np.sqrt(VarMNT1[ii]) for ii,k in enumerate(DesT1)]
        ErrT1 = np.array(ErrT1)

        ErrT2 = [k/np.sqrt(VarMNT2[ii]) for ii,k in enumerate(DesT2)]
        ErrT2 = np.array(ErrT2)

        return MesesM1, MesesMM1, MesesMD1,MesesME1,CiDT1,DesT1,ErrT1, MesesM2, MesesMM2, MesesMD2,MesesME2,CiDT2,DesT2,ErrT2

    def CiclDP(self,MonthsM,PathImg='',Name='',NameSt='',FlagMan=False,vmax=None,vmin=None,Flagcbar=True,FlagIng=False,Oper='Percen',VarInd='',VarL='Precipitación [%]',FlagG=True,FlagSeveral=False):
        '''
        DESCRIPTION:
            
            With this function you can get the diurnal cycle of the
            percentage of precipitation along the year.

            Also plots the graph.
        _______________________________________________________________________

        INPUT:
            + MonthsM: Precipitation of days for months.
            + PathImg: Path for the Graph.
            + Name: Name of the graph.
            + NameSt: Name of the Name of the Sation.
            + FlagMan: Flag to get the same cbar values.
            + vmax: Maximum value of the cbar.
            + vmin: Maximum value of the cbar.
            + Flagcbar: Flag to plot the cbar.
            + FlagIng: Flag to convert laels to english.
            + VarInd: Variable indicative (example: Prec).
        _______________________________________________________________________
        
        OUTPUT:
            This function plots and saves a graph.
            - MonthsMP: Directory with the monthly values of percentage
            - PorcP: Directory with the 
        '''
        
        if FlagIng:
            MM = ['Jan','Mar','May','Jul','Sep','Nov','Jan']
        else:
            MM = ['Ene','Mar','May','Jul','Sep','Nov','Ene']

        ProcP = np.empty((12,24)) # Porcentaje de todos los meses
        MonthsMP = dict()

        for ii,i in enumerate(range(1,13)):
            if Oper == 'Percen':
                MonthsMP[i] = PrecPor(MonthsM[i])
                ProcP[ii,:] = np.nanmean(MonthsMP[i],axis=0)*100
            else:
                MonthsMP[i] = MonthsM[i]
                ProcP[ii,:] = np.nanmean(MonthsMP[i],axis=0)

        if FlagG:
            HyPl.DalyAnCycle(ProcP,PathImg=PathImg,Name=Name,NameSt=NameSt,VarL=VarL,VarLL='Precipitación',VarInd=VarInd,FlagMan=FlagMan,vmax=vmax,vmin=vmin,Flagcbar=Flagcbar,FlagIng=FlagIng,FlagSeveral=FlagSeveral)

        return  MonthsMP,ProcP

    def CiclDV(self,MesesMM,PathImg='',Name='',Name2='',Var='',Var2='',Var3=''):
        '''
            DESCRIPTION:
        
        Con esta función se pretende realizar el ciclo diurno de cualquier variable
        a lo largo del año.

        Además pueden obtenerse las gráficas si así se desea.
        _______________________________________________________________________

            INPUT:
        + MesesMM: Variable de los promedios mensuales multianuales.
        + PathImg: Ruta para guardar las imágenes.
        + Name: Nombre de los documentos.
        + Name2: Nombre de la estación.
        + Var: Nombre corto de la variable.
        + Var2: Nombre completo de la variable
        + Var3: Nombre completo de la variable con unidades de medida.
        _______________________________________________________________________
        
            OUTPUT:
        Este código saca 1 gráfica teniendo el promedio de todos los meses.
        '''

        # Se crea la ruta en donde se guardarán los archivos
        utl.CrFolder(PathImg)

        # Se inicializan las variables
        MM = ['E','F','M','A','M','J','J','A','S','O','N','D','E']
        ProcP = np.empty((12,24))

        # Ciclo para extraer los promedios
        for ii,i in enumerate(range(1,13)):
            ProcP[ii,:] = MesesMM[i]

        x = np.arange(0,24)
        x3 = np.arange(0,25)

        # Se organizan las horas
        ProcP2 = np.hstack((ProcP[:,7:],ProcP[:,:7]))
        x2 = np.hstack((x[7:],x[:7]))
        for i in range(len(ProcP2)):
            ProcP22 = 0
            ProcP22 = np.hstack((ProcP2[i,:],ProcP2[i,0]))
            if i == 0:
                ProcP3 = ProcP22
            else:
                ProcP3 = np.vstack((ProcP3,ProcP22))

        ProcP3 = np.vstack((ProcP3,ProcP3[0,:]))
        
        # Datos para las gráficas
        # v = np.linspace(608, 8, 9, endpoint=True)
        # bounds=[0,1,2,3,4,5,6,7,8]
                

        # ProcP2 = np.hstack((ProcP2,ProcP2[:,0:1]))

        x2 = np.hstack((x2,x2[0]))

        F = plt.figure(figsize=(15,10))
        plt.rcParams.update({'font.size': 22})
        #plt.contourf(x3,np.arange(1,14),ProcP3,v,vmax=8,vmin=0)
        plt.contourf(x3,np.arange(1,14),ProcP3)
        plt.title('Ciclo diurno de la ' + Var2 + ' en el año en ' + Name2,fontsize=26 )  # Colocamos el título del gráfico
        plt.ylabel('Meses',fontsize=24)  # Colocamos la etiqueta en el eje x
        plt.xlabel('Horas',fontsize=24)  # Colocamos la etiqueta en el eje y
        axs = plt.gca()
        axs.yaxis.set_ticks(np.arange(1,14,1))
        axs.set_yticklabels(MM)
        axs.xaxis.set_ticks(np.arange(0,25,1))
        axs.set_xticklabels(x2)
        plt.tight_layout()
        #cbar = plt.colorbar(boundaries=bounds,ticks=v)
        cbar = plt.colorbar()
        cbar.set_label(Var3)
        plt.gca().invert_yaxis()
        plt.legend(loc=1)
        plt.grid()
        plt.savefig(PathImg + 'T' + Var +'_' + Name+'.png' )
        plt.close('all')

    def CiclA(self,VMes=None,Years=None,PathImg=None,Name=None,VarL=None,VarLL=None,C=None,NameArch=None,flagA=False,flagAH=False,oper='mean',FlagG=True):
        '''
        DESCRIPTION:
            This function calculates the annual cycle of a variable, also
            calculates the annual series of requiered.

            Additionally, this function can makes graphs of the annual cycle 
            and the annual series if asked.
        _______________________________________________________________________

        INPUT:

            :param VMes:     A ndarray, Variable with the monthly data.
            :param Years:    A list or ndarray, Vector with the initial and 
                                                final year.
            :param flagA:    A boolean, flag to know if the annual series 
                                        is requiered.
            :param oper:     A ndarray, Operation for the annual data.
            :param PathImg:  A str, Path to save the Graphs.
            :param Name:     A str, Name of the station.
            :param VarL:     A str, Label of the variable with units.
            :param VarLL:    A str, Label of the variable without units.
            :param C:        A str, Color of the line.
            :param NameArch: A str, Name of the File of the image.
            :param flagA:    A boolean, flag to know if the annual series
                                        is requiered.
            :param flagAH:   A boolean, flag to know if the graph is done with 
                                        the hydrological cycle.
            :param flagG:    A boolean, flag to make the graph.
        _______________________________________________________________________
        
        OUTPUT:
            This function returns a direcory with all the data.
        '''

        # --------------------
        # Error Managment
        # --------------------
        if VMes is None and self.VarM is None:
            raise utl.ShowError('CiclA','Hydro_Analysis','No Data added')
        if Years is None and self.DateM is None:
            raise utl.ShowError('CiclA','Hydro_Analysis','No Dates added')
        if Years is None: 
            Years = [self.DateM[0].year,self.DateM[-1].year]
        if VMes is None:
            VMes = self.VarM
        if FlagG is None:
            FlagG = self.FlagG
        if PathImg is None:
            PathImg = self.PathImg
        if NameArch is None:
            NameArch = self.NameA
        if Name is None:
            Name = self.Name
        if VarLL is None:
            VarLL = self.VarLL
        if VarL is None:
            VarL = self.VarL
        if C is None:
            C = self.color
        
        results = CCy.CiclA(VMes,Years,flagA=flagA,oper=oper)

        # Graph
        if FlagG:
            HyPl.AnnualCycle(results['MesM'],results['MesE'],VarL,VarLL,Name,NameArch,PathImg,flagAH,color=C)
        # --------------------
        # Annual Series
        # --------------------
        if flagA:
            Yi = int(Years[0])
            Yf = int(Years[1])
            x = [date(i,1,1) for i in range(Yi,Yf+1)]
            xx = [i for i in range(Yi,Yf+1)]
            if FlagG:
                HyPl.AnnualS(x,results['AnM'],results['AnE'],VarL,VarLL,Name,NameArch,PathImg+'Anual/',
                        color=C)

        return results

    def EstAnom(self,VMes):
        '''
        DESCRIPTION:
        
            This function takes the monthly data and generates the data 
            anomalies and the standarized data.

            The calculation is done using the following equation:

                Z = \frac{x-\mu}{\sigma}
        _______________________________________________________________________

        INPUT:
            :param VMes: a listo or ndArray, Mounthly average of the variable.
        _______________________________________________________________________
        
        OUTPUT:
        :param Anom:  a ndArray, Anomalie data results.
        :param StanA: a ndArray, Standarized data results.
        '''

        # Variable initialization
        Anom = np.empty(len(VMes))
        StanA = np.empty(len(VMes))


        # Variable reshape
        VarM = np.reshape(VMes[:],(-1,12))
        
        # Annual calculations
        MesM = np.nanmean(VarM,axis=0) # Annual Average.
        MesD = np.nanstd(VarM,axis=0) # Annual deviation.

        # Anomalie cycle
        x = 0
        for i in range(len(VarM)):
            for j in range(12):
                Anom[x] = VarM[i,j] - MesM[j] # Anomalies
                StanA[x] = (VarM[i,j] - MesM[j])/MesD[j] # Standarized data
                x += 1


        return Anom,StanA

    def MTrData(self,VMes):
        '''
            DESCRIPTION:
        
        This function takes the monthly data and generates a series with all the
        different months, as well as a series with the trimester data. The data must
        hace full years.
        _______________________________________________________________________

            INPUT:
        + VMes: Mounthly average of the variable.
        _______________________________________________________________________
        
            OUTPUT:
        - Months: Matrix with all the months in the column and years in the rows.
        - Trim: Matrix with all the trimesters in the columns and years in the rows.
        '''
        # Variable reshape
        Months = np.reshape(VMes[:],(-1,12))

        # Variable initialization
        Trim = np.empty((len(Months),4))

        DEF = np.empty(3)
        MAM = np.empty(3)
        JJA = np.empty(3)
        SON = np.empty(3)

        # Cycle to calculate the trimesters
        x = 0
        for i in range(len(Months)):
            if x == len(Months)-1:
                DEF = np.empty(3)*np.nan
            else:
                DEF[0] = Months[i,11]
                DEF[1:3] = Months[i+1,0:2]

            MAM = Months[i,1:4]
            JJA = Months[i,4:7]
            SON = Months[i,7:11]

            Trim[i,0] = np.mean(DEF)
            Trim[i,1] = np.mean(MAM)
            Trim[i,2] = np.mean(JJA)
            Trim[i,3] = np.mean(SON)

            x += 1

        return Months, Trim

    def PrecDryDays(self,Date,Prec,Ind=0.1):
        '''
            DESCRIPTION:
        
        This function calculates the number of consecutive days without 
        precipitation.
        _______________________________________________________________________

            INPUT:
        + Date: Vector of dates.
        + Prec: Precipitation vector in days.
        + Ind: Indicator of the minimum precipitation value.
        _______________________________________________________________________
        
            OUTPUT:
        - TotalNumDays: Vector with total number of consecutive days without 
                        precipitation.
        - MaxNumDays: Maximum number of consecutive days without precipitation.
        '''
        
        # Number of days below the indicated (Ind) value
        x = np.where(Prec <= Ind)[0]

        TotalNumDays = [] # List with all the days
        TotalPrecCount = [] # List for the total precipitation
        TotalDateB = [] # List of dates

        # Loop to calculate all the days
        DaysCount = 1 # Days counter
        PrecCount = Prec[x[0]] # Precipitation counter
        for i in range(1,len(x)):
            if x[i] == x[i-1]+1:
                DaysCount += 1
                PrecCount += Prec[x[i]]
            else:
                TotalNumDays.append(DaysCount)
                TotalPrecCount.append(PrecCount)
                TotalDateB.append(Date[x[i]-DaysCount])
                DaysCount = 1
                PrecCount = 0

        TotalNumDays = np.array(TotalNumDays)
        # Maximum number of days withput precipitation
        MaxNumDays = np.max(TotalNumDays)

        TotalPrecCount = np.array(TotalPrecCount)
        # Maximum value of total precipitation in those days
        MaxPrecCount = np.max(TotalPrecCount)

        # Beginning date of the maximum dry days
        TotalDateB = np.array(TotalDateB)
        xx = np.where(TotalNumDays == MaxNumDays)[0]
        DateMaxDays = TotalDateB[xx]

        return TotalNumDays,MaxNumDays,TotalPrecCount,MaxPrecCount,TotalDateB,DateMaxDays

    def PrecMoistDays(self,Date,Prec,Ind=0.1):
        '''
        DESCRIPTION:
        
            This function calculates the number of consecutive days with 
            precipitation above certain value.
        _______________________________________________________________________

        INPUT:
            + Date: Vector of dates.
            + Prec: Precipitation vector in days.
            + Ind: Indicator of the minimum precipitation value.
        _______________________________________________________________________
        
        OUTPUT:
            - TotalNumDays: Vector with total number of consecutive days 
                            without precipitation.
            - MaxNumDays: Maximum number of consecutive days without 
                          precipitation.
        '''
        
        # Number of days below the indicated (Ind) value
        x = np.where(Prec >= Ind)[0]

        TotalNumDays = [] # List with all the days
        TotalPrecCount = [] # List for the total precipitation
        TotalDateB = [] # List of dates

        # Loop to calculate all the days
        DaysCount = 1 # Days counter
        PrecCount = Prec[x[0]] # Precipitation counter
        for i in range(1,len(x)):
            if x[i] == x[i-1]+1:
                DaysCount += 1
                PrecCount += Prec[x[i]]
            else:
                TotalNumDays.append(DaysCount)
                TotalPrecCount.append(PrecCount)
                TotalDateB.append(Date[x[i]-DaysCount])
                DaysCount = 1
                PrecCount = 0

        TotalNumDays = np.array(TotalNumDays)
        # Maximum number of days with precipitation
        MaxNumDays = np.max(TotalNumDays)
        x = np.where(TotalNumDays == MaxNumDays)[0]

        TotalPrecCount = np.array(TotalPrecCount)
        # Maximum of precipitation of number of days that was max
        MaxPrecCount_MaxDay = np.max(TotalPrecCount[x])

        # Maximum value of total precipitation in those days
        MaxPrecCount = np.max(TotalPrecCount)
        x = np.where(TotalPrecCount == MaxPrecCount)[0]
        # Maximum number of days of the max precipitation
        MaxNumDays_MaxPrec = np.max(TotalNumDays[x])


        # Beginning date of the maximum dry days
        TotalDateB = np.array(TotalDateB)
        xx = np.where(TotalNumDays == MaxNumDays)[0]
        DateMaxDays = TotalDateB[xx]

        return TotalNumDays,MaxNumDays,TotalPrecCount,MaxPrecCount,TotalDateB,DateMaxDays,MaxPrecCount_MaxDay,MaxNumDays_MaxPrec

    def DaysOverOrLower(self,Data,Dates,Value,flagMonths=False,Comparation='Over'):
        '''
        DESCRIPTION:
    
            This function calculates the days over or lower one specific value
            in every year or month and year of the series.
        _______________________________________________________________________

        INPUT:
            + Data: Data that needs to be counted.
            + Dates: Dates of the data, can be in datetime or string vector.
                     if the Dates are in a string vector it has to be in
                     yyyy/mm/dd.
            + Value: Value to search days over or lower.
            + flagMonths: flag to know if the count would be monthly.
                          True: to make the count monthly.
                          False: to make the count annualy.
            + Comparation: String with the comparation that is going to be 
                           done.
                           
                           It can recognize the following strings:

                                 String           |       Interpretation 
                           'Over', 'over' or '>'  |             >
                           'Lower', 'lower' or '<'|             <
                                  '>='            |             >=
                                  '<='            |             <=

        _______________________________________________________________________
        
        OUTPUT:
        
            - results: number of days over or lower a value for every year or
                       month.
            - An: Dates where the operation was made.
        '''
        # Determine operation
        Comp = DM.Oper_Det(Comparation)
        # ----------------
        # Error managment
        # ----------------
        if Comp == -1:
            return -1, -1
        # Dates Error managment
        if isinstance(Dates[0],str):
            DatesP = DUtil.Dates_str2datetime(Dates)
            if list(DatesP)[0] == -1:
                return -1, -1
            elif isinstance(DatesP[0],datetime):
                Er = utl.ShowError('DaysOverOrLower','Hydro_Analysis','Dates are not in days')
                return Er, Er
        else:
            DatesP = Dates
            if isinstance(DatesP[0],datetime):
                Er = utl.ShowError('DaysOverOrLower','Hydro_Analysis','Dates are not in days')
                return Er, Er
        # ----------------
        # Dates managment
        # ----------------
        if flagMonths:
            An = DUtil.Dates_datetime2str(DatesP,Date_Format='%Y/%m')
            if list(An)[0] == -1:
                return -1,-1
        else:
            An = DUtil.Dates_datetime2str(DatesP,Date_Format='%Y')
            if list(An)[0] == -1:
                return -1,-1
        # ----------------
        # Calculations
        # ----------------
        results = []
        if flagMonths:      
            for i in range(DatesP[0].year,DatesP[-1].year+1):
                for j in range(1,13):
                    if j < 10:
                        m = '%s/0%s' %(i,j)
                    else:
                        m = '%s/%s' %(i,j)
                    x = np.where(An == m)[0]
                    P = Data[x]
                    xx = np.where(Comp(P,Value))[0]
                    if sum(~np.isnan(P)) >= 0.70*len(P):
                        results.append(len(xx))
                    else:
                        results.append(np.nan)
        else:
            for i in range(DatesP[0].year,DatesP[-1].year+1):
                x = np.where(An == str(i))[0]

                P = Data[x]
                xx = np.where(Comp(P,Value))[0]
                if sum(~np.isnan(P)) >= 0.70*len(P):
                    results.append(len(xx))
                else:
                    results.append(np.nan)

        return results,An

    def ConsDaysOverOrLower(self,Data,Dates,Value,Comparation='Over'):
        '''
        DESCRIPTION:
        
            This function calculates the number of consecutive days with 
            values over or lower a specific value and also gives the dates at
            the beginning and end of evey season.
        _______________________________________________________________________

        INPUT:
            + Data: Data that needs to be counted.
            + Dates: Dates of the data, can be in datetime or string vector.
                     if the Dates are in a string vector it has to be in
                     yyyy/mm/dd.
            + Value: Value to search days over or lower.
            + Comparation: String with the comparation that is going to be 
                           done.
                           
                           It can recognize the following strings:

                                 String           |       Interpretation 
                           'Over', 'over' or '>'  |             >
                           'Lower', 'lower' or '<'|             <
                                  '>='            |             >=
                                  '<='            |             <=
        _______________________________________________________________________
        
        OUTPUT:

            The output is the dictionary results with the following keys:

            - TotalNumDays: Vector with total number of consecutive days 
                            above or below the value.
            - TotalPrecCount: Vector with the total of values during the
                              different number of days, works with
                              precipitation, averages needs to be 
                              determined manually.
            - TotalDateB: Starting dates of the different events.
            - MaxNumDays: Maximum number of consecutive days above 
                          or below the value.
            - MaxPrecCount_MaxDay: Maximum values in the maximum day.
            - MaxNumDays_MaxPrec: Maximum days in the maximum values.
            - DateMaxDays: Beginnig date of the maximum days count.
        '''
        # keys
        keys = ['TotalNumDays','TotalPrecCount','TotalDateB','MaxNumDays',\
            'MaxPrecCount','DateMaxDays','MaxPrecCount_MaxDay','MaxNumDays_MaxPrec']
        # Determine operation
        Comp = DM.Oper_Det(Comparation)
        # ----------------
        # Error managment
        # ----------------
        if Comp == -1:
            return -1
        # Dates Error managment
        if isinstance(Dates[0],str):
            DatesP = DUtil.Dates_str2datetime(Dates)
            if list(DatesP)[0] == -1:
                return -1
            elif isinstance(DatesP[0],datetime):
                Er = utl.ShowError('DaysOverOrLower','Hydro_Analysis','Dates are not in days')
                return Er
        else:
            DatesP = Dates
            if isinstance(DatesP[0],datetime):
                Er = utl.ShowError('DaysOverOrLower','Hydro_Analysis','Dates are not in days')
                return Er
        # --------------
        # Calculations
        # --------------
        results = dict()
        results['TotalNumDays'] = [] # List with all the days
        results['TotalPrecCount'] = [] # List for the total
        results['TotalDateB'] = [] # List of dates
        x = np.where(Comp(Data,Value))[0]
        if len(x) > 1:
            # Loop to calculate all the days
            DaysCount = 1 # Days counter
            PrecCount = Data[x[0]] # Value counter
            for i in range(1,len(x)):
                if x[i] == x[i-1]+1:
                    DaysCount += 1
                    PrecCount += Data[x[i]]
                else:
                    results['TotalNumDays'].append(DaysCount)
                    results['TotalPrecCount'].append(PrecCount)
                    results['TotalDateB'].append(DatesP[x[i]-DaysCount])
                    DaysCount = 0
                    PrecCount = 0

                if i == len(x)-1:
                    results['TotalNumDays'].append(DaysCount)
                    results['TotalPrecCount'].append(PrecCount)
                    results['TotalDateB'].append(DatesP[x[i]-DaysCount])
                    DaysCount = 0
                    PrecCount = 0

            results['TotalNumDays'] = np.array(results['TotalNumDays'])
            # Maximum number of days
            results['MaxNumDays'] = np.max(results['TotalNumDays'])
            x = np.where(results['TotalNumDays'] == results['MaxNumDays'])[0]

            results['TotalPrecCount'] = np.array(results['TotalPrecCount'])
            # Maximum value counter of number of days that was max
            results['MaxPrecCount_MaxDay'] = np.max(results['TotalPrecCount'][x])

            # Maximum value in those days
            results['MaxPrecCount'] = np.max(results['TotalPrecCount'])
            x = np.where(results['TotalPrecCount'] == results['MaxPrecCount'])[0]
            # Maximum number of days of the maximum value
            results['MaxNumDays_MaxPrec'] = np.max(results['TotalNumDays'][x])

            # Beginning date of the maximum 
            results['TotalDateB'] = np.array(results['TotalDateB'])
            xx = np.where(results['TotalNumDays'] == results['MaxNumDays'])[0]
            results['DateMaxDays'] = results['TotalDateB'][xx]
        else:
            for ikey,key in enumerate(keys):
                if ikey > 2:
                    results[key] = 0
                else:
                    results[key] = np.array([0])

        return results

    def PrecCount(self,Prec,DatesEv,dt=1,M=60):
        '''
        DESCRIPTION:
            
            This functions calculates the duration of precipitation events 
            from composites.
        _________________________________________________________________________

        INPUT:
            + PrecC: composite of precipitation.
            + DatesEv: Matrix with all the events dates, format yyyy/mm/dd-HHMM
                       or datetime.
            + dt: Time delta in minutes
        _________________________________________________________________________

        OUTPUT:
            This functions returns a directory with the values of:
            - DurPrec: Precipitation duration in hours.
            - TotalPrec: Total of precipitation in that time.
            - IntPrec: Event Intensity.
            - MaxPrec: Maximum of precipitation during the event.
            - DatesEvst: Date where the event begins.
            - DatesEvend: Date where the event ends.
        '''

        # --------------------------------------
        # Error managment
        # --------------------------------------

        if not(isinstance(DatesEv[0][0],str)) and not(isinstance(DatesEv[0][0],datetime)):
            E = utl.ShowError('PrecCount','Hydro_Analysis','Not dates given, review format')
            raise E

        
        # --------------------------------------
        # Dates
        # --------------------------------------

        # Variables for benining and end of the event
        DatesEvst_Aft = []
        DatesEvend_Aft = []
        for i in range(len(DatesEv)):
            if isinstance(M,list):
                MP = M[i]
            else:
                MP = M
            x = [MP]
            # Minimum of precipitation
            if dt == 1:
                MinPrec = 0.001
            else:
                MinPrec = 0.10
            # Precipitation beginning
            xm = np.where(Prec[i,:MP]<=MinPrec)[0]
            k = 1
            a = len(xm)-1
            I = 10
            while k == 1:   
                if dt == 1:
                    if a == -1:
                        xmm = 0
                        k = 2
                        break
                    while a-I < 0:
                        I -= 1
                    if xm[a] == xm[a-I]+I:
                        xmm = xm[a]
                        k = 2
                    else:
                        a = a-1
                        if a == 0:
                            xmm = xm[0]
                            k = 2
                elif dt == 5:
                    if a == -1:
                        xmm = 0
                        k = 2
                        break
                    if xm[a] == xm[a-1]+1:
                        xmm = xm[a]
                        k = 2
                    else:
                        a = a-1
                        if a == 0:
                            xmm = xm[0]
                            k = 2                       
            
            # Precipitation ending
            xM = np.where(Prec[i,x[0]+1:]<=MinPrec)[0]+x[0]+1
            k = 1
            a = 0
            while k == 1:
                aa = len(xM)
                if aa == 1 or aa == 0:
                    xMM = len(Prec[i,:])-1
                    k = 2
                    break
                if dt == 1:
                    # print('a',a)
                    try:
                        if xM[a] == xM[a+10]-10:
                            xMM = xM[a]
                            k = 2
                        else:
                            a = a+1
                            if a == len(xM)-1:
                                xMM = xM[len(xM)-1]
                                k = 2
                    except:
                        try:
                            if xM[a] == xM[a+5]-5:
                                xMM = xM[a]
                                k = 2
                            else:
                                a = a+1
                                if a == len(xM)-1:
                                    xMM = xM[len(xM)-1]
                                    k = 2
                        except:
                            xMM = xM[a]
                            k = 2
                            
                elif dt == 5:
                    if xM[a] == xM[a+1]-1:
                        xMM = xM[a]
                        k = 2
                    else:
                        a = a+1
                        if a == len(xM)-1:
                            xMM = xM[len(xM)-1]
                            k = 2
                else:
                    xMM = xM[a]
                    k = 2
            DatesEvst_Aft.append(DatesEv[i][xmm])
            DatesEvend_Aft.append(DatesEv[i][xMM])
        
        DatesEvst = DUtil.Dates_str2datetime(DatesEvst_Aft,Date_Format=None)
        DatesEvend = DUtil.Dates_str2datetime(DatesEvend_Aft,Date_Format=None)
        DatesEvst_Aft = np.array(DatesEvst_Aft)
        DatesEvend_Aft = np.array(DatesEvend_Aft)
        # ---------------
        # Calculations
        # ---------------

        # Variables
        DurPrec = []
        TotalPrec = []
        IntPrec = []
        MaxPrec = []

        for i in range(len(DatesEv)):
            # Verify event data
            q = sum(~np.isnan(Prec[i]))
            if q <= len(DatesEv[i])*.60:
                DurPrec.append(np.nan)
                TotalPrec.append(np.nan)
                IntPrec.append(np.nan)
                MaxPrec.append(np.nan)
            else:
                # ------------------------
                # Rainfall duration
                # ------------------------
                Dxi = np.where(DatesEv[i] == DatesEvst_Aft[i])[0]
                Dxf = np.where(DatesEv[i] == DatesEvend_Aft[i])[0]
                DurPrec.append((Dxf[0]-Dxi[0]+1)*dt/60) # Duración en horas
                # Se verifica que haya información
                q = sum(~np.isnan(Prec[i,Dxi[0]:Dxf[0]+1]))
                if q <= len(Prec[i,Dxi[0]:Dxf[0]+1])*.50:
                    DurPrec[-1] = np.nan
                    TotalPrec.append(np.nan)
                    IntPrec.append(np.nan)
                    MaxPrec.append(np.nan)
                else:
                    # ------------------------
                    # Precipitation total
                    # ------------------------
                    TotalP = np.nansum(Prec[i,Dxi[0]:Dxf[0]+1])
                    TotalPrec.append(TotalP)
                    # ------------------------
                    # Intensity precipitation
                    # ------------------------
                    IntPrec.append(TotalP/DurPrec[-1])
                    # ------------------------
                    # Maximum Precipitation
                    # ------------------------
                    MaxPrec.append(np.nanmax(Prec[i,Dxi[0]:Dxf[0]+1]))

        DurPrec = np.array(DurPrec)
        TotalPrec = np.array(TotalPrec)
        IntPrec = np.array(IntPrec)
        MaxPrec = np.array(MaxPrec)

        Results = {'DurPrec':DurPrec,'TotalPrec':TotalPrec,'IntPrec':IntPrec
        ,'MaxPrec':MaxPrec,'DatesEvst':DatesEvst,'DatesEvend':DatesEvend}
        return Results



