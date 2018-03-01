# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 11/03/2016
#______________________________________________________________________________
#______________________________________________________________________________

'''

This functions generate the different cycles for meteorological data (less
than daily data). This functions are used in Hydro_Analysis.
   
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
from Utilities import DatesUtil as DUtil; DUtil=DUtil()
from Hydro_Analysis.Gen_Functions.Functions import *



def CiclD(Var,Years=None,Dates=None,DTH=24,flagZeros=False):
    '''
    DESCRIPTION:
    
        This function calculates the diurnal cycle of a variable from the 
        hourly data of a time series, it would obtain the monthly diurnal 
        cycle from all the different months and the total.
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
    _____________________________________________________________________
    
    OUTPUT:

        :return Resutls: A dict, The dictionary has:
            :return MonthsM:  A dict, Dictionary with the Monthly data.
            :return MonthsMM: A dict, Dictionary with the results 
                                      per month.
            :return MonthsME: A dict, Dictionary with the mean errors 
                                      per month.
            :return CiDT:     A ndarray, Mean complete diurnal cycle.
            :return ErrT:     A ndarray, Mean Errors.
    '''

    # Errors
    if Years == None and Dates == None:
        Er = utl.ShowError('CicloD','An_Hydro','No dates nor years were added')
        return
    elif Dates == None:
        FlagYears = True
    else:
        FlagYears = False

    # Variables
    # Months
    MonthsM = dict() # Data
    MonthsMM = dict() # Mean Data
    MonthsMMM = dict() 
    MonthsMD = dict() # Standard deviation
    MonthsME = dict() # Median Error
    TriM = dict() # Data trimestrales
    TriMM = dict() # Mean trimestral
    TriMD = dict() # Standard Deviation trimestral
    TriME = dict() # Median Error trimestral

    # Dates
    if FlagYears:
        Yi = int(Years[0])
        Yf = int(Years[1])
        dt = timedelta(0,24/DTH*3600)
        DateI = datetime(Years[0],1,1,0,0)
        DateI = datetime(Years[1],12,31,23,59)
        Date = DUtil.Dates_Comp(DateI,DateE,dtm=dt)
    elif isinstance(Dates[0],str):
        Date = DUtil.Dates_str2datetime(Dates)
    else:
        Date = Dates

    Months = np.array([i.month for i in Date])
    if flagZeros:
        q = np.where(Var == 0)
        Var[q] = np.nan

    # Months
    MaxV = []
    MinV = []
    for i in range(1,13):
        x = np.where(Months == i)
        MonthsM[i] = np.reshape(np.copy(Var)[x],(-1,DTH))
        MonthsMM[i], MonthsMD[i], MonthsME[i] =  MeanError(MonthsM[i],axis=0)
        MaxV.append(np.nanmax(MonthsMM[i])+np.nanmax(MonthsME[i]*1.2))
        MinV.append(np.nanmin(MonthsMM[i])-np.abs(np.nanmin(MonthsME[i]*1.2)))
    
    # Reshaped variable
    VarM = np.reshape(np.copy(Var),(-1,DTH))
    CiDT, DesT, ErrT =  MeanError(VarM,axis=0)
    Results = {'MonthsM':MonthsM,'MonthsMM':MonthsMM,'MonthsME':MonthsME,'CiDT':CiDT,'ErrT':ErrT}

    return Results

def CiclDPer(Var,Fecha,PV90=90,PV10=10,FlagG=True,PathImg='',NameA='',VarL='',VarLL='',C='k',Name='',flagTri=False,flagTriP=False,PathImgTri='',DTH=24):
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

def CiclDPvP(Var1,Var2,Fecha,DTH=24):
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

def CiclDP(MonthsM,PathImg='',Name='',NameSt='',FlagMan=False,vmax=None,vmin=None,Flagcbar=True,FlagIng=False,Oper='Percen',VarInd='',VarL='Precipitación [%]'):
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

    HyPl.DalyAnCycle(ProcP,PathImg=PathImg,Name=Name,NameSt=NameSt,VarL=VarL,VarLL='Precipitación',VarInd=VarInd,FlagMan=FlagMan,vmax=vmax,vmin=vmin,Flagcbar=Flagcbar,FlagIng=FlagIng,FlagSeveral=True)

    return  MonthsMP,ProcP

def CiclDV(MesesMM,PathImg='',Name='',Name2='',Var='',Var2='',Var3=''):
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

