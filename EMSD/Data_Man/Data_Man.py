# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 31/08/2017
#______________________________________________________________________________
#______________________________________________________________________________
# ------------------------
# Importing Modules
# ------------------------ 
# Manipulate Data
import numpy as np
import scipy.io as sio
from datetime import date, datetime, timedelta
# Open and saving data
import csv
import xlrd
import xlsxwriter as xlsxwl
# System
import sys
import os
import glob as gl
import re
import warnings
import platform

# ------------------
# Personal Modules
# ------------------
# Importing Modules
from Utilities import Utilities as utl
from Utilities import DatesUtil as DUtil; DUtil=DUtil()

# ------------------------
# Funciones
# ------------------------
def perdelta(start, end, delta):
    '''
    DESCRIPTION:
    
        Function that iterates dates.
    _________________________________________________________________________
    
    INPUT:
        + start: Inicial date.
        + end: Final date.
        + delta: Timedelta.
    _________________________________________________________________________
    
    OUTPUT:
        curr: Current date.
    '''
    curr = start
    while curr < end:
        yield curr
        curr += delta



# ------------------------
# Class
# ------------------------

# Data Managment
def CompD(Dates,V,dtm=None):
    '''
    DESCRIPTION:
    
        This function takes a data series and fill the missing dates with
        nan values, It would fill the entire year.
    _______________________________________________________________________

    INPUT:
        :param Dates: A list or ndarray, Data date, it must be a string 
                                         vector or a date or datetime 
                                         vector.
        :param V:     A list or ndarray, Variable that wants to be 
                                         filled. 
        :param dtm:   A list or ndarray, Time delta for the full data, 
                                         if None it would use the 
                                         timedelta from the 2 values of 
                                         the original data.
    _______________________________________________________________________
    
    OUTPUT:
        :return DateC: A ndarray, Comlete date string vector.
        :return VC:    A ndarray, Filled data values.
        :return DateN: A ndarray, Complete date Python datetime vector.
    '''
    V = np.array(V)
    Dates = np.array(Dates)
    # ---------------------
    # Error managment
    # ---------------------

    if isinstance(Dates[0],str) == False and isinstance(Dates[0],date) == False and isinstance(Dates[0],datetime) == False:
        Er = utl.ShowError('CompD','EDSM','not expected format in dates')
        return TypeError

    if len(Dates) != len(V):
        Er = utl.ShowError('CompD','EDSM','Date and V are different length')
        return Er
    if dtm != None and isinstance(dtm,timedelta) == False:
        Er = utl.ShowError('CompD','EDSM','Bad dtm format')
        return Er

    # Eliminate the errors in February
    if isinstance(Dates[0],str):
        lenDates = len(Dates)
        Dates2 = np.array([i[:10] for i in Dates])
        for iY,Y in enumerate(range(int(Dates2[0][:4]),int(Dates2[-1][:4]))):
            Fi = date(Y,2,1)
            Ff = date(Y,3,1)
            Dif = (Ff-Fi).days
            if Dif == 28:
                x = np.where(Dates2 == '%s/02/29' %(Y))
                Dates = np.delete(Dates,x)
                V = np.delete(V,x)
            x = np.where(Dates2 == '%s/02/30' %(Y))
            Dates = np.delete(Dates,x)
            V = np.delete(V,x)

    # ---------------------
    # Dates Calculations
    # ---------------------
    # Original Dates
    if isinstance(Dates[0],str):
        DatesO = DUtil.Dates_str2datetime(Dates)
    else:
        DatesO = Dates
    if dtm == None:
        dtm = DatesO[1]-DatesO[0]
    # Complete Dates
    if isinstance(DatesO[0],datetime):
        DateI = datetime(DatesO[0].year,1,1,0,0)
        DateE = datetime(DatesO[-1].year,12,31,23,59)
        DatesN = DUtil.Dates_Comp(DateI,DateE,dtm=dtm)
    else:
        DateI = date(DatesO[0].year,1,1)
        DateE = date(DatesO[-1].year,12,31)
        DatesN = DUtil.Dates_Comp(DateI,DateE,dtm=dtm)
    # Filled data
    VC = np.empty(len(DatesN))*np.nan
    DatesN = np.array(DatesN)
    DatesO = np.array(DatesO)
    V = np.array(V)
    # x = DatesN.searchsorted(DatesO)
    x = np.searchsorted(DatesN,DatesO) 

    try:
        VC[x] = V
    except ValueError:
        VC = np.array(['' for i in range(len(DatesN))])
        VC[x] = V
    
    DatesC = DUtil.Dates_datetime2str(DatesN)

    Results = {'DatesC':DatesC,'DatesN':DatesN,'VC':VC}
    return Results

def CompDC(Dates,V,DateI,DateE,dtm=None):
    '''
    DESCRIPTION:
    
        This function completes or cut data from specific dates.
    _____________________________________________________________________

    INPUT:
        + Dates: Data date, it must be a string like this 'Year/month/day' 
                the separator '/' could be change with any character. 
                It must be a string vector or a date or datetime vector.
        + VC: Variable. 
        + DateI: Initial Date in date or datetime format.
        + DateE: Final Date in date or datetime format.
        + dtm: Time delta for the full data, if None it would use the
               timedelta from the 2 values of the original data
    _____________________________________________________________________
    
    OUTPUT:
        - DatesC: Comlete date string vector.
        - V1C: Filled data values.
        - DatesN: Complete date Python datetime vector.
    
    '''
    
    V = np.array(V)
    Dates = np.array(Dates)
    # ---------------------
    # Error managment
    # ---------------------

    if isinstance(Dates[0],str) == False and isinstance(Dates[0],date) == False and isinstance(Dates[0],datetime) == False:
        Er = utl.ShowError('CompD','EDSM','Bad format in dates')
        raise Er
    if len(Dates) != len(V):
        Er = utl.ShowError('CompD','EDSM','Date and V are different length')
        raise Er
    if dtm != None and isinstance(dtm,timedelta) == False:
        Er = utl.ShowError('CompD','EDSM','Bad dtm format')
        raise Er

    # ---------------------
    # Dates Calculations
    # ---------------------
    # Original Dates
    if isinstance(Dates[0],str):
        DatesO = DUtil.Dates_str2datetime(Dates)
    else:
        DatesO = Dates

    if dtm == None:
        dtm = DatesO[1]-DatesO[0]
    DatesN = DUtil.Dates_Comp(DateI,DateE,dtm=dtm)
    
    # -------------------------------------------
    # Data Extraction
    # -------------------------------------------

    # Filled data
    if isinstance(V[0],str):
        VC = (np.empty(len(DatesN))*np.nan).astype(str)
    else:
        VC = (np.empty(len(DatesN))*np.nan)
    
    DatesN = np.array(DatesN)
    DatesO = np.array(DatesO)
    for iF,F in enumerate(DatesO):
        x = np.where(DatesN == F)[0]
        if len(x) == 1:
            VC[x] = V[iF]
        elif len(x) > 1:
            VC[x[0]] = V[iF]
    
    DatesC = DUtil.Dates_datetime2str(DatesN)

    Results = {'DatesC':DatesC,'DatesN':DatesN,'VC':VC}
    return Results

def Ca_E(FechaC,V1C,dt=24,escala=1,op='mean',flagMa=False,flagDF=False,flagNaN=True):
    '''
    DESCRIPTION:
    
        Con esta función se pretende cambiar de escala temporal los datos,
        agregándolos a diferentes escalas temporales, se deben insertar series
        completas de tiempo.

        Los datos faltantes deben estar como NaN.
    _______________________________________________________________________

    INPUT:
        + FechaC: Fecha de los datos organizada como 'año/mes/dia - HHMM' 
                  los '/' pueden ser substituidos por cualquier caracter. 
                  Debe ser un vector string y debe tener años enteros.
        + V1C: Variable que se desea cambiar de escala temporal. 
        + dt: Delta de tiempo para realizar la agregación, depende de 
              la naturaleza de los datos.
              Si se necesitan datos mensuales, el valor del dt debe ser 1.
        + escala: Escala a la cual se quieren pasar los datos:
                -1: de minutal.
                0: horario.
                1: a diario.
                2: a mensual, es necesario llevarlo primero a escala diaria.
        + op: Es la operación que se va a llevar a cabo para por ahora solo responde a:
              'mean': para obtener el promedio.
              'sum': para obtener la suma.
        + flagMa: Para ver si se quieren los valores máximos y mínimos.
                True: Para obtenerlos.
                False: Para no calcularos.
        + flagDF: Para ver si se quieren los datos faltantes por mes, solo funciona
                  en los datos que se dan diarios.
                True: Para calcularlos.
                False: Para no calcularos.
        + flagNaN: Flag to know if the user watns to include the data with low data.
    _______________________________________________________________________
    
    OUTPUT:
        - FechaEs: Nuevas fechas escaladas.
        - FechaNN: Nuevas fechas escaladas como vector fechas. 
        - VE: Variable escalada.
        - VEMax: Vector de máximos.
        - VEMin: Vector de mínimos.
    '''
    # Se desactivan los warnings en este codigo para que corra más rápido, los
    # warnings que se generaban eran por tener realizar promedios de datos NaN
    # no porque el código tenga un problema en los cálculos.
    warnings.filterwarnings('ignore')

    if escala > 2:
        utl.ShowError('EMSD','Ca_E','Todavía no se han programado estas escalas')

    # -------------------------------------------
    # Inicialización de variables
    # -------------------------------------------
    # Se inicializan las variables que se utilizarán
    FechaNN = ["" for k in range(1)]
    FechaEs = ["" for k in range(1)]
    VE = ["" for k in range(1)]
    VEMax = ["" for k in range(1)]
    VEMin = ["" for k in range(1)]

    NF = [] # Porcentaje de datos faltantes
    NNF = [] # Porcentaje de datos no faltantes
    rr = 0

    # -------------------------------------------
    # Vector de fechas
    # -------------------------------------------

    # Se toman los años
    yeari = int(FechaC[0][0:4]) # Año inicial
    yearf = int(FechaC[len(FechaC)-1][0:4]) # Año final
    Sep = FechaC[0][4] # Separador de la Fecha
    if isinstance(FechaC[0],str):
        DatesO = DUtil.Dates_str2datetime(FechaC)
    else:
        DatesO = FechaC

    # Los años se toman para generar el output de FechasEs
    if escala == -1:
        DateI = datetime(DatesO[0].year,1,1,0,0)
        DateE = datetime(DatesO[-1].year,12,31,23,59)
        dtm = timedelta(0,dt*60)
        FechaNN = DUtil.Dates_Comp(DateI,DateE,dtm=dtm)
        FechaEs = DUtil.Dates_datetime2str(FechaNN)

    elif escala == 0 or escala == 1: # Para datos horarios o diarios
        for result in perdelta(date(int(yeari), 1, 1), date(int(yearf)+1, 1, 1), timedelta(days=1)):
            FR = result.strftime('%Y'+Sep+'%m'+Sep+'%d') # Fecha
            if escala == 0:
                for i in range(0,24):
                    if rr == 0:
                        FechaNN[0] = result
                        if i < 10:
                            FechaEs[rr] = FR + '-0' +str(i)+'00'
                        else:
                            FechaEs[rr] = FR + '-' +str(i)+'00'
                    else:
                        FechaNN.append(result)
                        if i < 10:
                            FechaEs.append(FR + '-0' +str(i)+'00')
                        else:
                            FechaEs.append(FR + '-' +str(i)+'00')
                    rr += 1 # Se suman las filas
            elif escala == 1:
                if rr == 0:
                    FechaNN[0] = result
                    FechaEs[rr] = FR
                else:
                    FechaNN.append(result)
                    FechaEs.append(FR)
                rr += 1
    if escala == 2:
        x = 0
        for i in range(int(yeari),int(yearf)+1):
            for j in range(1,13):
                if i == int(yeari) and j == 1:
                    FechaNN[0] = date(i,j,1)
                    FechaEs[0] = FechaNN[0].strftime('%Y'+Sep+'%m')
                else:
                    FechaNN.append(date(i,j,1))
                    FechaEs.append(FechaNN[x].strftime('%Y'+Sep+'%m'))
                x += 1
    # -------------------------------------------
    # Cálculo del escalamiento
    # -------------------------------------------
    dtt = 0 # Contador de la diferencia
    if op == 'mean':
        if escala == 0 or escala == -1 or escala == 1: 
            # Ciclo para realizar el agregamiento de los datos
            for i in range(0,len(V1C),dt): 
                dtt = dtt + dt # Se aumenta el número de filas en el contador
                if i == 0:
                    q = np.isnan(V1C[i:dtt])
                    qq = sum(q)
                    qYes = sum(~np.isnan(V1C[i:dtt]))
                    if (qq > dt/2 and flagNaN) or qYes == 0:
                        VE[0] = float('nan')
                        if flagMa == True:
                            VEMax[0] = float('nan')
                            VEMin[0] = float('nan')
                    else:
                        try:
                            VE[0] = float(np.nanmean(V1C[i:dtt]))
                        except ValueError:
                            VE[0] = float('nan')
                        if flagMa == True:
                            try:
                                VEMax[0] = float(np.nanmax(V1C[i:dtt]))
                            except ValueError:
                                VEMax[0] = float('nan')
                            try:
                                VEMin[0] = float(np.nanmin(V1C[i:dtt]))
                            except ValueError:
                                VEMin[0] = float('nan')
                else:
                    q = np.isnan(V1C[i:dtt])
                    qq = sum(q)
                    qYes = sum(~np.isnan(V1C[i:dtt]))
                    if (qq > dt/2 and flagNaN) or qYes == 0:
                        VE.append(float('nan'))
                        if flagMa == True:
                            VEMax.append(float('nan'))
                            VEMin.append(float('nan'))
                    else:
                        try:
                            VE.append(float(np.nanmean(V1C[i:dtt])))
                        except ValueError:
                            VE.append(float('nan'))
                        if flagMa == True:
                            try:
                                VEMax.append(float(np.nanmax(V1C[i:dtt])))
                            except ValueError:
                                VEMax.append(float('nan'))
                            try:
                                VEMin.append(float(np.nanmin(V1C[i:dtt])))
                            except ValueError:
                                VEMin.append(float('nan'))
        elif escala == 2: # promedio mensual
            d = 0
            for i in range(int(yeari),int(yearf)+1):
                for j in range(1,13):
                    Fi = date(i,j,1)
                    if j == 12:
                        Ff = date(i+1,1,1)
                    else:
                        Ff = date(i,j+1,1)
                    DF = Ff-Fi
                    dtt = dtt + DF.days # Delta de días
                    if i == int(yeari) and j == 1:
                        q = np.isnan(V1C[d:dtt])
                        qq = sum(q)
                        NF.append(qq/len(V1C[d:dtt]))
                        NNF.append(1-NF[-1])    
                        qYes = sum(~np.isnan(V1C[i:dtt]))
                        if (qq > DF.days/2 and flagNaN) or qYes == 0:
                            VE[0] = float('nan')
                            if flagMa == True:
                                VEMax[0] = float('nan')
                                VEMin[0] = float('nan')
                        else:
                            try:
                                VE[0] = float(np.nanmean(V1C[d:dtt]))
                            except ValueError:
                                VE[0] = float('nan')
                            if flagMa == True:
                                try:
                                    VEMax[0] = float(np.nanmax(V1C[d:dtt]))
                                except ValueError:
                                    VEMax[0] = float('nan')
                                try:
                                    VEMin[0] = float(np.nanmin(V1C[d:dtt]))
                                except ValueError:
                                    VEMin[0] = float('nan')
                    else:
                        q = np.isnan(V1C[d:dtt])
                        qq = sum(q)
                        NF.append(qq/len(V1C[d:dtt]))
                        NNF.append(1-NF[-1])
                        qYes = sum(~np.isnan(V1C[i:dtt]))
                        if (qq > DF.days/2 and flagNaN) or qYes == 0:
                            VE.append(float('nan'))
                            if flagMa == True:
                                VEMax.append(float('nan'))
                                VEMin.append(float('nan'))
                        else:
                            try:
                                VE.append(float(np.nanmean(V1C[d:dtt])))
                            except ValueError:
                                VE.append(float('nan'))
                            if flagMa == True:
                                try:
                                    VEMax.append(float(np.nanmax(V1C[d:dtt])))
                                except ValueError:
                                    VEMax.append(float('nan'))
                                try:
                                    VEMin.append(float(np.nanmin(V1C[d:dtt])))
                                except ValueError:
                                    VEMin.append(float('nan'))
                    d = dtt


    elif op == 'sum':
        if escala == 0 or escala == -1 or escala == 1: 
            # Ciclo para realizar el agregamiento de los datos
            for i in range(0,len(V1C),dt): 
                dtt = dtt + dt # Se aumenta el número de filas en el contador
                if i == 0:
                    q = np.isnan(V1C[i:dtt])
                    qq = sum(q)
                    qYes = sum(~np.isnan(V1C[i:dtt]))
                    if (qq > dt/2 and flagNaN) or qYes == 0:
                        VE[0] = float('nan')
                        if flagMa == True:
                            VEMax[0] = float('nan')
                            VEMin[0] = float('nan')
                    else:
                        try:
                            VE[0] = float(np.nansum(V1C[i:dtt]))
                        except ValueError:
                            VE[0] = float('nan')
                        if flagMa == True:
                            try:
                                VEMax[0] = float(np.nanmax(V1C[i:dtt]))
                            except ValueError:
                                VEMax[0] = float('nan')
                            try:
                                VEMin[0] = float(np.nanmin(V1C[i:dtt]))
                            except ValueError:
                                VEMin[0] = float('nan')
                else:
                    q = np.isnan(V1C[i:dtt])
                    qq = sum(q)
                    qYes = sum(~np.isnan(V1C[i:dtt]))
                    if (qq > dt/2 and flagNaN) or qYes == 0:
                        VE.append(float('nan'))
                        if flagMa == True:
                            VEMax.append(float('nan'))
                            VEMin.append(float('nan'))
                    else:
                        try:
                            VE.append(float(np.nansum(V1C[i:dtt])))
                        except ValueError:
                            VE.append(float('nan'))
                        if flagMa == True:
                            try:
                                VEMax.append(float(np.nanmax(V1C[i:dtt])))
                            except ValueError:
                                VEMax.append(float('nan'))
                            try:
                                VEMin.append(float(np.nanmin(V1C[i:dtt])))
                            except ValueError:
                                VEMin.append(float('nan'))
        elif escala == 2: # Agregamiento mensual
            d = 0
            for i in range(int(yeari),int(yearf)+1):
                for j in range(1,13):
                    Fi = date(i,j,1)
                    if j == 12:
                        Ff = date(i+1,1,1)
                    else:
                        Ff = date(i,j+1,1)
                    DF = Ff-Fi
                    dtt = dtt + DF.days # Delta de días
                    if i == int(yeari) and j == 1:
                        q = np.isnan(V1C[d:dtt])
                        qq = sum(q)
                        NF.append(qq/len(V1C[d:dtt]))
                        NNF.append(1-NF[-1])    
                        qYes = sum(~np.isnan(V1C[i:dtt]))
                        if (qq > DF.days/2 and flagNaN) or qYes == 0:
                            VE[0] = float('nan')
                            if flagMa == True:
                                VEMax[0] = float('nan')
                                VEMin[0] = float('nan')
                        else:
                            try:
                                VE[0] = float(np.nansum(V1C[d:dtt]))
                            except ValueError:
                                VE[0] = float('nan')
                            if flagMa == True:
                                try:
                                    VEMax[0] = float(np.nanmax(V1C[d:dtt]))
                                except ValueError:
                                    VEMax[0] = float('nan')
                                try:
                                    VEMin[0] = float(np.nanmin(V1C[d:dtt]))
                                except ValueError:
                                    VEMin[0] = float('nan')
                    else:
                        q = np.isnan(V1C[d:dtt])
                        qq = sum(q)
                        NF.append(qq/len(V1C[d:dtt]))
                        NNF.append(1-NF[-1])
                        qYes = sum(~np.isnan(V1C[i:dtt]))
                        if (qq > DF.days/2 and flagNaN) or qYes == 0:
                            VE.append(float('nan'))
                            if flagMa == True:
                                VEMax.append(float('nan'))
                                VEMin.append(float('nan'))
                        else:
                            try:
                                VE.append(float(np.nansum(V1C[d:dtt])))
                            except ValueError:
                                VE.append(float('nan'))
                            if flagMa == True:
                                try:
                                    VEMax.append(float(np.nanmax(V1C[d:dtt])))
                                except ValueError:
                                    VEMax.append(float('nan'))
                                try:
                                    VEMin.append(float(np.nanmin(V1C[d:dtt])))
                                except ValueError:
                                    VEMin.append(float('nan'))
                    d = dtt

    # -------------------------------------------
    # Se dan los resultados
    # -------------------------------------------
    if flagMa == True:
        if  flagDF:
            return np.array(FechaEs), np.array(FechaNN), np.array(VE), np.array(VEMax), np.array(VEMin), np.array(NF),np.array(NNF)
        else:
            return np.array(FechaEs), np.array(FechaNN), np.array(VE), np.array(VEMax), np.array(VEMin)
    elif flagMa == False:
        if flagDF:
            return np.array(FechaEs), np.array(FechaNN), np.array(VE),np.array(NF),np.array(NNF)
        else:
            return np.array(FechaEs), np.array(FechaNN), np.array(VE)

def MIA(FechasC,Fechas,Data):
    '''
    DESCRIPTION:
    
        Con esta función se pretende encontrar la cantidad de datos faltantes de
        una serie.
    _______________________________________________________________________

    INPUT:
        + FechaC: Fecha inicial y final de la serie original.
        + Fechas: Vector de fechas completas de las series.
        + Data: Vector de fechas completo
    _______________________________________________________________________
    
    OUTPUT:
        N: Vector con los datos NaN.
        FechaNaN: Fechas en donde se encuentran los valores NaN
    '''
    # Se toman los datos 
    Ai = Fechas.index(FechasC[0])
    Af = Fechas.index(FechasC[1])

    DD = Data[Ai:Af+1]

    q = np.isnan(DD) # Cantidad de datos faltantes
    qq = ~np.isnan(DD) # Cantidad de datos no faltantes
    FechaNaN = [Fechas[k] for k in q] # Fechas en donde se encuentran los NaN

    # Se sacan los porcentajes de los datos faltantes
    NN = sum(q)/len(DD)
    NNN = sum(qq)/len(DD)

    N = [NN,NNN]

    return N, FechaNaN
        

