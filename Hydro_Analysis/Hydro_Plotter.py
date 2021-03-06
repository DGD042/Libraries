# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 18/02/2016
#______________________________________________________________________________
#______________________________________________________________________________

#______________________________________________________________________________
#
# DESCRIPCIÓN DE LA CLASE:
#   En esta clase se incluyen las rutinas para generar gráficos de información
#   hidrológica.
#
#   Esta libreria es de uso libre y puede ser modificada a su gusto, si tienen
#   algún problema se pueden comunicar con el programador al correo:
#   dagonzalezdu@unal.edu.co
#______________________________________________________________________________

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates
import matplotlib.mlab as mlab
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import sys
import warnings
from scipy import stats as st
from scipy.signal import butter, lfilter, freqz
# Se importan los paquetes para manejo de fechas
from datetime import date, datetime, timedelta

# ------------------
# Personal Modules
# ------------------
# Importing Modules
from Utilities import Utilities as utl
from Utilities import Data_Man as DM
try:
    from Hydro_Analysis.Dates.DatesC import DatesC
except ImportError:
    from Dates.DatesC import DatesC

class Hydro_Plotter:
    def __init__(self):
        '''
            DESCRIPTION:

        Este es el constructor por defecto, no realiza ninguna acción.
        '''
        self.font = 'Arial'

        # Indicators
        self.VarInd = {'precipitación':'Prec','precipitation':'Prec',
                       'temperatura':'Temp','temperature':'Temp',
                       'humedad relativa':'HR','relative humidity':'HR',
                       'humedad específica':'HS','specific humidity':'HS'}
        
        # Image dimensions
        self.fH = 20
        self.fV = self.fH*(2.0/3.0)

        # Image resolution
        self.dpi = 300

        # Image font size
        self.fontsize=15

    def monthlab(self,Date):
        '''             
            DESCRIPTION:
        Esta función pretende cambiar los ticks de las gráficas de años a 
        mes - año para compartir el mismo formato con las series de Excel.
        __________________________________________________________________
            
            INPUT:
        :param Date: Fechas en formato ordinal.
        __________________________________________________________________

            OUTPUT: 
        :return Labels: Labes que se ubicarán en los ejes.
        '''

        Year = [date.fromordinal(int(i)).year for i in Date]
        Month = [date.fromordinal(int(i)).month for i in Date]

        Meses = ['ene.','feb.','mar.','abr.','may.','jun.','jul.','ago.','sep.','oct.','nov.','dec.']

        Labels = [Meses[Month[i]-1] + ' - ' + str(Year[i])[2:] for i in range(len(Year))]
        return Labels

    def DalyS(self,Date,Value,Var_LUn,Var='',flagT=True,v='',PathImg='',**args):
        '''
        DESCRIPTION:
        
            Esta función permite hacer las gráficas de series temporales parecidas a
            a las presentadas en Excel para el proyecto.

        _________________________________________________________________________

        INPUT:
            :param Date: Vector de fechas en formato date.
            :param Value: Vector de valores de lo que se quiere graficar.
            :param Var_LUn: Label de la variable con unidades, por ejemplo 
                            Precipitación (mm).
            :param Var: Nombre de la imagen.
            :param flagT: Flag para saber si se incluye el título.
            :param v: Titulo de la Figura.
            :param PathImg: Ruta donde se quiere guardar el archivo.
            :param **args: Argumentos adicionales para la gráfica, como 
                   color o ancho de la línea.
        _________________________________________________________________________
        
        OUTPUT:
            Esta función arroja una gráfica y la guarda en la ruta desada.
        '''
        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)

        # Se genera la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': self.fontsize,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
            ,'xtick.labelsize': self.fontsize,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': self.fontsize+1,'ytick.major.size': 12,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})
        plt.rcParams['agg.path.chunksize'] = 20000
        plt.tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='on',direction='out')
        plt.tick_params(axis='y',which='both',left='on',right='off',\
            labelleft='on')
        plt.tick_params(axis='y',which='major',direction='inout') 
        plt.grid()
        # Se realiza la figura 
        plt.plot(Date,Value,**args)
        # Se arreglan los ejes
        axes = plt.gca()
        plt.xlim([min(Date),max(Date)]) # Incluyen todas las fechas
        # Se incluyen los valores de los minor ticks
        yTL = axes.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # Minor tick value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().yaxis.set_minor_locator(minorLocatory)
        # Se cambia el label de los ejes
        xTL = axes.xaxis.get_ticklocs() # List of position in x
        Labels2 = self.monthlab(xTL)
        plt.xticks(xTL, Labels2) # Se cambia el label de los ejes
        # Se rotan los ejes
        for tick in plt.gca().get_xticklabels():
            tick.set_rotation(45)
        # Labels
        if flagT:
            plt.title(v,fontsize=self.fontsize)
        plt.ylabel(Var_LUn,fontsize=self.fontsize+1)
        # Se arregla el espaciado de la figura
        plt.tight_layout()
        # Se guarda la figura
        plt.savefig(PathImg + Var + '.png',format='png',dpi=self.dpi)
        plt.close('all')

    def FreqPrec(self,Percen,Value,PerVer=None,limits=None,Var_LUn='',Var='',flagT=True,v='',PathImg='',**args):
        '''
        DESCRIPTION:
        
            Function to plot the Precipitation Frequency. 

        _________________________________________________________________________

        INPUT:
            :param Percen: A ndarray, Percentiles.
            :param Value: A ndarray, vector of sorted values.
            :param PerVer: A list, Vertical Percentiles.
            :param Var_LUn: Label de la variable con unidades, por ejemplo 
                            Precipitación (mm).
            :param Var: Nombre de la imagen.
            :param flagT: Flag para saber si se incluye el título.
            :param v: Titulo de la Figura.
            :param PathImg: Ruta donde se quiere guardar el archivo.
            :param **args: Argumentos adicionales para la gráfica, como 
                   color o ancho de la línea.
        _________________________________________________________________________
        
        OUTPUT:
            Esta función arroja una gráfica y la guarda en la ruta desada.
        '''
        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)

        # Se genera la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': self.fontsize,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
            ,'xtick.labelsize': self.fontsize,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': self.fontsize+1,'ytick.major.size': 12,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})
        plt.rcParams['agg.path.chunksize'] = 20000
        plt.tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='on',direction='out')
        plt.tick_params(axis='y',which='both',left='on',right='off',\
            labelleft='on')
        plt.tick_params(axis='y',which='major',direction='inout') 
        plt.grid()
        # Se realiza la figura 
        plt.plot(Percen,Value,**args)
        if not(limits is None):
            plt.ylim([limits[0],limits[1]])
        if not(PerVer is None):
            axes = plt.gca()
            yTL = axes.yaxis.get_ticklocs() # List of Ticks in x
            Lines = ['--','-.',':']
            for iV,V in enumerate(PerVer):
                plt.plot([V,V],[yTL[0],yTL[-1]],Lines[iV],color='r',label=str(V))
            plt.legend(loc=0)
        # Se arreglan los ejes
        axes = plt.gca()
        # plt.xlim([-1,101]) # Incluyen todas las fechas
        # Se incluyen los valores de los minor ticks
        yTL = axes.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # Minor tick value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().yaxis.set_minor_locator(minorLocatory)
        xTL = axes.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # Minor tick value
        minorLocatorx = MultipleLocator(MxL)
        plt.gca().xaxis.set_minor_locator(minorLocatorx)
        # Se cambia el label de los ejes
        xTL = axes.xaxis.get_ticklocs() # List of position in x
        # Labels
        if flagT:
            plt.title(v,fontsize=self.fontsize)
        plt.ylabel(Var_LUn,fontsize=self.fontsize+1)
        # Se arregla el espaciado de la figura
        plt.tight_layout()
        # Se guarda la figura
        plt.savefig(PathImg + Var + '.png',format='png',dpi=self.dpi)
        plt.close('all')

    def NaNMGr(self,Date,NNF,NF,Var='',flagT=True,Var_L='',Names='',PathImg=''):
        '''
            DESCRIPTION:
        
        Esta función permite hacer las gráficas de datos faltantes mensuales.

        _________________________________________________________________________

            INPUT:
        + Date: Vector de fechas en formato date.
        + Value: Vector de valores de lo que se quiere graficar.
        + Var: Nombre de la imagen.
        + flagT: Flag para saber si se incluye el título.
        + Var_L: Label de la variable sin unidades.
        + Names: Nombre de la estación para el título.
        + PathImg: Ruta donde se quiere guardar el archivo.
        + **args: Argumentos adicionales para la gráfica, como color o ancho de la línea.
        _________________________________________________________________________
        
            OUTPUT:
        Esta función arroja una gráfica y la guarda en la ruta desada.
        '''
        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)

        NN = np.array(NNF)*100
        N = np.array(NF)*100

        # Parámetros de la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
            ,'xtick.labelsize': 15,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': 16,'ytick.major.size': 6,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})
        plt.tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='on',direction='out')
        plt.tick_params(axis='y',which='both',left='on',right='off',\
            labelleft='on')
        plt.tick_params(axis='y',which='major',direction='out') 
        # Se realiza la figura 
        p1 = plt.bar(Date,NN,color='#006600',width=31,edgecolor='none') # Disponibles
        p2 = plt.bar(Date,N,color='#E46C0A',bottom=NN,width=31,edgecolor='none') # Faltantes
        # Se arreglan los ejes
        axes = plt.gca()
        axes.set_ylim([0,100])
        # Se cambia el label de los eje
        xTL = axes.xaxis.get_ticklocs() # List of position in x
        Labels2 = self.monthlab(xTL)
        plt.xticks(xTL, Labels2) # Se cambia el valor de los ejes
        plt.legend((p2[0],p1[0]), ('Faltantes','Disponibles'),loc=4)
        # Labels
        if flagT:
            # Título
            plt.title('Estado de la información de ' + Var_L + ' en la estación ' + Names,fontsize=15 )
        plt.xlabel(u'Mes',fontsize=16)  # Colocamos la etiqueta en el eje x
        plt.ylabel('Porcentaje de datos',fontsize=16)  # Colocamos la etiqueta en el eje y
        plt.tight_layout()
        plt.savefig(PathImg + Var +'_NaN_Mens' + '.png',format='png',dpi=self.dpi )
        plt.close('all')

    def NaNMGrC(self,Date,NNF,NF,Var='',flagT=True,Var_L='',Names='',PathImg=''):
        '''
            DESCRIPTION:
        
        Esta función permite hacer las gráficas de datos faltantes mensuales.

        _________________________________________________________________________

            INPUT:
        + Date: Vector de fechas en formato date.
        + Value: Vector de valores de lo que se quiere graficar.
        + Var: Nombre de la imagen.
        + flagT: Flag para saber si se incluye el título.
        + Var_L: Label de la variable sin unidades.
        + Names: Número de la cuenca.
        + PathImg: Ruta donde se quiere guardar el archivo.
        _________________________________________________________________________
        
            OUTPUT:
        Esta función arroja una gráfica y la guarda en la ruta desada.
        '''
        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)

        NN = np.array(NNF)*100
        N = np.array(NF)*100

        # Parámetros de la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
            ,'xtick.labelsize': 15,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': 16,'ytick.major.size': 6,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})
        plt.tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='on',direction='out')
        plt.tick_params(axis='y',which='both',left='on',right='off',\
            labelleft='on')
        plt.tick_params(axis='y',which='major',direction='out') 
        # Se realiza la figura 
        p1 = plt.bar(Date,NN,color='#006600',width=31,edgecolor='none') # Disponibles
        p2 = plt.bar(Date,N,color='#E46C0A',bottom=NN,width=31,edgecolor='none') # Faltantes
        # Se arreglan los ejes
        axes = plt.gca()
        axes.set_ylim([0,100])
        # Se cambia el label de los eje
        xTL = axes.xaxis.get_ticklocs() # List of position in x
        Labels2 = self.monthlab(xTL)
        plt.xticks(xTL, Labels2) # Se cambia el valor de los ejes
        plt.legend((p2[0],p1[0]), ('Faltantes','Disponibles'),loc=4)
        # Labels
        if flagT:
            # Título
            plt.title('Estado de la información de ' + Var_L + ' en la cuenca ' + Names,fontsize=15 )
        plt.xlabel(u'Mes',fontsize=16)  # Colocamos la etiqueta en el eje x
        plt.ylabel('Porcentaje de datos',fontsize=16)  # Colocamos la etiqueta en el eje y
        plt.tight_layout()
        plt.savefig(PathImg + Var +'_NaN_Mens' + '.png',format='png',dpi=self.dpi )
        plt.close('all')

    # Cycles
    def DalyCycle(self,HH,CiDT,ErrT,VarL='',VarLL='',MaxMin=None,Name='',
            NameA='Figura',PathImg='',vlimits=None,flagIng=True,**args):
        '''
        DESCRIPTION:
        
            Esta función permite hacer las gráficas del ciclo diurno
        _________________________________________________________________________

        INPUT:
            :param HH:      Vector de horas.
            :param CiDT:    Vector de datos horarios promedio.
            :param ErrT:    Barras de error de los datos.
            :param VarL:    Label de la variable con unidades, por ejemplo 
                            Precipitación (mm).
            :param VarLL:   Label de la variable sin unidades.
            :param MaxMin:  A ndarray, array with the Maximum and Minimum values
                            to be graph.
            :param Name:    Nombre de la Estación.
            :param NameA:   Nombre del archivo.
            :param PathImg: Ruta donde se quiere guardar el archivo.
            :param vlimits: A list, 2 length list with the min and max y axis 
                            limit.
            :param **args:  Argumentos adicionales para la gráfica, como color 
                            o ancho de la línea.
        _________________________________________________________________________
        
            OUTPUT:
        Esta función arroja una gráfica y la guarda en la ruta desada.
        '''
        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)

        # Se genera la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': self.fontsize,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
            ,'xtick.labelsize': self.fontsize,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': self.fontsize,'ytick.major.size': 12,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})
        plt.tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='on',direction='in')
        plt.tick_params(axis='x',which='major',direction='inout')
        plt.tick_params(axis='y',which='both',left='on',right='off',\
            labelleft='on')
        plt.tick_params(axis='y',which='major',direction='inout') 
        plt.grid()
        # Argumentos que se deben incluir color=C,label=VarLL,lw=1.5
        if MaxMin is None:
            plt.errorbar(HH,CiDT,yerr=ErrT,fmt='-',**args)
        else:
            plt.plot(HH,CiDT,**args)
            plt.plot(HH,MaxMin[0,:],'--',**args)
            plt.plot(HH,MaxMin[1,:],'--',**args)

        if self.fontsize > 19:
            plt.title(Name,fontsize=self.fontsize)  # Colocamos el título del gráfico
        else:
            plt.title('Ciclo Diurno de ' + VarLL + ' en ' + Name,fontsize=self.fontsize)  # Colocamos el título del gráfico
        plt.ylabel(VarL,fontsize=self.fontsize)  # Colocamos la etiqueta en el eje x
        if flagIng:
            plt.xlabel('Hours',fontsize=self.fontsize)  # Colocamos la etiqueta en el eje y
        else:
            plt.xlabel('Horas',fontsize=self.fontsize)  # Colocamos la etiqueta en el eje y
        ax = plt.gca()
        plt.xlim([0,23])
        if not(vlimits is None):
            plt.ylim([vlimits[0],vlimits[1]])

        # The minor ticks are included
        xTL = ax.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
        minorLocatorx = MultipleLocator(MxL)
        yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # minorLocatory value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().xaxis.set_minor_locator(minorLocatorx)
        plt.gca().yaxis.set_minor_locator(minorLocatory)
        plt.tight_layout()
        plt.savefig(PathImg + 'CTErr_' + NameA+'.png',format='png',dpi=self.dpi )
        plt.close('all')

    def DalyCyclePer(self,HH,zM,zMed,P10,P90,PV90,PV10,VarL,VarLL,Name,NameA,PathImg):
        '''
            DESCRIPTION:
        
        Esta función permite hacer las gráficas del ciclo diurno con área de
        porcentaje.
        _________________________________________________________________________

            INPUT:
        + HH: Vector de horas.
        + zM: Media.
        + zMed: Mediana.
        + P10: Percentil inferior.
        + P90: Percentil superior.
        + PV90: Valor que se quiere tomar para el percentil superrior.
        + PV10: Valor que se quiere tomar para el percentil inferior.
        + VarL: Label de la variable con unidades, por ejemplo Precipitación (mm).
        + VarLL: Label de la variable sin unidades.
        + Name: Nombre de la Estación.
        + NameA: Nombre del archivo.
        + PathImg: Ruta donde se quiere guardar el archivo.
        _________________________________________________________________________
        
            OUTPUT:
        Esta función arroja una gráfica y la guarda en la ruta desada.
        '''
        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)

        # Se genera la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
            ,'xtick.labelsize': 15,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': 16,'ytick.major.size': 12,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})
        plt.tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='on',direction='in')
        plt.tick_params(axis='x',which='major',direction='inout')
        plt.tick_params(axis='y',which='both',left='on',right='off',\
            labelleft='on')
        plt.tick_params(axis='y',which='major',direction='inout') 
        plt.grid()
        # Argumentos que se deben incluir color=C,label=VarLL,lw=1.5
        plt.plot(HH,zM,'k-',label='Media',lw=1.5)
        plt.plot(HH,zMed,'k--',label='Mediana',lw=1.5)
        plt.fill_between(HH,P10,P90,color='silver',label=r'P$_{%s}$ a P$_{%s}$' %(PV10,PV90))
        plt.plot(HH,P10,'w-',lw=0.0001)
        plt.legend(loc=9)
        plt.xlim([0,23])
        ax = plt.gca()
        yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # minorLocatory value
        plt.ylim([yTL[0]-2*MyL,yTL[-1]+2*MyL])
        plt.title('Ciclo Diurno de ' + VarLL + ' en ' + Name,fontsize=16 )  # Colocamos el título del gráfico
        plt.ylabel(VarL,fontsize=16)  # Colocamos la etiqueta en el eje x
        plt.xlabel('Horas',fontsize=16)  # Colocamos la etiqueta en el eje y
        # The minor ticks are included
        ax = plt.gca()
        xTL = ax.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
        minorLocatorx = MultipleLocator(MxL)
        yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # minorLocatory value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().xaxis.set_minor_locator(minorLocatorx)
        plt.gca().yaxis.set_minor_locator(minorLocatory)
        plt.tight_layout()
        plt.savefig(PathImg + 'CTPer_' + NameA+'.png',format='png',dpi=self.dpi )
        plt.close('all')
        return

    def DalyAnCycle(self,MonthsM,PathImg='',Name='',NameSt='',VarL='',
            VarLL='',VarInd='',FlagMan=False,vmax=None,vmin=None,
            Flagcbar=True,FlagIng=False,FlagSeveral=True):
        '''
        DESCRIPTION:
            
            With this function you can get the diurnal cycle of the
            percentage of precipitation along the year.

            Also plots the graph.
        _______________________________________________________________________

        INPUT:
            :param MonthsMM: Variable of days for months.
            :param PathImg: Path for the Graph.
            :param Name: Name of the graph.
            :param NameSt: Name of the Name of the Sation.
            :param FlagMan: Flag to get the same cbar values.
            :param vmax: Maximum value of the cbar.
            :param vmin: Maximum value of the cbar.
            :param VarL: Variable label with units.
            :param VarLL: Variable label without units.
            :param VarInd: Variable indicative (example: Prec).
            :param Flagcbar: Flag to plot the cbar.
            :param FlagIng: Flag to convert labels to english.
            :param FlagSeveral: Flag to make big labels.
        _______________________________________________________________________
        
        OUTPUT:
            This function plots and saves a graph.
            - MonthsMP: Directory with the monthly values of percentage
            - PorcP: Directory with the 
        '''

        if VarInd == '':
            VarInd = self.VarInd[VarLL.lower()]
        
        if FlagIng:
            MM = ['Jan','Mar','May','Jul','Sep','Nov','Jan']
        else:
            MM = ['Ene','Mar','May','Jul','Sep','Nov','Ene']

        # Input data from 7 to 7 and from Jan to Jan
        x = np.arange(0,24)
        x3 = np.arange(0,25)
        ProcP2 = np.hstack((MonthsM[:,7:],MonthsM[:,:7]))
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
        if vmax != None:
            v = np.linspace(vmin, vmax, 9, endpoint=True)
            bounds = np.arange(vmin,vmax+0.1,1)

        x2 = np.hstack((x2,x2[0]))
        x22 = np.array([x2[i] for i in range(0,len(x2),3)])

        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)

        # Labels fontsize
        if FlagSeveral:
            LabFont = 28
            LabTi = 32
        else:
            LabFont = 16
            LabTi = 18

        # Se genera la gráfica
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': LabFont,'font.family': 'sans-serif'\
            ,'font.sans-serif': 'Arial'\
            ,'xtick.labelsize': LabFont,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': LabFont,'ytick.major.size': 6,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        plt.tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='on',direction='out')
        plt.tick_params(axis='x',which='major',direction='out')
        plt.tick_params(axis='y',which='both',left='on',right='off',\
            labelleft='on',direction='out')
        plt.tick_params(axis='y',which='major',direction='out') 
        plt.grid()

        if FlagMan:
            plt.contourf(x3,np.arange(1,14),ProcP3,v,vmax=vmax,vmin=vmin,cmap=cm.jet)
        else:
            plt.contourf(x3,np.arange(1,14),ProcP3,cmap=cm.jet)
        plt.title(NameSt,fontsize=LabTi)  # Colocamos el título del gráfico
        # plt.ylabel('Meses',fontsize=15)  # Colocamos la etiqueta en el eje x
        # plt.xlabel('Horas',fontsize=15)  # Colocamos la etiqueta en el eje y
        axs = plt.gca()
        axs.yaxis.set_ticks(np.arange(1,14,2))
        axs.set_yticklabels(MM)
        axs.xaxis.set_ticks(np.arange(0,25,3))
        axs.set_xticklabels(x22)
        plt.tight_layout()
        if Flagcbar:
            if FlagMan:
                cbar = plt.colorbar(boundaries=bounds,ticks=v)
            else:
                cbar = plt.colorbar()
            cbar.set_label(VarL)
        plt.gca().invert_yaxis()
        # plt.legend(loc=1)
        plt.grid()
        # The minor ticks are included
        ax = plt.gca()
        xTL = ax.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/3 # minorLocatorx value
        minorLocatorx = MultipleLocator(MxL)
        yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/2 # minorLocatory value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().xaxis.set_minor_locator(minorLocatorx)
        plt.gca().yaxis.set_minor_locator(minorLocatory)
        plt.tight_layout()
        plt.savefig(PathImg + 'T'+VarInd+'_' + Name+'.png',format = 'png',dpi=self.dpi )
        plt.close('all')

    def AnnualCycle(self,MesM,MesE=None,VarL='',VarLL='',Name='',NameA='Test',
            PathImg='',AH=False,flagE=True,colors=None,labels=None,v=None,flagIng=False,**args):
        '''
            DESCRIPTION:
        
        Esta función permite hacer las gráficas del ciclo anual

        _________________________________________________________________________

            INPUT:
        :param MesM: Valor medio del mes, debe ir desde enero hasta diciembre.
        :param MesE: Barras de error de los valores.
        :param VarL: Labes de la variable con unidades.
        :param VarLL: Label de la variable sin unidades.
        :param Name: Nombre de la Estación.
        :param NameA: Nombre del archivo.
        :param PathImg: Ruta donde se quiere guardar el archivo.
        :param AH: Este es un flag para saber si se hace el gráfico con el año hidrológico.
        :param flagE: A boolean, flag to include Errors.
        :param colors: a list, list with colors of the lines.
        :param labels: a list, list with the labels of the lines.
        :param v: a list, list with two values with the min and max.
        :param flagIng: A boolean, flag to change titles for english.
        :param **args: Argumentos adicionales para la gráfica, como color o ancho de la línea.
        _________________________________________________________________________
        
            OUTPUT:
        Esta función arroja una gráfica y la guarda en la ruta desada.
        '''
        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)
        # Vector de meses
        if flagIng:
            Months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        else:
            Months = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dec']
        if AH:
            Months2 = Months[5:]+Months[:5]
            if len(MesM.shape) > 1:
                MesM2 = np.empty(MesM.shape)*np.nan
                for i in range(MesM.shape[0]):
                    MesM2[i] = np.hstack((MesM[i,5:],MesM[i,:5]))
            else:
                MesM2 = np.hstack((MesM[5:],MesM[:5]))
            if flagE:
                if len(MesE.shape) > 1:
                    MesE2 = np.empty(MesE.shape)*np.nan
                    for i in range(MesM.shape[0]):
                        MesE2 = np.hstack((MesE[i,5:],MesE[i,:5]))
                else:
                    MesE2 = np.hstack((MesE[5:],MesE[:5]))
        else:
            Months2 = Months
            MesM2 = MesM
            if flagE:
                MesE2 = MesE

        # Se genera la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
            ,'xtick.labelsize': 15,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': 16,'ytick.major.size': 12,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})
        plt.tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='on',direction='in')
        plt.tick_params(axis='x',which='major',direction='inout')
        plt.tick_params(axis='y',which='both',left='on',right='off',\
            labelleft='on')
        plt.tick_params(axis='y',which='major',direction='inout') 
        plt.grid()
        # Argumentos que se deben incluir color=C,label=VarLL,lw=1.5
        if flagE:
            if len(MesM2.shape) > 1:
                for i in range(MesM2.shape[0]):
                    if colors is None:
                        if labels is None:
                            plt.errorbar(np.arange(1,13),MesM2[i],yerr=MesE2[i],fmt='-')
                        else:
                            plt.errorbar(np.arange(1,13),MesM2[i],yerr=MesE2[i],fmt='-',label=labels[i])
                    else:
                        if labels is None:
                            plt.errorbar(np.arange(1,13),MesM2[i],yerr=MesE2[i],fmt='-',color=colors[i])
                        else:
                            plt.errorbar(np.arange(1,13),MesM2[i],yerr=MesE2[i],fmt='-',color=colors[i],label=labels[i])
            else:
                plt.errorbar(np.arange(1,13),MesM2,yerr=MesE2,fmt='-',**args)
        else:
            if len(MesM2.shape) > 1:
                for i in range(MesM2.shape[0]):
                    if colors is None:
                        if labels is None:
                            plt.plot(np.arange(1,13),MesM2[i])
                        else:
                            plt.plot(np.arange(1,13),MesM2[i],label=labels[i])
                    else:
                        if labels is None:
                            plt.plot(np.arange(1,13),MesM2[i],color=colors[i])
                        else:
                            plt.plot(np.arange(1,13),MesM2[i],color=colors[i],label=labels[i])
            else:
                plt.plot(np.arange(1,13),MesM2,**args)
        if flagIng:
            plt.title('Annual Cycle of '+ VarLL +' in ' + Name,fontsize=15 )  # Colocamos el título del gráfico
            plt.xlabel('Months',fontsize=16)  # Colocamos la etiqueta en el eje y
        else:
            plt.title('Ciclo anual de '+ VarLL +' en ' + Name,fontsize=15 )  # Colocamos el título del gráfico
            plt.xlabel('Meses',fontsize=16)  # Colocamos la etiqueta en el eje y
        plt.ylabel(VarL,fontsize=16)  # Colocamos la etiqueta en el eje x
        if not(v is None):
            plt.ylim(v)
        # The minor ticks are included
        ax = plt.gca()
        yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # minorLocatory value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().yaxis.set_minor_locator(minorLocatory)
        ax.set_xlim([0.5,12.5])
        plt.xticks(np.arange(1,13), Months2) # Se cambia el valor de los ejes
        if not(labels is None):
            plt.legend(loc=0)
        plt.tight_layout()
        plt.savefig(PathImg + 'CAErr_' + NameA+'.png',format='png',dpi=self.dpi )
        plt.close('all')

    def AnnualCycleBoxPlot(self,MesM,VarL,VarLL,Name,NameA,PathImg,AH=False):
        '''
            DESCRIPTION:
        
        Esta función permite hacer las gráficas del ciclo anual

        _________________________________________________________________________

            INPUT:
        + MesM: Valor medio del mes, debe ir desde enero hasta diciembre.
        + VarL: Labes de la variable con unidades.
        + VarLL: Label de la variable sin unidades.
        + Name: Nombre de la Estación.
        + NameA: Nombre del archivo.
        + PathImg: Ruta donde se quiere guardar el archivo.
        + AH: Este es un flag para saber si se hace el gráfico con el año hidrológico.
        + **args: Argumentos adicionales para la gráfica, como color o ancho de la línea.
        _________________________________________________________________________
        
            OUTPUT:
        Esta función arroja una gráfica y la guarda en la ruta desada.
        '''
        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)
        # Vector de meses
        Months = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dec']
        if AH:
            Months2 = Months[5:]+Months[:5]
            MesM2 = np.hstack((MesM[5:],MesM[:5]))
        else:
            Months2 = Months
            MesM2 = MesM

        MesMM = np.reshape(MesM2,(-1,12))

        # Se genera la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
            ,'xtick.labelsize': 15,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': 16,'ytick.major.size': 12,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})
        plt.tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='on',direction='in')
        plt.tick_params(axis='x',which='major',direction='inout')
        plt.tick_params(axis='y',which='both',left='on',right='off',\
            labelleft='on')
        plt.tick_params(axis='y',which='major',direction='inout') 
        plt.grid()
        ax = plt.gca()
        ii = 0
        for i in range(12):
            PMM2 = MesMM[:,i]
            NoNaN = ~np.isnan(PMM2)
            
            bp = plt.boxplot(PMM2[NoNaN], positions=[ii],widths=0.6,showbox=False,sym='o')

            # Se cambia cada caja
            plt.setp(bp['boxes'], color='#B9CDE5')
            plt.setp(bp['whiskers'], color='black',linestyle='-')
            plt.setp(bp['medians'], color='#B9CDE5',linestyle='-')
            plt.setp(bp['fliers'], color='white',markerfacecolor='black', marker='o'\
                ,markersize=3,alpha=0.8)
            # Estadísticos
            median = np.median(PMM2[NoNaN])
            upper_quartile = np.percentile(PMM2[NoNaN], 75)
            lower_quartile = np.percentile(PMM2[NoNaN], 25)
            Dif1 = median-lower_quartile
            Dif2 = upper_quartile-median
            ax.add_patch(patches.Rectangle((ii-0.3, lower_quartile), 0.6, Dif1, fill=True,\
                facecolor='#2B65AB',edgecolor="none"))
            ax.add_patch(patches.Rectangle((ii-0.3, median), 0.6, Dif2, fill=True,\
                facecolor='#B9CDE5',edgecolor="none"))
            
            ii += 1
        # Se calcula el promedio
        PMMM = np.nanmean(MesMM,axis=0)
        plt.plot(np.arange(0,ii,1),PMMM,'+--',color='k',lw=1)
        plt.title('Ciclo anual de '+ VarLL +' en ' + Name,fontsize=16 )  # Colocamos el título del gráfico
        plt.ylabel(VarL,fontsize=16)  # Colocamos la etiqueta en el eje x
        plt.xlabel('Meses',fontsize=16)  # Colocamos la etiqueta en el eje y
        # The minor ticks are included
        ax = plt.gca()
        yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # minorLocatory value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().yaxis.set_minor_locator(minorLocatory)
        ax.set_xlim([-0.5,11.5])
        plt.xticks(np.arange(0,12), Months2) # Se cambia el valor de los ejes
        plt.tight_layout()
        plt.savefig(PathImg + 'CAErr_' + NameA+'.png',format='png',dpi=self.dpi )
        plt.close('all')

    def AnnualS(self,Fecha,AnM,AnE,VarL,VarLL,Name,NameA,PathImg,**args):
        '''
            DESCRIPTION:
        
        Esta función permite hacer la gráfica de la serie anual

        _________________________________________________________________________

            INPUT:
        + AnM: Valor medio del An, debe ir desde enero hasta diciembre.
        + AnE: Barras de error de los valores.
        + VarL: Labes de la variable con unidades.
        + VarLL: Label de la variable sin unidades.
        + Name: Nombre de la Estación.
        + NameA: Nombre del archivo.
        + PathImg: Ruta donde se quiere guardar el archivo.
        + **args: Argumentos adicionales para la gráfica, como color o ancho de la línea.
        _________________________________________________________________________
        
            OUTPUT:
        Esta función arroja una gráfica y la guarda en la ruta desada.
        '''
        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)

        # Se genera la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
            ,'xtick.labelsize': 15,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': 16,'ytick.major.size': 12,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})
        plt.tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='on',direction='in')
        plt.tick_params(axis='x',which='major',direction='inout')
        plt.tick_params(axis='y',which='both',left='on',right='off',\
            labelleft='on')
        plt.tick_params(axis='y',which='major',direction='inout') 
        plt.grid()
        # Argumentos que se deben incluir color=C,label=VarLL,lw=1.5
        plt.errorbar(Fecha,AnM,yerr=AnE,fmt='-',**args)
        plt.title('Serie Anual de '+ VarLL +' en ' + Name,fontsize=16 )  # Colocamos el título del gráfico
        plt.ylabel(VarL,fontsize=16)  # Colocamos la etiqueta en el eje x
        plt.xlabel('Años',fontsize=16)  # Colocamos la etiqueta en el eje y
        # The minor ticks are included
        ax = plt.gca()
        yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # minorLocatory value
        minorLocatory = MultipleLocator(MyL)
        ax.set_xlim([Fecha[0]-timedelta(60),Fecha[-1]+timedelta(30)])
        plt.gca().yaxis.set_minor_locator(minorLocatory)
        plt.tight_layout()
        plt.savefig(PathImg + 'SAnErr_' + NameA+'.png',format='png',dpi=self.dpi )
        plt.close('all')

    def GCorr(self,CP,VarL1,VarL2,Names,NameA,PathImg='',**args):
        '''
            DESCRIPTION:
        
        Esta función permite hacer las gráficas de correlación.

        _________________________________________________________________________

            INPUT:
        + CP: Valores de correlación.
        + VarL1: Label de la primera variable.
        + VarL2: Label de la segunda variable.
        + Names: Nombre de las estaciones.
        + NamesA: Nombre de la gráfica.
        + PathImg: Ruta donde se quiere guardar el archivo.
        + **args: Argumentos adicionales para la gráfica, como color o ancho de la línea.
        _________________________________________________________________________
        
            OUTPUT:
        Esta función arroja una gráfica y la guarda en la ruta desada.
        '''
        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)

        N = len(CP)

        # Parámetros de la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
            ,'xtick.labelsize': 15,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': 16,'ytick.major.size': 6,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})
        plt.tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='on',direction='out')
        plt.tick_params(axis='y',which='both',left='on',right='off',\
            labelleft='on')
        plt.tick_params(axis='y',which='major',direction='out') 
        # Se realiza la figura 
        p1 = plt.bar(np.arange(0,N),CP,width=0.8,**args)
        plt.title('Correlación entre la ' + VarL1 + ' y la ' + VarL2,fontsize=16 )  # Colocamos el título del gráfico
        plt.xlabel(u'Estaciones',fontsize=16)  # Colocamos la etiqueta en el eje x
        plt.ylabel('Correlación',fontsize=16)  # Colocamos la etiqueta en el eje y
        plt.xlim([-0.2,N])
        axes = plt.gca()
        axes.xaxis.set_ticks(np.arange(0.4,N+0.4,1))
        axes.set_xticklabels(Names)
        yTL = axes.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # minorLocatory value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().yaxis.set_minor_locator(minorLocatory)
        plt.tight_layout()
        plt.savefig(PathImg + 'Correlaciones_'+ NameA +'.png',format='png',dpi=self.dpi )
        plt.close('all')

    def AnnualCycleCompS(self,V1,V2,Lab1,Lab2,VarLL,Var_LUn,Var='',PathImg=''):
        '''
            DESCRIPTION:
        
        Esta función permite comparar dos ciclos anuales.
        ________________________________________________________________________

            INPUT:
        + V1: Valores 1.
        + V2: Valores 2.
        + Lab1: Nombre de la serie 1 que se está comparando.
        + Lab2: Nombre de la serie 2 que se está comparando.
        + VarLL: Variable sin unidades
        + Var_LUn: Label de la variable con unidades, por ejemplo Precipitación (mm).
        + Var: Nombre de la imagen.
        + PathImg: Ruta donde se quiere guardar el archivo.
        + **args: Argumentos adicionales para la gráfica, como color o ancho de la línea.
        _________________________________________________________________________
        
            OUTPUT:
        Esta función arroja una gráfica y la guarda en la ruta desada.
        '''

        # Se calcula el ciclo anual de cada variable
        V1M = np.reshape(V1,(-1,12))
        V1MM = np.nanmean(V1M,axis=0)
        V2M = np.reshape(V2,(-1,12))
        V2MM = np.nanmean(V2M,axis=0)
        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)
        # Vector de meses
        Months = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dec']
        Months2 = Months

        # Se genera la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
            ,'xtick.labelsize': 15,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': 16,'ytick.major.size': 12,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})
        plt.tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='on',direction='in')
        plt.tick_params(axis='x',which='major',direction='inout')
        plt.tick_params(axis='y',which='both',left='on',right='off',\
            labelleft='on')
        plt.tick_params(axis='y',which='major',direction='inout') 
        plt.grid()
        # Argumentos que se deben incluir color=C,label=VarLL,lw=1.5
        plt.plot(np.arange(1,13),V1MM, 'r-', lw = 1,label=Lab1)
        plt.plot(np.arange(1,13),V2MM, 'b-', lw = 1,label=Lab2)
        plt.legend(loc=0)
        plt.title('Comparación del ciclo anual de '+ VarLL,fontsize=16 )  # Colocamos el título del gráfico
        plt.ylabel(Var_LUn,fontsize=16)  # Colocamos la etiqueta en el eje x
        plt.xlabel('Meses',fontsize=16)  # Colocamos la etiqueta en el eje y
        # The minor ticks are included
        ax = plt.gca()
        yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # minorLocatory value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().yaxis.set_minor_locator(minorLocatory)
        ax.set_xlim([0.5,12.5])
        plt.xticks(np.arange(1,13), Months2) # Se cambia el valor de los ejes
        plt.tight_layout()
        plt.savefig(PathImg + Var+'.png',format='png',dpi=self.dpi )
        plt.close('all')

    def CompS(self,Date,V1,V2,Lab1,Lab2,Var_LUn,Var='',flagT=True,v='',PathImg=''):
        '''
            DESCRIPTION:
        
        Esta función permite comparar dos series de datos
        ________________________________________________________________________

            INPUT:
        + Date: Vector de fechas en formato date.
        + V1: Valores 1.
        + V2: Valores 2.
        + Lab1: Nombre de la serie 1 que se está comparando.
        + Lab2: Nombre de la serie 2 que se está comparando.
        + Var_LUn: Label de la variable con unidades, por ejemplo Precipitación (mm).
        + Var: Nombre de la imagen.
        + flagT: Flag para saber si se incluye el título.
        + v: Variable que se está comparando.
        + PathImg: Ruta donde se quiere guardar el archivo.
        + **args: Argumentos adicionales para la gráfica, como color o ancho de la línea.
        _________________________________________________________________________
        
            OUTPUT:
        Esta función arroja una gráfica y la guarda en la ruta desada.
        '''
        warnings.filterwarnings('ignore')
        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)

        # Se genera la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
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
        # Se realiza la figura 
        plt.plot(Date,V1, 'r.', lw = 1,label=Lab1)
        plt.plot(Date,V2, 'b.', lw = 1,label=Lab2)
        # Se arreglan los ejes
        axes = plt.gca()
        plt.xlim([min(Date),max(Date)]) # Incluyen todas las fechas
        # Se incluyen los valores de los minor ticks
        yTL = axes.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # Minor tick value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().yaxis.set_minor_locator(minorLocatory)
        # Se cambia el label de los ejes
        xTL = axes.xaxis.get_ticklocs() # List of position in x
        Labels2 = self.monthlab(xTL)
        plt.xticks(xTL, Labels2) # Se cambia el label de los ejes
        # Se rotan los ejes
        for tick in plt.gca().get_xticklabels():
            tick.set_rotation(45)
        # Labels
        if flagT:
            plt.title('Comparación de '+v,fontsize=18)
        plt.ylabel(Var_LUn,fontsize=16)
        plt.legend(loc=0)
        # Se arregla el espaciado de la figura
        plt.tight_layout()
        # Se guarda la figura
        plt.savefig(PathImg +Var +'_Comp' + '.png',format='png',dpi=self.dpi)
        plt.close('all')

        # Se genera el gráfico de los errores de estimación
        Err = V1-V2
        ErrM = np.nanmean(Err)

        # Se genera la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
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
        # Se realiza la figura 
        plt.plot(Date,Err, 'k.', lw = 1)
        plt.plot([Date[0],Date[-1]],[ErrM,ErrM], 'k--', lw = 1)
        # Se arreglan los ejes
        axes = plt.gca()
        plt.xlim([min(Date),max(Date)]) # Incluyen todas las fechas
        # Se incluyen los valores de los minor ticks
        yTL = axes.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # Minor tick value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().yaxis.set_minor_locator(minorLocatory)
        # Se cambia el label de los ejes
        xTL = axes.xaxis.get_ticklocs() # List of position in x
        Labels2 = self.monthlab(xTL)
        plt.xticks(xTL, Labels2) # Se cambia el label de los ejes
        # Se rotan los ejes
        for tick in plt.gca().get_xticklabels():
            tick.set_rotation(45)
        # Labels
        if flagT:
            plt.title('Error en la medida de '+v,fontsize=18)
        plt.ylabel(Var_LUn,fontsize=16)
        # Se calcula la correlación
        # q = ~(np.isnan(V1) | np.isnan(V2))
        # CCP,sig = st.pearsonr(V1[q],V2[q])
        # plt.text(Date[5],yTL[1], r'Correlación de Pearson:', fontsize=14)
        # gg = 5+680
        # if sig <= 0.05:
        #   plt.text(Date[gg],yTL[1], r'%s' %(round(CCP,3)), fontsize=14,color='blue')
        # else:
        #   plt.text(Date[gg],yTL[1], r'%s' %(round(CCP,3)), fontsize=14,color='red')
        # Se arregla el espaciado de la figura
        plt.tight_layout()
        # Se guarda la figura
        plt.savefig(PathImg + Var+'_Err' + '.png',format='png',dpi=self.dpi)
        plt.close('all')

        # the histogram of the data
        q = ~np.isnan(Err)
        # Se encuentra el histograma
        DH,DBin = np.histogram(Err[q],bins=30); [float(i) for i in DH]
        DH = DH/float(DH.sum())*100;

        # Se organizan los valores del histograma
        widthD = 1 * (DBin[1] - DBin[0])
        centerD = (DBin[:-1] + DBin[1:]) / 2

        # Se encuentran los diferentes momentos estadísticos
        A = np.nanmean(Err)
        B = np.nanstd(Err)

        # Tamaño de la Figura
        fH=25 # Largo de la Figura
        fV = fH*(2.0/3.0) # Ancho de la Figura

        # Se incluye el histograma y el diagrama de dispersión
        fig, axs = plt.subplots(1,2, figsize=DM.cm2inch(fH,fV), facecolor='w', edgecolor='k')
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
            ,'xtick.labelsize': 15,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': 16,'ytick.major.size': 12,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})
        axs = axs.ravel() # Para hacer un loop con los subplots
        axs[0].tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='on',direction='out')
        axs[0].tick_params(axis='y',which='both',left='on',right='off',\
            labelleft='on')
        axs[0].tick_params(axis='y',which='major',direction='inout') 
        axs[0].grid()
        axs[1].tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='on',direction='out')
        axs[1].tick_params(axis='y',which='both',left='on',right='off',\
            labelleft='on')
        axs[1].tick_params(axis='y',which='major',direction='inout') 
        axs[1].grid()

        # n, bins, patches = axs[0].hist(E_MM,bins=30, normed=1, facecolor='blue', alpha=0.5)
        p1 = axs[0].bar(DBin[:-1],DH,color='dodgerblue',width=widthD,edgecolor="none")#,edgecolor='none')
        # Se cambia el valor de los ejes.
        #axs[0].set_xticks(centerD) # Se cambia el valor de los ejes

        # add a 'best fit' line
        axs[0].set_title('Histograma del Error de la medida',fontsize=16)
        axs[0].set_xlabel(Var_LUn,fontsize=16)
        axs[0].set_ylabel('Probabilidad',fontsize=16)
        # Se incluyen los valores de los minor ticks
        yTL = axs[0].yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # Minor tick value
        minorLocatory = MultipleLocator(MyL)
        axs[0].yaxis.set_minor_locator(minorLocatory)
        # Se agrega la línea de promedio y desviación estándar
        axs[0].plot([A,A],[0,yTL[-1]],'k')
        axs[0].plot([A+B,A+B],[0,yTL[-1]],'k--')
        axs[0].plot([A-B,A-B],[0,yTL[-1]],'k--')

        # Diagrama de dispersión
        
        axs[1].scatter(V1, V2, linewidth='0')
        axs[1].set_title('Diagrama de Dispersión',fontsize=16)
        axs[1].set_xlabel(Lab1 + ' ' + Var_LUn,fontsize=16)
        axs[1].set_ylabel(Lab2  + ' ' + Var_LUn,fontsize=16)
        axs[1].set_ylim([np.nanmin(V2)-2,np.nanmax(V2)+2])
        axs[1].set_xlim([np.nanmin(V1)-2,np.nanmax(V1)+2])
        x = np.linspace(*axs[1].get_xlim())
        axs[1].plot(x,x, 'k-')
        yTL = axs[1].yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # Minor tick value
        minorLocatory = MultipleLocator(MyL)
        axs[1].yaxis.set_minor_locator(minorLocatory)

        xTL = axs[1].xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # Minor tick value
        minorLocatorx = MultipleLocator(MxL)
        axs[1].xaxis.set_minor_locator(minorLocatorx)
        plt.tight_layout()
        plt.savefig(PathImg + Var+'_Hist' + '.png',format='png',dpi=self.dpi)
        plt.close('all')

    def PorHid(self,PorP,PorI,PorE,Var='',flagT=True,Names='',PathImg=''):
        '''
            DESCRIPTION:
        
        Esta función permite hacer las gráficas de datos faltantes mensuales.

        _________________________________________________________________________

            INPUT:
        + Date: Vector de fechas en formato date.
        + Value: Vector de valores de lo que se quiere graficar.
        + Var: Nombre de la imagen.
        + flagT: Flag para saber si se incluye el título.
        + Var_L: Label de la variable sin unidades.
        + Names: Nombre de la estación para el título.
        + PathImg: Ruta donde se quiere guardar el archivo.
        + **args: Argumentos adicionales para la gráfica, como color o ancho de la línea.
        _________________________________________________________________________
        
            OUTPUT:
        Esta función arroja una gráfica y la guarda en la ruta desada.
        '''

        Months2 = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dec']
        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)

        # Parámetros de la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
            ,'xtick.labelsize': 15,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': 16,'ytick.major.size': 6,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})
        plt.tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='on',direction='out')
        plt.tick_params(axis='y',which='both',left='on',right='off',\
            labelleft='on')
        plt.tick_params(axis='y',which='major',direction='out') 
        plt.grid()
        # Se realiza la figura (,edgecolor='none')
        p1 = plt.bar(np.arange(0.5,12),PorP,color='#006600',width=1) # Precipitación
        p2 = plt.bar(np.arange(0.5,12),PorI,color='blue',bottom=PorP,width=1) # Interceptación
        p3 = plt.bar(np.arange(0.5,12),-PorE,color='red',bottom=0,width=1) # Evapotranspiración
        plt.plot([0,13],[0,0],'k-')
        # Se arreglan los ejes
        axes = plt.gca()
        axes.set_ylim([-100,100])
        # Se cambia el label de los eje
        ax = plt.gca()
        yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # minorLocatory value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().yaxis.set_minor_locator(minorLocatory)
        plt.xticks(np.arange(1,13), Months2) # Se cambia el valor de los ejes
        ax.set_xlim([0,13])
        # plt.xticks(xTL, Labels2) # Se cambia el valor de los ejes
        plt.legend((p1[0],p2[0],p3[0]), ('Precipitación','Interceptación','Evapotranspiración'),loc=4\
            ,fontsize = 14)
        # Labels
        if flagT:
            # Título
            plt.title('Porcentajes hidrológicos en la cuenca ' + Names,fontsize=15 )
        plt.xlabel(u'Mes',fontsize=16)  # Colocamos la etiqueta en el eje x
        plt.ylabel('Porcentaje',fontsize=16)  # Colocamos la etiqueta en el eje y
        plt.tight_layout()
        plt.savefig(PathImg + Var +'Hidro_Por_Mens' + '.png',format='png',dpi=self.dpi )
        plt.close('all')

    def EventPres(self,Date,Value,Datest,PCxi,PCxf,PCxfB,Dxi,Dxf,Name='',PathImg='',Nameout=''):
        '''
        DESCRIPTION:
        
            Esta función permite .

        _________________________________________________________________________

        INPUT:
            + Date: Vector de fechas en formato date.
            + Value: Vector de valores de lo que se quiere graficar.
            + Datest: Fecha de comienzan del evento de lluvia en string.
            + PCxi: Posición del mínimo de presión.
            + PCxf: Posición del máximo de presión durante el evento.
            + PCxfB: Posición del máximo de presión antes el evento.
            + Dxi: Posición del inicio del evento.
            + Dxf: Posición del final del evento.
            + Name: Nombre de la estación para el título.
            + PathImg: Ruta donde se quiere guardar el archivo.
            + Nameout: Ruta y nombre del documento que se quiere guardar
        _________________________________________________________________________
        
            OUTPUT:
        Esta función arroja una gráfica y la guarda en la ruta desada.
        '''
        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)

        # Se genera la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
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
        a11 = plt.plot(Date,Value,'-k', label = 'Presión')
        plt.title(Name+r" Evento " + Datest,fontsize=16)
        plt.xlabel("Tiempo",fontsize=16)
        plt.ylabel('Presión [hPa]',fontsize=16)

        if ~np.isnan(PCxi):
            L1 = plt.plot([Date[PCxi],Date[PCxi]],[np.nanmin(Value),np.nanmax(Value)],'--b', label = 'Min Pres') # Punto mínimo
            if ~np.isnan(PCxfB):
                L2 = plt.plot([Date[PCxfB],Date[PCxfB]],[np.nanmin(Value),np.nanmax(Value)],'--r', label = 'Max Pres Antes') # Punto máximo B
            L3 = plt.plot([Date[PCxf],Date[PCxf]],[np.nanmin(Value),np.nanmax(Value)],'--g', label = 'Max Pres Después') # Punto máximo A

        # Líneas para la precipitación
        L4 = plt.plot([Date[Dxi],Date[Dxi]],[np.nanmin(Value),np.nanmax(Value)],'-.b', label = 'Inicio Prec') # Inicio del aguacero
        L5 = plt.plot([Date[Dxf],Date[Dxf]],[np.nanmin(Value),np.nanmax(Value)],'-.g', label = 'Fin Prec') # Fin del aguacero

        # added these three lines
        if ~np.isnan(PCxfB):
            lns = a11+L1+L2+L3+L4+L5
        else:
            lns = a11+L1+L3+L4+L5
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc=0,fontsize=12)
        
        # Axis
        axes = plt.gca()
        xTL = axes.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
        minorLocatorx = MultipleLocator(MxL)
        yTL = axes.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # minorLocatory value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().xaxis.set_minor_locator(minorLocatorx)
        plt.gca().yaxis.set_minor_locator(minorLocatory)

        plt.tight_layout()
        plt.savefig(Nameout + '.png',format='png',dpi=self.dpi )
        plt.close('all')

    def SPvDPPotAdj(self,DurPrec,PresRate,Name='',PathImg='',Nameout='',FlagA=True,FlagAn=False):
        '''
            DESCRIPTION:
        
        Esta función permite hacer las gráficas de datos faltantes mensuales.

        _________________________________________________________________________

            INPUT:
        + DurPrec: Vector de duración de las tormentas en horas.
        + PresRate: Vector de valores tasa de cambio de presión.
        + Name: Nombre de la estación para el título.
        + PathImg: Ruta donde se quiere guardar el archivo.
        + FlagA: Indicador si se quiere realizar el ajuste.
        + FlagAn: Indicador para anotar el número del punto.
        _________________________________________________________________________
        
            OUTPUT:
        Esta función arroja una gráfica y la guarda en la ruta desada.
        '''

        # Se calcula el ajuste
        if FlagA:
            # Se importa el paquete de Fit
            from CFitting import CFitting
            CF = CFitting()

            # Se realiza la regresión
            Coef, perr,R2 = CF.FF(DurPrec,PresRate,2)

            # Se toman los datos para ser comparados posteriormente
            DD,PP = utl.NoNaN(DurPrec,PresRate,False)
            N = len(DD)
            a = Coef[0]
            b = Coef[1]
            desv_a = perr[0]
            desv_b = perr[1]
            # Se garda la variable
            CC = np.array([N,a,b,desv_a,desv_b,R2])
            
            
            # Se realiza el ajuste a ver que tal dió
            x = np.linspace(np.nanmin(DurPrec),np.nanmax(DurPrec),100)
            PresRateC = Coef[0]*x**Coef[1]

        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)

        # Se genera la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
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
        plt.scatter(DurPrec,PresRate)
        plt.title('Cambios en Presión Atmosférica en '+ Name,fontsize=16)
        plt.xlabel(u'Duración de la Tormenta [h]',fontsize=16)
        plt.ylabel('Tasa de Cambio de Presión [hPa/h]',fontsize=16)

        if FlagAn:
            # Número de cada punto
            n = np.arange(0,len(DurPrec))
            for i, txt in enumerate(n):
                plt.annotate(txt, (DurPrec[i],PresRate[i]),fontsize=8)

        # Axes
        ax = plt.gca()
        xTL = ax.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
        plt.xlim([0,np.nanmax(DurPrec)+2*MxL])

        xTL = ax.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
        minorLocatorx = MultipleLocator(MxL)
        yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
        MyL = np.abs(np.abs(yTL[1])-np.abs(yTL[0]))/5 # minorLocatory value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().xaxis.set_minor_locator(minorLocatorx)
        plt.gca().yaxis.set_minor_locator(minorLocatory)

        # Se incluye el ajuste
        if FlagA:
            plt.plot(x,PresRateC,'k--')
            # Se incluye la ecuación
            if np.nanmin(PresRate) < 0:
                plt.text(xTL[-4],yTL[2]+2*MyL, r'$\Delta = %sD^{%s}$' %(round(a,3),round(b,3)), fontsize=15)
                plt.text(xTL[-4],yTL[2], r'$R^2 = %s$' %(round(R2,4)), fontsize=14)
            else:
                plt.text(xTL[-4],yTL[-2], r'$\Delta = %sD^{%s}$' %(round(a,3),round(b,3)), fontsize=15)
                plt.text(xTL[-4],yTL[-2]-2*MyL, r'$R^2 = %s$' %(round(R2,4)), fontsize=14)

        plt.tight_layout()
        plt.savefig(Nameout+'.png',format='png',dpi=self.dpi )
        plt.close('all')

        if FlagA:
            return CC

    def SPvDPPotGen(self,V1,V2,Fit='',Title='',xLabel='',yLabel='',Name='',PathImg='',FlagA=True,FlagAn=False):
        '''
        DESCRIPTION:
        
            This function allows the user to create a Scatter plot and adjust
            the best line to the data.
        _________________________________________________________________________

        INPUT:
            :param V1:       a ndarray, Vector with the X values. 
            :param V2: a ndarray, Vector With the Y values.
            :param Name:     a str, Title of the graph.
            :param PathImg:  a str, Ruta donde se quiere guardar el archivo.
            :param FlagA:    a bolean, Indicador si se quiere realizar el ajuste.
            :param FlagAn:   a bolean, Indicador para anotar el número del punto.
        _________________________________________________________________________
        
            OUTPUT:
        Esta función arroja una gráfica y la guarda en la ruta desada.
        '''

        # Se calcula el ajuste
        if FlagA:
            # Se importa el paquete de Fit
            from AnET import CFitting
            CF = CFitting()
            # Se ajusta la curva
            FitC = CF.FF(V1,V2,F=Fit)

            # Se toman los datos para ser comparados posteriormente
            DD,PP = DM.NoNaN(V1,V2,False)
            N = len(DD)
            # Se guarda la variable
            
            # Se realiza el ajuste a ver que tal dió
            x = np.linspace(np.nanmin(V1),np.nanmax(V1),100)
            VC = FitC['Function'](x, *FitC['Coef'])

        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)

        # Se genera la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
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
        plt.title(Title,fontsize=16)
        plt.xlabel(xLabel,fontsize=16)
        plt.ylabel(yLabel,fontsize=16)

        if FlagAn:
            # Número de cada punto
            n = np.arange(0,len(V1))
            for i, txt in enumerate(n):
                plt.annotate(txt, (V1[i],V2[i]),fontsize=8)

        # Axes
        ax = plt.gca()
        xTL = ax.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
        plt.xlim([np.nanmin(V1)-2*MxL,np.nanmax(V1)+2*MxL])

        xTL = ax.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
        minorLocatorx = MultipleLocator(MxL)
        yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
        MyL = np.abs(np.abs(yTL[1])-np.abs(yTL[0]))/5 # minorLocatory value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().xaxis.set_minor_locator(minorLocatorx)
        plt.gca().yaxis.set_minor_locator(minorLocatory)

        # Se incluye el ajuste
        if FlagA:
            Label = FitC['FunctionEq']+'\n'+r'$R^2=%.3f$'
            plt.plot(x,VC,'k--',label=Label %tuple(list(FitC['Coef'])+[FitC['R2']]))
            plt.legend(loc=1,fontsize=12)

        try:
            plt.tight_layout()
        except RuntimeError:
            plt.close('all')
            return
        Nameout = PathImg+Name
        plt.savefig(Nameout+'.png',format='png',dpi=self.dpi )
        plt.close('all')

        if FlagA:
            return 

    def ScatterGen(self,V1,V2,Fit='',Title='',xLabel='',yLabel='',Name='',PathImg='',FlagA=True,FlagAn=False,FlagInv=False,FlagInvAxis=False,Annotations=None,**args):
        '''
        DESCRIPTION:
        
            This function allows the user to create a Scatter plot and adjust
            the best line to the data.
        _________________________________________________________________________

        INPUT:
            :param V1:       a ndarray, Vector with the X values. 
            :param V2: a ndarray, Vector With the Y values.
            :param Name:     a str, Title of the graph.
            :param PathImg:  a str, Ruta donde se quiere guardar el archivo.
            :param FlagA:    a bolean, Indicador si se quiere realizar el ajuste.
            :param FlagAn:   a bolean, Indicador para anotar el número del punto.
            :param FlagInv:  a bolean, Indicator to make inverse adjustment.
            :param FlagInvAxis:  a bolean, Indicator to make inverse adjustment.
            :param Annotations:  a ndArray, annotations in the scatter plot.
        _________________________________________________________________________
        
        OUTPUT:
            This function saves an image.
        '''

        # Se calcula el ajuste
        if FlagA:
            # Se importa el paquete de Fit
            from AnET import CFitting
            CF = CFitting()
            if FlagInv:
                # Se ajusta la curva
                FitC = CF.FF(V2,V1,F=Fit)
                # Se realiza el ajuste a ver que tal dió
                x = np.linspace(np.nanmin(V2),np.nanmax(V2),100)
                VC = FitC['Function'](x, *FitC['Coef'])
            else:
                # Se ajusta la curva
                FitC = CF.FF(V1,V2,F=Fit)
                # Se realiza el ajuste a ver que tal dió
                x = np.linspace(np.nanmin(V1),np.nanmax(V1),100)
                VC = FitC['Function'](x, *FitC['Coef'])

        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)

        # Se genera la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
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
        plt.scatter(V1,V2,**args)
        plt.title(Title,fontsize=16)
        plt.xlabel(xLabel,fontsize=16)
        plt.ylabel(yLabel,fontsize=16)

        if FlagAn:
            if Annotations == None:
                n = np.arange(0,len(V1))
                for i, txt in enumerate(n):
                    plt.annotate(txt, (V1[i],V2[i]),fontsize=8)
            else:
                for i, txt in enumerate(Annotations):
                    plt.annotate(txt, (V1[i],V2[i]),fontsize=8)

        # Axes
        ax = plt.gca()
        xTL = ax.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
        plt.xlim([np.nanmin(V1)-2*MxL,np.nanmax(V1)+2*MxL])

        xTL = ax.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
        minorLocatorx = MultipleLocator(MxL)
        yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
        MyL = np.abs(np.abs(yTL[1])-np.abs(yTL[0]))/5 # minorLocatory value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().xaxis.set_minor_locator(minorLocatorx)
        plt.gca().yaxis.set_minor_locator(minorLocatory)

        # Se incluye el ajuste
        if FlagA:
            Label = FitC['FunctionEq']+'\n'+r'$R^2=%.3f$'
            if FlagInv:
                plt.plot(VC,x,'k--',label=Label %tuple(list(FitC['Coef'])+[FitC['R2']]))
            else:
                plt.plot(x,VC,'k--',label=Label %tuple(list(FitC['Coef'])+[FitC['R2']]))
            plt.legend(loc=1,fontsize=12)

        plt.tight_layout()
        Nameout = PathImg+Name
        plt.savefig(Nameout+'.png',format='png',dpi=self.dpi )
        plt.close('all')

        if FlagA:
            return 

    def SPAvPD(self,PresRateB,PresRateA,Name='',PathImg='',Nameout='',FlagA=False,FEn=False):
        '''
            DESCRIPTION:
        
        Esta función permite hacer las gráficas de datos faltantes mensuales.

        _________________________________________________________________________

            INPUT:
        + PresRateB: Vector de valores tasa de cambio de presión antes.
        + PresRateA: Vector de valores tasa de cambio de presión durante.
        + Name: Nombre de la estación para el título.
        + PathImg: Ruta donde se quiere guardar el archivo.
        + FlagA: Bandera para generar el ajuste.
        + FEn: Bandera para tener el gráfico en inglés.
        _________________________________________________________________________
        
            OUTPUT:
        Esta función arroja una gráfica y la guarda en la ruta desada.
        '''

        # Se calcula el ajuste
        if FlagA:
            # Se importa el paquete de Fit
            from CFitting import CFitting
            CF = CFitting()

            # Se realiza la regresión
            Coef, perr,R2 = CF.FF(PresRateB,PresRateA,1)

            # Se toman los datos para ser comparados posteriormente
            DD,PP = utl.NoNaN(PresRateB,PresRateA,False)
            N = len(DD)
            a = Coef[0]
            b = Coef[1]
            desv_a = perr[0]
            desv_b = perr[1]
            # Se garda la variable
            CC = np.array([N,a,b,desv_a,desv_b,R2])
            
            
            # Se realiza el ajuste a ver que tal dió
            x = np.linspace(np.nanmin(PresRateB),np.nanmax(PresRateB),100)
            PresRateC = Coef[0]*x+Coef[1]

        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)

        # Se genera la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
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
        # Scatter
        plt.scatter(PresRateB,PresRateA)
        if FEn:
            plt.title('Surface Pressure Rate Comparison in '+ Name,fontsize=16)
            plt.xlabel(u'Pressure Rate (Before the Event) [hPa/h]',fontsize=16)
            plt.ylabel('Pressure Rate (During the Event) [hPa/h]',fontsize=16)
        else:
            plt.title('Comparación de cambios en Presión Atmosférica en '+ Name,fontsize=16)
            plt.xlabel(u'Tasa de Cambio de Presión Antes [hPa/h]',fontsize=16)
            plt.ylabel('Tasa de Cambio de Presión Durante [hPa/h]',fontsize=16)

        # if FlagAn:
        #   # Número de cada punto
        #   n = np.arange(0,len(DurPrec))
        #   for i, txt in enumerate(n):
        #       plt.annotate(txt, (DurPrec[i],PresRate[i]),fontsize=8)

        # Se incluye el ajuste
        if FlagA:
            plt.plot(x,PresRateC,'k--')

        # Axes
        ax = plt.gca()
        xTL = ax.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
        
        xTL = ax.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
        minorLocatorx = MultipleLocator(MxL)
        yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
        MyL = np.abs(np.abs(yTL[1])-np.abs(yTL[0]))/5 # minorLocatory value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().xaxis.set_minor_locator(minorLocatorx)
        plt.gca().yaxis.set_minor_locator(minorLocatory)

        if FlagA:
            plt.text(xTL[-4],yTL[-2], r'$\Delta_d = %s\Delta_b+%s$' %(round(a,3),round(b,3)), fontsize=15)
            plt.text(xTL[-4],yTL[-2]-2*MyL, r'$R^2 = %s$' %(round(R2,4)), fontsize=14)

        plt.tight_layout()
        plt.savefig(Nameout+'.png',format='png',dpi=self.dpi )
        plt.close('all')

        if FlagA:
            return CC

    def Histogram2d(self,Data1,Data2,Bins,Title='',Var1='',Var2='',Name='',PathImg='',M=True,FlagHour=False,FlagTitle=False,FlagBig=False):
        '''
        DESCRIPTION:
        
            Esta función permite hacer un histograma doble a partir de un set de
            datos.

        _________________________________________________________________________

        INPUT:
            :param Data1:    A ndarray, Vector de valores de una variable.
            :param Data2:    A ndarray, Vector de valores de la otra variable.
            :param Bins:     A List, List with 2 valuesfor the bins of the 
                                     data.
            :param Title:    A str, Título de la imágen.
            :param Var1:     A str, Variable.
            :param Var2:     A str, Variable.
            :param Names:    A str, Nombre de la imágen.
            :param PathImg:  A str, Ruta donde se quiere guardar el archivo.
            :param M:        A str, Metodo para hacer el histograma.
            :param FlagHour: A bool, flag para poner el eje x en horas.
            :param FlagTitle:A bool, flag para incluir el título.
            :param FlagBig:A bool, flag para hacer el texto más grande.
        _________________________________________________________________________
        
        OUTPUT:
            Esta función arroja una gráfica y la guarda en la ruta desada.
        '''
        warnings.filterwarnings('ignore')
        # Se encuentra el histograma
        q = ~(np.isnan(Data1) | np.isnan(Data2))
        # Se encuentra el histograma
        H, xedges, yedges = np.histogram2d(Data1[q],Data2[q]
                ,bins=Bins,normed=M)

        # Se organizan los valores del histograma
        centerDx = (xedges[:-1] + xedges[1:]) / 2
        centerDy = (yedges[:-1] + yedges[1:]) / 2

        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)

        # Parámetros de la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        if FlagBig:
            plt.rcParams.update({'font.size': 20,'font.family': 'sans-serif'\
                ,'font.sans-serif': self.font\
                ,'xtick.labelsize': 23,'xtick.major.size': 6,'xtick.minor.size': 4\
                ,'xtick.major.width': 1,'xtick.minor.width': 1\
                ,'ytick.labelsize': 23,'ytick.major.size': 6,'ytick.minor.size': 4\
                ,'ytick.major.width': 1,'ytick.minor.width': 1\
                ,'axes.linewidth':1\
                ,'grid.alpha':0.1,'grid.linestyle':'-'})
        else:
            plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
                ,'font.sans-serif': self.font\
                ,'xtick.labelsize': 14,'xtick.major.size': 6,'xtick.minor.size': 4\
                ,'xtick.major.width': 1,'xtick.minor.width': 1\
                ,'ytick.labelsize': 16,'ytick.major.size': 6,'ytick.minor.size': 4\
                ,'ytick.major.width': 1,'ytick.minor.width': 1\
                ,'axes.linewidth':1\
                ,'grid.alpha':0.1,'grid.linestyle':'-'})
        plt.tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='on',direction='out')
        plt.tick_params(axis='y',which='both',left='on',right='off',\
            labelleft='on')
        plt.tick_params(axis='y',which='major',direction='out') 
        # Se realiza la figura 
        plt.contourf(centerDx,centerDy,H.T)
        # Se grafica el Colorbar
        # cbar = plt.colorbar()
        # if M:
        #     cbar.ax.set_ylabel('Densidad')
        # else:
        #     cbar.ax.set_ylabel('Datos')

        # Se cambia el valor de los ejes.
        if FlagHour:
            CCStr = []
            for C in centerD:
                hour = int(C)
                minutes = int((C*60) % 60)
                CCStr.append('%d:%02d' %(hour,minutes))
            ax = plt.gca()
            ax.set_xticklabels(CCStr)
        # Se arreglan los ejes
        ax = plt.gca()
        # Se cambia el label de los eje
        xTL = ax.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
        yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
        MyL = np.abs(np.abs(yTL[1])-np.abs(yTL[0]))/5 # minorLocatory value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().yaxis.set_minor_locator(minorLocatory)

        # Labels
        # Título
        if FlagTitle:
            plt.title('Histograma de '+Title)
        plt.xlabel(Var1)  # Colocamos la etiqueta en el eje x
        plt.ylabel(Var2)  # Colocamos la etiqueta en el eje y
        plt.tight_layout()
        plt.savefig(PathImg + Name +'_Hist' + '.png',format='png',dpi=self.dpi )
        plt.close('all')

    def HistogramNP(self,Data,Bins,Title='',Var='',Name='',PathImg='',M='porcen',FEn=False,Left=True,FlagHour=False,flagEst=True,FlagTitle=False,FlagBig=False,vmax=None,FlagMonths=False):
        '''
        DESCRIPTION:
        
            Esta función permite hacer un histograma a partir de unos datos 
            y graficarlo.

        _________________________________________________________________________

        INPUT:
            + Data: Vector de valores con los que se quiere realizar el histograma.
            + Bins: Cantidad de intervalos de clase.
            + Title: Título de la imágen.
            + Var: Variable.
            + Names: Nombre de la imágen.
            + PathImg: Ruta donde se quiere guardar el archivo.
            + M: Metodo para hacer el histograma.
            + FEn: Flag para saber si quiere tener el histograma en inglés.
            + Left: flag para poner los estadísticos a la izquierda.
            + FlagHour: flag para poner el eje x en horas.
            + flagEst: flag para poner los estadísticos.
            + FlagTitle: flag incluir el título.
            + FlagBig: flag para poner la letra grande.
            + vmax: valor máximo del gráfico.
            + FlagMonths: flag para incluir meses.
        _________________________________________________________________________
        
        OUTPUT:
            Esta función arroja una gráfica y la guarda en la ruta desada.
        '''
        warnings.filterwarnings('ignore')
        # Se encuentra el histograma
        q = (~np.isnan(Data) & ~np.isinf(Data)) 
        # Se encuentra el histograma
        DH,DBin = np.histogram(Data[q],bins=Bins); [float(i) for i in DH]

        if M.lower() == 'porcen':
            # Se encuentra la frecuencia relativa del histograma
            DH = DH/float(DH.sum())*100;

        # Se organizan los valores del histograma
        widthD = 1 * (DBin[1] - DBin[0])
        centerD = (DBin[:-1] + DBin[1:]) / 2

        # Se encuentran los diferentes momentos estadísticos
        A = np.nanmean(Data)
        B = np.nanstd(Data)
        C = st.skew(Data[q])
        D = st.kurtosis(Data[q])

        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)

        # Parámetros de la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        if FlagBig:
            plt.rcParams.update({'font.size': 18,'font.family': 'sans-serif'\
                ,'font.sans-serif': self.font\
                ,'xtick.labelsize': 18,'xtick.major.size': 6,'xtick.minor.size': 4\
                ,'xtick.major.width': 1,'xtick.minor.width': 1\
                ,'ytick.labelsize': 18,'ytick.major.size': 6,'ytick.minor.size': 4\
                ,'ytick.major.width': 1,'ytick.minor.width': 1\
                ,'axes.linewidth':1\
                ,'grid.alpha':0.1,'grid.linestyle':'-'})
        else:
            plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
                ,'font.sans-serif': self.font\
                ,'xtick.labelsize': 14,'xtick.major.size': 6,'xtick.minor.size': 4\
                ,'xtick.major.width': 1,'xtick.minor.width': 1\
                ,'ytick.labelsize': 16,'ytick.major.size': 6,'ytick.minor.size': 4\
                ,'ytick.major.width': 1,'ytick.minor.width': 1\
                ,'axes.linewidth':1\
                ,'grid.alpha':0.1,'grid.linestyle':'-'})
        plt.tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='on',direction='out')
        plt.tick_params(axis='y',which='both',left='on',right='off',\
            labelleft='on')
        plt.tick_params(axis='y',which='major',direction='out') 
        # Se realiza la figura 
        # p1 = plt.bar(DBin[:-1],DH,color='dodgerblue',width=widthD,edgecolor='k')
        p1 = plt.bar(centerD,DH,color='dodgerblue',width=widthD,edgecolor=['k']*len(centerD))
        # Se cambia el valor de los ejes.
        plt.xticks(centerD) # Se cambia el valor de los ejes
        ax = plt.gca()
        if isinstance(Bins,list) or isinstance(Bins,np.ndarray):
            plt.xlim([Bins[0],Bins[-1]])
        else:
            plt.xlim([DBin[0],DBin[-1]])
        if FlagHour:
            CCStr = []
            for C in centerD:
                hour = int(C)
                minutes = int((C*60) % 60)
                CCStr.append('%d:%02d' %(hour,minutes))
            ax.set_xticklabels(CCStr)
        if FlagMonths:
            Meses = ['Ene','Feb','Mar','Abr','May','Jun','Jul',
                    'Ago','Sep','Oct','Nov','Dec']
            ax.set_xticklabels(Meses)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

        ax = plt.gca()
        xTL = ax.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
        yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
        MyL = np.abs(np.abs(yTL[1])-np.abs(yTL[0]))/5 # minorLocatory value
        if vmax != None:
            plt.ylim([0,vmax])
        if flagEst:
            plt.plot([A,A],[0,yTL[-1]],'k-')
            plt.plot([A+B,A+B],[0,yTL[-1]],'k--')
            plt.plot([A-B,A-B],[0,yTL[-1]],'k--')
            # print('DBin',DBin)
            # print('len(DBin)',len(DBin))
            # print('xTL',xTL)
            # print('len',len(xTL))
            # incluyen los valores
            if Left:
                plt.text(xTL[1]-2*MxL,yTL[-2], r'$\mu = %s$' %(round(A,3)), fontsize=14)
                plt.text(xTL[1]-2*MxL,yTL[-2]-2*MyL, r'$\sigma = %s$' %(round(B,3)), fontsize=14)
                plt.text(xTL[1]-2*MxL,yTL[-2]-4*MyL, r'$\gamma = %s$' %(round(C,3)), fontsize=14)
                plt.text(xTL[1]-2*MxL,yTL[-2]-6*MyL, r'$\kappa = %s$' %(round(D,3)), fontsize=14)
            else:
                plt.text(xTL[-2]-2*MxL,yTL[-2], r'$\mu = %s$' %(round(A,3)), fontsize=14)
                plt.text(xTL[-2]-2*MxL,yTL[-2]-2*MyL, r'$\sigma = %s$' %(round(B,3)), fontsize=14)
                plt.text(xTL[-2]-2*MxL,yTL[-2]-4*MyL, r'$\gamma = %s$' %(round(C,3)), fontsize=14)
                plt.text(xTL[-2]-2*MxL,yTL[-2]-6*MyL, r'$\kappa = %s$' %(round(D,3)), fontsize=14)
        # Labels
        # Título
        if FlagTitle:
            plt.title(Title)
        plt.xlabel(Var)  # Colocamos la etiqueta en el eje x
        if M == 'porcen':
            plt.ylabel('Porcentaje [%]')  # Colocamos la etiqueta en el eje y
        else:
            plt.ylabel('Número de Datos')  # Colocamos la etiqueta en el eje y

        # Se arreglan los ejes
        ax = plt.gca()
        # Se cambia el label de los eje
        xTL = ax.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
        yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
        MyL = np.abs(np.abs(yTL[1])-np.abs(yTL[0]))/5 # minorLocatory value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().yaxis.set_minor_locator(minorLocatory)
        plt.tight_layout()
        plt.savefig(PathImg + Name +'_Hist' + '.png',format='png',dpi=self.dpi )
        plt.close('all')

    def ButterworthGraph(self,FiltData,fs,order,worN=2000,PathImg='',Name='Filt'):
        '''
        DESCRIPTION:

            Esta función permite comparar dos ciclos anuales.
        ________________________________________________________________________

        INPUT:
            + FiltData: b and a of the filter. 
            + order: order of the filter.
            + worN: See freqz documentation.
        _________________________________________________________________________

        OUTPUT:
            
        '''
        warnings.filterwarnings('ignore')
        b = FiltData[0]
        a = FiltData[1]
        w, h = freqz(b, a, worN=worN)

        # Figure size
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        if PathImg != '':
            utl.CrFolder(PathImg)
        # Se genera la gráfica
        F = plt.figure(figsize=DM.cm2inch(fH,fV))
        # Parámetros de la Figura
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
            ,'xtick.labelsize': 15,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': 16,'ytick.major.size': 12,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})
        plt.tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='on',direction='in')
        plt.tick_params(axis='x',which='major',direction='inout')
        plt.tick_params(axis='y',which='both',left='on',right='off',\
            labelleft='on')
        plt.tick_params(axis='y',which='major',direction='inout') 
        plt.grid()
        # plt.plot((fs*0.5/np.pi)*w, abs(h), label="orden = %d" % order)
        # plt.plot((fs*0.5)*w, abs(h), label="orden = %d" % order)
        # plt.plot(w, abs(h), label="orden = %d" % order)
        plt.semilogx(((1/w)*5)/60*np.pi,  abs(h), label="30 min - 5 h Filtro de paso banda")
        plt.title('Butterworth Filter')
        plt.xlabel('Period [h]')
        plt.ylabel(r'(Response Magnitud)$^2$')
        plt.legend(loc=1)
        # The minor ticks are included
        ax = plt.gca()
        yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # minorLocatory value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().yaxis.set_minor_locator(minorLocatory)
        plt.tight_layout()
        plt.savefig(PathImg + Name +'.png',format='png',dpi=self.dpi )
        plt.close('all')
        return
    
    def FilComp(self,dates,Data,DataF,a,b,VarU,PathImg='',Name='Estación',Var='Pres',Filt=''):
        '''
        DESCRIPTION:
        
            Esta función permite comparar la serie normal con la serie 
            filtrada.
        _________________________________________________________________________

        INPUT:
            + Date: Vector de fechas en formato date.
        _________________________________________________________________________
        
        OUTPUT:
            Esta función arroja una gráfica y la guarda en la ruta desada.
        '''
        warnings.filterwarnings('ignore')
        # Tamaño de la Figura
        fH = self.fH # Largo de la Figura
        fV = self.fV # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg)

        plt.rcParams.update({'font.size': 14,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
            ,'xtick.labelsize': 14,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': 10,'ytick.major.size': 12,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})

        # Se genera la gráfica
        fig, axs = plt.subplots(2,1, figsize=DM.cm2inch(fH,fV))
        axs = axs.ravel() # Para hacer un loop con los subplots

        axs[0].plot(dates[a:b],Data[a:b],'k-')
        axs[0].axes.get_xaxis().set_visible(False)
        axs[0].set_title('No Filtered Series',fontsize=16)
        axs[0].set_ylabel(VarU,fontsize=16)
        axs[1].plot(dates[a:b],DataF[a:b],'k-')
        axs[1].set_title('Filtered Series',fontsize=16)
        axs[1].set_ylabel(VarU,fontsize=16)
        # axs[1].set_xlabel(u'Fecha',fontsize=16)
        # for tick in axs[0].get_xticklabels():
        #   tick.set_rotation(45)
        for tick in axs[1].get_xticklabels():
            tick.set_rotation(45)
        plt.tight_layout()
        plt.savefig(PathImg+ Name + '_' +Var+Filt+'_SeriesSep.png',format='png',dpi=300)
        plt.close('all')

        # ---------------------
        plt.rcParams.update({'font.size': 14,'font.family': 'sans-serif'\
            ,'font.sans-serif': self.font\
            ,'xtick.labelsize': 14,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': 10,'ytick.major.size': 12,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})
        fig= plt.figure(figsize=DM.cm2inch(fH,fV))

        plt.plot(dates[a:b],Data[a:b],'b-',label='Sin filtrar')
        plt.plot(dates[a:b],DataF[a:b],'r--',label='Filtrada')
        plt.title('Serie',fontsize=16)
        plt.ylabel(VarU,fontsize=16)
        # plt.xlabel(u'Fecha',fontsize=16)
        plt.legend(loc=1,fontsize=13)

        # for tick in axs[0].get_xticklabels():
        #   tick.set_rotation(45)
        for tick in plt.gca().get_xticklabels():
            tick.set_rotation(45)
        plt.tight_layout()
        plt.savefig(PathImg+ Name + '_' +Var+Filt+'_Series.png',format='png',dpi=300)
        plt.close('all')

    def EventsSeriesGen(self,DatesEv,Data,DataV=None,DataKeyV=None,DataKey=None,PathImg='',
            Name='',NameArch='',GraphInfoV={'color':['-.b'],'label':['Inicio del Evento']},
            GraphInfo={'ylabel':['Precipitación [mm]'],'color':['b'],'label':['Precipitación']},
            flagBig=False,vm={'vmax':[],'vmin':[]},Ev=0,flagV=False,
            flagAverage=False,dt=1,Date='',flagEvent=False,N=None,flagEng=True):
        '''
        DESCRIPTION:

            This method plots several lines in different axis.
        _______________________________________________________________________
        INPUT:
            :param DatesEv:     A ndarray, Dates of the events.
            :param Data:        A dict, Diccionario con las variables
                                            que se graficarán.
            :param DataV:       A dict, Diccionario con las líneas verticales
                                        estos deben ser fechas en formato
                                        datetime.
            :param DataKeyV:    A list, Lista con keys de los valores verticales.

        '''
        # Se organizan las fechas
        if flagAverage:
            H = int(len(DatesEv)/2)
            FechaEvv = np.arange(-H,H,1)
            FechaEvv = FechaEvv*dt/60
        else:
            FechaEvv = DatesEv

        if DataKey is None:
            flagSeveral = False
        elif len(DataKey) >= 1:
            flagSeveral = True

        if len(vm['vmax']) == 0 and len(vm['vmin']) == 0:
            flagVm = False
        else:
            flagVm = True

        if flagVm:
            if len(vm['vmax']) >= 1 and len(vm['vmin']) >= 1:
                flagVmax = True
                flagVmin = True
            elif len(vm['vmax']) >= 1:
                flagVmax = True
                flagVmin = False
            elif len(vm['vmin']) >= 1:
                flagVmax = False
                flagVmin = True

        # -------------------------
        # Se grafican los eventos
        # -------------------------

        # Se grafican las dos series
        fH=30 # Largo de la Figura
        fV = fH*(2/3) # Ancho de la Figura


        if flagBig:
            fH=30 # Largo de la Figura
            fV = fH*(2/3) # Ancho de la Figura
            lensize=16
            plt.rcParams.update({'font.size': 28,'font.family': 'sans-serif'\
                ,'font.sans-serif': 'Arial Narrow'\
                ,'xtick.labelsize': 28,'xtick.major.size': 6,'xtick.minor.size': 4\
                ,'xtick.major.width': 1,'xtick.minor.width': 1\
                ,'ytick.labelsize': 28,'ytick.major.size': 12,'ytick.minor.size': 4\
                ,'ytick.major.width': 1,'ytick.minor.width': 1\
                ,'axes.linewidth':1\
                ,'grid.alpha':0.1,'grid.linestyle':'-'})
            offNew = 10
        else:
            lensize=15
            plt.rcParams.update({'font.size': 18,'font.family': 'sans-serif'\
                ,'font.sans-serif': 'Arial Narrow'\
                ,'xtick.labelsize': 18,'xtick.major.size': 6,'xtick.minor.size': 4\
                ,'xtick.major.width': 1,'xtick.minor.width': 1\
                ,'ytick.labelsize': 18,'ytick.major.size': 12,'ytick.minor.size': 4\
                ,'ytick.major.width': 1,'ytick.minor.width': 1\
                ,'axes.linewidth':1\
                ,'grid.alpha':0.1,'grid.linestyle':'-'})
            offNew = 0

        plt.xticks(rotation=45)
        f = plt.figure(figsize=DM.cm2inch(fH,fV))
        ax = host_subplot(111, axes_class=AA.Axes)
        ax.tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='on',direction='out')
        ax.tick_params(axis='y',which='both',left='on',right='off',\
            labelleft='on')
        ax.tick_params(axis='y',which='major',direction='inout') 
        if flagSeveral:
            if len(DataKey) >= 1:
                DataP = Data[DataKey[0]]
        else:
            DataP = Data

        # Se grafica el gráfico principal
        ax.plot(FechaEvv,DataP,color=GraphInfo['color'][0],label=GraphInfo['label'][0])
        if not(flagAverage):
            ax.axis["bottom"].major_ticklabels.set_rotation(30)
            ax.axis["bottom"].major_ticklabels.set_ha("right")
            ax.axis["bottom"].label.set_pad(30)
            ax.axis["bottom"].format_xdata = mdates.DateFormatter('%H%M')
        ax.axis["left"].label.set_color(color=GraphInfo['color'][0])
        ax.set_ylabel(GraphInfo['ylabel'][0])

        # Se escala el eje 
        if flagVm:
            if (flagVmax and flagVmin) and (not(vm['vmin'][0] is None) and not(vm['vmax'][0] is None)):
                ax.set_ylim([vm['vmin'][0],vm['vmax'][0]])
            elif flagVmax and not(vm['vmax'][0] is None):
                ax.set_ylim(ymax=vm['vmax'][0])
            elif flagVmin and not(vm['vmin'][0] is None):
                ax.set_ylim(ymin=vm['vmin'][0])

        yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
        xTL = ax.xaxis.get_ticklocs() # List of Ticks in x

        # Se grafican las líneas verticales
        if flagV:
            for ilab,lab in enumerate(DataKeyV):
                ax.plot([DataV[lab],DataV[lab]],[yTL[0],yTL[-1]],
                        GraphInfoV['color'][ilab],label=GraphInfoV['label'][ilab])

        # Se organizan los ejes 
        MyL = (yTL[1]-yTL[0])/5 # minorLocatory value
        MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
        minorLocatory = MultipleLocator(MyL)
        ax.yaxis.set_minor_locator(minorLocatory)

        if flagAverage:
            if N != None:
                ax.text(xTL[-4]+2*MxL,yTL[-1]-5*MyL,'$N=%s$'%(N))

        # Se realizan los demás gráficos
        if flagSeveral:
            axi = [ax.twinx() for i in range(len(DataKey)-1)]
            for ilab,lab in enumerate(DataKey):
                if ilab >= 1:
                    axi[ilab-1].plot(FechaEvv,Data[lab],color=GraphInfo['color'][ilab],
                            label=GraphInfo['label'][ilab])
                    axi[ilab-1].set_ylabel(GraphInfo['ylabel'][ilab])
                    if flagVm and len(vm['vmax']) > 1:
                        if (not(vm['vmin'][ilab] is None) and not(vm['vmax'][ilab] is None)):
                            axi[ilab-1].set_ylim([vm['vmin'][ilab],vm['vmax'][ilab]])
                        elif not(vm['vmax'][ilab] is None):
                            axi[ilab-1].set_ylim(ymax=vm['vmax'][ilab])
                        elif not(vm['vmin'][ilab] is None):
                            axi[ilab-1].set_ylim(ymin=vm['vmin'][ilab])

                    if ilab == 2:
                        offset = 80+offNew
                        new_fixed_axis = axi[ilab-1].get_grid_helper().new_fixed_axis
                        axi[ilab-1].axis["right"] = new_fixed_axis(loc="right",
                                                        axes=axi[ilab-1],
                                                        offset=(offset, 0))
                        axi[ilab-1].axis["right"].label.set_color(color=GraphInfo['color'][ilab])
                    elif ilab == 3:
                        # axi[ilab-1].spines['right'].set_position(('axes',-0.25))
                        offset = -65-offNew
                        new_fixed_axis = axi[ilab-1].get_grid_helper().new_fixed_axis
                        axi[ilab-1].axis["right"] = new_fixed_axis(loc="left",
                                                        axes=axi[ilab-1],
                                                        offset=(offset, 0))
                        axi[ilab-1].axis["right"].label.set_color(color=GraphInfo['color'][ilab])
                    else:
                        offset = 0
                        new_fixed_axis = axi[ilab-1].get_grid_helper().new_fixed_axis
                        axi[ilab-1].axis["right"] = new_fixed_axis(loc="right",
                                                        axes=axi[ilab-1],
                                                        offset=(offset, 0))
                        axi[ilab-1].axis["right"].label.set_color(color=GraphInfo['color'][ilab])

                    # Se organizan los ejes 
                    yTL = axi[ilab-1].yaxis.get_ticklocs() # List of Ticks in y
                    MyL = (yTL[1]-yTL[0])/5 # minorLocatory value
                    minorLocatory = MultipleLocator(MyL)
                    axi[ilab-1].yaxis.set_minor_locator(minorLocatory)
                    axi[ilab-1].format_xdata = mdates.DateFormatter('%H%M')

        if flagAverage:
            if flagEng:
                ax.set_xlabel('Time [h]')
                ax.set_title(r"Composites in "+Name)
            else:
                ax.set_xlabel('Tiempo [h]')
                ax.set_title(r"Diagrama de Compuestos en "+Name)
            if flagBig:
                ax.set_title(Name)
        else:
            if Date == '':
                ax.set_title(Name)
            else:
                ax.set_title(Name+r" Evento "+Date)
        if not(flagBig):
            plt.legend(loc=4,framealpha=0.6,fontsize=lensize)
        ax.set_xlabel('Time [h]')
        # plt.grid()
        if flagAverage:
            # Se crea la carpeta para guardar la imágen
            utl.CrFolder(PathImg + 'Average/')
            Nameout = PathImg + 'Average/' + NameArch 
        elif flagEvent:
            # Se crea la carpeta para guardar la imágen
            DateL = Date.split('/')
            utl.CrFolder(PathImg + DateL[0]+DateL[1]+DateL[2] + '/')
            Nameout = PathImg + '/' + DateL[0]+DateL[1]+DateL[2] + '/'\
                    + NameArch + '_Ev_' + str(Ev)
        else:
            # Se crea la carpeta para guardar la imágen
            utl.CrFolder(PathImg + NameArch + '/')
            Nameout = PathImg + NameArch + '/' + NameArch + '_Ev_'+str(Ev)
        plt.tight_layout()

        plt.savefig(Nameout+'.png',format='png',dpi=200)
        plt.close('all')
