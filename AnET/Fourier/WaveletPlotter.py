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
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
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



def WaveletPlot(time,Data1,period,power,sig95,coi,global_ws,global_signif,
        Var_LUn='',Un3='',Title1='',Title2='',Title3='',
        Name='',PathImg='',**args):
    '''
    DESCRIPTION:
    
        Esta función permite hacer las gráficas de series temporales parecidas a
        a las presentadas en Excel para el proyecto.

    _________________________________________________________________________

    INPUT:
        :param time: a ndarray, Years
        :param Data1: a ndarray, Vector with the Data 
        :param Var_LUn: A str, Label with units.
        :param Title1: A str, title of the image.
        :param Var: Nombre de la imagen.
        :param flagT: Flag para saber si se incluye el título.
        :param v: Titulo de la Figura.
        :param PathImg: Ruta donde se quiere guardar el archivo.
        :param **args: Argumentos adicionales para la gráfica, 
                       como color o ancho de la línea.
    _________________________________________________________________________
    
        OUTPUT:
    Esta función arroja una gráfica y la guarda en la ruta desada.
    '''
    font = 'Arial'
    # Image dimensions
    fH = 30
    fV = fH*(2.0/3.0)
    # Image resolution
    dpi = 120

    # Tamaño de la Figura
    fH = fH # Largo de la Figura
    fV = fV # Ancho de la Figura
    # Se crea la carpeta para guardar la imágen
    utl.CrFolder(PathImg)

    # Se genera la gráfica
    F = plt.figure(figsize=DM.cm2inch(fH,fV))
    # Parámetros de la Figura
    plt.rcParams.update({'font.size': 12,'font.family': 'sans-serif'\
        ,'font.sans-serif': font\
        ,'xtick.labelsize': 12,'xtick.major.size': 6,'xtick.minor.size': 4\
        ,'xtick.major.width': 1,'xtick.minor.width': 1\
        ,'ytick.labelsize': 12,'ytick.major.size': 12,'ytick.minor.size': 4\
        ,'ytick.major.width': 1,'ytick.minor.width': 1\
        ,'axes.linewidth':1\
        ,'grid.alpha':0.1,'grid.linestyle':'-'})
    plt.rcParams['agg.path.chunksize'] = 20000
    plt.subplot(211)
    plt.plot(time,Data1,**args)
    plt.xlim(time[0],time[-1])
    plt.xlabel('Time [years]')
    plt.ylabel(Var_LUn)
    plt.title('a) '+Title1,fontsize=12)

    #--- Contour plot wavelet power spectrum
    plt3 = plt.subplot(223)
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    CS = plt.contourf(time, period, np.log2(power), len(levels),cmap=plt.get_cmap('jet'))  #*** or use 'contour'
    im = plt.contourf(CS, levels=np.log2(levels),cmap=plt.get_cmap('jet'))
    plt.xlabel('Time [year]')
    plt.ylabel('Period [years]')
    plt.title('b) '+Title2,fontsize=12)
    plt.xlim(time[0],time[-1])
    # 95# significance contour, levels at -99 (fake) and 1 (95# signif)
    plt.contour(time, period, sig95, [-99, 1], colors='k')
    # cone-of-influence, anything "below" is dubious
    plt.plot(time, coi, 'k')
    # format y-scale
    plt3.set_yscale('log', basey=2, subsy=None)
    plt.ylim([np.min(period), np.max(period)])
    ax = plt.gca().yaxis
    ax.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt3.ticklabel_format(axis='y', style='plain')
    plt3.invert_yaxis()
    # set up the size and location of the colorbar
    divider = make_axes_locatable(plt3)
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    plt.colorbar(im, cax=cax, orientation='horizontal')

    #--- Plot global wavelet spectrum
    plt4 = plt.subplot(224)
    plt.plot(global_ws, period)
    plt.plot(global_signif, period, '--')
    plt.xlabel(r'Power '+Un3)
    plt.ylabel('Period [years]')
    plt.title('c) '+Title3,fontsize=12)
    plt.xlim([0, 1.25 * np.max(global_ws)])
    # format y-scale
    plt4.set_yscale('log', basey=2, subsy=None)
    plt.ylim([np.min(period), np.max(period)])
    ax = plt.gca().yaxis
    ax.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt4.ticklabel_format(axis='y', style='plain')
    plt4.invert_yaxis()
    plt.tight_layout()
    # Se guarda la figura
    plt.savefig(PathImg + Name + '.png',format='png',dpi=dpi)
    plt.close('all')

