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
from Utilities import DatesUtil as DUtil
from AnET import CorrSt as cr
from AnET import CFitting as CF
from Hydro_Analysis import Hydro_Plotter as HyPL
from Hydro_Analysis import Hydro_Analysis as HA
from Hydro_Analysis import Thermo_An as TA
from AnET import AnET as anet
from EMSD import EMSD
from BPumpL.BPumpL import BPumpL as BP


class MapSeriesGen(object): 
    '''
    DESCRIPTION:

    Clase para abrir los documentos que se necesitan para hacer los diferentes
    estudios de los diagramas de dispersión.
    
    '''

    def __init__(self,PathImg='',Fol='201303060940',V=['Prec','Pres']):
        MapFold = np.array(utl.GetFolders(PathImg+'Maps/'))
        SeriesFold = np.array(utl.GetFolders(PathImg+'Series/'))
        RadarFold = np.array(utl.GetFolders(PathImg+'Radar/'))

        xFolMap = np.where(MapFold== Fol)[0]
        xFolSer = np.where(SeriesFold== Fol)[0]
        xFolRad= np.where(RadarFold== Fol)[0]

        if len(xFolSer) == 0:
            E = utl.ShowError('__init__','MapSeriesGen','No se encuentra la carpeta en los dos directorios')
            raise E

        if len(xFolMap) != 0:
            self.PathMaps = PathImg+'Maps/'+MapFold[xFolMap[0]]+'/'
            self.ArchMap = gl.glob(self.PathMaps+'*.png')
            self.Names = [i[len(self.PathMaps):-4] for i in self.ArchMap]

        self.PathSeries = PathImg+'Series/'+SeriesFold[xFolSer[0]]+'/'+V[1]+'/'
        self.ArchSeries = gl.glob(self.PathSeries+'*.png')
        self.NamesSeries = [i[len(self.PathSeries):-4] for i in self.ArchSeries]

        if len(xFolRad) != 0:
            # Se cargan los archivos, tener en cuenta la variable del radar 'DBZH'
            self.PathRadar1 = PathImg+'Radar/'+SeriesFold[xFolSer[0]]+'/'+V[0]+'/DBZH/'
            self.PathRadar2 = PathImg+'Radar/'+SeriesFold[xFolSer[0]]+'/'+V[1]+'/DBZH/'
            self.ArchRadar1 = gl.glob(self.PathRadar1 +'*.png')
            self.ArchRadar2 = gl.glob(self.PathRadar2 +'*.png')
            self.NamesRadar1 = [i[len(self.PathRadar1):-4] for i in self.ArchRadar1]
            self.NamesRadar2 = [i[len(self.PathRadar2):-4] for i in self.ArchRadar2]

        return
    
    def GraphComp(self,PathImg=''):
        '''
        DESCRIPTION:

            Función para hacer la combinación de gráficos.
        '''
        
        # Tamaño de la Figura
        fH=40 # Largo de la Figura
        fV = 20 # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg+ self.Names[0])
        for iar,ar in enumerate(self.ArchMap):
            if self.Names[iar] == self.NamesSeries[iar]:
                # Se extraen los datos
                MapImg = mpimg.imread(self.ArchMap[iar])
                SeriesImg = mpimg.imread(self.ArchSeries[iar])
                f = plt.figure(figsize=utl.cm2inch(fH,fV),facecolor='w',edgecolor='w')
                a = f.add_subplot(1,2,1)
                a.imshow(SeriesImg)
                a.axes.get_xaxis().set_visible(False)
                a.axes.get_yaxis().set_visible(False)
                a = f.add_subplot(1,2,2)
                a.imshow(MapImg)
                a.axes.get_xaxis().set_visible(False)
                a.axes.get_yaxis().set_visible(False)
                plt.tight_layout()
                plt.savefig(PathImg + self.Names[0] + '/' + 'Image' + '_%02d.png' % iar,format='png',dpi=200)
                plt.close('all')
    
    def GraphCompRadar(self,PathImg=''):
        '''
        DESCRIPTION:

            Función para hacer la combinación de gráficos.
        '''
        
        # Tamaño de la Figura
        fH=40 # Largo de la Figura
        fV = 20 # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg+ self.NamesSeries[0])
        for iar,ar in enumerate(self.ArchRadar1):
            if self.NamesRadar1[iar] == self.NamesSeries[iar]:
                # Se extraen los datos
                RadarImg1 = mpimg.imread(self.ArchRadar1[iar])
                RadarImg2 = mpimg.imread(self.ArchRadar2[iar])
                SeriesImg = mpimg.imread(self.ArchSeries[iar])
                f = plt.figure(figsize=utl.cm2inch(fH,fV),facecolor='w',edgecolor='w')
                a = f.add_subplot(1,3,1)
                a.imshow(SeriesImg)
                a.axes.get_xaxis().set_visible(False)
                a.axes.get_yaxis().set_visible(False)
                a = f.add_subplot(1,3,2)
                a.imshow(RadarImg1)
                a.axes.get_xaxis().set_visible(False)
                a.axes.get_yaxis().set_visible(False)
                a = f.add_subplot(1,3,3)
                a.imshow(RadarImg2)
                a.axes.get_xaxis().set_visible(False)
                a.axes.get_yaxis().set_visible(False)
                plt.tight_layout()
                plt.savefig(PathImg + self.NamesRadar1[0] + '/' + 'Image' + '_%02d.png' % iar,format='png',dpi=200)
                plt.close('all')
        return
    
    def GraphCompRadarOnly(self,PathImg=''):
        '''
        DESCRIPTION:

            Función para hacer la combinación de gráficos.
        '''
        
        # Tamaño de la Figura
        fH=40 # Largo de la Figura
        fV = 20 # Ancho de la Figura
        # Se crea la carpeta para guardar la imágen
        utl.CrFolder(PathImg+ self.NamesSeries[0])
        for iar,ar in enumerate(self.ArchRadar1):
            if self.NamesRadar1[iar] == self.NamesRadar2[iar]:
                # Se extraen los datos
                RadarImg1 = mpimg.imread(self.ArchRadar1[iar])
                RadarImg2 = mpimg.imread(self.ArchRadar2[iar])
                SeriesImg = mpimg.imread(self.ArchSeries[iar])
                f = plt.figure(figsize=utl.cm2inch(fH,fV),facecolor='w',edgecolor='w')
                # a = f.add_subplot(1,3,1)
                # a.imshow(SeriesImg)
                # a.axes.get_xaxis().set_visible(False)
                # a.axes.get_yaxis().set_visible(False)
                a = f.add_subplot(1,2,1)
                a.imshow(RadarImg1)
                a.axes.get_xaxis().set_visible(False)
                a.axes.get_yaxis().set_visible(False)
                a = f.add_subplot(1,2,2)
                a.imshow(RadarImg2)
                a.axes.get_xaxis().set_visible(False)
                a.axes.get_yaxis().set_visible(False)
                plt.tight_layout()
                plt.savefig(PathImg + self.NamesRadar1[0] + '/' + 'Image' + '_%02d.png' % iar,format='png',dpi=200)
                plt.close('all')
        return
    
    def VidComp(self,PathImg='',Ev=0):
        '''
        DESCRIPTION:

            Función para realizar el video.
        '''     

        PathAct = os.getcwd()
        os.chdir(PathImg+self.NamesSeries[Ev])
        print(os.getcwd())
        # Se hace el video
        '''ffmpeg -i postgrados_ATrTEv.mov -vf scale=720:-1 
        -c:v libx264 -profile:v high -pix_fmt yuv420p -g 30 
        -r 30 postgrados_ATrTEv2.mov'''

        subprocess.call(['ffmpeg','-y','-framerate','2','-i',
            'Image_%02d.png',
            '-vcodec','libx264',
            '-vf', 'scale=1700:-2',
            '-pix_fmt','yuv420p',
            self.NamesSeries[Ev]+'_Video.mp4'])

        os.chdir(PathAct)
    
