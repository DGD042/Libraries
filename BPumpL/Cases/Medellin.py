# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 16/11/2017
#______________________________________________________________________________
#______________________________________________________________________________


# Data Management
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from math import cos, sin, asin, sqrt, radians
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.dates as mdates
from scipy import stats as st
# System Management
import sys
import warnings
import re 
import os
import glob as gl
# Librerías de extracción de datos
import netCDF4 as nc
import xlsxwriter as xlsxwl
import xlrd
from fastkml.kml import KML
# from pykml import parser
# Map Management
from osgeo import gdal
from osgeo import osr
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
from pyproj import Proj, transform
from mpl_toolkits.basemap import Basemap
import pyart
import pyart.io.cfradial as CFR
# Dates
from datetime import date, datetime, timedelta
# ------------------
# Personal Modules
# ------------------
# Importing Modules
import BPumpL as BPL
from Utilities import Data_Man as DMan
from GeoF.GeoTIFF import Functions as GF
from Utilities import Utilities as utl
from Hydro_Analysis import Hydro_Plotter as HyPl; HyPl=HyPl()
from EMSD.Dates import DatesFunctions as DUtil

# ------------------
# Class
# ------------------
class Medellin(object):
    '''
    DESCRIPTION:
        This class develops the interpolations.
    _________________________________________________________________
    INPUT:
        :param DateI: A datetime, a datetime object with the initial
                      date for the graph.
        :param DateE: A datetime, a datetime object with the ending 
                      date for the graph.
        :param endingmat: A str, string denoting the end of the .mat
                          file.
    '''
    def __init__(self,DateI,DateE,endingmat=''):
        '''
        '''
        # ----------------
        # Error Managment
        # ----------------
        if not(isinstance(endingmat,str)):
            self.ShowError('__init__','Medellin','endingmat must be a string')
        # ----------------
        # Paths
        # ----------------
        # ----------------
        # Constants
        # ----------------
        self.ImgFolder = {'Medellin':'Medellin/Cases'+endingmat+'/'}
        Labels = ['ID','Name','Latitud','Longitud']
        self.DateI = DateI
        self.DateE = DateE
        lenData = int((self.DateE-self.DateI).seconds/60)
        self.DatesSt = [DateI]
        for iEv in range(lenData):
            self.DatesSt.append(self.DatesSt[-1]+timedelta(0,60))
        self.DatesSt = np.array(self.DatesSt)
        DatesStr = DUtil.Dates_datetime2str([DateI],Date_Format='%Y%m')[0]
        self.PathImg = 'Tesis_MscR/02_Docs/01_Tesis_Doc/Kap5/Img/Cases/'+DatesStr+'/'
        # ----------------
        # Load Information
        # ----------------
        self.SC = BPL.Scatter_Gen(DataBase='Medellin',
                endingmat=endingmat,PathImg=self.PathImg)
        # ---------------------
        # Station information
        # ---------------------
        self.ID = self.SC.ID
        self.St_Info = {}
        for iSt,St in enumerate(self.ID):
            self.St_Info[St] = {}
            for Lab in Labels:
                self.St_Info[St][Lab] = self.SC.StInfo['Medellin'][Lab][iSt]
            self.St_Info[St]['CodesNames'] = self.SC.StInfo['Medellin']['ID'][iSt]+ ' ' + self.SC.StInfo['Medellin']['Name'][iSt]
        return

    def LoadStations(self):
        '''
        DESCRIPTION:
            This method loads the information of the different stations.
        _________________________________________________________________
        INPUT:
        '''
        # ----------------
        # Constants
        # ----------------
        lenArch = len(self.SC.Arch)
        lenData = int((self.DateE-self.DateI).seconds/60)
        self.DataSt = {}
        Labels = ['FechaCP','Prec','Pres_F','T_F','HR_F','W_F','q_F']
        self.vmax = {}
        self.vmin = {}
        for Lab in Labels[1:]:
            self.vmax[Lab] = []
            self.vmin[Lab] = []


        # ----------------
        # Extract Data
        # ----------------
        for iar in range(lenArch):
            self.DataSt[self.ID[iar]] = {}
            self.SC.LoadData(irow=iar)
            xi = np.where(self.SC.f['FechaCP'] == self.DateI)[0]
            xf = np.where(self.SC.f['FechaCP'] == self.DateE)[0]
            if len(xi) == 0 or len(xf) == 0:
                for Lab in Labels:
                    self.DataSt[self.ID[iar]][Lab] = np.array(lenData)
                continue
            for iLab,Lab in enumerate(Labels):
                self.DataSt[self.ID[iar]][Lab] = self.SC.f[Lab][xi:xf+1]
                if iLab > 0:
                    self.vmax[Lab].append(np.nanmax(self.SC.f[Lab][xi:xf+1]))
                    self.vmin[Lab].append(np.nanmin(self.SC.f[Lab][xi:xf+1]))

        for Lab in Labels[1:]:
            self.vmax[Lab] = np.nanmax(self.vmax[Lab])+0.1
            if Lab == 'Prec':
                self.vmin[Lab] = np.nanmin(self.vmin[Lab])
            else:
                self.vmin[Lab] = np.nanmin(self.vmin[Lab])-0.1


        return
    
    def LoadRadar(self):
        '''
        DESCRIPTION:
            This method loads the radar information.
        _________________________________________________________________
        INPUT:
        '''
        PathRadar = '/Users/DGD042/Documents/Est_Information/SIATA/Radar/01_nc/PPIVol/'
        # ---------------------
        # Radar Files
        # ---------------------
        self.DateRI = self.DateI+timedelta(0,5*60*60)
        self.DateRE = self.DateE+timedelta(0,5*60*60)
        DateR = '%04i%02i/'%(self.DateRI.year,self.DateRE.month)
        self.PathRadar = PathRadar+DateR
        Files = gl.glob(PathRadar+DateR+'*.gz')
        if len(Files) == 0:
            self.ShowError('LoadRadar','Medellin','No Radar Files Found')
        self.Files = gl.glob(PathRadar+DateR+'*.gz')
        self.RadarDates = np.array([i[-32:-32+8]+i[-32+9:-32+8+5] for i in Files])
        self.RadarDatesP = DUtil.Dates_str2datetime(self.RadarDates,Date_Format='%Y%m%d%H%M')
        # ---------------------
        # Dates
        # ---------------------
        xi = np.where(self.RadarDatesP == self.DateRI)[0]
        while len(xi) == 0:
            xi = np.where(self.RadarDatesP == self.DateRI+timedelta(0,60))[0]
        xf = np.where(self.RadarDatesP == self.DateRE)[0]
        while len(xf) == 0:
            xf = np.where(self.RadarDatesP == self.DateRE-timedelta(0,60))[0]

        self.ArchRadar = Files[xi:xf+1]
        self.RadarDates = self.RadarDates[xi:xf+1]
        self.RadarDatesP = DUtil.Dates_str2datetime(self.RadarDates,Date_Format='%Y%m%d%H%M')
        return

    def PlotData(self,Time=0):
        '''
        DESCRIPTION:
            This method plots all the case information.
        _________________________________________________________________
        INPUT:
        '''
        FilesA = []
        for it,t in enumerate(self.DatesSt):
            if it >= Time:
                F = t
                xEv = np.where(self.RadarDatesP == F+timedelta(0,5*60*60))[0]
                if len(xEv) == 1:
                    FilesA.append(self.ArchRadar[xEv[0]][len(self.PathRadar):])
                RadarFile = self.Opengz(FilesA[-1],self.PathRadar,self.PathRadar)
                # Se corrige la información de Radar
                VELH = RadarFile.fields['VELH']
                DBZH  = RadarFile.fields['DBZH']
                NCPH = RadarFile.fields['NCPH']
                DBZHC = DBZH
                masks = [~(np.ma.getmaskarray(VELH['data'])) & (np.ma.getdata(VELH['data'])==0),
                        ~(np.ma.getmaskarray(DBZH['data'])) & (np.ma.getdata(DBZH['data'])<=10),
                        ~(np.ma.getmaskarray(NCPH['data'])) & (np.ma.getdata(NCPH['data'])<=0.75)]
                total_mask = masks[0] | masks[1] | masks [2]

                DBZHC['data'] = np.ma.masked_where(total_mask,DBZHC['data'])
                RadarFile.add_field('DBZHC', DBZHC)

                # Attenuation
                spec_at, cor_z = pyart.correct.calculate_attenuation(
                    RadarFile, 0, refl_field='DBZH',
                    ncp_field='NCPH', rhv_field='SNRHC',
                    phidp_field='PHIDP')
                RadarFile.add_field('specific_attenuation', spec_at)
                RadarFile.add_field('corrected_reflectivity_horizontal', cor_z)
                self.RadarFile = RadarFile

                # Se genera el mapa
                self.RadarGraphs(RadarFile,t.strftime('%Y/%m/%d %H:%M'),t,vmin=10)
                utl.CrFolder(self.PathImg+'DBZH'+'/')
                plt.savefig(self.PathImg+'DBZH'+'/'+'%04i Image'%(it)+'.png' ,format='png',dpi=200)
                plt.close('all')
        return

    def Opengz(self,File,PathData,PathData2):
        '''
        DESCRIPTION:
            This method open a gz file, loads the data an then deletes
            it
        _________________________________________________________________
        INPUT:
        '''
        os.system('gzip -d --keep ' + PathData2 + File)
        RadarFile = CFR.read_cfradial(PathData+File[:-3])
        os.remove(PathData+File[:-3])
        return RadarFile

    def RadarGraphs(self,RadarFile,DateT,DateTP,vmin=None,vmax=None):
        '''
        DESCRIPTION:
            This method plots the radar information
        _________________________________________________________________
        INPUT:
            :param RadarFile: A pyart object, object with the information 
                              of the radar.
        '''
        # -------------
        # Radar Graph
        # -------------
        plt.rcParams.update({'font.size': 18,'font.family': 'sans-serif'\
            ,'font.sans-serif': 'Arial'\
            ,'xtick.labelsize': 18,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': 18,'ytick.major.size': 12,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})
        ax1 = host_subplot(111, axes_class=AA.Axes)
        display = pyart.graph.RadarMapDisplay(RadarFile)
        display.plot_ppi_map('DBZH', 0,vmin=vmin,vmax=vmax,
             projection='lcc',
             min_lon=-75.75,max_lon=-75.15,
             min_lat=6.00,max_lat=6.50,
             resolution='i',
             lat_0=RadarFile.latitude['data'][0],
             lon_0=RadarFile.longitude['data'][0],
             title=DateT,
             colorbar_flag=False,
             colorbar_label='Reflectividad horizontal equivalente [dBZ]',ax=ax1
             )

        # Graph Generation
        fig = plt.gcf()
        fH = 45
        fV = 30
        fig.set_size_inches(DMan.cm2inch(fH,fV))

        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': 'Arial'\
            ,'xtick.labelsize': 15,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': 15,'ytick.major.size': 12,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})

        # Station Points
        for iID,ID in enumerate(self.ID):
            if ID == '68' or ID == '202' or ID == '201' or ID == '203':
                display.plot_point(self.St_Info[ID]['Longitud'],self.St_Info[ID]['Latitud'],
                        symbol='o',color='gray',markersize=10,markeredgecolor='k',
                        )#label_text=ID)
            else:
                display.plot_point(self.St_Info[ID]['Longitud'],self.St_Info[ID]['Latitud'],
                        symbol='o',color='gray',markersize=10,markeredgecolor='k',
                        label_text=ID)

        # plot range rings at 10, 20, 30 and 40km
        display.plot_range_ring(10.0, line_style='k-')
        display.plot_range_ring(20.0, line_style='k--')
        display.plot_range_ring(30.0, line_style='k-')
        display.plot_range_ring(50.0, line_style='k--')

        # plots cross hairs
        display.plot_line_xy(np.array([-40000.0, 40000.0]), np.array([0.0, 0.0]),
                             line_style='k-')
        display.plot_line_xy(np.array([0.0, 0.0]), np.array([-20000.0, 200000.0]),
                             line_style='k-')

        # Lines of the Zoom-in
        display.plot_line_geo(np.array([-75.60,-75.60]),np.array([6.225,6.280]),line_style='k-')
        display.plot_line_geo(np.array([-75.55,-75.55]),np.array([6.225,6.280]),line_style='k-')
        display.plot_line_geo(np.array([-75.60,-75.55]),np.array([6.280,6.280]),line_style='k-')
        display.plot_line_geo(np.array([-75.60,-75.55]),np.array([6.225,6.225]),line_style='k-')

        display.plot_line_geo(np.array([-75.60,-75.355]),np.array([6.225,6.001]),line_style='k-')
        display.plot_line_geo(np.array([-75.55,-75.15]),np.array([6.280,6.225]),line_style='k-')

        # Indicate the radar location with a point
        # display.plot_point(RadarFile.longitude['data'][0], RadarFile.latitude['data'][0])

        pos1 = [0.27,0.27,0.445,0.445]
        ax1.set_position(pos1)
        
        # -----
        # Stations
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': 'Arial Narrow'\
            ,'xtick.labelsize': 15,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': 15,'ytick.major.size': 12,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})

        Var = 'CodesNames'
        GraphL = 0.19
        GraphH = 0.17
        Rows = [0.80,0.56,0.31,0.06]
        Col = [0.04,0.38,0.72]
        x = 0
        xx = 0
        for R in Rows:
            for C in Col:
                if xx == 4 or xx == 7:
                    xx += 1
                    continue
                ID = self.ID[x]
                pos2 = [C,R,GraphL,GraphH]
                ax = host_subplot(111, axes_class=AA.Axes)
                ax.set_position(pos2)
                DataV = {'Tiempo':[DateTP]}
                DataKeyV = ['Tiempo']
                self.EventsSeriesGen(ax,self.DataSt[ID]['FechaCP'],self.DataSt[ID],
                        DataV,DataKeyV,DataKey=['Prec','Pres_F','T_F'],
                    PathImg='',Name=self.St_Info[ID][Var],NameArch='',
                    GraphInfo={'ylabel':['Precipitación [mm]','Presión [hPa]','Temperatura [°C]'],
                        'color':['b','k','r'],'label':['Precipitación','Presión','Temperatura']},
                    GraphInfoV={'color':['-.r'],'label':['Inicio del Evento']},
                    flagBig=True,
                    vm={'vmax':[self.vmax['Prec'],self.vmax['Pres_F'],self.vmax['T_F']],
                        'vmin':[self.vmin['Prec'],self.vmin['Pres_F'],self.vmin['T_F']]},Ev=0,flagV=True,
                    flagAverage=False,dt=1,Date='',flagEvent=False)
                x += 1
                xx += 1

        # Se grafica el mapa pequeño
        axx = plt.axes([0.51,0.27,0.2,0.2])
        display.plot_ppi_map('DBZH', 0,vmin=vmin,vmax=vmax,
                 projection='lcc',
                 min_lon=-75.60,max_lon=-75.55,
                 min_lat=6.225,max_lat=6.280,
                 resolution='i',
                 lat_0=RadarFile.latitude['data'][0],
                 lon_0=RadarFile.longitude['data'][0],
                 title='',
                 colorbar_flag=False,colorbar_label='Reflectividad horizontal equivalente [dBZ]',
                 ax=axx,title_flag=False)

        for iID,ID in enumerate(self.ID):
            if ID == '68' or ID == '202' or ID == '201' or ID == '203':
                display.plot_point(self.St_Info[ID]['Longitud'],self.St_Info[ID]['Latitud'],
                        symbol='o',label_text=ID,color='gray',
                        markersize=15,label_offset=[-0.005,0.003])

        # plot range rings at 10, 20, 30 and 40km
        display.plot_range_ring(10.0, line_style='k-')
        display.plot_range_ring(20.0, line_style='k--')
        display.plot_range_ring(30.0, line_style='k-')
        display.plot_range_ring(50.0, line_style='k--')

        # plots cross hairs
        display.plot_line_xy(np.array([-40000.0, 40000.0]), np.array([0.0, 0.0]),
                             line_style='k-')
        display.plot_line_xy(np.array([0.0, 0.0]), np.array([-20000.0, 200000.0]),
                             line_style='k-')

        return

    def EventsSeriesGen(self,ax,DatesEv,Data,DataV,DataKeyV,DataKey=None,
            PathImg='',Name='',NameArch='',
            GraphInfo={'ylabel':['Precipitación [mm]'],'color':['b'],'label':['Precipitación']},
            GraphInfoV={'color':['-.b'],'label':['Inicio del Evento']},
            flagBig=False,vm={'vmax':[],'vmin':[]},Ev=0,flagV=True,
            flagAverage=False,dt=1,Date='',flagEvent=False):
        '''
        DESCRIPTION:

            Esta función realiza los gráficos de los diferentes eventos
            solamente de los eventos de precipitación.
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
            if not(isinstance(DatesEv[0],datetime)):
                FechaEvv = DUtil.Dates_str2datetime(DatesEv)
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
        # fH=30 # Largo de la Figura
        # fV = fH*(2/3) # Ancho de la Figura


        lensize=17
        plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
            ,'font.sans-serif': 'Arial Narrow'\
            ,'xtick.labelsize': 13,'xtick.major.size': 6,'xtick.minor.size': 4\
            ,'xtick.major.width': 1,'xtick.minor.width': 1\
            ,'ytick.labelsize': 15,'ytick.major.size': 12,'ytick.minor.size': 4\
            ,'ytick.major.width': 1,'ytick.minor.width': 1\
            ,'axes.linewidth':1\
            ,'grid.alpha':0.1,'grid.linestyle':'-'})

        # plt.tick_params(
        #     axis='x',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     bottom='off',      # ticks along the bottom edge are off
        #     top='off',         # ticks along the top edge are off
        #     labelbottom='off') 

        plt.xticks(rotation=45)
        # f = plt.figure(figsize=DM.cm2inch(fH,fV))
        # ax = host_subplot(111, axes_class=AA.Axes)
        ax.tick_params(axis='x',which='both',bottom='on',top='off',\
            labelbottom='off',direction='out')
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
            # ax.axis["bottom"].set_visible(False)
            # ax.axes().get_xaxis().set_visible(False)
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

        # Se grafican las líneas verticales
        if flagV:
            for ilab,lab in enumerate(DataKeyV):
                ax.plot([DataV[lab][0],DataV[lab][0]],[yTL[0],yTL[-1]],
                        GraphInfoV['color'][ilab],label=GraphInfoV['label'][ilab])

        # Se organizan los ejes 
        MyL = (yTL[1]-yTL[0])/5 # minorLocatory value
        minorLocatory = MultipleLocator(MyL)
        ax.yaxis.set_minor_locator(minorLocatory)

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
                        offset = 60
                        new_fixed_axis = axi[ilab-1].get_grid_helper().new_fixed_axis
                        axi[ilab-1].axis["right"] = new_fixed_axis(loc="right",
                                                        axes=axi[ilab-1],
                                                        offset=(offset, 0))
                        axi[ilab-1].axis["right"].label.set_color(color=GraphInfo['color'][ilab])
                    elif ilab == 3:
                        # axi[ilab-1].spines['right'].set_position(('axes',-0.25))
                        offset = -65
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
        ax.set_title(Name)
        return

    def __str__(self):
        '''
        '''
        a = '\nDATA INFORMATION:'
        b = '\n DATES ADDED FROM: '+self.Dates.str[0] +' TO '+ self.Dates.str[-1]
        c = '\n SHAPE OF THE DATA ADDED: '+str(self.Data['Data'].shape)
        return a+b+c

    def ShowError(self,fn,cl,msg):
        '''
        DESCRIPTION:

            This method manages errors, and shows them. 
        _______________________________________________________________________
        INPUT:
            :param fn:  A str, Function that produced the error.
            :param cl:  A str, Class that produced the error.
            :param msg: A str, Message of the error.
        _______________________________________________________________________
        OUTPUT:
           :return: An int, Error managment -1. 
        '''

        raise Exception('ERROR: Method <'+fn+'> Class <'+cl+'>: '+msg)


