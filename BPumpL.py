# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#						Coded by Daniel González Duque
#						    Last revised 04/03/2016
#______________________________________________________________________________
#______________________________________________________________________________

#______________________________________________________________________________
#
# DESCRIPCIÓN DE LA CLASE:
# 	En esta clase se incluyen las rutinas para realizar el estudio relacioneado
#	con la Bomba biótica de humedad.
#
#	Estos códigos serán utilizados para el desarrollo de la Tesis de Maestría
#	del estudiante Daniel González Duque.
#______________________________________________________________________________

import numpy as np
import sys
import csv
import xlrd # Para poder abrir archivos de Excel
import xlsxwriter as xlsxwl
import scipy.io as sio # Para poder trabajar con archivos .mat
from scipy import stats as st # Para realizar las regresiones
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates # Manejo de dateticks
import matplotlib.mlab as mlab
import time
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from mpl_toolkits.mplot3d import Axes3D
import os
from Hydro_Plotter import Hydro_Plotter

import warnings

# Se importan los paquetes para manejo de fechas
from datetime import date, datetime, timedelta

from UtilitiesDGD import UtilitiesDGD
from CorrSt import CorrSt
from CFitting import CFitting

utl = UtilitiesDGD()
cr = CorrSt()
HyPl = Hydro_Plotter()
CF = CFitting()

class BPumpL:

	def __init__(self):

		'''
			DESCRIPTION:

		Este es el constructor por defecto, no realiza ninguna acción.
		'''
	def GraphIDEA(self,FechaN,PrecC,TempC,HRC,PresBC,Var,PathImg,Tot,V=1):
		'''
			DESCRIPTION:
		
		Con esta función se pretende graficar los datos extraídos de las estaciones
		de IDEA para poder visualizar las series completas.
		_________________________________________________________________________

			INPUT:
		+ FechaN: Vector de fechas con los valores en formato datetime.
		+ PrecC: Precipitación completa.
		+ TempC: Temperatura completa.
		+ HRC: Humedad relativa completa.
		+ PresBC: Presión Barométrica completa.
		+ Var: Nombre de variables a gráficar.
		+ PathImg: Ruta para guardar las imágenes.
		+ V: número de la figura.
		_________________________________________________________________________
		
			OUTPUT:
		Esta función libera un subplot con todas las figuras de las diferentes 
		variables.
		'''
		# Se crea la ruta en donde se guardarán los archivos
		utl.CrFolder(PathImg+ 'Manizales/Series/')
		plt.close('all')
		lfs = 16
		fig, axs = plt.subplots(2,2, figsize=(15, 8), facecolor='w', edgecolor='k')
		plt.rcParams.update({'font.size': lfs-6})
		axs = axs.ravel() # Para hacer un loop con los subplots
		#myFmt = mdates.DateFormatter('%d/%m/%y') # Para hacer el formato de fechas
		#xlabels = ['Ticklabel %i' % i for i in range(10)] # Para rotar los ejes
		for i in range(4):
			if i == 0:
				P1 = axs[i].plot(FechaN,PrecC,'b-', label = Var[i])
				axs[i].set_ylabel(Var[i] + '[mm]',fontsize=lfs-4)
				axs[i].legend(loc='best')
			elif i == 1:
				P1 = axs[i].plot(FechaN,TempC,'r-', label = Var[i])
				axs[i].set_ylabel(Var[i] + '[°C]',fontsize=lfs-4)
				axs[i].legend(loc='best')
			elif i == 2:
				P1 = axs[i].plot(FechaN,HRC,'g-', label = Var[i])
				axs[i].set_ylabel(Var[i] + '[%]',fontsize=lfs-4)
				axs[i].legend(loc='best')
				axs[i].set_xlabel(u'Fechas',fontsize=14)
			elif i == 3:
				P1 = axs[i].plot(FechaN,PresBC,'k-', label = Var[i])
				axs[i].set_ylabel(Var[i] + '[mmHg]',fontsize=lfs-4)
				axs[i].legend(loc='best')
				axs[i].set_xlabel(u'Fechas',fontsize=14)
			axs[i].set_title(Var[i],fontsize=lfs)
		plt.savefig(PathImg + 'Manizales/Series/' + Tot[71:-4] + '_Series_'+ str(V) +'.png' )
		plt.close('all')

	def NaNEl(self,V):
		'''
			DESCRIPTION:
		
		Con esta función se pretende interpolar los datos NaN que se encuentran
		cercanos y dejar los demás datos como NaN.
		_________________________________________________________________________

			INPUT:
		+ V: Variable que se quiere interpolar.
		_________________________________________________________________________
		
			OUTPUT:
		- VV: Datos con NaN interpolados.
		'''
		# En esta sección se eliminan o se interpolan los datos faltantes
		q = np.isnan(V)
		qq = np.where(q == True)[0]
		VV = V.copy() # Se crea la otra variable
		for ii,i in enumerate(qq):
			if ii == len(qq)-1:
				VV[i] = V[i] # Se dejan los datos NaN.
			else:
				if i == qq[ii+1]-1:
					VV[i] = V[i] # Se dejan los datos NaN.
				else:
					VV[i] = utl.Interp(1,V[i-1],2,3,V[i+1])

		return VV

	def ExEv(self,Prec,V,Fecha,Ci=60,Cf=60,m=0.8):
		'''
			DESCRIPTION:
		
		Con esta función se pretende realizar los diferentes diagramas de
		compuestos.
		_________________________________________________________________________

			INPUT:
		+ Prec: Precipitación
		+ V: Variable que se quiere tener en el diagrama de compuestos.
		+ Ci: Minutos atrás.
		+ Cf: Minutos adelante.
		+ m: Valor mínimo para extraer los datos.
		_________________________________________________________________________
		
			OUTPUT:
		- PrecC: Precipitación en compuestos.
		- VC: Diagrama de la variable.
		'''
		# Se inicializan las variables
		FechaEv = []
		Prec2 = Prec.copy()
		maxPrec = np.nanmax(Prec2)
		
		x = 0
		while maxPrec > m:
			q = np.where(Prec2 == maxPrec)[0]
			if x == 0:
				PrecC = Prec[q[0]-Ci:q[0]+Cf]
				VC = V[q[0]-Ci:q[0]+Cf]
			else:
				PrecC = np.vstack((PrecC,Prec[q[0]-Ci:q[0]+Cf]))
				VC = np.vstack((VC,V[q[0]-Ci:q[0]+Cf]))
			FechaEv.append(Fecha[q[0]-Ci:q[0]+Cf])

			Prec2[q[0]-Ci:q[0]+Cf] = np.nan
			maxPrec = np.nanmax(Prec2)
			x += 1

		return PrecC, VC,FechaEv

	def ExEvSea(self,PrecC,VC,FechaEv):
		'''
			DESCRIPTION:
		
		Con esta función se pretende separar los diferentes trimestres a partir de
		los diagramas de compuestos ya obtenidos.
		_________________________________________________________________________

			INPUT:
		+ PrecC: Diagrama de compuestos de Precipitación
		+ VC: Diagrama de compuestos de la variable que se quiere tener.
		_________________________________________________________________________
		
			OUTPUT:
		- PrecCS: Precipitación en compuestos por trimestre.
		- VCS: Diagrama de la variable por trimestre.
		- Fechas: Fechas en donde se dan los diferentes eventoss.
		'''

		# Se inician las variables de los trimestres
		PrecCS = dict()
		VCS = dict()
		Fechas = dict()

		Months=[]

		# Se extraen los datos de los diferentes trimestres.
		for i in range(len(FechaEv)):
			Months.append(FechaEv[i][0][5:7])

		x = [0 for k in range(4)]
		# Se extraen los diferentes trimestres
		for ii,i in enumerate(Months):
			M = int(i)
			if M == 1 or M == 2 or M == 12:
				if x[0] == 0:
					PrecCS[0] = PrecC[ii]
					VCS[0] = VC[ii]
					Fechas[0] = FechaEv[ii]
					x[0] += 1
				else:
					PrecCS[0] = np.vstack((PrecCS[0],PrecC[ii]))
					VCS[0] = np.vstack((VCS[0],VC[ii]))
					Fechas[0] = np.vstack((Fechas[0],FechaEv[ii]))
			if M == 3 or M == 4 or M == 5:
				if x[1] == 0:
					PrecCS[1] = PrecC[ii]
					VCS[1] = VC[ii]
					Fechas[1] = FechaEv[ii]
					x[1] += 1
				else:
					PrecCS[1] = np.vstack((PrecCS[1],PrecC[ii]))
					VCS[1] = np.vstack((VCS[1],VC[ii]))
					Fechas[1] = np.vstack((Fechas[1],FechaEv[ii]))
			if M == 6 or M == 7 or M == 8:
				if x[2] == 0:
					PrecCS[2] = PrecC[ii]
					VCS[2] = VC[ii]
					Fechas[2] = FechaEv[ii]
					x[2] += 1
				else:
					PrecCS[2] = np.vstack((PrecCS[2],PrecC[ii]))
					VCS[2] = np.vstack((VCS[2],VC[ii]))
					Fechas[2] = np.vstack((Fechas[2],FechaEv[ii]))
			if M == 9 or M == 10 or M == 11:
				if x[3] == 0:
					PrecCS[3] = PrecC[ii]
					VCS[3] = VC[ii]
					Fechas[3] = FechaEv[ii]
					x[3] += 1
				else:
					PrecCS[3] = np.vstack((PrecCS[3],PrecC[ii]))
					VCS[3] = np.vstack((VCS[3],VC[ii]))
					Fechas[3] = np.vstack((Fechas[3],FechaEv[ii]))

		return PrecCS, VCS, Fechas

	def ExEvDN(self,PrecC,VC,FechaEv,Mid):
		'''
			DESCRIPTION:
		
		Con esta función se pretende separar los eventos que se presentan de día
		y de noche.
		_________________________________________________________________________

			INPUT:
		+ PrecC: Diagrama de compuestos de Precipitación
		+ VC: Diagrama de compuestos de la variable que se quiere tener.
		+ FechaEv: Fecha de cada uno de los eventos.
		+ Mid: Valor del medio.
		_________________________________________________________________________
		
			OUTPUT:
		- PrecCS: Precipitación en compuestos por trimestre.
		- VCS: Diagrama de la variable por trimestre.
		- Fechas: Fechas en donde se dan los diferentes eventoss.
		'''

		# Se inician las variables de los trimestres
		PrecCS = dict()
		VCS = dict()
		Fechas = dict()

		Hours=[]

		# Se extraen los datos de las diferentes horas.
		for i in range(len(FechaEv)):
			Hours.append(FechaEv[i][Mid][11:13])

		x = [0 for k in range(2)]
		# Se extraen las diferentes horas.
		for ii,i in enumerate(Hours):
			M = int(i)

			if M >=6 and M <= 17:
				if x[0] == 0:
					PrecCS[0] = PrecC[ii]
					VCS[0] = VC[ii]
					Fechas[0] = FechaEv[ii]
					x[0] += 1
				else:
					PrecCS[0] = np.vstack((PrecCS[0],PrecC[ii]))
					VCS[0] = np.vstack((VCS[0],VC[ii]))
					Fechas[0] = np.vstack((Fechas[0],FechaEv[ii]))
			else:
				if x[1] == 0:
					PrecCS[1] = PrecC[ii]
					VCS[1] = VC[ii]
					Fechas[1] = FechaEv[ii]
					x[1] += 1
				else:
					PrecCS[1] = np.vstack((PrecCS[1],PrecC[ii]))
					VCS[1] = np.vstack((VCS[1],VC[ii]))
					Fechas[1] = np.vstack((Fechas[1],FechaEv[ii]))

		return PrecCS, VCS, Fechas

	def ExEvDH(self,PrecC,VC,FechaEv,Mid,Hi=6,Hf=17):
		'''
			DESCRIPTION:
		
		Con esta función se pretende separar los eventos a dos espacios de horas
		diferentes.
		_________________________________________________________________________

			INPUT:
		+ PrecC: Diagrama de compuestos de Precipitación
		+ VC: Diagrama de compuestos de la variable que se quiere tener.
		+ FechaEv: Fecha de cada uno de los eventos.
		+ Mid: Valor del medio.
		+ Hi: Hora inicial.
		+ Hf: Hora final.
		_________________________________________________________________________
		
			OUTPUT:
		- PrecCS: Precipitación en compuestos por trimestre.
		- VCS: Diagrama de la variable por trimestre.
		- Fechas: Fechas en donde se dan los diferentes eventoss.
		'''

		# Se inician las variables de los trimestres
		PrecCS = dict()
		VCS = dict()
		Fechas = dict()

		Hours=[]

		# Se extraen los datos de las diferentes horas.
		for i in range(len(FechaEv)):
			Hours.append(FechaEv[i][Mid][11:13])

		x = [0 for k in range(2)]
		# Se extraen las diferentes horas.
		for ii,i in enumerate(Hours):
			M = int(i)

			if M >=Hi and M <= Hf:
				if x[0] == 0:
					PrecCS[0] = PrecC[ii]
					VCS[0] = VC[ii]
					Fechas[0] = FechaEv[ii]
					x[0] += 1
				else:
					PrecCS[0] = np.vstack((PrecCS[0],PrecC[ii]))
					VCS[0] = np.vstack((VCS[0],VC[ii]))
					Fechas[0] = np.vstack((Fechas[0],FechaEv[ii]))
			else:
				if x[1] == 0:
					PrecCS[1] = PrecC[ii]
					VCS[1] = VC[ii]
					Fechas[1] = FechaEv[ii]
					x[1] += 1
				else:
					PrecCS[1] = np.vstack((PrecCS[1],PrecC[ii]))
					VCS[1] = np.vstack((VCS[1],VC[ii]))
					Fechas[1] = np.vstack((Fechas[1],FechaEv[ii]))

		return PrecCS, VCS, Fechas

	def ExEvENSO(self,PrecC,VC,FechaEv,Nino,Nina,Normal):
		'''
			DESCRIPTION:
		
		Con esta función se pretende separar los diferentes periodos en meses Niño,
		Niña y Normal.
		_________________________________________________________________________

			INPUT:
		+ PrecC: Diagrama de compuestos de Precipitación
		+ VC: Diagrama de compuestos de la variable que se quiere tener.
		+ FechaEv: Fecha de los eventos.
		+ Nino: Matriz para los meses Niño, Se debe incluir una matriz con filas
				los años y columnas los diferentes meses.
		+ Nina: Matriz para los meses Niña, Se debe incluir una matriz con filas
				los años y columnas los diferentes meses.
		+ Normal: Matriz para los meses Normales, Se debe incluir una matriz con filas
				  los años y columnas los diferentes meses.
		Las matrices de los años Niño, Niña y Normal deben estar desde 1950
		_________________________________________________________________________
		
			OUTPUT:
		- PrecCS: Precipitación en compuestos por trimestre.
		- VCS: Diagrama de la variable por trimestre.
		- Fechas: Fechas en donde se dan los diferentes eventoss.
		'''

		# Se inician las variables de los trimestres
		PrecCS = dict()
		VCS = dict()
		Fechas = dict()

		Months = []
		Year = []
		# Se extraen los datos de los diferentes trimestres.
		for i in range(len(FechaEv)):
			Months.append(FechaEv[i][0][5:7])
			Year.append(FechaEv[i][0][0:4])

		
		YearsN = [k for k in range(1950,2051)]
		YearsN = np.array(YearsN)

		x1 = 0
		x2 = 0
		x3 = 0
		# Niño 0, Niña 1, Normal 2.

		# Se extraen los diferentes periodos
		for ii,i in enumerate(Months):
			M = int(i)
			Y = int(Year[ii])
			if Y == 2016:
				continue
			else: 
				x = np.where(YearsN == Y)[0]
				
				if Nino[x,M-1] == 1:
					if x1 == 0:
						PrecCS[0] = PrecC[ii]
						VCS[0] = VC[ii]
						Fechas[0] = FechaEv[ii]
						x1 += 1
					else:
						PrecCS[0] = np.vstack((PrecCS[0],PrecC[ii]))
						VCS[0] = np.vstack((VCS[0],VC[ii]))
						Fechas[0] = np.vstack((Fechas[0],FechaEv[ii]))
				if Nina[x,M-1] == 1:
					if x2 == 0:
						PrecCS[1] = PrecC[ii]
						VCS[1] = VC[ii]
						Fechas[1] = FechaEv[ii]
						x2 += 1
					else:
						PrecCS[1] = np.vstack((PrecCS[1],PrecC[ii]))
						VCS[1] = np.vstack((VCS[1],VC[ii]))
						Fechas[1] = np.vstack((Fechas[1],FechaEv[ii]))

				if Normal[x,M-1] == 1:
					if x3 == 0:
						PrecCS[2] = PrecC[ii]
						VCS[2] = VC[ii]
						Fechas[2] = FechaEv[ii]
						x3 += 1
					else:
						PrecCS[2] = np.vstack((PrecCS[2],PrecC[ii]))
						VCS[2] = np.vstack((VCS[2],VC[ii]))
						Fechas[2] = np.vstack((Fechas[2],FechaEv[ii]))

		return PrecCS, VCS, Fechas

	def graphEv(self,Prec_Ev,Pres_F_Ev,T_F_Ev,V1,V2,V3,V11,V22,V33,Ax1,Ax2,Ax3,L1,L2,L3,L11,L22,L33,ii=1,ix='pos',PathImg='',FlagT=True,DTT='5',flagLim=False,Lim='none',Lim1=0,Lim2=0,Lim3=0):
		'''
			DESCRIPTION:
		
		Con esta función se pretende realizar graficos de los 6 primeros eventos
		del diagrama de compuestos
		_________________________________________________________________________

			INPUT:
		+ Prec_Ev: Diagrama de compuestos de Precipitación.
		+ Pres_F_Ev: Diagrama de compuestos de la variable.
		+ V1: Variable 1.
		+ V2: Variable 2.
		+ V11: Label variable 1
		+ V22: Label variable 2
		+ AX1: y1label.
		+ AX2: y2label.
		+ L1: Line type.
		+ L2: Line type.
		+ ii: Número Figura.
		+ ix: Estación.
		_________________________________________________________________________
		
			OUTPUT:
		Esta función libera 3 gráficas.
		'''
		warnings.filterwarnings('ignore')
		# ---- 

		# Se crea la ruta en donde se guardarán los archivos
		utl.CrFolder(PathImg)
		utl.CrFolder(PathImg + 'Average/')
		utl.CrFolder(PathImg + 'Histograms/')

		# Se organizan las variables para generar el 0 en el punto máximo de la
		# precipitación.

		Time = Prec_Ev.shape[1]
		Time_G = np.arange(-Time/2,Time/2)

		# Valores generales para los gráficos
		Afon = 14; Tit = 18; Axl = 16
		# Se grafican los primeros 6 eventos 

		f, ((ax11,ax12,ax13), ((ax21,ax22,ax23))) = plt.subplots(2,3, figsize=(20,10))
		plt.rcParams.update({'font.size': Afon})
		# Precipitación Ev 1
		a11 = ax11.plot(Time_G,Prec_Ev[0],L1, label = V1)
		ax11.set_title(r"Evento 1",fontsize=Tit)
		#ax11.set_xlabel("Tiempo [cada 5 min]",fontsize=Axl)
		ax11.set_ylabel(Ax1,fontsize=Axl)
		#ax11.set_xlim([Time_G[0],Time_G[len(Prec_Ev[0])-1]+1])
		# Presión barométrica
		axx11 = ax11.twinx()
		a112 = axx11.plot(Time_G,Pres_F_Ev[0],L2, label = V2)
		#axx11.set_ylabel("Presión [hPa]",fontsize=Axl)

		# added these three lines
		lns = a11+a112
		labs = [l.get_label() for l in lns]
		#ax11.legend(lns, labs, loc=0)

		# Precipitación Ev 2
		a12 = ax12.plot(Time_G,Prec_Ev[1],L1, label = u'Precipitación')
		ax12.set_title(r"Evento 2",fontsize=Tit)
		#ax11.set_xlabel("Tiempo [cada 5 min]",fontsize=Axl)
		#ax12.set_ylabel("Precipitación [mm]",fontsize=Axl)

		# Presión barométrica
		axx12 = ax12.twinx()
		a122 = axx12.plot(Time_G,Pres_F_Ev[1],L2, label = u'Presión Barométrica')
		#axx11.set_ylabel("Presión [hPa]",fontsize=Axl)

		# added these three lines
		lns = a12+a122
		labs = [l.get_label() for l in lns]
		#ax12.legend(lns, labs, loc=0)

		# Precipitación Ev 3
		a13 = ax13.plot(Time_G,Prec_Ev[2],L1, label = V1)
		ax13.set_title(r"Evento 3",fontsize=Tit)
		#ax11.set_xlabel("Tiempo [cada 5 min]",fontsize=Axl)
		#ax12.set_ylabel("Precipitación [mm]",fontsize=Axl)

		# Presión barométrica
		axx13 = ax13.twinx()
		a132 = axx13.plot(Time_G,Pres_F_Ev[2],L2, label = V2)
		axx13.set_ylabel(Ax2,fontsize=Axl)

		# added these three lines
		lns = a13+a132
		labs = [l.get_label() for l in lns]
		ax13.legend(lns, labs, loc=1)
		

		# Precipitación Ev 4
		a21 = ax21.plot(Time_G,Prec_Ev[3],L1, label = u'Precipitación')
		ax21.set_title(r"Evento 4",fontsize=Tit)
		ax21.set_xlabel("Tiempo [cada "+ DTT + " min]",fontsize=Axl)
		ax21.set_ylabel(Ax1,fontsize=Axl)

		# Presión barométrica
		axx21 = ax21.twinx()
		a212 = axx21.plot(Time_G,Pres_F_Ev[3],L2, label = u'Presión Barométrica')
		#axx11.set_ylabel("Presión [hPa]",fontsize=Axl)

		# added these three lines
		lns = a21+a212
		labs = [l.get_label() for l in lns]
		#ax11.legend(lns, labs, loc=0)

		# Precipitación Ev 5
		a22 = ax22.plot(Time_G,Prec_Ev[4],L1, label = u'Precipitación')
		ax22.set_title(r"Evento 5",fontsize=Tit)
		ax22.set_xlabel("Tiempo [cada "+ DTT + " min]",fontsize=Axl)
		#ax12.set_ylabel("Precipitación [mm]",fontsize=Axl)

		# Presión barométrica
		axx22 = ax22.twinx()
		a222 = axx22.plot(Time_G,Pres_F_Ev[4],L2, label = u'Presión Barométrica')
		#axx11.set_ylabel("Presión [hPa]",fontsize=Axl)

		# added these three lines
		lns = a22+a222
		labs = [l.get_label() for l in lns]
		#ax12.legend(lns, labs, loc=0)
		ev = 5
		# Precipitación Ev 3
		a23 = ax23.plot(Time_G,Prec_Ev[ev],L1, label = u'Precipitación')
		ax23.set_title(r"Evento %s" %(ev+1),fontsize=Tit)
		ax23.set_xlabel("Tiempo [cada "+ DTT + " min]",fontsize=Axl)
		#ax12.set_ylabel("Precipitación [mm]",fontsize=Axl)

		# Presión barométrica
		axx23 = ax23.twinx()
		a232 = axx23.plot(Time_G,Pres_F_Ev[ev],L2, label = u'Presión Barométrica')
		axx23.set_ylabel(Ax2,fontsize=Axl)

		# added these three lines
		lns = a23+a232
		labs = [l.get_label() for l in lns]
		#ax13.legend(lns, labs, loc=0)
		plt.tight_layout()
		plt.savefig(PathImg + ix + '_' + V11 + 'V' + V22 + '_Ev_' + str(ii) + '.png')
		plt.close('all')

		# -----------------------
		# Se saca el promedio de los eventos
		Prec_EvM = np.nanmean(Prec_Ev,axis=0)
		Pres_F_EvM = np.nanmean(Pres_F_Ev,axis=0)
		T_F_EvM = np.nanmean(T_F_Ev,axis=0)
		# Desviaciones
		Prec_EvD = np.nanstd(Prec_Ev,axis=0)
		Pres_F_EvD = np.nanstd(Pres_F_Ev,axis=0)
		T_F_EvD = np.nanstd(T_F_Ev,axis=0)
		# Error
		PrecNT = []
		PresNT = []
		TNT = []

		for i in range(len(Prec_Ev[0])):
			PrecNT.append(sum(~np.isnan(Prec_Ev[:,i])))
			PresNT.append(sum(~np.isnan(Pres_F_Ev[:,i])))
			TNT.append(sum(~np.isnan(T_F_Ev[:,i])))

		Prec_EvE = [k/np.sqrt(PresNT[ii]) for ii,k in enumerate(Prec_EvD)]
		Pres_F_EvE = [k/np.sqrt(PresNT[ii]) for ii,k in enumerate(Pres_F_EvD)]
		T_F_EvE = [k/np.sqrt(TNT[ii]) for ii,k in enumerate(T_F_EvD)]

		Prec_EvE = np.array(Prec_EvE)
		Pres_F_EvE = np.array(Pres_F_EvE)
		T_F_EvE = np.array(T_F_EvE)

		# -----------------------

		f, (ax11) = plt.subplots( figsize=(20,10))
		plt.rcParams.update({'font.size': 18})
		# Precipitación
		a11 = ax11.plot(Time_G,Prec_EvM,L1, label = V1)
		ax11.set_title(r'Promedio de Eventos',fontsize=24)
		ax11.set_xlabel("Tiempo [cada "+ DTT + " min]",fontsize=20)
		ax11.set_ylabel(Ax1,fontsize=20)

		# Presión barométrica
		axx11 = ax11.twinx()

		a112 = axx11.plot(Time_G,Pres_F_EvM,L2, label = V2)
		axx11.set_ylabel(Ax2,fontsize=20)

		# added these three lines
		lns = a11+a112
		labs = [l.get_label() for l in lns]
		ax11.legend(lns, labs, loc=0)

		plt.savefig(PathImg +'Average/' + ix + '_' + 'P_' + V11 + 'V' + V22 +'_' + str(ii) + '.png')
		plt.close('all')
		xx = Time_G
		
		# -----------------------
		if V11 == 'Prec':
			if FlagT==True:

				

				# Se obtienen las correlaciones
				CCP,CCS,QQ = cr.CorrC(Pres_F_EvM,T_F_EvM,True,0)
				CCP2,CCS,QQ2 = cr.CorrC(Pres_F_EvM,Prec_EvM,True,0)
				#print(CCP2)
				QQMP = np.max(QQ[0],QQ[2])
				QQMP2 = np.max(QQ2[0],QQ2[2])

				# Promedio de tres eventos
				#f, ax11 = plt.subplots(111, axes_class=AA.Axes, figsize=(20,10))
				f = plt.figure(figsize=(20,10))
				ax11 = host_subplot(111, axes_class=AA.Axes)
				plt.rcParams.update({'font.size': 18})
				# Precipitación
				a11 = ax11.errorbar(xx, Prec_EvM, yerr=Prec_EvE, fmt='o-', color=L11, label = V1)		
				ax11.set_title(r'Promedio de Eventos',fontsize=24)
				if DTT == '1 h':
					ax11.set_xlabel("Tiempo [cada "+ DTT + "]",fontsize=20)
				else:
					ax11.set_xlabel("Tiempo [cada "+ DTT + " min]",fontsize=20)
				ax11.set_ylabel(Ax1,fontsize=20)

				# Presión barométrica
				axx11 = ax11.twinx()
				axxx11 = ax11.twinx()

				a112 = axx11.errorbar(xx, Pres_F_EvM, yerr=Pres_F_EvE, fmt='o-', color=L22, label = V2)
				a113 = axxx11.errorbar(xx,T_F_EvM,yerr=T_F_EvE,fmt='o-',color=L33, label = V3)

				axx11.set_ylabel(Ax2,fontsize=20)
				axxx11.set_ylabel(Ax3,fontsize=20)

				offset = 80
				new_fixed_axis = axxx11.get_grid_helper().new_fixed_axis
				axxx11.axis["right"] = new_fixed_axis(loc="right",
												axes=axxx11,
												offset=(offset, 0))

				axxx11.axis["right"].toggle(all=True)
				# added these three lines
				# lns = a11+a112+a113
				# labs = [l.get_label() for l in lns]
				# ax11.legend(lns, labs, loc=0)
				ax11.legend(loc=0)

				ax11.axis["left"].label.set_color(color=L11)
				axx11.axis["right"].label.set_color(color=L22)
				axxx11.axis["right"].label.set_color(color=L33)

				# Se incluyen las correlaciones
				# Valores para el posicionamiento
				LM = np.max(Prec_EvM)
				Lm= np.min(Prec_EvM)
				#L = (LM+Lm)/2
				
				
				if DTT == '5' or DTT == '15':
					L = LM
					SLP = 0.15
					SLS = 0.0
					Lx = -50
					Sx = 21
				elif DTT == '30':
					L = LM
					SLP = 0.30
					SLS = 0.15
					Lx = -10
					Sx = 4.5
				elif DTT == '1 h':
					L = LM-1
					SLP = 0.5
					SLS = 0.0
					Lx = 1
					Sx = 2

				
				FS = 20

				ax11.text(Lx,L+SLP, r'$r_{Pearson}(%s,%s)=$' %(V22,V33), fontsize=FS)
				ax11.text(Lx,L+SLS, r'$r_{Pearson}(%s,%s)=$' %(V11,V22), fontsize=FS)
				if CCP >= 0: # Cuando la correlación es positiva
					if CCP >= QQMP:
						ax11.text(Lx+Sx,L+SLP, r'$%s$' %(round(CCP,3)), fontsize=FS,color='blue')
					else:
						ax11.text(Lx+Sx,L+SLP, r'$%s$' %(round(CCP,3)), fontsize=FS,color='red')
				elif CCP <0: # Cuando la correlación es negativa
					if CCP <= QQMP:
						ax11.text(Lx+Sx,L+SLP, r'$%s$' %(round(CCP,3)), fontsize=FS,color='blue')
					else:
						ax11.text(Lx+Sx,L+SLP, r'$%s$' %(round(CCP,3)), fontsize=FS,color='red')

				if CCP2 >= 0: # Cuando la correlación es positiva
					if CCP2 >= QQMP2:
						ax11.text(Lx+Sx,L+SLS, r'$%s$' %(round(CCP2,3)), fontsize=FS,color='blue')
					else:
						ax11.text(Lx+Sx,L+SLS, r'$%s$' %(round(CCP2,3)), fontsize=FS,color='red')
				elif CCP2 <0: # Cuando la correlación es negativa
					if CCP2 <= QQMP2:
						ax11.text(Lx+Sx,L+SLS, r'$%s$' %(round(CCP2,3)), fontsize=FS,color='blue')
					else:
						ax11.text(Lx+Sx,L+SLS, r'$%s$' %(round(CCP2,3)), fontsize=FS,color='red')

				if Lim == 'all':
					ax11.set_ylim(Lim1)
					axx11.set_ylim(Lim2)
					axxx11.set_ylim(Lim3)
				elif Lim == '1':
					ax11.set_ylim(Lim1)
				elif Lim == '2':
					axx11.set_ylim(Lim2)
				elif Lim == '3':
					axxx11.set_ylim(Lim2)

				ax11.set_xlim([min(xx)-1,max(xx)+1])

				#plt.tight_layout()
				plt.savefig(PathImg+'Average/' + ix + '_' + 'PE_' + V11 + 'V' + V22+'V' + V33 +'_' + str(ii) + '.png')
				plt.close('all')

				f = plt.figure(figsize=(20,10))
				ax11 = host_subplot(111, axes_class=AA.Axes)
				plt.rcParams.update({'font.size': 18})
				# Precipitación
				a11 = ax11.errorbar(xx, Prec_EvM, yerr=Prec_EvD, fmt='o-', color=L11, label = V1)
				ax11.set_title(r'Promedio de Eventos',fontsize=24)
				if DTT == '1 h':
					ax11.set_xlabel("Tiempo [cada "+ DTT + "]",fontsize=20)
				else:
					ax11.set_xlabel("Tiempo [cada "+ DTT + " min]",fontsize=20)
				ax11.set_ylabel(Ax1,fontsize=20)

				# Presión barométrica
				axx11 = ax11.twinx()
				axxx11 = ax11.twinx()

				a112 = axx11.errorbar(xx, Pres_F_EvM, yerr=Pres_F_EvD, fmt='o-', color=L22, label = V2)
				a113 = axxx11.errorbar(xx,T_F_EvM,yerr=T_F_EvD,fmt='o-',color=L33, label = V3)

				axx11.set_ylabel(Ax2,fontsize=20)
				axxx11.set_ylabel(Ax3,fontsize=20)

				offset = 80
				new_fixed_axis = axxx11.get_grid_helper().new_fixed_axis
				axxx11.axis["right"] = new_fixed_axis(loc="right",
												axes=axxx11,
												offset=(offset, 0))

				axxx11.axis["right"].toggle(all=True)
				# added these three lines
				# lns = a11+a112+a113
				# labs = [l.get_label() for l in lns]
				# ax11.legend(lns, labs, loc=0)
				ax11.legend(loc=0)

				ax11.axis["left"].label.set_color(color=L11)
				axx11.axis["right"].label.set_color(color=L22)
				axxx11.axis["right"].label.set_color(color=L33)

				ax11.set_xlim([min(xx)-1,max(xx)+1])

				#plt.tight_layout()
				plt.savefig(PathImg +'Average/' + ix + '_' + 'PD_' + V11 + 'V' + V22+'V' + V33 +'_' + str(ii) + '.png')
				plt.close('all')



				# Se calculan los histogramas

				qPrec = ~np.isnan(Prec_Ev)
				qPres = ~np.isnan(Pres_F_Ev)
				qT = ~np.isnan(T_F_Ev)

				n = 100 # Número de bins
				PrecH,PrecBin = np.histogram(Prec_Ev[qPrec],bins=n); [float(i) for i in PrecH]
				PresH,PresBin = np.histogram(Pres_F_Ev[qPres],bins=n); [float(i) for i in PresH]
				TH,TBin = np.histogram(T_F_Ev[qT],bins=n); [float(i) for i in TH]

				PrecH = PrecH/float(PrecH.sum()); PresH = PresH/float(PresH.sum()); TH = TH/float(TH.sum())

				widthPres = 0.7 * (PresBin[1] - PresBin[0])
				centerPres = (PresBin[:-1] + PresBin[1:]) / 2

				widthT = 0.7 * (TBin[1] - TBin[0])
				centerT = (TBin[:-1] + TBin[1:]) / 2
				
				APres = np.nanmean(Pres_F_Ev)
				BPres = np.nanstd(Pres_F_Ev)
				CPres = st.skew(Pres_F_Ev[qPres])
				DPres = st.kurtosis(Pres_F_Ev[qPres])

				AT = np.nanmean(T_F_Ev)
				BT = np.nanstd(T_F_Ev)
				CT = st.skew(T_F_Ev[qT])
				DT = st.kurtosis(T_F_Ev[qT])

				# Se grafícan los histogramas
				lfs = 18
				fig, axs = plt.subplots(1,2, figsize=(15, 8), facecolor='w', edgecolor='k')
				plt.rcParams.update({'font.size': lfs-6})
				axs = axs.ravel() # Para hacer un loop con los subplots
				axs[0].bar(centerPres, PresH*100, align='center', width=widthPres)
				axs[0].set_title('Histograma de '+V2,fontsize=lfs)
				axs[0].set_xlabel(Ax2,fontsize=lfs-4)
				axs[0].set_ylabel('Probabilidad (%)',fontsize=lfs-4)
				axs[0].text(PresBin[1],max(PresH*100), r'$\mu=$ %s' %(round(APres,3)), fontsize=12)
				axs[0].text(PresBin[1],max(PresH*100)-0.5, r'$\sigma=$ %s' %(round(BPres,3)), fontsize=12)
				axs[0].text(PresBin[1],max(PresH*100)-1, r'$\gamma=$ %s' %(round(CPres,3)), fontsize=12)
				axs[0].text(PresBin[1],max(PresH*100)-1.5, r'$\kappa=$ %s' %(round(DPres,3)), fontsize=12)

				axs[1].bar(centerT, TH*100, align='center', width=widthT)
				axs[1].set_title('Histograma de '+V3,fontsize=lfs)
				axs[1].set_xlabel(Ax3,fontsize=lfs-4)

				axs[1].text(TBin[1],max(TH*100), r'$\mu=$ %s' %(round(AT,3)), fontsize=12)
				axs[1].text(TBin[1],max(TH*100)-0.5, r'$\sigma=$ %s' %(round(BT,3)), fontsize=12)
				axs[1].text(TBin[1],max(TH*100)-1, r'$\gamma=$ %s' %(round(CT,3)), fontsize=12)
				axs[1].text(TBin[1],max(TH*100)-1.5, r'$\kappa=$ %s' %(round(DT,3)), fontsize=12)

				plt.savefig(PathImg +'Histograms/' + ix + '_' + 'Hist_' + V22+'V' + V33 +'_' + str(ii) + '.png')
				plt.close('all')
				return PrecH,PrecBin,PresH,PresBin,TH,TBin
			else:
				# Se obtiene la correlació 
				CCP2,CCS,QQ2 = cr.CorrC(Pres_F_EvM,Prec_EvM,True,0)
				QQMP2 = np.max(QQ2[0],QQ2[2])

				# Promedio de tres eventos
				#f, ax11 = plt.subplots(111, axes_class=AA.Axes, figsize=(20,10))
				f = plt.figure(figsize=(20,10))
				ax11 = host_subplot(111, axes_class=AA.Axes)
				plt.rcParams.update({'font.size': 18})
				# Precipitación
				a11 = ax11.errorbar(xx, Prec_EvM, yerr=Prec_EvE, fmt='o-', color=L11, label = V1)		
				ax11.set_title(r'Promedio de Eventos',fontsize=24)
				ax11.set_xlabel("Tiempo [cada "+ DTT + " min]",fontsize=20)
				ax11.set_ylabel(Ax1,fontsize=20)

				# Presión barométrica
				axx11 = ax11.twinx()

				a112 = axx11.errorbar(xx, Pres_F_EvM, yerr=Pres_F_EvE, fmt='o-', color=L22, label = V2)

				axx11.set_ylabel(Ax2,fontsize=20)
				
				# added these three lines
				# lns = a11+a112+a113
				# labs = [l.get_label() for l in lns]
				# ax11.legend(lns, labs, loc=0)
				ax11.legend(loc=0)

				ax11.axis["left"].label.set_color(color=L11)
				axx11.axis["right"].label.set_color(color=L22)

				# Se incluyen las correlaciones
				# Valores para el posicionamiento
				LM = np.max(Prec_EvM)
				Lm= np.min(Prec_EvM)
				#L = (LM+Lm)/2
				L = LM
				
				SLP = 0.1
				SLS = 0.0
				
				Lx = 3
				if DTT == '30':
					Sx = 4.3
				elif DTT == '15':
					Sx = 8.5
				elif DTT == '5':
					Sx = 21
				elif DTT == '1':
					Sx = 90
				
				FS = 20

				ax11.text(Lx,L+SLS, r'$r_{Pearson}(%s,%s)=$' %(V11,V22), fontsize=FS)
				#ax11.text(3,3, r'$r_{Pearson}(%s,%s)=$' %(V11,V22), fontsize=FS)

				if CCP2 >= 0: # Cuando la correlación es positiva
					if CCP2 >= QQMP2:
						ax11.text(Lx+Sx,L+SLS, r'$%s$' %(round(CCP2,3)), fontsize=FS,color='blue')
					else:
						ax11.text(Lx+Sx,L+SLS, r'$%s$' %(round(CCP2,3)), fontsize=FS,color='red')
				elif CCP2 <0: # Cuando la correlación es negativa
					if CCP2 <= QQMP2:
						ax11.text(Lx+Sx,L+SLS, r'$%s$' %(round(CCP2,3)), fontsize=FS,color='blue')
					else:
						ax11.text(Lx+Sx,L+SLS, r'$%s$' %(round(CCP2,3)), fontsize=FS,color='red')

				#plt.tight_layout()
				plt.savefig(PathImg+'Average/' + ix + '_' + 'PE_' + V11 + 'V' + V22+'V' + V33 +'_' + str(ii) + '.png')
				plt.close('all')

				#f, ax11 = plt.subplots(111, axes_class=AA.Axes, figsize=(20,10))
				f = plt.figure(figsize=(20,10))
				ax11 = host_subplot(111, axes_class=AA.Axes)
				plt.rcParams.update({'font.size': 18})
				# Precipitación
				a11 = ax11.errorbar(xx, Prec_EvM, yerr=Prec_EvD, fmt='o-', color=L11, label = V1)
				ax11.set_title(r'Promedio de Eventos',fontsize=24)
				ax11.set_xlabel("Tiempo [cada "+ DTT + " min]",fontsize=20)
				ax11.set_ylabel(Ax1,fontsize=20)

				# Presión barométrica
				axx11 = ax11.twinx()
				
				a112 = axx11.errorbar(xx, Pres_F_EvM, yerr=Pres_F_EvD, fmt='o-', color=L22, label = V2)
				
				axx11.set_ylabel(Ax2,fontsize=20)
				
				# added these three lines
				# lns = a11+a112+a113
				# labs = [l.get_label() for l in lns]
				# ax11.legend(lns, labs, loc=0)
				ax11.legend(loc=0)

				ax11.axis["left"].label.set_color(color=L11)
				axx11.axis["right"].label.set_color(color=L22)


				#plt.tight_layout()
				plt.savefig(PathImg +'Average/' + ix + '_' + 'PD_' + V11 + 'V' + V22+'V' + V33 +'_' + str(ii) + '.png')
				plt.close('all')

	def graphEvA(self,Prec_Ev,Pres_F_Ev,T_F_Ev,V1,V2,V3,V11,V22,V33,Ax1,Ax2,Ax3,L1,L2,L3,L11,L22,L33,ii=1,ix='pos',PathImg='',flagPel=False,EvN=1,C1=0,C2=120):
		'''
			DESCRIPTION:
		
		Con esta función se pretende realizar graficos de los CE primeros eventos
		del diagrama de compuestos, adicionalmente puede sacar una información para
		tener los datos para hacer una película.
		_________________________________________________________________________

			INPUT:
		+ Prec_EV: Diagrama de compuestos de Precipitación.
		+ Pres_F_Ev: Diagrama de compuestos de la variable.
		+ V1: Variable 1.
		+ V2: Variable 2.
		+ V11: Label variable 1
		+ V22: Label variable 2
		+ AX1: y1label.
		+ AX2: y2label.
		+ L1: Line type.
		+ L2: Line type.
		+ ii: Número Figura.
		+ ix: Estación.
		+ PathImg: Ruta para guardar los Datos.
		+ flagPel: Activador de película.
		+ EvN: Número del evento que se le quiere hacer la película.
		_________________________________________________________________________
		
			OUTPUT:
		Esta función libera una 
		'''
		lfs = 14
		fig = plt.figure(figsize=(20,10))
		plt.rcParams.update({'font.size': lfs-6})
		# Evento 1
		for i in range(1,13):
			axs = fig.add_subplot(4,3,i, projection='3d')
			axs.plot(Pres_F_Ev[i-1],T_F_Ev[i-1],Prec_Ev[i-1])
			axs.set_title('Evento '+str(i),fontsize=lfs)
			axs.set_xlabel(Ax2,fontsize=lfs-4)
			axs.set_ylabel(Ax3,fontsize=lfs-4)
			axs.set_zlabel(Ax1,fontsize=lfs-4)
			#axs.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
		plt.tight_layout()
		plt.savefig(PathImg +'Atractors/' + ix + '_' + 'AtrEv_'+ V11 +'V' + V22+'V' + V33 +'_' + str(ii) + '.png')
		plt.close('all')

		lfs = 28
		fig = plt.figure(figsize=(20,10))
		plt.rcParams.update({'font.size': lfs-6})
		axs = fig.add_subplot(111, projection='3d')
		# Evento 1
		CE = 20
		for i in range(1,CE+1):
			axs.plot(Pres_F_Ev[i-1],T_F_Ev[i-1],Prec_Ev[i-1])
		#axs.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
		axs.set_title(str(CE)+' Eventos',fontsize=lfs)
		axs.set_xlabel(Ax2,fontsize=lfs-4)
		axs.set_ylabel(Ax3,fontsize=lfs-4)
		axs.set_zlabel(Ax1,fontsize=lfs-4)
		plt.tight_layout()
		plt.savefig(PathImg +'Atractors/' + ix + '_' + 'AtrTEv_'+ V11 +'V' + V22+'V' + V33 +'_' + str(ii) + '.png')
		plt.close('all')

		if flagPel == True:
			# Se mira si la ruta existe o se crea
			if not os.path.exists(PathImg+'Atractors/Mov_1Ev/'+ix+'/'+str(EvN)):
				os.makedirs(PathImg+'Atractors/Mov_1Ev/'+ix+'/'+str(EvN))
			lfs = 28
			CE = 20
			i = EvN
			for pp in range(C1,C2+1):
				# fig = plt.figure(figsize=(20,10))
				# plt.rcParams.update({'font.size': lfs-10})
				# axs = fig.add_subplot(111, projection='3d')	
				# axs.plot(Pres_F_Ev[i-1][C1:C2+1],T_F_Ev[i-1][C1:C2+1],Prec_Ev[i-1][C1:C2+1])
				# if ~np.isnan(Pres_F_Ev[i-1][pp]) or ~np.isnan(T_F_Ev[i-1][pp]) or ~np.isnan(Prec_Ev[i-1][pp]):
				# 	if Prec_Ev[i-1][pp] > 0:
				# 		axs.scatter(Pres_F_Ev[i-1][pp],T_F_Ev[i-1][pp],Prec_Ev[i-1][pp],s=50,color='r')
				# 	else:
				# 		axs.scatter(Pres_F_Ev[i-1][pp],T_F_Ev[i-1][pp],Prec_Ev[i-1][pp],s=50)
				# #axs.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
				# axs.set_title('Evento Número '+str(i),fontsize=lfs)
				# axs.set_xlabel(Ax2,fontsize=lfs-4)
				# axs.set_ylabel(Ax3,fontsize=lfs-4)
				# axs.set_zlabel(Ax1,fontsize=lfs-4)
				# plt.gca().grid()
				# plt.tight_layout()
				# if pp <100:
				# 	if pp <10:
				# 		plt.savefig(PathImg +'Atractors/Mov_1Ev/' + ix + '/' + ix + '_' + 'AtrTEv_'+ V11 +'V' + V22+'V' + V33 + '_00' + str(pp) +'.png')
				# 	else:
				# 		plt.savefig(PathImg +'Atractors/Mov_1Ev/' + ix + '/' + ix + '_' + 'AtrTEv_'+ V11 +'V' + V22+'V' + V33 + '_0' + str(pp) +'.png')
				# else:
				# 	plt.savefig(PathImg +'Atractors/Mov_1Ev/' + ix + '/' + ix + '_' + 'AtrTEv_'+ V11 +'V' + V22+'V' + V33 + '_' + str(pp) +'.png')
				# plt.close('all')

				# Se hace con dos gráficas
				fig = plt.figure(figsize=(20,10))
				plt.rcParams.update({'font.size': lfs-10})
				axs = fig.add_subplot(121, projection='3d')	
				axs.plot(Pres_F_Ev[i-1][C1:C2+1],T_F_Ev[i-1][C1:C2+1],Prec_Ev[i-1][C1:C2+1])
				if ~np.isnan(Pres_F_Ev[i-1][pp]) or ~np.isnan(T_F_Ev[i-1][pp]) or ~np.isnan(Prec_Ev[i-1][pp]):
					if Prec_Ev[i-1][pp] > 0:
						axs.scatter(Pres_F_Ev[i-1][pp],T_F_Ev[i-1][pp],Prec_Ev[i-1][pp],s=50,color='r')
					else:
						axs.scatter(Pres_F_Ev[i-1][pp],T_F_Ev[i-1][pp],Prec_Ev[i-1][pp],s=50)
				#axs.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
				axs.set_title('Evento Número '+str(i),fontsize=lfs)
				axs.set_xlabel(Ax2,fontsize=lfs-4)
				axs.set_ylabel(Ax3,fontsize=lfs-4)
				axs.set_zlabel(Ax1,fontsize=lfs-4)

				axs = fig.add_subplot(122, projection='3d')	
				axs.plot(T_F_Ev[i-1][C1:C2+1],Pres_F_Ev[i-1][C1:C2+1],Prec_Ev[i-1][C1:C2+1])
				if ~np.isnan(Pres_F_Ev[i-1][pp]) or ~np.isnan(T_F_Ev[i-1][pp]) or ~np.isnan(Prec_Ev[i-1][pp]):
					if Prec_Ev[i-1][pp] > 0:
						axs.scatter(T_F_Ev[i-1][pp],Pres_F_Ev[i-1][pp],Prec_Ev[i-1][pp],s=50,color='r')
					else:
						axs.scatter(T_F_Ev[i-1][pp],Pres_F_Ev[i-1][pp],Prec_Ev[i-1][pp],s=50)
					#axs.scatter(T_F_Ev[i-1][pp],Pres_F_Ev[i-1][pp],Prec_Ev[i-1][pp],s=50)
				#axs.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
				axs.set_title('Evento Número '+str(i),fontsize=lfs)
				axs.set_xlabel(Ax3,fontsize=lfs-4)
				axs.set_ylabel(Ax2,fontsize=lfs-4)
				axs.set_zlabel(Ax1,fontsize=lfs-4)

				plt.gca().grid()
				plt.tight_layout()
				if pp <100:
					if pp <10:
						plt.savefig(PathImg +'Atractors/Mov_1Ev/' + ix + '/' + str(EvN) + '/' + ix + '_' + 'AtrTEv_'+ V11 +'V' + V22+'V' + V33 + '_00' + str(pp) +'.png')
					else:
						plt.savefig(PathImg +'Atractors/Mov_1Ev/' + ix + '/' + str(EvN) + '/' + ix + '_' + 'AtrTEv_'+ V11 +'V' + V22+'V' + V33 + '_0' + str(pp) +'.png')
				else:
					plt.savefig(PathImg +'Atractors/Mov_1Ev/' + ix + '/' + str(EvN) + '/' + ix + '_' + 'AtrTEv_'+ V11 +'V' + V22+'V' + V33 + '_' + str(pp) +'.png')
				plt.close('all')

	def RainDT(self,Prec):
		'''
		DESCRIPTION:
	
		Con esta función se pretende encontrar los diferentes tiempos de
		antes y durante las tormentas.
		_________________________________________________________________________

			INPUT:
		+ Prec: Precipitación.
		_________________________________________________________________________

			OUTPUT:

		'''
		# Se calculan las zonas en donde se presentan los diferentes eventos de 
		# precipitación.
		x = np.where(Prec >= 0.1)[0]
		ds = np.array([0.0])
		dp = np.array([0.0])
		In = np.array([0.0])
		ii = 0
		while ii <= len(x)-2:

			if ii == 0:
				# Se mira la cantidad de valores NaN antes de la tormenta.
				q = np.isnan(Prec[0:x[ii]])
				qq = sum(q)
				if qq <= 4:
					ds[0] = (x[0]-1)*5/60.0
					xx = 1 # Contador
					k = 0 # Activador
					while k == 0:
						a = x[ii+xx]
						if x[ii] == a-xx:
							xx += 1
						else:
							break
					
					dp[0] = xx*5/60.0
					In = np.nanmax(Prec[x[ii]:x[ii+xx]+1])				
					ii = ii + xx
				else:
					ii = ii + 1
				
			else:
				q = np.isnan(Prec[x[ii-1]+1:x[ii]])
				qq = sum(q)
				if qq <= 4:
					ds = np.hstack((ds,(x[ii]-x[ii-1])*5/60.0))
					xx = 1 # Contador
					k = 0 # Activador
					while k == 0:
						if ii+xx >= len(x)-2:
							break
						a = x[ii+xx]
						if x[ii] == a-xx:
							xx += 1
						else:
							break
						

					dp = np.hstack((dp,xx*5/60.0))
					In = np.hstack((In,np.nanmax(Prec[x[ii]:x[ii+xx]+1])))
					ii = ii + xx
				else:
					ii = ii + 1

		return ds, dp, In

	def PRvDP(self,PrecC,PresC,dt=1,M=60*4,flagEv=False,PathImg='',Name=''):
		'''
		DESCRIPTION:
	
		Con esta función se pretende encontrar la tasa de cambio de presión junto
		con la duración de las diferentes tormentas para luego ser gráficada por
		aparte.
		_________________________________________________________________________

			INPUT:
		+ PrecC: Diagrama de compuestos de precipitación.
		+ PresC: Diagrama de compuestos de presión barométrica.
		+ dt: delta de tiempo que se tienen en los diagramas de compuestos.
		+ M: Mitad en donde se encuentran los datos.
		+ flagEV: 
		_________________________________________________________________________

			OUTPUT:
		- DurPrec: Duración del evento de precipitación.
		- MaxPrec: Máximo de precipitación.
		- PresRateB: Tasa de cambio de presión antes.
		- PresRateA: Tasa de cambio de presión después.
		- PresChangeB: Cambio de presión antes.
		- PresChangeA: Cambio de presión después.
		- DurPres: Duración de presión.
		'''

		# Se inicializan las variables que se necesitan
		DurPrec = np.empty(len(PrecC))*np.nan
		MaxPrec = np.empty(len(PrecC))*np.nan
		PresRateB = np.empty(len(PrecC))*np.nan
		PresRateA = np.empty(len(PrecC))*np.nan
		PresChangeB = np.empty(len(PrecC))*np.nan
		PresChangeA = np.empty(len(PrecC))*np.nan
		#DurPres = np.empty(len(PrecC))*np.nan
		xx = []

		PosiminPres = []
		PosimaxPres = []
		PosimaxPresA = []

		PrecPre = []
		PrecPos = []
		# Ciclo para todos los diferentes eventos
		for i in range(len(PrecC)):
			x = 0
			xm = 0
			# Se encuentra se encuentra la posición del máximo de precipitación
			MaxPrec[i] = np.nanmax(PrecC[i,:])
			#x = np.where(PrecC[i,:] == MaxPrec[i])[0]
			xx.append(x)
			x = [M]
			# Se encuentra el mínimo de precipitación antes de ese punto
			xm = np.where(PrecC[i,:x[0]]<=0.10)[0]
			#print(xm)

			# Se mira si es mínimo de antes por 10 minutos consecutivos de mínimos
			k = 1
			a = len(xm)-1
			while k == 1:
				
				if dt == 1:
					utl.ExitError('PRvDP','BPumpL','No se ha realizado esta subrutina')
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
				else:
					if a == -1:
						xmm = 0
					else:
						xmm = xm[a]
					k = 2
				# elif dt == 1:
				# 	if xm[a] == xm[a-1]+1 and xm[a] == xm[a-2]+2 and xm[a] == xm[a-3]+3 and\
				# 		xm[a] == xm[a-4]+4 and:

			# Se encuentra el máximo de precipitación antes de ese punto
			xM = np.where(PrecC[i,x[0]+1:]<=0.10)[0]+x[0]+1


			print(x[0])
			#print('i='+str(i))
			print('xM='+str(xM))

			# Se busca el mínimo después del máximo
			k = 1
			a = 0
			while k == 1:
				aa = len(xM)
				if aa == 1 or aa == 0:
					xMM = len(PrecC[i,:])-1
					k = 2
					break
				if dt == 1:
					utl.ExitError('PRvDP','BPumpL','No se ha realizado esta subrutina')
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

			# print(xMM)
			# print(xmm)

			DurPrec[i] = (xMM-xmm+1)*dt/60

			PrecPre.append(xmm)
			PrecPos.append(xMM)

			# Se hace el proceso para los datos de presión
			if dt == 1:
				utl.ExitError('PRvDP','BPumpL','No se ha realizado esta subrutina')
			elif dt == 5:
				# Filtrado adicional de la serie
				qq = np.isnan(PrecC[i,:x[0]+1])
				sq = sum(qq)
				if sq >= x[0]/2 or xmm == 0:
					PresChangeB[i] = np.nan
					PresRateB[i] = np.nan
					PresChangeA[i] = np.nan
					PresRateA[i] = np.nan
				else:
					# Se encuentra el mínimo de presión antes del evento
					PresMin = np.nanmin(PresC[i,xmm-10:xmm+5]) # Valor del mínimo
					xpm = np.where(PresC[i,xmm-10:xmm+5] == PresMin)[0]+xmm-10 # Posición del mínimo

					# print('xpm=',xpm)
					# print('xmm=',xmm)

					# Se encuentra el cambio de presión antes del evento
					if xpm[0] <= 15:					
						PresMaxB = PresMin
						xpM = xpm
					else:
						try:
							PresMaxB = np.nanmax(PresC[i,xmm-35:xpm+1]) # Valor máximo antes
							xpM = np.where(PresC[i,xmm-35:xpm+1] == PresMaxB)[0]+xmm-35 # Posición del máximo antes
						except:
							PresMaxB = np.nanmax(PresC[i,:xpm+1]) # Valor máximo antes
							xpM = np.where(PresC[i,:xpm+1] == PresMaxB)[0] # Posición del máximo antes
					# print('Before')
					# print('xpm='+str(xpm))
					# print('xpM='+str(xpM))
					if np.isnan(PresMaxB) or np.isnan(PresMin) or xpM == xpm:
						PresChangeB[i] = np.nan
						PresRateB[i] = np.nan
					else:
						PresChangeB[i] = PresMaxB - PresMin
						PresRateB[i] = PresChangeB[i]/((xpM-xpm-1)*dt/60) # Rate en hPa/h
					
					PosiminPres.append(xpm)
					PosimaxPres.append(xpM)

					# print('PresChangeB='+str(PresChangeB[i]))
					# print('PresRateB='+str(PresRateB[i]))

					# Se encuentra el cambio de presión durante el evento.
					PresMaxA = np.nanmax(PresC[i,xpm:x[0]+10]) # Valor máximo
					xpM = np.where(PresC[i,xpm:x[0]+10] == PresMaxA)[0]+xpm # Posición del máximo antes

					if np.isnan(PresMaxA) or np.isnan(PresMin) or xpM == xpm:
						PresChangeA[i] = np.nan
						PresRateA[i] = np.nan
					else:
						PresChangeA[i] = PresMaxA - PresMin
						PresRateA[i] = PresChangeA[i]/((xpM-xpm+1)*dt/60) # Rate en hPa/h
					# if i == 426:
					# 	print('After')
					# 	print('xpm='+str(xpm))
					# 	print('xpM='+str(xpM))
					# 	print('PresChangeA='+str(PresChangeA[i]))
					# 	print('PresRateBA='+str(PresRateA[i]))

					PosimaxPresA.append(xpM)
			elif dt < 60 and dt > 5:
				# Filtrado adicional de la serie
				qq = np.isnan(PrecC[i,:x[0]+1])
				sq = sum(qq)
				if sq >= x[0]/2 or xmm <= 4:
					PresChangeB[i] = np.nan
					PresRateB[i] = np.nan
					PresChangeA[i] = np.nan
					PresRateA[i] = np.nan
				else:

					# print('Before')
					print('xmm='+str(xmm))

					# Se encuentra el mínimo de presión antes del evento
					PresMin = np.nanmin(PresC[i,xmm-4:xmm+2]) # Valor del mínimo
					xpm = np.where(PresC[i,xmm-4:xmm+2] == PresMin)[0]+xmm-4 # Posición del mínimo
					c = len(xpm)
					if c >=2:
						xpm = xpm[len(xpm)-1]
					print('xpm=',xpm)
					# print('xmm=',xmm)
					print('xMM=',xMM)
					
					if xpm <= 1:
					# Se encuentra el cambio de presión antes del evento
						PresMaxB = PresMin
						xpM = xpm
					else:
						try:
							PresMaxB = np.nanmax(PresC[i,xmm-5:xpm]) # Valor máximo antes
							#xpM = np.where(PresC[i,xmm-(dt-1)/3:xpm+1] == PresMaxB)[0]+xmm-(dt-1)/3 # Posición del máximo antes
							xpM = np.where(PresC[i,xmm-5:xpm] == PresMaxB)[0]+xmm-5 # Posición del máximo antes
							c = len(xpM)
							if c >=2:
								xpM = xpM[len(xpM)-1]
						except:
							PresMaxB = np.nanmax(PresC[i,:xpm+1]) # Valor máximo antes
							xpM = np.where(PresC[i,:xpm+1] == PresMaxB)[0] # Posición del máximo antes
							c = len(xpM)
							if c >=2:
								xpM = xpM[len(xpM)-1]

					print('xpM=',xpM)
					
					


					
					# print('xpm='+str(xpm))
					# print('xpM='+str(xpM))
					print(np.isnan(PresMaxB))
					print(np.isnan(PresMin))
					if np.isnan(PresMaxB) or np.isnan(PresMin) or xpM == xpm:
						PresChangeB[i] = np.nan
						PresRateB[i] = np.nan
					else:
						PresChangeB[i] = PresMaxB - PresMin
						PresRateB[i] = PresChangeB[i]/((xpM-xpm-1)*dt/60) # Rate en hPa/h
					
					PosiminPres.append(xpm)
					PosimaxPres.append(xpM)


					# print('PresChangeB='+str(PresChangeB[i]))
					# print('PresRateB='+str(PresRateB[i]))

					# Se encuentra el cambio de presión durante el evento.
					PresMaxA = np.nanmax(PresC[i,xmm+1:xMM]) # Valor máximo
					xpM = np.where(PresC[i,xmm+1:xMM] == PresMaxA)[0]+xmm+1 # Posición del máximo antes

					c = len(xpM)
					if c >=2:
						xpM = xpM[len(xpM)-1]

					if np.isnan(PresMaxA) or np.isnan(PresMin) or xpM == xpm:
						PresChangeA[i] = np.nan
						PresRateA[i] = np.nan
					else:
						PresChangeA[i] = PresMaxA - PresMin
						PresRateA[i] = PresChangeA[i]/((xpM-xpm+1)*dt/60) # Rate en hPa/h
					# if i == 426:
					# 	print('After')
					# 	print('xpm='+str(xpm))
					# 	print('xpM='+str(xpM))
					# 	print('PresChangeA='+str(PresChangeA[i]))
					# 	print('PresRateBA='+str(PresRateA[i]))


					PosimaxPresA.append(xpM)
			else:
				# Filtrado adicional de la serie
				qq = np.isnan(PrecC[i,:x[0]+1])
				sq = sum(qq)
				if sq >= x[0]/2 or xmm <= 1:
					PresChangeB[i] = np.nan
					PresRateB[i] = np.nan
					PresChangeA[i] = np.nan
					PresRateA[i] = np.nan
				else:

					# Se encuentra el mínimo de presión antes del evento
					PresMin = np.nanmin(PresC[i,xmm-2:xmm+3]) # Valor del mínimo
					xpm = np.where(PresC[i,xmm-2:xmm+3] == PresMin)[0]+xmm-2 # Posición del mínimo
					# print('xpm=',xpm)
					# print('xmm=',xmm)
					print(xpm)
					if xpm == 0:
					# Se encuentra el cambio de presión antes del evento
						PresMaxB = PresMin
						xpM = xpm
					else:
						try:
							PresMaxB = np.nanmax(PresC[i,xmm-(dt-1)/3:xpm+1]) # Valor máximo antes
							xpM = np.where(PresC[i,xmm-(dt-1)/3:xpm+1] == PresMaxB)[0]+xmm-(dt-1)/3 # Posición del máximo antes
						except:
							PresMaxB = np.nanmax(PresC[i,:xpm[0]+1]) # Valor máximo antes
							xpM = np.where(PresC[i,:xpm[0]+1] == PresMaxB)[0] # Posición del máximo antes
					# print('Before')
					# print('xpm='+str(xpm))
					# print('xpM='+str(xpM))
					if np.isnan(PresMaxB) or np.isnan(PresMin) or xpM == xpm:
						PresChangeB[i] = np.nan
						PresRateB[i] = np.nan
					else:
						PresChangeB[i] = PresMaxB - PresMin
						PresRateB[i] = PresChangeB[i]/((xpM-xpm-1)*dt/60) # Rate en hPa/h
					
					# print('PresChangeB='+str(PresChangeB[i]))
					# print('PresRateB='+str(PresRateB[i]))

					# Se encuentra el cambio de presión durante el evento.
					PresMaxA = np.nanmax(PresC[i,xpm:x[0]+5]) # Valor máximo
					xpM = np.where(PresC[i,xpm:x[0]+6] == PresMaxA)[0]+xpm # Posición del máximo antes

					if np.isnan(PresMaxA) or np.isnan(PresMin) or xpM == xpm:
						PresChangeA[i] = np.nan
						PresRateA[i] = np.nan
					else:
						PresChangeA[i] = PresMaxA - PresMin
						PresRateA[i] = PresChangeA[i]/((xpM-xpm+1)*dt/60) # Rate en hPa/h
					# if i == 426:
					# 	print('After')
					# 	print('xpm='+str(xpm))
					# 	print('xpM='+str(xpM))
					# 	print('PresChangeA='+str(PresChangeA[i]))
					# 	print('PresRateBA='+str(PresRateA[i]))

			


		if flagEv == True:
			print('\n Se desarrollan las gráficas')
			# Se crea la ruta en donde se guardarán los archivos
			utl.CrFolder(PathImg+Name+'/')
			# Tiempo
			Time = PrecC.shape[1]
			Time_G = np.arange(-Time/2,Time/2)
			
			# Valores generales para los gráficos
			Afon = 18; Tit = 22; Axl = 20

			for i in range(len(PrecC)):
				if i <= 946:
					f = plt.figure(figsize=(20,10))
					plt.rcParams.update({'font.size': Afon})
					ax11 = host_subplot(111, axes_class=AA.Axes)
					# Precipitación
					a11 = ax11.plot(Time_G,PrecC[i],'-b', label = 'Precipitación')
					ax11.set_title(Name+r" Evento " + str(i),fontsize=Tit)
					ax11.set_xlabel("Tiempo [cada 5 min]",fontsize=Axl)
					ax11.set_ylabel('Precipitación [mm]',fontsize=Axl)
					#ax11.set_xlim([Time_G[0],Time_G[len(Prec_Ev[0])-1]+1])
					# Presión barométrica
					axx11 = ax11.twinx()
					a112 = axx11.plot(Time_G,PresC[i],'-k', label = 'Presión Barométrica')
					axx11.set_ylabel("Presión [hPa]",fontsize=Axl)

					# print(PosiminPres[i])
					# print(PosimaxPres[i])
					# print(Time_G.shape)
					# Líneas de eventos
					L1 = ax11.plot([Time_G[PosiminPres[i]],Time_G[PosiminPres[i]]],[0,np.nanmax(PrecC[i])],'--b', label = 'Min Pres') # Punto mínimo
					L2 = ax11.plot([Time_G[PosimaxPres[i]],Time_G[PosimaxPres[i]]],[0,np.nanmax(PrecC[i])],'--r', label = 'Max Pres Antes') # Punto máximo B
					L3 = ax11.plot([Time_G[PosimaxPresA[i]],Time_G[PosimaxPresA[i]]],[0,np.nanmax(PrecC[i])],'--g', label = 'Max Pres Después') # Punto máximo A

					# Líneas para la precipitación
					L4 = ax11.plot([Time_G[PrecPre[i]],Time_G[PrecPre[i]]],[0,np.nanmax(PrecC[i])],'-.b', label = 'Inicio Prec') # Inicio del aguacero
					L5 = ax11.plot([Time_G[PrecPos[i]],Time_G[PrecPos[i]]],[0,np.nanmax(PrecC[i])],'-.g', label = 'Fin Prec') # Fin del aguacero

					# added these three lines
					lns = a11+a112+L1+L2+L3+L4+L5
					labs = [l.get_label() for l in lns]
					ax11.legend(lns, labs, loc=0)
					
					#plt.tight_layout()
					plt.savefig(PathImg + Name + '/' + Name + '_' + 'Ev_' + str(i) + '.png')
					plt.close('all')

		return DurPrec, MaxPrec, PresRateB,PresRateA,PresChangeB,PresChangeA,xx


	def PRvDP_T(self,PrecC,PresC,dt=1,M=60*4,flagEv=False,PathImg='',Name=''):
		'''
		DESCRIPTION:
	
		Con esta función se pretende encontrar la tasa de cambio de presión junto
		con la duración de las diferentes tormentas para luego ser gráficada por
		aparte.
		_________________________________________________________________________

			INPUT:
		+ PrecC: Diagrama de compuestos de precipitación.
		+ PresC: Diagrama de compuestos de presión barométrica.
		+ dt: delta de tiempo que se tienen en los diagramas de compuestos.
		+ M: Mitad en donde se encuentran los datos.
		+ flagEV: 
		_________________________________________________________________________

			OUTPUT:
		- DurPrec: Duración del evento de precipitación.
		- MaxPrec: Máximo de precipitación.
		- PresRateB: Tasa de cambio de presión antes.
		- PresRateA: Tasa de cambio de presión después.
		- PresChangeB: Cambio de presión antes.
		- PresChangeA: Cambio de presión después.
		- DurPres: Duración de presión.
		'''

		# Se inicializan las variables que se necesitan
		DurPrec = np.empty(len(PrecC))*np.nan
		MaxPrec = np.empty(len(PrecC))*np.nan
		PresRateB = np.empty(len(PrecC))*np.nan
		PresRateA = np.empty(len(PrecC))*np.nan
		PresChangeB = np.empty(len(PrecC))*np.nan
		PresChangeA = np.empty(len(PrecC))*np.nan
		#DurPres = np.empty(len(PrecC))*np.nan
		xx = []

		PosiminPres = []
		PosimaxPres = []
		PosimaxPresA = []

		PrecPre = []
		PrecPos = []
		# Ciclo para todos los diferentes eventos
		for i in range(len(PrecC)):
			x = 0
			xm = 0
			# Se encuentra se encuentra la posición del máximo de precipitación
			MaxPrec[i] = np.nanmax(PrecC[i,:])
			#x = np.where(PrecC[i,:] == MaxPrec[i])[0]
			xx.append(x)
			x = [M]
			# Se encuentra el mínimo de precipitación antes de ese punto
			xm = np.where(PrecC[i,:x[0]]<=0.10)[0]
			#print(xm)

			# Se mira si es mínimo de antes por 10 minutos consecutivos de mínimos
			k = 1
			a = len(xm)-1
			while k == 1:
				
				if dt == 1:
					utl.ExitError('PRvDP','BPumpL','No se ha realizado esta subrutina')
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
				else:
					if a == -1:
						xmm = 0
					else:
						xmm = xm[a]
					k = 2
				# elif dt == 1:
				# 	if xm[a] == xm[a-1]+1 and xm[a] == xm[a-2]+2 and xm[a] == xm[a-3]+3 and\
				# 		xm[a] == xm[a-4]+4 and:

			# Se encuentra el máximo de precipitación antes de ese punto
			xM = np.where(PrecC[i,x[0]+1:]<=0.10)[0]+x[0]+1


			#print(x[0])
			#print('i='+str(i))
			#print('xM='+str(xM))

			# Se busca el mínimo después del máximo
			k = 1
			a = 0
			while k == 1:
				aa = len(xM)
				if aa == 1 or aa == 0:
					xMM = len(PrecC[i,:])-1
					k = 2
					break
				if dt == 1:
					utl.ExitError('PRvDP','BPumpL','No se ha realizado esta subrutina')
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

			# print(xMM)
			# print(xmm)

			DurPrec[i] = (xMM-xmm+1)*dt/60

			PrecPre.append(xmm)
			PrecPos.append(xMM)

			# Se hace el proceso para los datos de presión
			if dt == 1:
				utl.ExitError('PRvDP','BPumpL','No se ha realizado esta subrutina')
			elif dt == 5:
				# Filtrado adicional de la serie
				qq = np.isnan(PrecC[i,:x[0]+1])
				sq = sum(qq)
				if sq >= x[0]/2 or xmm == 0:
					PresChangeB[i] = np.nan
					PresRateB[i] = np.nan
					PresChangeA[i] = np.nan
					PresRateA[i] = np.nan
				else:

					# Se encuentra el mínimo de presión antes del evento
					PresMin = np.nanmin(PresC[i,xmm-10:xmm+5]) # Valor del mínimo
					xpm = np.where(PresC[i,xmm-10:xmm+5] == PresMin)[0]+xmm-10 # Posición del mínimo

					# Se encuentra el cambio de presión antes del evento
					PresRateBM = [(PresC[i,xmm]-PresC[i,xmm-kk])/((kk)*dt/60) for kk in range(1,xmm+1)]
					PresChangeBM = [(PresC[i,xmm]-PresC[i,xmm-kk]) for kk in range(1,xmm+1)]
					PresRateB[i] = np.nanmin(PresRateBM)
					xpM = xmm-1-np.where(np.array(PresRateBM) == PresRateB[i])[0]
					m = np.where(np.array(PresRateBM) == PresRateB[i])[0]
					PresChangeB[i] = PresChangeBM[m]
					
					PosiminPres.append(xpm)
					PosimaxPres.append(xpM)

					# print('PresChangeB='+str(PresChangeB[i]))
					# print('PresRateB='+str(PresRateB[i]))

					# Se encuentra el cambio de presión durante el evento.
					PresRateAM = [(PresC[i,kk]-PresC[i,xmm])/((kk-xmm)*dt/60) for kk in range(xmm+1,len(PresC[i]))]
					PresChangeAM = [(PresC[i,kk]-PresC[i,xmm]) for kk in range(xmm+1,len(PresC[i]))]
					PresRateA[i] = np.nanmax(PresRateAM)
					xpM = xmm+np.where(np.array(PresRateAM) == PresRateA[i])[0]
					m = np.where(np.array(PresRateAM) == PresRateA[i])[0]
					PresChangeA[i] = PresChangeAM[m]

					PosimaxPresA.append(xpM)
			elif dt < 60 and dt > 5:
				# Filtrado adicional de la serie
				qq = np.isnan(PrecC[i,:x[0]+1])
				sq = sum(qq)
				if sq >= x[0]/2 or xmm <= 4:
					PresChangeB[i] = np.nan
					PresRateB[i] = np.nan
					PresChangeA[i] = np.nan
					PresRateA[i] = np.nan
				else:

					# print('Before')
					print('xmm='+str(xmm))

					# Se encuentra el mínimo de presión antes del evento
					PresMin = np.nanmin(PresC[i,xmm-4:xmm+2]) # Valor del mínimo
					xpm = np.where(PresC[i,xmm-4:xmm+2] == PresMin)[0]+xmm-4 # Posición del mínimo
					c = len(xpm)
					if c >=2:
						xpm = xpm[len(xpm)-1]
					print('xpm=',xpm)
					# print('xmm=',xmm)
					print('xMM=',xMM)
					
					if xpm <= 1:
					# Se encuentra el cambio de presión antes del evento
						PresMaxB = PresMin
						xpM = xpm
					else:
						try:
							PresMaxB = np.nanmax(PresC[i,xmm-5:xpm]) # Valor máximo antes
							#xpM = np.where(PresC[i,xmm-(dt-1)/3:xpm+1] == PresMaxB)[0]+xmm-(dt-1)/3 # Posición del máximo antes
							xpM = np.where(PresC[i,xmm-5:xpm] == PresMaxB)[0]+xmm-5 # Posición del máximo antes
							c = len(xpM)
							if c >=2:
								xpM = xpM[len(xpM)-1]
						except:
							PresMaxB = np.nanmax(PresC[i,:xpm+1]) # Valor máximo antes
							xpM = np.where(PresC[i,:xpm+1] == PresMaxB)[0] # Posición del máximo antes
							c = len(xpM)
							if c >=2:
								xpM = xpM[len(xpM)-1]

					print('xpM=',xpM)
					
					


					
					# print('xpm='+str(xpm))
					# print('xpM='+str(xpM))
					print(np.isnan(PresMaxB))
					print(np.isnan(PresMin))
					if np.isnan(PresMaxB) or np.isnan(PresMin) or xpM == xpm:
						PresChangeB[i] = np.nan
						PresRateB[i] = np.nan
					else:
						PresChangeB[i] = PresMaxB - PresMin
						PresRateB[i] = PresChangeB[i]/((xpM-xpm-1)*dt/60) # Rate en hPa/h
					
					PosiminPres.append(xpm)
					PosimaxPres.append(xpM)


					# print('PresChangeB='+str(PresChangeB[i]))
					# print('PresRateB='+str(PresRateB[i]))

					# Se encuentra el cambio de presión durante el evento.
					PresMaxA = np.nanmax(PresC[i,xmm+1:xMM]) # Valor máximo
					xpM = np.where(PresC[i,xmm+1:xMM] == PresMaxA)[0]+xmm+1 # Posición del máximo antes

					c = len(xpM)
					if c >=2:
						xpM = xpM[len(xpM)-1]

					if np.isnan(PresMaxA) or np.isnan(PresMin) or xpM == xpm:
						PresChangeA[i] = np.nan
						PresRateA[i] = np.nan
					else:
						PresChangeA[i] = PresMaxA - PresMin
						PresRateA[i] = PresChangeA[i]/((xpM-xpm+1)*dt/60) # Rate en hPa/h
					# if i == 426:
					# 	print('After')
					# 	print('xpm='+str(xpm))
					# 	print('xpM='+str(xpM))
					# 	print('PresChangeA='+str(PresChangeA[i]))
					# 	print('PresRateBA='+str(PresRateA[i]))


					PosimaxPresA.append(xpM)
			else:
				# Filtrado adicional de la serie
				qq = np.isnan(PrecC[i,:x[0]+1])
				sq = sum(qq)
				if sq >= x[0]/2 or xmm <= 1:
					PresChangeB[i] = np.nan
					PresRateB[i] = np.nan
					PresChangeA[i] = np.nan
					PresRateA[i] = np.nan
				else:

					# Se encuentra el mínimo de presión antes del evento
					PresMin = np.nanmin(PresC[i,xmm-2:xmm+3]) # Valor del mínimo
					xpm = np.where(PresC[i,xmm-2:xmm+3] == PresMin)[0]+xmm-2 # Posición del mínimo
					# print('xpm=',xpm)
					# print('xmm=',xmm)
					print(xpm)
					if xpm == 0:
					# Se encuentra el cambio de presión antes del evento
						PresMaxB = PresMin
						xpM = xpm
					else:
						try:
							PresMaxB = np.nanmax(PresC[i,xmm-(dt-1)/3:xpm+1]) # Valor máximo antes
							xpM = np.where(PresC[i,xmm-(dt-1)/3:xpm+1] == PresMaxB)[0]+xmm-(dt-1)/3 # Posición del máximo antes
						except:
							PresMaxB = np.nanmax(PresC[i,:xpm[0]+1]) # Valor máximo antes
							xpM = np.where(PresC[i,:xpm[0]+1] == PresMaxB)[0] # Posición del máximo antes
					# print('Before')
					# print('xpm='+str(xpm))
					# print('xpM='+str(xpM))
					if np.isnan(PresMaxB) or np.isnan(PresMin) or xpM == xpm:
						PresChangeB[i] = np.nan
						PresRateB[i] = np.nan
					else:
						PresChangeB[i] = PresMaxB - PresMin
						PresRateB[i] = PresChangeB[i]/((xpM-xpm-1)*dt/60) # Rate en hPa/h
					
					# print('PresChangeB='+str(PresChangeB[i]))
					# print('PresRateB='+str(PresRateB[i]))

					# Se encuentra el cambio de presión durante el evento.
					PresMaxA = np.nanmax(PresC[i,xpm:x[0]+5]) # Valor máximo
					xpM = np.where(PresC[i,xpm:x[0]+6] == PresMaxA)[0]+xpm # Posición del máximo antes

					if np.isnan(PresMaxA) or np.isnan(PresMin) or xpM == xpm:
						PresChangeA[i] = np.nan
						PresRateA[i] = np.nan
					else:
						PresChangeA[i] = PresMaxA - PresMin
						PresRateA[i] = PresChangeA[i]/((xpM-xpm+1)*dt/60) # Rate en hPa/h
					# if i == 426:
					# 	print('After')
					# 	print('xpm='+str(xpm))
					# 	print('xpM='+str(xpM))
					# 	print('PresChangeA='+str(PresChangeA[i]))
					# 	print('PresRateBA='+str(PresRateA[i]))

			


		if flagEv == True:
			print('\n Se desarrollan las gráficas')
			# Se crea la ruta en donde se guardarán los archivos
			utl.CrFolder(PathImg+Name+'/')
			# Tiempo
			Time = PrecC.shape[1]
			Time_G = np.arange(-Time/2,Time/2)
			
			# Valores generales para los gráficos
			Afon = 18; Tit = 22; Axl = 20

			for i in range(len(PrecC)):
				if i <= 946:
					f = plt.figure(figsize=(20,10))
					plt.rcParams.update({'font.size': Afon})
					ax11 = host_subplot(111, axes_class=AA.Axes)
					# Precipitación
					a11 = ax11.plot(Time_G,PrecC[i],'-b', label = 'Precipitación')
					ax11.set_title(Name+r" Evento " + str(i),fontsize=Tit)
					ax11.set_xlabel("Tiempo [cada 5 min]",fontsize=Axl)
					ax11.set_ylabel('Precipitación [mm]',fontsize=Axl)
					#ax11.set_xlim([Time_G[0],Time_G[len(Prec_Ev[0])-1]+1])
					# Presión barométrica
					axx11 = ax11.twinx()
					a112 = axx11.plot(Time_G,PresC[i],'-k', label = 'Presión Barométrica')
					axx11.set_ylabel("Presión [hPa]",fontsize=Axl)

					# print(PosiminPres[i])
					# print(PosimaxPres[i])
					# print(Time_G.shape)
					# Líneas de eventos
					L1 = ax11.plot([Time_G[PosiminPres[i]],Time_G[PosiminPres[i]]],[0,np.nanmax(PrecC[i])],'--b', label = 'Min Pres') # Punto mínimo
					L2 = ax11.plot([Time_G[PosimaxPres[i]],Time_G[PosimaxPres[i]]],[0,np.nanmax(PrecC[i])],'--r', label = 'Max Pres Antes') # Punto máximo B
					L3 = ax11.plot([Time_G[PosimaxPresA[i]],Time_G[PosimaxPresA[i]]],[0,np.nanmax(PrecC[i])],'--g', label = 'Max Pres Después') # Punto máximo A

					# Líneas para la precipitación
					L4 = ax11.plot([Time_G[PrecPre[i]],Time_G[PrecPre[i]]],[0,np.nanmax(PrecC[i])],'-.b', label = 'Inicio Prec') # Inicio del aguacero
					L5 = ax11.plot([Time_G[PrecPos[i]],Time_G[PrecPos[i]]],[0,np.nanmax(PrecC[i])],'-.g', label = 'Fin Prec') # Fin del aguacero

					# added these three lines
					lns = a11+a112+L1+L2+L3+L4+L5
					labs = [l.get_label() for l in lns]
					ax11.legend(lns, labs, loc=0)
					
					#plt.tight_layout()
					plt.savefig(PathImg + Name + '/' + Name + '_' + 'Ev_' + str(i) + '.png')
					plt.close('all')

		return DurPrec, MaxPrec, PresRateB,PresRateA,PresChangeB,PresChangeA,xx

	def PRvDP_C(self,PrecC,PresC,FechaEv,FechaEvst_Aft=0,FechaEvend_Aft=0,Mar=0.8,dt=1,M=60*4,flagEv=False,PathImg='',Name=''):
		'''
		DESCRIPTION:
	
		Con esta función se pretende encontrar la tasa de cambio de presión junto
		con la duración de las diferentes tormentas para luego ser gráficada por
		aparte.
		_________________________________________________________________________

			INPUT:
		+ PrecC: Diagrama de compuestos de precipitación.
		+ PresC: Diagrama de compuestos de presión barométrica.
		+ FechaEv: Matriz con las fechas de los eventos en formato yyyy/mm/dd-HHMM.
		+ FechaEvst_Aft: Fecha de comienzo del evento, en el mismo formato que FechaEv.
		+ FechaEvend_Aft: Fecha de finalización del evento, en el mismo formato que FechaEv.
		+ Mar: Valor del cambio de presión mínimo para calcular las tasas antes
			   del evento de precipitación.
		+ dt: delta de tiempo que se tienen en los diagramas de compuestos.
		+ M: Mitad en donde se encuentran los datos.
		+ flagEV: Bandera para graficar los eventos.
		_________________________________________________________________________

			OUTPUT:
		- DurPrec: Duración del evento de precipitación.
		- PresRateB: Tasa de cambio de presión antes.
		- PresRateA: Tasa de cambio de presión después.
		'''
		
		# Se filtran las alertas
		warnings.filterwarnings('ignore')
		# --------------------------------------
		# Se arreglan las fechas de los eventos
		# --------------------------------------

		FechaEvv = []
		for k in range(len(FechaEv)):
			FechaEvv.append([datetime(int(i[0:4]),int(i[5:7]),int(i[8:10]),int(i[11:13]),int(i[13:15])) for i in FechaEv[k]])
		FechaEvv = np.array(FechaEvv)
		FechaEvst = np.array([datetime(int(i[0:4]),int(i[5:7]),int(i[8:10]),int(i[11:13]),int(i[13:15])) for i in FechaEvst_Aft])
		FechaEvend = np.array([datetime(int(i[0:4]),int(i[5:7]),int(i[8:10]),int(i[11:13]),int(i[13:15])) for i in FechaEvend_Aft])

		if FechaEvst_Aft == 0:
			print('Generar el cálculo para el inicio de las tormentas')

		# Se inicializan las variables
		DurPrec = []
		PresRateA = []
		PresRateB = []
		PCxii = []
		PCxff = []
		PCxfBf = []
		Pii = [] # Position of the eventos
		Dxii = []
		Dxff = []
		G = 0

		for i in range(len(FechaEv)):
			# Se verifica que haya información
			q = sum(~np.isnan(PresC[i]))
			if q <= len(PresC[i])*.60:
				DurPrec.append(np.nan)
				PresRateA.append(np.nan)
				PresRateB.append(np.nan)

				PCxii.append(np.nan)
				PCxff.append(np.nan)
				PCxfBf.append(np.nan)
				Dxii.append(np.nan)
				Dxff.append(np.nan)
			else:
				# ------------------------
				# Duración de la tormenta
				# ------------------------
				Dxi = np.where(FechaEvv[i] ==FechaEvst[i])[0]
				Dxf = np.where(FechaEvv[i] ==FechaEvend[i])[0]
				DurPrec.append((Dxf[0]-Dxi[0]+1)*dt/60) # Duración en horas
				# Se verifica que haya información
				q = sum(~np.isnan(PresC[i,Dxi[0]:Dxf[0]+1]))
				if q <= len(PresC[i,Dxi[0]:Dxf[0]+1])*.50:
					DurPrec[-1] = np.nan
					PresRateA.append(np.nan)
					PresRateB.append(np.nan)

					PCxii.append(np.nan)
					PCxff.append(np.nan)
					PCxfBf.append(np.nan)
					Dxii.append(np.nan)
					Dxff.append(np.nan)
				else:

					# --------------------------
					# Tasa de cambio de presión
					# --------------------------

					# Mínimo
					if dt <= 10:
						# Valores para posiciones antes y después
						# basados en el tiempo.
						ValBef = int(10/dt) # 10 min 
						ValAft = int(20/dt) # 20 min

					PCAi1 = np.nanmin(PresC[i,Dxi[0]-ValBef:Dxi[0]])
					PCxi1 = np.where(PresC[i] == PCAi1)[0]
					PCAi2 = np.nanmin(PresC[i,Dxi[0]:Dxi[0]+ValAft])
					PCxi2 = np.where(PresC[i] == PCAi2)[0]

					PMax = np.nanmax(PresC[i,Dxi[0]-ValBef:Dxi[0]+1])
					xMax = np.where(PresC[i] == PMax)[0]

					if ((PCAi1 > PCAi2) and (PMax > PCAi1)) or np.isnan(PCAi1):
						PCAi = PCAi2
						PCxi = PCxi2
					else:
						PCAi = PCAi1
						PCxi = PCxi1
					if len(PCxi) == 0:
						DurPrec[-1] = np.nan
						PresRateA.append(np.nan)
						PresRateB.append(np.nan)

						PCxii.append(np.nan)
						PCxff.append(np.nan)
						PCxfBf.append(np.nan)
						Dxii.append(np.nan)
						Dxff.append(np.nan)
					else:
						# Máximo After
						# Mar = 0.8
						# Método de Validación
						# PCAf = []
						# PCAxf = []
						# DPA = []
						# PRA = []
						
						# for j in range(1,len(FechaEvv[i])-PCxi[0]):
						# 	PCAf.append(PresC[i,PCxi[0]+j])
						# 	PCAxf.append(PCxi[0]+j)
						# 	DPA.append(j*5/60)
						# 	PRA.append((PCAf[-1]-PCAi)/DPA[-1])
						# PCAf = np.array(PCAf)
						# PCAxf = np.array(PCAxf)
						# DPA = np.array(DPA)
						# PRA = np.array(PRA)

						# DifPR = np.abs(PresRatePos[i]-PRA)
						# DifPRmx = np.where(DifPR == np.nanmin(DifPR))[0]
						# PCxf = PCAxf[DifPRmx[0]]
						# PCA = PCAf[DifPRmx[0]]
						# PRAA = PRA[DifPRmx[0]]

						# Primer método
						PCAf = np.nanmax(PresC[i,Dxi[0]:Dxf[0]+ValBef])
						PCAxf = np.where(PresC[i,Dxi[0]:Dxf[0]+ValBef] == PCAf)[0][-1]
						PCxf = PCAxf + len(PresC[i,:Dxi[0]])
						DPB = (PCxf - PCxi[0])*dt/60
						PRAA = (PCAf-PCAi)/DPB
						PCA = PCAf-PCAi
						# ---------------------

						# if PCA <= Mar:
						# 	PRAA = np.nan
						# # else:
						# # 	# Se obitenen los eventos que pasaron
						# # 	Nameout = PathImg + '/US_MesoWest/Scatter/Pos/Adjusted/Events_Aft/' + Name + '/' + Name + '_' + 'Ev_' + str(i)
						# # 	HyPl.EventPres(FechaEvv[i],PresC[i],FechaEvst_Aft[i],PCxi,PCxf,np.nan,Dxi,Dxf,Name,PathImg + 'US_MesoWest/Scatter/Pos/Adjusted/Events_Aft/' + Name + '/',Nameout)

						# Se guarda la tasa de cambio de presión durante
						PresRateA.append(PRAA)


						# Máximo Before
						# Primer método
						# PCBf = []
						# PCBxf = []
						# DPB = []
						# PRB = []
						
						# for j in range(1,(Dxf[0]-Dxi[0]+1)):
						# 	PCBf.append(PresC[i,PCxi[0]-j])
						# 	PCBxf.append(PCxi[0]-j)
						# 	DPB.append(j*5/60)
						# 	PRB.append((PCAi-PCBf[-1])/DPB[-1])
						# PCBf = np.array(PCBf)
						# PCBxf = np.array(PCBxf)
						# DPB = np.array(DPB)
						# PRB = np.array(PRB)

						# PRBM = np.nanmin(PRB)
						# PRMxB = np.where(PRB == PRBM)[0]
						# PCxfB = PCBxf[PRMxB[0]]
						# PCB = np.abs(PCBf[PRMxB[0]] - PCAi)
						# if PCB <= 0.8:
						# 	PRBM = np.nan
						# else:
						# 	# Se obitenen los eventos que pasaron
						# 	Nameout = PathImg + '/US_MesoWest/Scatter/Pos/Adjusted/Events_1/' + Name + '/' + Name + '_' + 'Ev_' + str(i)
						# 	HyPl.EventPres(FechaEvv[i],PresC[i],FechaEvst_Aft[i],PCxi,PCxf,PCxfB,Dxi,Dxf,Name,PathImg + 'US_MesoWest/Scatter/Pos/Adjusted/Events_1/' + Name + '/',Nameout)

						# Segundo método
						# Mar = 0.8 # Margen
						# BE = 2 # Buscador antes del evento
						# # Se encuentran varios máximos antes del evento
						# MaxBP1 = np.nanmax(PresC[i,PCxi[0]-Dxf[0]-Dxi[0]-BE:PCxi[0]])
						# if np.isnan(MaxBP1):
						# 	PCBf = MaxBP1
						# else:
						# 	MaxBPx1 = np.where(PresC[i,PCxi[0]-Dxf[0]-Dxi[0]-BE:PCxi[0]] == MaxBP1)[0][-1]
						# 	MaxBPx1 += len(PresC[i,:PCxi[0]-Dxf[0]-Dxi[0]-BE])
						# 	PresC2 = np.copy(PresC)
						# 	PresC2[i,MaxBPx1-1:MaxBPx1+2] = np.nan
						# 	if np.abs(Dxf[0]-Dxi[0]) > 3:
						# 		# Se encuentran los otros dos máximos
						# 		MaxBP2 = []
						# 		MaxBPx2 = []
						# 		for j in range(2):
						# 			MaxBP2.append(np.nanmax(PresC2[i,PCxi[0]-Dxf[0]-Dxi[0]-BE:PCxi[0]]))
						# 			if np.isnan(MaxBP2[-1]):
						# 				MaxBPx2.append(np.nan)	
						# 			else:
						# 				MaxBPx2.append(np.where(PresC2[i,PCxi[0]-Dxf[0]-Dxi[0]-BE:PCxi[0]] == MaxBP2[-1])[0][-1])
						# 				MaxBPx2[-1] += len(PresC[i,:PCxi[0]-Dxf[0]-Dxi[0]-BE])
						# 				PresC2[i,MaxBPx2[-1]-1:MaxBPx2[-1]+2] = np.nan
						# 		# Se encuentra la diferencia
						# 		MaxBP2 = np.array(MaxBP2)
						# 		MaxBPx2 = np.array(MaxBPx2)
						# 		DMaxBP = np.abs(MaxBP1 - MaxBP2)
								
						# 			# Se escoge el máximo más cercano
						# 		for jj,j in enumerate(DMaxBP):
						# 			if MaxBPx2[jj] > MaxBPx1 and j < 0.05:
						# 				PCBf = MaxBP2[jj]
						# 				break
						# 			else:
						# 				PCBf = MaxBP1
						# 	else:
						# 		PCBf = MaxBP1

						# if np.isnan(PCBf):
						# 	PCxfB = np.nan
						# 	DPB = np.nan
						# 	PRBM = np.nan
						# else:
						# 	PCxfB = np.where(PresC[i,PCxi[0]-Dxf[0]-Dxi[0]-BE:PCxi[0]] == PCBf)[0][-1]
						# 	PCxfB += len(PresC[i,:PCxi[0]-Dxf[0]-Dxi[0]-BE])
						# 	DPB = (PCxi[0] - PCxfB)*5/60
						# 	PRBM = (PCAi - PCBf)/DPB
						# 	PCB = np.abs(PCAi-PCBf)
						# 	if PCB <= Mar:
						# 		PRBM = np.nan
							# else:
							# 	# Se obitenen los eventos que pasaron
							# 	Nameout = PathImg + '/US_MesoWest/Scatter/Pos/Adjusted/Events_2/' + Name + '/' + Name + '_' + 'Ev_' + str(i)
							# 	HyPl.EventPres(FechaEvv[i],PresC[i],FechaEvst_Aft[i],PCxi,PCxf,PCxfB,Dxi,Dxf,Name,PathImg + 'US_MesoWest/Scatter/Pos/Adjusted/Events_2/' + Name + '/',Nameout)

						# Tercer método
						PCBf = []
						PCBxf = []
						DPB = []
						PRB = []

						BE = ValBef # Valor de posición antes.

						MaxBP = np.nanmax(PresC[i,PCxi[0]-Dxf[0]-Dxi[0]-BE:PCxi[0]])
						if np.isnan(MaxBP):
							PRBM = np.nan
							PCxfB = np.nan
						else:
							MaxBPx = np.where(PresC[i,PCxi[0]-Dxf[0]-Dxi[0]-BE:PCxi[0]] == MaxBP)[0][-1]
							MaxBPx += len(PresC[i,:PCxi[0]-Dxf[0]-Dxi[0]-BE])
							
							for j in range(1,(Dxi[0]-MaxBPx+1)):
							# for j in range(1,(Dxf[0]-Dxi[0]+4)):
							# for j in range(1,PCxi-2):
								PCBf.append(PresC[i,PCxi[0]-j])
								PCBxf.append(PCxi[0]-j)
								DPB.append(j*dt/60)
								PRB.append((PCAi-PCBf[-1])/DPB[-1])
							PCBf = np.array(PCBf)
							PCBxf = np.array(PCBxf)
							DPB = np.array(DPB)
							PRB = np.array(PRB)

							qq = sum(~np.isnan(PRB))
							if qq <= len(PRB)*.70:
								PCxfB = np.nan
								PRBM = np.nan
							else:
								DifPRB = np.abs(PresRateA[-1]+PRB)
								DifPRmxB = np.where(DifPRB == np.nanmin(DifPRB))[0]
								PCxfB = PCBxf[DifPRmxB[0]]
								PRBM = PRB[DifPRmxB[0]]
								PCB = np.abs(PCAi-PCBf[DifPRmxB[0]])
								if PCB <= Mar:
									PRBM = np.nan
								else:
									Pii.append(i)
									# if G <= 10:
									# 	
									# 	Nameout = PathImg + '/US_MesoWest/Scatter/Pos/Adjusted/Events/' + Name + '/' + Name + '_' + 'Ev_' + str(i)
									# 	HyPl.EventPres(FechaEvv[i],PresC[i],FechaEvst_Aft[i],PCxi,PCxf,PCxfB,Dxi,Dxf,Name,PathImg + 'US_MesoWest/Scatter/Pos/Adjusted/Events/' + Name + '/',Nameout)
									# 	G += 1

							
						# Se guarda la tasa de cambio de presión antes
						PresRateB.append(PRBM)
						# Se llenan los valores de las posiciones 
						PCxii.append(PCxi[0])
						PCxff.append(PCxf)
						PCxfBf.append(PCxfB)
						Dxii.append(Dxi[0])
						Dxff.append(Dxf[0])
						

		# -------------------------
		# Se verifican los eventos
		# -------------------------
						
		DurPrec = np.array(DurPrec)
		PresRateA = np.array(PresRateA)
		PresRateB = np.array(PresRateB)

		if flagEv:
			for i in Pii:

				Nameout = PathImg + Name + '/' + Name + '_Ev_'+str(i)

				# Se grafican las dos series
				fH=30 # Largo de la Figura
				fV = fH*(2/3) # Ancho de la Figura

				# Se crea la carpeta para guardar la imágen
				utl.CrFolder(PathImg + Name + '/')

				f, ((ax12,ax13), (ax22,ax23)) = plt.subplots(2, 2, figsize=utl.cm2inch(fH,fV))
				plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
					,'font.sans-serif': 'Arial Narrow'\
					,'xtick.labelsize': 15,'xtick.major.size': 6,'xtick.minor.size': 4\
					,'xtick.major.width': 1,'xtick.minor.width': 1\
					,'ytick.labelsize': 16,'ytick.major.size': 12,'ytick.minor.size': 4\
					,'ytick.major.width': 1,'ytick.minor.width': 1\
					,'axes.linewidth':1\
					,'grid.alpha':0.1,'grid.linestyle':'-'})
				ax12.tick_params(axis='x',which='both',bottom='on',top='off',\
					labelbottom='on',direction='out')
				ax12.tick_params(axis='y',which='both',left='on',right='off',\
					labelleft='on')
				ax12.tick_params(axis='y',which='major',direction='inout') 
				# Precipitación 
				ax12.scatter(DurPrec,PresRateB)
				ax12.scatter(DurPrec[i],PresRateB[i],color='red')
				ax12.set_title('Cambios en Presión Atmosférica en '+ Name,fontsize=16)
				ax12.set_xlabel(u'Duración de la Tormenta [h]',fontsize=15)
				ax12.set_ylabel('Tasa de Cambio de Presión [hPa/h]',fontsize=15)
				ax12.grid()

				
				xTL = ax12.xaxis.get_ticklocs() # List of Ticks in x
				MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
				plt.xlim([0,np.nanmax(DurPrec)+2*MxL])

				xTL = ax12.xaxis.get_ticklocs() # List of Ticks in x
				MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
				minorLocatorx = MultipleLocator(MxL)
				yTL = ax12.yaxis.get_ticklocs() # List of Ticks in y
				MyL = np.abs(np.abs(yTL[1])-np.abs(yTL[0]))/5 # minorLocatory value
				minorLocatory = MultipleLocator(MyL)
				ax12.xaxis.set_minor_locator(minorLocatorx)
				ax12.yaxis.set_minor_locator(minorLocatory)

				# Se realiza la regresión
				Coef, perr,R2 = CF.FF(DurPrec,PresRateB,2)

				# Se toman los datos para ser comparados posteriormente
				DD,PP = utl.NoNaN(DurPrec,PresRateB,False)
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


				ax12.plot(x,PresRateC,'k--')
				# Se incluye la ecuación
				if np.nanmin(PresRateB) < 0:
					ax12.text(xTL[-4],yTL[3]+3*MyL, r'$\Delta = %sD^{%s}$' %(round(a,3),round(b,3)), fontsize=15)
					ax12.text(xTL[-4],yTL[3], r'$R^2 = %s$' %(round(R2,4)), fontsize=14)
				else:
					ax12.text(xTL[-4],yTL[32], r'$\Delta = %sD^{%s}$' %(round(a,3),round(b,3)), fontsize=15)
					ax12.text(xTL[-4],yTL[32]-3*MyL, r'$R^2 = %s$' %(round(R2,4)), fontsize=14)

				# -----------
				ax13.tick_params(axis='x',which='both',bottom='on',top='off',\
					labelbottom='on',direction='out')
				ax13.tick_params(axis='y',which='both',left='on',right='off',\
					labelleft='on')
				ax13.tick_params(axis='y',which='major',direction='inout') 
				ax13.scatter(DurPrec,PresRateA)
				ax13.scatter(DurPrec[i],PresRateA[i],color='red')
				ax13.set_title('Cambios en Presión Atmosférica en '+ Name,fontsize=16)
				ax13.set_xlabel(u'Duración de la Tormenta [h]',fontsize=15)
				#ax13.set_ylabel('Tasa de Cambio de Presión [hPa/h]',fontsize=15)
				ax13.grid()

				xTL = ax13.xaxis.get_ticklocs() # List of Ticks in x
				MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
				plt.xlim([0,np.nanmax(DurPrec)+2*MxL])

				xTL = ax13.xaxis.get_ticklocs() # List of Ticks in x
				MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
				minorLocatorx = MultipleLocator(MxL)
				yTL = ax13.yaxis.get_ticklocs() # List of Ticks in y
				MyL = np.abs(yTL[1]-yTL[0])/5 # minorLocatory value
				minorLocatory = MultipleLocator(MyL)
				ax13.xaxis.set_minor_locator(minorLocatorx)
				ax13.yaxis.set_minor_locator(minorLocatory)

				# Se realiza la regresión
				Coef, perr,R2 = CF.FF(DurPrec,PresRateA,2)

				# Se toman los datos para ser comparados posteriormente
				DD,PP = utl.NoNaN(DurPrec,PresRateA,False)
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

				ax13.plot(x,PresRateC,'k--')
				# Se incluye la ecuación
				if np.nanmin(PresRateA) < 0:
					ax13.text(xTL[-3],yTL[2]+3*MyL, r'$\Delta = %sD^{%s}$' %(round(a,3),round(b,3)), fontsize=15)
					ax13.text(xTL[-3],yTL[2], r'$R^2 = %s$' %(round(R2,4)), fontsize=14)
				else:
					ax13.text(xTL[-3],yTL[-2], r'$\Delta = %sD^{%s}$' %(round(a,3),round(b,3)), fontsize=15)
					ax13.text(xTL[-3],yTL[-2]-3*MyL, r'$R^2 = %s$' %(round(R2,4)), fontsize=14)

				plt.subplot(212)
				plt.tick_params(axis='x',which='both',bottom='on',top='off',\
					labelbottom='on',direction='out')
				plt.tick_params(axis='y',which='both',left='on',right='off',\
					labelleft='on')
				plt.tick_params(axis='y',which='major',direction='inout') 
				#plt.bar(x[q],C[q],lw=2, label = u'Correlación')
				a11 = plt.plot(FechaEvv[i],PresC[i],'-k', label = 'Presión')
				plt.title(Name+r" Evento "+FechaEvst_Aft[i],fontsize=16)
				plt.xlabel("Tiempo",fontsize=15)
				plt.ylabel('Presión [hPa]',fontsize=15)

				if ~np.isnan(PCxii[i]):
					L1 = plt.plot([FechaEvv[i,PCxii[i]],FechaEvv[i,PCxii[i]]],[np.nanmin(PresC[i]),np.nanmax(PresC[i])],'--b', label = 'Min Pres') # Punto mínimo
					if ~np.isnan(PCxfBf[i]):
						L2 = plt.plot([FechaEvv[i,PCxfBf[i]],FechaEvv[i,PCxfBf[i]]],[np.nanmin(PresC[i]),np.nanmax(PresC[i])],'--r', label = 'Max Pres Antes') # Punto máximo B
					L3 = plt.plot([FechaEvv[i,PCxff[i]],FechaEvv[i,PCxff[i]]],[np.nanmin(PresC[i]),np.nanmax(PresC[i])],'--g', label = 'Max Pres Después') # Punto máximo A

				# Líneas para la precipitación
				L4 = plt.plot([FechaEvv[i,Dxii[i]],FechaEvv[i,Dxii[i]]],[np.nanmin(PresC[i]),np.nanmax(PresC[i])],'-.b', label = 'Inicio Prec') # Inicio del aguacero
				L5 = plt.plot([FechaEvv[i,Dxff[i]],FechaEvv[i,Dxff[i]]],[np.nanmin(PresC[i]),np.nanmax(PresC[i])],'-.g', label = 'Fin Prec') # Fin del aguacero

				xTL = ax13.xaxis.get_ticklocs() # List of Ticks in x
				MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
				minorLocatorx = MultipleLocator(MxL)
				yTL = ax13.yaxis.get_ticklocs() # List of Ticks in y
				MyL = (yTL[1]-yTL[0])/5 # minorLocatory value
				minorLocatory = MultipleLocator(MyL)
				ax13.yaxis.set_minor_locator(minorLocatory)

				# added these three lines
				if ~np.isnan(PCxfB):
					lns = a11+L1+L2+L3+L4+L5
				else:
					lns = a11+L1+L3+L4+L5
				labs = [l.get_label() for l in lns]
				plt.legend(lns, labs, loc=3,fontsize=13)
				
				plt.grid()
				plt.tight_layout()
				plt.savefig(Nameout + '.png',format='png',dpi=300 )
				plt.close('all')

		return DurPrec, PresRateA, PresRateB






