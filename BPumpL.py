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
import matplotlib.dates as mdates # Manejo de dateticks
import matplotlib.mlab as mlab
import time
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

import warnings

# Se importan los paquetes para manejo de fechas
from datetime import date, datetime, timedelta

from UtilitiesDGD import UtilitiesDGD
from CorrSt import CorrSt

utl = UtilitiesDGD()
cr = CorrSt()

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

	def ExEvDN(self,PrecC,VC,FechaEv,Mid=0):
		'''
			DESCRIPTION:
		
		Con esta función se pretende separar los eventos que se presentan de día
		y de noche.
		_________________________________________________________________________

			INPUT:
		+ PrecC: Diagrama de compuestos de Precipitación
		+ VC: Diagrama de compuestos de la variable que se quiere tener.
		+ FechaEv: Fecha de cada uno de los eventos.
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

		# Se extraen los datos de los diferentes trimestres.
		for i in range(len(FechaEv)):
			Hours.append(FechaEv[i][Mid][11:13])

		x = [0 for k in range(2)]
		# Se extraen los diferentes trimestres
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
		_________________________________________________________________________
		
			OUTPUT:
		Esta función libera 3 gráficas.
		'''
		warnings.filterwarnings('ignore')
		# ---- 
		# Valores generales para los gráficos

		Afon = 14; Tit = 18; Axl = 16


		# Se grafican los primeros 6 eventos 

		f, ((ax11,ax12,ax13), ((ax21,ax22,ax23))) = plt.subplots(2,3, figsize=(20,10))
		plt.rcParams.update({'font.size': Afon})
		# Precipitación Ev 1
		a11 = ax11.plot(np.arange(0,len(Prec_Ev[0])),Prec_Ev[0],L1, label = V1)
		ax11.set_title(r"Evento 1",fontsize=Tit)
		#ax11.set_xlabel("Tiempo [cada 5 min]",fontsize=Axl)
		ax11.set_ylabel(Ax1,fontsize=Axl)
		ax11.set_xlim([0,len(Prec_Ev[0]+1)])
		# Presión barométrica
		axx11 = ax11.twinx()
		a112 = axx11.plot(np.arange(0,len(Prec_Ev[0])),Pres_F_Ev[0],L2, label = V2)
		axx11.set_xlim([0,len(Pres_F_Ev[0]+1)])
		#axx11.set_ylabel("Presión [hPa]",fontsize=Axl)

		# added these three lines
		lns = a11+a112
		labs = [l.get_label() for l in lns]
		#ax11.legend(lns, labs, loc=0)

		# Precipitación Ev 2
		a12 = ax12.plot(Prec_Ev[1],L1, label = u'Precipitación')
		ax12.set_title(r"Evento 2",fontsize=Tit)
		#ax11.set_xlabel("Tiempo [cada 5 min]",fontsize=Axl)
		#ax12.set_ylabel("Precipitación [mm]",fontsize=Axl)

		# Presión barométrica
		axx12 = ax12.twinx()
		a122 = axx12.plot(Pres_F_Ev[1],L2, label = u'Presión Barométrica')
		#axx11.set_ylabel("Presión [hPa]",fontsize=Axl)

		# added these three lines
		lns = a12+a122
		labs = [l.get_label() for l in lns]
		#ax12.legend(lns, labs, loc=0)

		# Precipitación Ev 3
		a13 = ax13.plot(Prec_Ev[2],L1, label = V1)
		ax13.set_title(r"Evento 3",fontsize=Tit)
		#ax11.set_xlabel("Tiempo [cada 5 min]",fontsize=Axl)
		#ax12.set_ylabel("Precipitación [mm]",fontsize=Axl)

		# Presión barométrica
		axx13 = ax13.twinx()
		a132 = axx13.plot(Pres_F_Ev[2],L2, label = V2)
		axx13.set_ylabel(Ax2,fontsize=Axl)

		# added these three lines
		lns = a13+a132
		labs = [l.get_label() for l in lns]
		ax13.legend(lns, labs, loc=1)
		

		# Precipitación Ev 4
		a21 = ax21.plot(Prec_Ev[3],L1, label = u'Precipitación')
		ax21.set_title(r"Evento 4",fontsize=Tit)
		ax21.set_xlabel("Tiempo [cada "+ DTT + " min]",fontsize=Axl)
		ax21.set_ylabel(Ax1,fontsize=Axl)

		# Presión barométrica
		axx21 = ax21.twinx()
		a212 = axx21.plot(Pres_F_Ev[3],L2, label = u'Presión Barométrica')
		#axx11.set_ylabel("Presión [hPa]",fontsize=Axl)

		# added these three lines
		lns = a21+a212
		labs = [l.get_label() for l in lns]
		#ax11.legend(lns, labs, loc=0)

		# Precipitación Ev 5
		a22 = ax22.plot(Prec_Ev[4],L1, label = u'Precipitación')
		ax22.set_title(r"Evento 5",fontsize=Tit)
		ax22.set_xlabel("Tiempo [cada "+ DTT + " min]",fontsize=Axl)
		#ax12.set_ylabel("Precipitación [mm]",fontsize=Axl)

		# Presión barométrica
		axx22 = ax22.twinx()
		a222 = axx22.plot(Pres_F_Ev[4],L2, label = u'Presión Barométrica')
		#axx11.set_ylabel("Presión [hPa]",fontsize=Axl)

		# added these three lines
		lns = a22+a222
		labs = [l.get_label() for l in lns]
		#ax12.legend(lns, labs, loc=0)
		ev = 5
		# Precipitación Ev 3
		a23 = ax23.plot(Prec_Ev[ev],L1, label = u'Precipitación')
		ax23.set_title(r"Evento %s" %(ev+1),fontsize=Tit)
		ax23.set_xlabel("Tiempo [cada "+ DTT + " min]",fontsize=Axl)
		#ax12.set_ylabel("Precipitación [mm]",fontsize=Axl)

		# Presión barométrica
		axx23 = ax23.twinx()
		a232 = axx23.plot(Pres_F_Ev[ev],L2, label = u'Presión Barométrica')
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
		a11 = ax11.plot(Prec_EvM,L1, label = V1)
		ax11.set_title(r'Promedio de Eventos',fontsize=24)
		ax11.set_xlabel("Tiempo [cada "+ DTT + " min]",fontsize=20)
		ax11.set_ylabel(Ax1,fontsize=20)

		# Presión barométrica
		axx11 = ax11.twinx()

		a112 = axx11.plot(Pres_F_EvM,L2, label = V2)
		axx11.set_ylabel(Ax2,fontsize=20)

		# added these three lines
		lns = a11+a112
		labs = [l.get_label() for l in lns]
		ax11.legend(lns, labs, loc=0)

		plt.savefig(PathImg +'Average/' + ix + '_' + 'P_' + V11 + 'V' + V22 +'_' + str(ii) + '.png')
		plt.close('all')
		xx = np.arange(0,len(Prec_EvM))
		
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
					Lx = 3
					Sx = 21
				elif DTT == '30':
					L = LM
					SLP = 0.25
					SLS = 0.15
					Lx = 2
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

				#plt.tight_layout()
				plt.savefig(PathImg +'Average/' + ix + '_' + 'PD_' + V11 + 'V' + V22+'V' + V33 +'_' + str(ii) + '.png')
				plt.close('all')



				# Se calculan los histogramas

				qPrec = ~np.isnan(Prec_Ev)
				qPres = ~np.isnan(Pres_F_Ev)
				qT = ~np.isnan(T_F_Ev)

				n = 100 # Número de bins
				PrecH,PrecBin = np.histogram(Prec_Ev[qPres],bins=n); [float(i) for i in PrecH]
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

	def PRvDP(self,PrecC,PresC,dt=1,M=60*4):
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
			xm = np.where(PrecC[i,:x[0]+1]<=0.10)[0]
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

			# Se encuentra el mínimo de precipitación antes de ese punto
			xM = np.where(PrecC[i,x[0]:]<=0.10)[0]+x[0]

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
					PresMin = np.nanmin(PresC[i,xmm-2:xmm+3]) # Valor del mínimo
					xpm = np.where(PresC[i,xmm-2:xmm+3] == PresMin)[0]+xmm-2 # Posición del mínimo
					
					# print('xpm=',xpm)
					# print('xmm=',xmm)
					
					if xpm[0] <= 10:
					# Se encuentra el cambio de presión antes del evento
						PresMaxB = PresMin
						xpM = xpm
					else:
						try:
							PresMaxB = np.nanmax(PresC[i,xmm-12:xpm+1]) # Valor máximo antes
							xpM = np.where(PresC[i,xmm-12:xpm+1] == PresMaxB)[0]+xmm-12 # Posición del máximo antes
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
			elif dt < 60:
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
					
					if xpm[0] <= 1:
					# Se encuentra el cambio de presión antes del evento
						PresMaxB = PresMin
						xpM = xpm
					else:
						try:
							PresMaxB = np.nanmax(PresC[i,xmm-(dt-1)/3:xpm+1]) # Valor máximo antes
							xpM = np.where(PresC[i,xmm-(dt-1)/3:xpm+1] == PresMaxB)[0]+xmm-(dt-1)/3 # Posición del máximo antes
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


		return DurPrec, MaxPrec, PresRateB,PresRateA,PresChangeB,PresChangeA,xx







