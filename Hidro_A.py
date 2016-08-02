# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#						Coded by Daniel González Duque
#						    Last revised 11/03/2016
#______________________________________________________________________________
#______________________________________________________________________________

#______________________________________________________________________________
#
# CLASS DESCRIPTION:
# 	This class have different routines for hydrological analysis. 
#
#	This class do not use Pandas in any function, it uses directories and save
#	several images in different folders. It is important to include the path 
#	to save the images.
#	
#______________________________________________________________________________

import numpy as np
import sys
import csv
import xlrd 
import xlsxwriter as xlsxwl
import scipy.io as sio
from scipy import stats as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.mlab as mlab
import time
import warnings

from datetime import date, datetime, timedelta

from UtilitiesDGD import UtilitiesDGD
utl = UtilitiesDGD()


class Hidro_A:

	def __init__(self):
		'''
			DESCRIPTION:

		This is the build up function.
		'''

	def CiclD(self,Var,Fecha,FlagG=True,PathImg='',NameA='',VarL='',VarLL='',C='k',Name='',flagTri=False,flagTriP=False,PathImgTri='',DTH=24):
		'''
			DESCRIPTION:
		
		Con esta función se pretende realizar el ciclo diurno de una variable a
		partir de los datos horarios de una serie de datos, se obtendrá el ciclo
		diurno para todos los datos totales y los datos mensuales.

		Además pueden obtenerse las gráficas si así se desea.
		_________________________________________________________________________

			INPUT:
		+ Var: Variable que se desea tratar.
		+ Fecha: Variable con la Fecha inicial y la fecha final de los datos.
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
		_________________________________________________________________________
		
			OUTPUT:
		Este código saca 2 gráficas, una por cada mes y el promedio de todos los
		meses.
		- MesesMM: Variable directorio con los resultados por mes
		
		'''
		# Se quitan los warnings de las gráficas
		warnings.filterwarnings('ignore')

		# Se inicializan las variables
		MesesM = dict() # Meses
		MesesMM = dict() # Meses
		MesesMMM = dict() # Meses
		MesesMD = dict() # Meses
		MesesME = dict()
		TriM = dict() # Datos trimestrales
		TriMM = dict() # Promedio trimestral
		TriMD = dict() # Desviación estándar trimestral
		TriME = dict() # Error medio trimestral

		VarM = np.reshape(Var[:],(-1,DTH))
		
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
					MesesM[j] = VarM[d:dtt,:]
					
				else:
					MesesM[j] = np.vstack((MesesM[j],VarM[d:dtt,:]))

				if flagTri == True:
					# Condicional para el inicial del trimestre
					if x == 0:
						if j == 1 or j == 3 or j == 6 or j == 9:
							xx = 0
						else:
							xx = 1
					else:
						xx = 1
					# Se mira en qué trimestre esta
					#print('xx='+str(xx))
					if xx == 0:
						if j == 1:
							TriM[1] = VarM[d:dtt,:]
							#print(j)
						if j == 3:
							TriM[2] = VarM[d:dtt,:]
						if j == 6:
							TriM[3] = VarM[d:dtt,:]
						if j == 9:
							TriM[4] = VarM[d:dtt,:]
					elif xx == 1:
						if j == 1 or j == 2 or j == 12:
							#print(j)
							TriM[1] = np.vstack((TriM[1],VarM[d:dtt,:]))
						if j == 3 or j == 4 or  j == 5:
							TriM[2] = np.vstack((TriM[2],VarM[d:dtt,:]))
						if j == 6 or j == 7 or  j == 8:
							TriM[3] = np.vstack((TriM[3],VarM[d:dtt,:]))
						if j == 9 or j == 10 or  j == 11:
							TriM[4] = np.vstack((TriM[4],VarM[d:dtt,:]))

				d = dtt
			x += 1
		if VarLL == 'Precipitación':
			TM = []
			# Se eliminan todos los ceros de las variables.
			for i in range(1,13):
				q = np.where(MesesM[i] <= 0.1)
				MesesM[i][q] = float('nan')

			for i in range(1,13): # Meses
				MesesMM[i] = np.nanmean(MesesM[i],axis=0)
				MesesMD[i] = np.nanstd(MesesM[i],axis=0)
				MesesME[i] = [k/np.sqrt(len(MesesM[i])) for k in MesesMD[i]]

				TM.append(np.sum(MesesMM[i]))
				MesesMMM[i] = [(k/TM[i-1])*100 for k in MesesMM[i]]
		else:			
			for i in range(1,13): # Meses
				MesesMM[i] = np.nanmean(MesesM[i],axis=0)
				MesesMD[i] = np.nanstd(MesesM[i],axis=0)
				MesesME[i] = [k/np.sqrt(len(MesesM[i])) for k in MesesMD[i]]
			if flagTri == True:
				for i in range(1,5):
					TriMM[i] = np.nanmean(TriM[i],axis=0)
					TriMD[i] = np.nanstd(TriM[i],axis=0)
					TriME[i] = [k/np.sqrt(len(TriM[i])) for k in TriMD[i]]

		if VarLL == 'Precipitación':
			q = np.where(VarM <= 0.1)
			VarM[q] = np.nan

			# Se calcula el ciclo diurno para todos los datos
			CiDT = np.nanmean(VarM, axis=0)
			DesT = np.nanstd(VarM, axis=0)
		else:
			# Se calcula el ciclo diurno para todos los datos
			CiDT = np.nanmean(VarM, axis=0)
			DesT = np.nanstd(VarM, axis=0)
		VarMNT = []
		# Se calcula el número total de datos
		for i in range(len(VarM[0])):
			VarMNT.append(sum(~np.isnan(VarM[:,i])))

		ErrT = [k/np.sqrt(VarMNT[ii]) for ii,k in enumerate(DesT)]
		ErrT = np.array(ErrT)
		#ErrT = DesT
		
		# Se realizan los labels para las horas
		if DTH == 24:
			HH = np.arange(0,24)
			HH2 = np.arange(0,24)
			HHL = np.arange(0,24)
		elif DTH == 48:
			HH = np.arange(0,48)
			HH2 = np.arange(0,48,2)
			HHL = np.arange(0,24)

		if FlagG == True:
			Mes = ['Enero','Febrero','Marzo','Abril','Mayo','Junio',\
				'Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'];
			lfs = 16
			# Se realiza la curva para todos los meses
			fig, axs = plt.subplots(4,3, figsize=(15, 8), facecolor='w', edgecolor='k')
			plt.rcParams.update({'font.size': lfs-6})
			axs = axs.ravel() # Para hacer un loop con los subplots

			
				# for i in HH:
				# 	for j in range(2):
				# 		if j == 0:
				# 			if i < 10:
				# 				HHL.append('0'+str(i)+'00')
				# 			else:
				# 				HHL.append(str(i)+'00')
				# 		else:
				# 			if i < 10:
				# 				HHL.append('0'+str(i)+'30')
				# 			else:
				# 				HHL.append(str(i)+'30')


			for ii,i in enumerate(range(1,13)):
				axs[ii].errorbar(HH,MesesMM[i],yerr=MesesME[i],fmt='-',color=C,label=VarLL)
				axs[ii].set_title(Mes[ii],fontsize=lfs)
				axs[ii].grid()
				axs[ii].set_xlim([0,23])
				axs[ii].xaxis.set_ticks(HH2)
				axs[ii].set_xticklabels(HHL,rotation=90)

				if ii == 0 or ii == 3 or ii == 6 or ii == 9:
					axs[ii].set_ylabel(VarL,fontsize=lfs-2)
				if ii >=9:
					axs[ii].set_xlabel('Horas',fontsize=lfs-2)
				if ii == 2:
					axs[ii].legend(loc='best')

			plt.tight_layout()
			plt.savefig(PathImg + 'CMErr_' + NameA +'.png' )
			plt.close('all')

			# Gráficos con desviación estándar
			fig, axs = plt.subplots(4,3, figsize=(15, 8), facecolor='w', edgecolor='k')
			plt.rcParams.update({'font.size': lfs-6})
			axs = axs.ravel() # Para hacer un loop con los subplots
			for ii,i in enumerate(range(1,13)):
				axs[ii].errorbar(HH,MesesMM[i],yerr=MesesMD[i],fmt='-',color=C,label=VarLL)
				axs[ii].set_title(Mes[ii],fontsize=lfs)
				axs[ii].grid()
				axs[ii].set_xlim([0,23])
				axs[ii].xaxis.set_ticks(HH2)
				axs[ii].set_xticklabels(HHL,rotation=90)

				if ii == 0 or ii == 3 or ii == 6 or ii == 9:
					axs[ii].set_ylabel(VarL,fontsize=lfs-2)
				if ii >=9:
					axs[ii].set_xlabel('Horas',fontsize=lfs-2)
				if ii == 2:
					axs[ii].legend(loc=1)

			plt.tight_layout()
			plt.savefig(PathImg + 'CMD_' + NameA +'.png' )
			plt.close('all')


			if VarLL == 'Precipitación':
				fig, axs = plt.subplots(4,3, figsize=(15, 8), facecolor='w', edgecolor='k')
				plt.rcParams.update({'font.size': lfs-6})
				axs = axs.ravel() # Para hacer un loop con los subplots
				for ii,i in enumerate(range(1,13)):
					axs[ii].plot(HH,MesesMMM[i],'-',color=C,label=VarLL)
					axs[ii].set_title(Mes[ii],fontsize=lfs)
					axs[ii].grid()
					axs[ii].set_xlim([0,23])
					axs[ii].xaxis.set_ticks(HH2)
					axs[ii].set_xticklabels(HHL,rotation=90)

					if ii == 0 or ii == 3 or ii == 6 or ii == 9:
						axs[ii].set_ylabel('Precipitación [%]',fontsize=lfs-2)
					if ii >=9:
						axs[ii].set_xlabel('Horas',fontsize=lfs-2)
					if ii == 2:
						axs[ii].legend(loc='best')

				plt.tight_layout()
				plt.savefig(PathImg + 'CMPPrec_' + NameA +'.png' )
				plt.close('all')


			# Se grafica la curva anual
			F = plt.figure(figsize=(15,10))
			plt.rcParams.update({'font.size': 22})
			#plt.plot(CiDT, 'k-', lw = 1.5)
			plt.errorbar(HH,CiDT,yerr=ErrT,fmt='-',color=C,label=VarLL)
			plt.title('Ciclo diurno de ' + Name,fontsize=26 )  # Colocamos el título del gráfico
			plt.ylabel(VarL,fontsize=24)  # Colocamos la etiqueta en el eje x
			plt.xlabel('Horas',fontsize=24)  # Colocamos la etiqueta en el eje y
			axes = plt.gca()
			axes.set_xlim([0,23])
			plt.gca().xaxis.set_ticks(HH2)
			plt.gca().set_xticklabels(HHL,rotation=90)
			plt.legend(loc=1)
			plt.grid()
			plt.savefig(PathImg + 'CTErr_' + NameA+'.png' )
			plt.close('all')

			F = plt.figure(figsize=(15,10))
			plt.rcParams.update({'font.size': 22})
			#plt.plot(CiDT, 'k-', lw = 1.5)
			plt.errorbar(HH,CiDT,yerr=DesT,fmt='-',color=C,label=VarLL)
			plt.title('Ciclo diurno de ' + Name,fontsize=26 )  # Colocamos el título del gráfico
			plt.ylabel(VarL,fontsize=24)  # Colocamos la etiqueta en el eje x
			plt.xlabel('Horas',fontsize=24)  # Colocamos la etiqueta en el eje y
			axes = plt.gca()
			axes.set_xlim([0,23])
			plt.gca().xaxis.set_ticks(HH2)
			plt.gca().set_xticklabels(HHL,rotation=90)
			plt.legend(loc=1)
			plt.grid()
			plt.savefig(PathImg + 'CTDes_' + NameA+'.png' )
			plt.close('all')

		if flagTri == True:
			if flagTriP == True:
				# Se grafican los trimestres
				Trim = ['DEF','MAM','JJA','SON']
				lfs = 16
				# Se realiza la curva para todos los meses
				fig, axs = plt.subplots(2,2, figsize=(15, 8), facecolor='w', edgecolor='k')
				plt.rcParams.update({'font.size': lfs-6})
				axs = axs.ravel() # Para hacer un loop con los subplots
				for ii,i in enumerate(range(1,5)):
					axs[ii].errorbar(HH,TriMM[i],yerr=TriME[i],fmt='-',color=C,label=VarLL)
					axs[ii].set_title(Trim[ii],fontsize=lfs)
					axs[ii].grid()
					axs[ii].set_xlim([0,23])
					axs[ii].xaxis.set_ticks(HH2)
					axs[ii].set_xticklabels(HHL,rotation=90)	

					if ii == 0 or ii == 2:
						axs[ii].set_ylabel(VarL,fontsize=lfs-2)
					if ii >=2:
						axs[ii].set_xlabel('Horas',fontsize=lfs-2)
					if ii == 1:
						axs[ii].legend(loc='best')

				plt.tight_layout()
				plt.savefig(PathImgTri + 'CMErr_' + NameA +'.png' )
				plt.close('all')

				# Se realiza la curva para todos los meses
				fig, axs = plt.subplots(2,2, figsize=(15, 8), facecolor='w', edgecolor='k')
				plt.rcParams.update({'font.size': lfs-6})
				axs = axs.ravel() # Para hacer un loop con los subplots
				for ii,i in enumerate(range(1,5)):
					axs[ii].errorbar(HH,TriMM[i],yerr=TriMD[i],fmt='-',color=C,label=VarLL)
					axs[ii].set_title(Trim[ii],fontsize=lfs)
					axs[ii].grid()
					axs[ii].set_xlim([0,23])
					axs[ii].xaxis.set_ticks(HH2)
					axs[ii].set_xticklabels(HHL,rotation=90)

					if ii == 0 or ii == 2:
						axs[ii].set_ylabel(VarL,fontsize=lfs-2)
					if ii >=2:
						axs[ii].set_xlabel('Horas',fontsize=lfs-2)
					if ii == 1:
						axs[ii].legend(loc='best')

				plt.tight_layout()
				plt.savefig(PathImgTri + 'CMD_' + NameA +'.png' )
				plt.close('all')


			return MesesM, MesesMM, MesesMD,MesesME,CiDT,DesT,ErrT,TriM,TriMM,TriMD,TriME
		else:
			return MesesM, MesesMM, MesesMD,MesesME,CiDT,DesT,ErrT

	def CiclDPvP(self,Var1,Var2,Fecha,DTH=24):
		'''
			DESCRIPTION:
		
		Con esta función se pretende realizar el ciclo diurno de una variable
		que depende de otra variable.

		Además pueden obtenerse las gráficas si así se desea.
		_________________________________________________________________________

			INPUT:
		+ Var1: Variable que se desea tratar debe ser precipitación.
		+ Var2: Variable que se desea tratar.
		+ Fecha: Variable con la Fecha inicial y la fecha final de los datos.
		+ DTH: Cuantos datos se dan para completar un día, se asume que son 24 
			   datos porque podrían estar en horas
		_________________________________________________________________________
		
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

	def CiclDP(self,MesesMM,PathImg='',Name='',Name2='',FlagMan=True):
		'''
			DESCRIPTION:
		
		Con esta función se pretende realizar el ciclo diurno de la precipitación
		a lo largo del año.

		Además pueden obtenerse las gráficas si así se desea.
		_________________________________________________________________________

			INPUT:
		+ MesesMM: Variable de los promedios mensuales multianuales.
		+ PathImg: Ruta para guardar las imágenes.
		+ Name: Nombre de las estaciones.
		_________________________________________________________________________
		
			OUTPUT:
		Este código saca 2 gráficas, una por cada mes y el promedio de todos los
		meses.
		- PorcP: Variable directorio con los resultados por mes
		'''
		# Se inicializan las variables
		MM = ['E','F','M','A','M','J','J','A','S','O','N','D','E']
		ProcP = np.empty((12,24)) # Porcentaje de todos los meses

		TT = sum(MesesMM)

		for ii,i in enumerate(range(1,13)):

			ProcP[ii,:] = MesesMM[i]/TT*100

		x = np.arange(0,24)
		x3 = np.arange(0,25)
		# for i in range(8):
		# 	ProcP = np.roll(ProcP,1,axis=1)
		# 	x = np.roll(x,1,axis=1)

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
		v = np.linspace(0, 8, 9, endpoint=True)
		bounds=[0,1,2,3,4,5,6,7,8]
				

		# ProcP2 = np.hstack((ProcP2,ProcP2[:,0:1]))

		x2 = np.hstack((x2,x2[0]))

		F = plt.figure(figsize=(15,10))
		plt.rcParams.update({'font.size': 22})
		#plt.plot(CiDT, 'k-', lw = 1.5)
		if FlagMan == True:
			plt.contourf(x3,np.arange(1,14),ProcP3,v,vmax=8,vmin=0)
		else:
			plt.contourf(x3,np.arange(1,14),ProcP3)
		plt.title('Ciclo diurno de la precipitación en el año en ' + Name2,fontsize=26 )  # Colocamos el título del gráfico
		plt.ylabel('Meses',fontsize=24)  # Colocamos la etiqueta en el eje x
		plt.xlabel('Horas',fontsize=24)  # Colocamos la etiqueta en el eje y
		axs = plt.gca()
		axs.yaxis.set_ticks(np.arange(1,14,1))
		axs.set_yticklabels(MM)
		axs.xaxis.set_ticks(np.arange(0,25,1))
		axs.set_xticklabels(x2)
		plt.tight_layout()
		if FlagMan == True:
			cbar = plt.colorbar(boundaries=bounds,ticks=v)
		else:
			cbar = plt.colorbar()
		cbar.set_label('Precipitación [%]')
		plt.gca().invert_yaxis()
		plt.legend(loc=1)
		plt.grid()
		plt.savefig(PathImg + 'TPrec_' + Name+'.png' )
		plt.close('all')

		return ProcP3

	def CiclDV(self,MesesMM,PathImg='',Name='',Name2=''):
		'''
			DESCRIPTION:
		
		Con esta función se pretende realizar el ciclo diurno de cualquier variable
		a lo largo del año.

		Además pueden obtenerse las gráficas si así se desea.
		_________________________________________________________________________

			INPUT:
		+ MesesMM: Variable de los promedios mensuales multianuales.
		+ PathImg: Ruta para guardar las imágenes.
		+ Name: Nombre de las estaciones.
		_________________________________________________________________________
		
			OUTPUT:
		Este código saca 2 gráficas, una por cada mes y el promedio de todos los
		meses.
		- PorcP: Variable directorio con los resultados por mes
		'''
		# Se inicializan las variables
		MM = ['E','F','M','A','M','J','J','A','S','O','N','D','E']
		ProcP = np.empty((12,24)) # Porcentaje de todos los meses

		TT = sum(MesesMM)

		for ii,i in enumerate(range(1,13)):

			ProcP[ii,:] = MesesMM[i]/TT*100

		x = np.arange(0,24)
		x3 = np.arange(0,25)
		# for i in range(8):
		# 	ProcP = np.roll(ProcP,1,axis=1)
		# 	x = np.roll(x,1,axis=1)

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
		v = np.linspace(0, 8, 9, endpoint=True)
		bounds=[0,1,2,3,4,5,6,7,8]
				

		# ProcP2 = np.hstack((ProcP2,ProcP2[:,0:1]))

		x2 = np.hstack((x2,x2[0]))

		F = plt.figure(figsize=(15,10))
		plt.rcParams.update({'font.size': 22})
		#plt.plot(CiDT, 'k-', lw = 1.5)
		plt.contourf(x3,np.arange(1,14),ProcP3,v,vmax=8,vmin=0)
		plt.title('Ciclo diurno de la precipitación en el año en ' + Name2,fontsize=26 )  # Colocamos el título del gráfico
		plt.ylabel('Meses',fontsize=24)  # Colocamos la etiqueta en el eje x
		plt.xlabel('Horas',fontsize=24)  # Colocamos la etiqueta en el eje y
		axs = plt.gca()
		axs.yaxis.set_ticks(np.arange(1,14,1))
		axs.set_yticklabels(MM)
		axs.xaxis.set_ticks(np.arange(0,25,1))
		axs.set_xticklabels(x2)
		plt.tight_layout()
		cbar = plt.colorbar(boundaries=bounds,ticks=v)
		cbar.set_label('Precipitación [%]')
		plt.gca().invert_yaxis()
		plt.legend(loc=1)
		plt.grid()
		plt.savefig(PathImg + 'TPrec_' + Name+'.png' )
		plt.close('all')

		return ProcP3

	def CiclA(self,VMes,Fecha,PathImg='',Name='',VarL='',VarLL='',C='k'):
		'''
			DESCRIPTION:
		
		Con esta función se pretende realizar el ciclo anual de la variable
		que se quiera.

		Además pueden obtenerse las gráficas si así se desea.
		_________________________________________________________________________

			INPUT:
		+ VMes: Variable de los promedios mensuales.
		+ PathImg: Ruta para guardar las imágenes.
		+ Name: Nombre de las estaciones.
		+ VarL: Label de la variable.
		+ VarLL: Nombre de la variable
		+ C: Color de la línea.
		_________________________________________________________________________
		
			OUTPUT:
		Este código saca 1 gráficas, una por cada mes y el promedio de todos los
		meses.
		- PorcP: Variable directorio con los resultados por mes
		'''
		# Se inicializan las variables
		Yi = int(Fecha[0])
		Yf = int(Fecha[1])
		Meses = np.empty((1,12))
		VarM = np.reshape(VMes[:],(-1,12))

		# Calculo del ciclo anual
		MesM = np.nanmean(VarM,axis=0) # Promedio anual.
		MesD = np.nanstd(VarM,axis=0) # Desviación anual.
		VarMNT = []
		# Se calcula el número total de datos
		for i in range(len(VarM[0])):
			VarMNT.append(sum(~np.isnan(VarM[:,i])))

		MesE = [k/np.sqrt(VarMNT[ii]) for ii,k in enumerate(MesD)] # Error anual
		#MesE = [k/np.sqrt(len(VarM)) for k in MesD] # Error anual

		# Figuras
		F = plt.figure(figsize=(15,10))
		plt.rcParams.update({'font.size': 22})
		plt.errorbar(np.arange(1,13),MesM,yerr=MesE,fmt='-',color=C,label=VarLL)
		plt.title('Ciclo anual de '+ VarLL +' de ' + Name,fontsize=26 )  # Colocamos el título del gráfico
		plt.ylabel(VarL,fontsize=24)  # Colocamos la etiqueta en el eje x
		plt.xlabel('Tiempo [Meses]',fontsize=24)  # Colocamos la etiqueta en el eje y
		axes = plt.gca()
		axes.set_xlim([1,12])
		plt.legend(loc=1)
		plt.grid()
		plt.savefig(PathImg + 'CAErr_' + Name+'.png' )
		plt.close('all')

		F = plt.figure(figsize=(15,10))
		plt.rcParams.update({'font.size': 22})
		plt.errorbar(np.arange(1,13),MesM,yerr=MesD,fmt='-',color=C,label=VarLL)
		plt.title('Ciclo anual de '+ VarLL +' de ' + Name,fontsize=26 )  # Colocamos el título del gráfico
		plt.ylabel(VarL,fontsize=24)  # Colocamos la etiqueta en el eje x
		plt.xlabel('Tiempo [Meses]',fontsize=24)  # Colocamos la etiqueta en el eje y
		axes = plt.gca()
		axes.set_xlim([1,12])
		plt.legend(loc=1)
		plt.grid()
		plt.savefig(PathImg + 'CAD_' + Name+'.png' )
		plt.close('all')

		return MesM,MesD,MesE

	def EstAnom(self,VMes):
		'''
			DESCRIPTION:
		
		This function takes the monthly data and generates the data anomalies and the
		standarized data.

		The calculation is done using the following equation:

			Z = \frac{x-\mu}{\sigma}
		_________________________________________________________________________

			INPUT:
		+ VMes: Mounthly average of the variable.
		_________________________________________________________________________
		
			OUTPUT:
		- Anom: Anomalie data results.
		- StanA: Standarized data results.
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
		_________________________________________________________________________

			INPUT:
		+ VMes: Mounthly average of the variable.
		_________________________________________________________________________
		
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



