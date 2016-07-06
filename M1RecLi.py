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
# 	En esta clase se incluyen las rutinas para realizar la reconstrucción
# 	Horaria de la información necesitada por ISAGEN con la metodología 1.
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

import warnings

# Se importan los paquetes para manejo de fechas
from datetime import date, datetime, timedelta

from UtilitiesDGD import UtilitiesDGD
from CorrSt import CorrSt

utl = UtilitiesDGD()
cr = CorrSt()

class M1RecLi:

	def __init__(self):

		'''
			DESCRIPTION:

		Este es el constructor por defecto, no realiza ninguna acción.
		'''

	def ExDH(self,Tot,TotalH,i,FlagN=False,NT=0):
		
		R = sio.loadmat(Tot) # Se cargan los archivos
		# Se cargan las variables
		TCH = R['TCH']
		# Se guadran las variables
		TotalH[i,:] = TCH
		if FlagN == True:
			N = R['N'] # Datos faltantes
			NT[i,:] = N
			return TotalH,NT
		else:
			return TotalH		

	def ExDD(self,Tot,TotalD,TotalmaxD,TotalminD,i,FlagN=False,NT=0):
		'''
			DESCRIPTION:
		
		Con esta función se pretende extraer la información diaria de cada archivo
		.mat y organizarla en la matriz original, es necesario que las columnas de
		cada variable sean iguales.
		_________________________________________________________________________

			INPUT:
		+ Tot: Ruta a los archivos.
		+ TotalD, ...maxD y ...minD: Matriz completa con toda la información.
		+ i: Número de la estación que se va a llenar
		+ FlagN: ¿Se van a extraer porcentajes de datos faltantes?
		+ NT: Vector de datos faltantes.
		_________________________________________________________________________
		
			OUTPUT:
		- TotalD, ...maxD y ...minD: Libera la matriz completa.
		'''
		R = sio.loadmat(Tot) # Se cargan los archivos
		# Se cargan las variables
		TCDdia = R['TCDdia']
		TCDmaxD = R['TCDmaxD']
		TCDminD = R['TCDminD']
		# Se guadran las variables
		TotalD[i,:] = TCDdia
		TotalmaxD[i,:] = TCDmaxD
		TotalminD[i,:] = TCDminD
		if FlagN == True:
			N = R['N'] # Datos faltantes
			NT[i,:] = N
			return TotalD, TotalmaxD, TotalminD,NT
		else:
			return TotalD, TotalmaxD, TotalminD

	def CurveG(self,TotalH,Ai,Af,D):
		'''
			DESCRIPTION:
		
		Con esta función se pretende generar las diferentes curvas diarias de los
		datos horarios que se tienen.
		_________________________________________________________________________

			INPUT:
		+ TotalH: Matriz con los datos totales.
		+ Ai: Año incial de la toma de datos.
		+ Af: Año final de la toma de datos.
		+ D: Filas en donde se encuentran los datos de los sensores.
		_________________________________________________________________________
		
			OUTPUT:
		- CurvasC y CurvasCC: Curvas de todos los datos.
		- CurvasS y CurvasSS: Curvas con todos los sensores.
		- CurvasG y CurvasGG: Curvas con todos los datos.
		'''
		warnings.filterwarnings('ignore')
		# --------------------------------------------------
		# Se inicializan las variables
		# --------------------------------------------------

		Data = dict() # Directorio en donde se encuentran los datos
		CurvasSen = dict() # Curvas de los sensores
		CurvasC = dict() # Directorio en donde se encuentran todas las curvas
		CurvasCC = dict() # Directorio en donde se encuentran todas las curvas

		# --------------------------------------------------
		# Se Calculan las curvas
		# --------------------------------------------------

		for i in range(len(TotalH)):
			Data[i] = np.reshape(TotalH[i,:],(-1,24)) # Se realiza el reshape de los datos
			Meses  = dict() # Directorio para los meses 1: enero, 2: Febrero ...
			DDF = 0 # Contador para los días en general
			for j in range(Ai,Af+1): # Para los años
				x = 0 # Contador de meses
				for g in range(1,13): # Para los meses
					# Contador de meses
					Fi = date(j,g,1)
					if g == 12:
						Ff = date(j+1,1,1)
					else:
						Ff = date(j,g+1,1)
					DF = Ff-Fi
					if j == Ai:
						Meses[x] = Data[i][DDF:DF.days+DDF,:]
					else: 
						Meses[x] = np.vstack((Meses[x],Data[i][DDF:DF.days+DDF,:]))
					x += 1
					DDF = DDF + DF.days # Se suman los días

			# Se calculan los promedios por mes
			x = 0
			for g in range(1,13):
				a = np.nanmean(Meses[x], axis=0)
				b = np.min(a)
				c = np.max(a)
				Dif = c-b
				if g == 1:
					CurvasC[i] = a
					CurvasCC[i] = (a-b)/Dif
				else:
					CurvasC[i] = np.vstack((CurvasC[i],a))
					CurvasCC[i] = np.vstack((CurvasCC[i],(a-b)/Dif))
				x += 1


		Meses = dict()
		for i in range(len(CurvasC)):
			
			for g in range(0,12):
				if i == 0:
					Meses[g] = CurvasC[i][g]
				else:
					Meses[g] = np.vstack((Meses[g],CurvasC[i][g]))
				

		# Se calculan los promedios por mes para todos los datos
		for x in range(0,12):
			a = np.nanmean(Meses[x], axis=0)
			b = np.min(a)
			c = np.max(a)
			Dif = c-b
			if x == 0:
				CurvasG = a
				CurvasGG = (a-b)/Dif
			else:
				CurvasG = np.vstack((CurvasG,a))
				CurvasGG = np.vstack((CurvasGG,(a-b)/Dif))
			

		# Se realiza el mismo cálculo anterior para la curva general de los sensores
		x = 0
		for i in D: # Se toman las curvas de los sensores
			CurvasSen[x] = CurvasC[i]
			x += 1
		
		Meses = dict()
		for i in range(len(CurvasSen)):
			
			for g in range(0,12):
				if i == 0:
					Meses[g] = CurvasSen[i][g]
				else:
					Meses[g] = np.vstack((Meses[g],CurvasSen[i][g]))
				
		# Se calculan los promedios por mes para todos los datos de los sensores
		for x in range(0,12):
			a = np.nanmean(Meses[x], axis=0)
			b = np.min(a)
			c = np.max(a)
			Dif = c-b
			if x == 0:
				CurvasS = a
				CurvasSS = (a-b)/Dif
			else:
				CurvasS = np.vstack((CurvasS,a))
				CurvasSS = np.vstack((CurvasSS,(a-b)/Dif))

		return CurvasC, CurvasCC, CurvasS, CurvasSS, CurvasG, CurvasGG

	def PVTCalc(self,ZT,TotalH,TotalD,TotalmaxD,TotalminD,Ai,Af,NEst,FlagP=True,PathImg=''):
		'''
			DESCRIPTION:
		
		Con esta función se pretende generar los diferentes Perfiles Verticales
		de Temperatura (PVT), asimismo se pretende tener las fechas en donde 
		no se puede calcular el PVT debido a problemas con la cantidad de datos o
		con la pendiente positiva.

		Adicionalmente se presentarán los problemas en donde se tiene pendiente
		positiva o cruzamiento de PVTs en los datos diarios y se recomendará la
		acción a seguir.
		_________________________________________________________________________

			INPUT:
		Esta función guarda gráficos con la información
		+ ZT: Alturas de todos los datos
		+ TotalH: Matriz con los datos horarios.
		+ TotalD, ...maxD y ...minD: Matriz completa diaria.
		+ Ai: Año inicial.
		+ Af: Año final.
		_________________________________________________________________________
		
			OUTPUT:
		Con esta función se obtendrán gráficos con los errores en la reconstrucción
		diaria.
		- Fecha: Vector de Fecha con todos los datos.
		- fTotalH, ...D, ...maxD, ...minD: Matrices de regresiones lineales.
		- N...s: Cantidad de datos con valores NaN de cada reconstrucción.
		- Caso: Vector con todos los casos que se usarán.
		- Hu: Errores en la reconstrucción diaria.
		- ZTH: Alturas de cada reconstrucción horaria.
		'''

		# --------------------------------------------------
		# Se inicializan las variables
		# --------------------------------------------------
		Fecha = ["" for k in range(1)] # Fechas localizadoras
		# Se obtiene le vector fecha
		x = 0
		for result in utl.perdelta(date(Ai, 1, 1), date(Af+1, 1, 1), timedelta(days=1)):
			if x == 0: 
				Fecha[0] = result.strftime('%Y'+'-'+'%m'+'-'+'%d')
			else:
				Fecha.append(result.strftime('%Y'+'-'+'%m'+'-'+'%d'))

			x += 1

		Caso = [1 for k in range(len(Fecha))] # Casos para la reconstrucción
		ZTH = dict() # Directorio de las alturas horarias
		ZTD = dict() # Directorio de las alturas diarias
		ZTmaxD = dict() # Directorio de las alturas diarias
		ZTminD = dict() # Directorio de las alturas diarias
		TD = dict() # Directorio de las temperaturas diarias
		TmaxD = dict() # Directorio de las temperaturas diarias
		TminD = dict() # Directorio de las temperaturas diarias
		NH = ["" for k in range(1)] # Cantidad de datos NaN Horarios
		ND = ["" for k in range(1)] # Cantidad de datos NaN Diarios
		NmaxD = ["" for k in range(1)] # Cantidad de datos NaN Diarios
		NminD = ["" for k in range(1)] # Cantidad de datos NaN Diarios
		Hu = dict() # Directorio de los errores Diarios
		std_errH = [] # Error horario 
		std_errD = [] # Error diario medio
		std_errDM = [] # Error diario máximo
		std_errDm = [] # Error diario máximo

		start_time = time.time() # Para calcular el tiempo estimado de cálculo

		# --------------------------------------------------
		# Se calculan los PVTs Horarios
		# --------------------------------------------------
		print('\nHorarios:')
		# Se calculan los PVTs horarios
		for i in range(len(TotalH[0])):
			if i == 0:

				XX,YY,NH[0] = utl.NoNaN(ZT,TotalH[:,i])
				slope, intercept, r_value, p_value, std_err = st.linregress(XX,YY)
				fTotalH = [slope,intercept,r_value,std_err]
				ZTH[i] = XX
				# Se incluyen los errores
				std_errH.append(std_err)

			else:
				XX,YY,N = utl.NoNaN(ZT,TotalH[:,i])
				NH.append(N)
				slope, intercept, r_value, p_value, std_err = st.linregress(XX,YY)
				fTotalH = np.vstack((fTotalH,[slope,intercept,r_value,std_err]))
				ZTH[i] = XX
				# Se incluyen los errores
				std_errH.append(std_err)

		print("--- Timestep: %s seg ---" % (time.time() - start_time))
		# --------------------------------------------------
		# Se calculan los PVTs Diarios
		# --------------------------------------------------
		print('\nDiarios:')
		x = 0
		# Se calculan los PVTs horarios
		for i in range(len(TotalD[0])):

			if i == 0:

				# Datos medios
				XX,YY,ND[0] = utl.NoNaN(ZT,TotalD[:,i])
				slope, intercept, r_value, p_value, std_err = st.linregress(XX,YY)
				fTotalD = [slope,intercept,r_value,std_err]
				
				# Se agrega el error
				std_errD.append(std_err)

				ZTD[i] = XX
				TD[i] = YY

				# Datos máximos
				XX,YY,NmaxD[0] = utl.NoNaN(ZT,TotalmaxD[:,i])
				slope, intercept, r_value, p_value, std_err = st.linregress(XX,YY)
				fTotalmaxD = [slope,intercept,r_value,std_err]
				
				# Se agrega el error
				std_errDM.append(std_err)

				ZTmaxD[i] = XX
				TmaxD[i] = YY

				# Datos mínimos
				XX,YY,NminD[0] = utl.NoNaN(ZT,TotalminD[:,i])
				slope, intercept, r_value, p_value, std_err = st.linregress(XX,YY)
				fTotalminD = [slope,intercept,r_value,std_err]

				# Se agrega el error
				std_errDm.append(std_err)

				ZTminD[i] = XX
				TminD[i] = YY

			else:
				# Datos medios
				XX,YY,N = utl.NoNaN(ZT,TotalD[:,i])
				ND.append(N)
				slope, intercept, r_value, p_value, std_err = st.linregress(XX,YY)
				fTotalD = np.vstack((fTotalD,[slope,intercept,r_value,std_err]))

				# Se agrega el error
				std_errD.append(std_err)

				ZTD[i] = XX
				TD[i] = YY

				# Datos máximos
				XX,YY,N = utl.NoNaN(ZT,TotalmaxD[:,i])
				NmaxD.append(N)
				slope, intercept, r_value, p_value, std_err = st.linregress(XX,YY)
				fTotalmaxD = np.vstack((fTotalmaxD,[slope,intercept,r_value,std_err]))

				# Se agrega el error
				std_errDM.append(std_err)

				ZTmaxD[i] = XX
				TmaxD[i] = YY

				# Datos máximos
				XX,YY,N = utl.NoNaN(ZT,TotalminD[:,i])
				NminD.append(N)
				slope, intercept, r_value, p_value, std_err = st.linregress(XX,YY)
				fTotalminD = np.vstack((fTotalminD,[slope,intercept,r_value,std_err]))
				
				# Se agrega el error
				std_errDm.append(std_err)

				ZTminD[i] = XX
				TminD[i] = YY

			# Se miran las restricciones de los horarios
			
			for j in range(24):
				c = sum(np.where(ZTH[x] > 2900,1,0))
				cc = sum(np.where(ZTH[x] <= 2900,1,0))
				#print(c,' ',cc)

				if NH[x] <=4 or c == 0 or cc == 0 or fTotalH[x,0] >= 0:
					Caso[i] = 2
				x += 1
		x = 0
		# Se mira si los PVT se cruzan
		for i in range(len(TotalD[0])):
			if fTotalmaxD[i,0] is not fTotalminD[i,0]:

				a1 = fTotalmaxD[i,0]
				a2 = fTotalminD[i,0]

				b1 = fTotalmaxD[i,1]
				b2 = fTotalminD[i,1]

				HH = (((b2)-(b1))/(a1-a2))

				if ((HH >= 500 and HH <= 4000) or (a1 >= 0) or (a2 >= 0)) and Caso[i] == 2:

					Hu[x] = [Fecha[i],HH,a1,a2,b1,b2,Caso[i]]
					
					Mmax = [(a1*4000+b1),(a1*100+b1)]
					Mmin = [(a2*4000+b2),(a2*100+b2)]
					Z = [4000,100]
					if FlagP == True:
						# Se grafican los datos
						F = plt.figure(figsize=(15,10))
						plt.rcParams.update({'font.size': 22})
						plt.plot(Mmax,Z, 'r-', lw = 1.5)
						plt.scatter(TmaxD[i],ZTmaxD[i],color='red')
						plt.plot(Mmin,Z, 'b-', lw = 1.5)
						plt.scatter(TminD[i],ZTminD[i],color='blue')
						plt.title(Fecha[i],fontsize=26 )  # Colocamos el título del gráfico
						plt.xlabel(u'Temperatura [°C]',fontsize=24)  # Colocamos la etiqueta en el eje x
						plt.ylabel('Altura [msnm]',fontsize=24)  # Colocamos la etiqueta en el eje y
						axes = plt.gca()
						axes.set_ylim([0,4500])
						plt.grid()
						plt.savefig(PathImg + 'NEst_' + str(NEst) + Fecha[i]  +'.png' )
						plt.close('all')

					x += 1

		print("--- Timestep: %s seg ---" % (time.time() - start_time))

		return Fecha,fTotalH,NH,fTotalD,ND,fTotalmaxD,NmaxD,fTotalminD,NminD,Caso,Hu,ZTH,std_errH,std_errD,std_errDM,std_errDm
				

	def RecData(self,fTotalHE,Z,FlagD=True,fTotalDE=0,fTotalmaxDE=0,fTotalminDE=0,Curva=0,An=0,FlagHu=False,Fecha=0,Hu=0):
		'''
			DESCRIPTION:
		
		Con esta función se pretende realizar la reconstrucción de cualquier serie
		de temperatura en las zona escogida y en el periodo de tiempo designado.
		_________________________________________________________________________

			INPUT:
		+ fTotalHE, ...DE, ...maxDE, ...minDE: Matrices de regresiones lineales escodigas.
		+ Z: Vector de alturas.
		+ Curva: Curva para la reconstrucción diaria escogida.
		+ An: Año de reconstrucción
		+ FlagD: ¿Reconstrucción Diaria u Horaria?
		+ FlagHu: ¿Es necesario realizar alguna corrección?
		+ Fecha: Fechas diarias.
		+ Hu: Vector con las fechas en donde se encuentra un error.
		_________________________________________________________________________
		
			OUTPUT:
		- RecH: Reconstrucción horaria.
		- RecD: Reconstrucción Diaria media.
		- RecDH: Reconstrucción de Diaria a Horaria.
		- Deltas: Deltas de temperatura.
		'''
		# --------------------------------------------------
		# Se Realiza la reconstrucción
		# --------------------------------------------------
		if FlagD == False:
			RecH = np.empty(len(fTotalHE)) # Variable de reconstrucción
			m = fTotalHE[:,0] # Pendientes
			b = fTotalHE[:,1] # Interceptos
			# Ciclo para la reconstrucción 
			for i in range(len(fTotalHE)):
				RecH[i] = m[i]*Z+b[i] # Reconstrucción lineal

			return RecH

		elif FlagD == True:
			RecD = np.empty(len(fTotalDE)) # Variable de reconstrucción Diaria media
			DatamaxD = np.empty(len(fTotalDE)) # Variable de reconstrucción Diaria media
			DataminD = np.empty(len(fTotalDE)) # Variable de reconstrucción Diaria media
			RecDH = np.empty(len(fTotalHE)) # Variable de reconstrucción Diaria a horaria
			mD = fTotalDE[:,0] # Pendientes
			mmaxD = fTotalmaxDE[:,0] # Pendientes
			mminD = fTotalminDE[:,0] # Pendientes
			bD = fTotalDE[:,1] # Interceptos
			bmaxD = fTotalmaxDE[:,1] # Interceptos
			bminD = fTotalminDE[:,1] # Interceptos

			# Ciclo para la reconstrucción 
			for i in range(len(fTotalDE)): # Reconstrucción media diaria
				RecD[i] = mD[i]*Z+bD[i] # Reconstrucción lineal
				DatamaxD[i] = mmaxD[i]*Z+bmaxD[i] # Reconstrucción lineal
				DataminD[i] = mminD[i]*Z+bminD[i] # Reconstrucción lineal
			# Se realizan correcciones lineales de datos
			if FlagHu == True:
				for i in Hu:
					P = np.where(Fecha == i)[0]
					DatamaxD[P] = utl.Interp(1,DatamaxD[P-1],2,3,DatamaxD[P-1])
					DataminD[P] = utl.Interp(1,DataminD[P-1],2,3,DataminD[P-1])

			Delta = DatamaxD-DataminD
			x = 0 # Contador de meses
			xx = 0 # Contador de días
			xxx = 0 # Contador de las horas
			# Ciclo Diaria - Horaria 
			for i in range(1,13): # Ciclo para los meses
				# Contador de meses
				Fi = date(An,i,1)
				if i == 12:
					Ff = date(An+1,1,1)
				else:
					Ff = date(An,i+1,1)
				DF = Ff-Fi

				for j in range(DF.days): # Ciclo para los días
					for g in range(24): # Ciclo para las horas
						RecDH[xxx] = (Delta[xx]*Curva[x][g])+DataminD[xx]
						xxx += 1
					xx += 1
				x += 1
			



			return RecD, RecDH, Delta

	def GraphH(self,RecH,FechaHE,FechaHVE,Year,PathImg,FlagV=False,ValR=0,Name='',Method=1):
		'''
			DESCRIPTION:
		
		Con esta función se pretende graficas los datos de Reconstrucción horarios
		por año. A su vez pued realizar la validación de los datos.
		_________________________________________________________________________

			INPUT:
		+ RecH: Serie reconstruida.
		+ FechaHE: Serie de Fechas Escogidas.
		+ FechaHVE: Serie de Fechas para graficar.
		+ PathImg: Ruta para guardar las imágenes, de debe tener una carpeta /Validation/
				   en la ruta para que se guarden ahí.
		+ FlagV: ¿Es Necesario realizar reconstrucción?
		+ ValR: Serie de validación
		+ Name: Nombre de la serie que se reconstruyó.
		_________________________________________________________________________
		
			OUTPUT:
		Este archivo
		- 
		'''
		# --------------------------------------------------
		# Se inicializan las variables
		# --------------------------------------------------
		# Variables de meses
		plt.close('all')
		Mes = ['Enero','Febrero','Marzo','Abril','Mayo','Junio',\
		'Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'];
		lfs = 18

		# Se comienzan a hacer las gráficas
		# Los primeros 6 meses
		fig, axs = plt.subplots(2,3, figsize=(15, 8), facecolor='w', edgecolor='k')
		plt.rcParams.update({'font.size': lfs-6})
		axs = axs.ravel() # Para hacer un loop con los subplots
		myFmt = mdates.DateFormatter('%d/%m/%y') # Para hacer el formato de fechas
		xlabels = ['Ticklabel %i' % i for i in range(10)] # Para rotar los ejes
		x = 1
		for i in range(6): # Primeros 6 valores
			# Se obtiene la información de las fechas
			Fi = FechaHE.index(str(Year)+ '-0' + str(i+1) +'-01-0000') # Año incial
			Ff = FechaHE.index(str(Year)+ '-0' + str(i+2) + '-01-0000') # Año incial

			# Se extraen los datos
			RecHi = RecH[Fi:Ff]
			FechaHVi = FechaHVE[Fi:Ff]

			# Se grafican los datos
			axs[i].set_color_cycle([(1.0,0.3,0.0), (0.4,0.4,1.0)])
			axs[i].plot(FechaHVi,RecHi,'-', label = u'Reconstruidos')
			axs[i].set_title(Mes[i],fontsize=lfs)
			# Se nombran los ejes
			if i == 0 or i == 3:
				axs[i].set_ylabel(u'Temperatura[°C]',fontsize=lfs-4)
			if i >= 3:
				axs[i].set_xlabel(u'Fechas',fontsize=14)
			# Se grafíca la validación
			if FlagV == True:
				ValRi = ValR[Fi:Ff]
				axs[i].plot(FechaHVi,ValRi,'-', label = u'Reales')
				axs[2].legend(loc='best')
			axs[i].set_xticklabels(xlabels, rotation=40)
			axs[i].xaxis.set_major_formatter(myFmt)
			x += 1
		# Se plotea la legenda y se guarda la imágen
		plt.tight_layout()
		if FlagV == True:
			plt.savefig(PathImg + '/Validation/ValH_'+ str(Method) + '_'+ str(Year) + '_' + Name + '_1' +'.png' )
		elif FlagV == False:
			plt.savefig(PathImg + '/Rec/Img/' + str(Year) + '_1' +'.png' )
		plt.close('all')

		# Los otros 6 meses
		fig, axs = plt.subplots(2,3, figsize=(15, 8), facecolor='w', edgecolor='k')
		plt.rcParams.update({'font.size': lfs-6})
		axs = axs.ravel() # Para hacer un loop con los subplots
		myFmt = mdates.DateFormatter('%d/%m/%y') # Para hacer el formato de fechas
		xlabels = ['Ticklabel %i' % i for i in range(10)] # Para rotar los ejes

		for i in range(6): # Primeros 6 valores
			# Se obtiene la información de las fechas
			if x < 10:
				Fi = FechaHE.index(str(Year)+ '-0' + str(x) +'-01-0000') # Año incial
				if x == 9:
					Ff = FechaHE.index(str(Year)+ '-' + str(x+1) + '-01-0000') # Año incial
				else:
					Ff = FechaHE.index(str(Year)+ '-0' + str(x+1) + '-01-0000') # Año incial
			else:
				Fi = FechaHE.index(str(Year)+ '-' + str(x) +'-01-0000') # Año incial
				if x == 12: # Mes 12
					Ff = FechaHE.index(str(Year)+ '-' + str(x) + '-31-0000') # Año incial
				else:
					Ff = FechaHE.index(str(Year)+ '-' + str(x+1) + '-01-0000') # Año incial

			# Se extraen los datos
			RecHi = RecH[Fi:Ff]
			FechaHVi = FechaHVE[Fi:Ff]

			# Se grafican los datos
			axs[i].set_color_cycle([(1.0,0.3,0.0), (0.4,0.4,1.0)])
			axs[i].plot(FechaHVi,RecHi,'-', label = u'Reconstruidos')
			axs[i].set_title(Mes[x-1],fontsize=lfs)
			# Se nombran los ejes
			if i == 0 or i == 3:
				axs[i].set_ylabel(u'Temperatura[°C]',fontsize=lfs-4)
			if i >= 3:
				axs[i].set_xlabel(u'Fechas',fontsize=14)
			# Se grafíca la validación
			if FlagV == True:
				ValRi = ValR[Fi:Ff]
				axs[i].plot(FechaHVi,ValRi,'-', label = u'Reales')
				axs[2].legend(loc='best')
			axs[i].set_xticklabels(xlabels, rotation=40)
			axs[i].xaxis.set_major_formatter(myFmt)
			x += 1
		# Se plotea la legenda y se guarda la imágen
		plt.tight_layout()
		if FlagV == True:
			plt.savefig(PathImg + '/Validation/ValH_'+ str(Method) + '_'+ str(Year) + '_' + Name + '_2' +'.png' )
		elif FlagV == False:
			plt.savefig(PathImg + '/Rec/Img/' + str(Year) + '_2' +'.png' )

		# Se realiza la segunda parte de la validación
		if FlagV == True:
			# Se calcula el error de medición
			E_M = ValR-RecH # Real - Reconstruido

			# Se comienzan a hacer las gráficas
			# Los primeros 6 meses
			fig, axs = plt.subplots(2,3, figsize=(15, 8), facecolor='w', edgecolor='k')
			plt.rcParams.update({'font.size': lfs-6})
			axs = axs.ravel() # Para hacer un loop con los subplots
			myFmt = mdates.DateFormatter('%d/%m/%y') # Para hacer el formato de fechas
			xlabels = ['Ticklabel %i' % i for i in range(10)] # Para rotar los ejes

			x = 1
			for i in range(6): # Primeros 6 valores
				# Se obtiene la información de las fechas
				Fi = FechaHE.index(str(Year)+ '-0' + str(i+1) +'-01-0000') # Año incial
				Ff = FechaHE.index(str(Year)+ '-0' + str(i+2) + '-01-0000') # Año incial
				
				RecHi = RecH[Fi:Ff]
				ValRi = ValR[Fi:Ff]
				# Se calcula la correlación
				XX, YY, N = utl.NoNaN(RecHi,ValRi)
				CCP,CCS,QQ = cr.CorrC(XX,YY,True,0,0.05)

				# Se extraen los datos
				E_Mi = E_M[Fi:Ff]
				FechaHVi = FechaHVE[Fi:Ff]
				# Se grafican los datos
				axs[i].plot(FechaHVi,E_Mi,'k-', label = u'Reconstruidos')
				axs[i].plot([FechaHVi[0],FechaHVi[len(FechaHVi)-1]],[np.nanmean(E_Mi),np.nanmean(E_Mi)],'k--')
				#axs[i].text(max(E_Mi)-0.5, FechaHVE[1] , u'Correlación de Pearson: \color{blue} '+str(round(CCP,3)), fontsize=15)
				# Se ponen las correlaciones
				L = 7
				axs[i].text(FechaHVi[1],L, r'Correlación de Pearson:', fontsize=12)
				axs[i].text(FechaHVi[1],L-1, r'Correlación de Spearman:', fontsize=12)
				gg = 400
				if CCP >= QQ[0]:
					axs[i].text(FechaHVi[gg],L, r'%s' %(round(CCP,3)), fontsize=12,color='blue')
				else:
					axs[i].text(FechaHVi[gg],L, r'%s' %(round(CCP,3)), fontsize=12,color='red')

				if CCS >= QQ[1]:
					axs[i].text(FechaHVi[gg],L-1, r'%s' %(round(CCS,3)), fontsize=12,color='blue')
				else:
					axs[i].text(FechaHVi[gg],L-1, r'%s' %(round(CCS,3)), fontsize=12,color='red')
				axs[i].set_title(Mes[i],fontsize=lfs)

				# Se nombran los ejes
				if i == 0 or i == 3:
					axs[i].set_ylabel(u'Error de Estimación [°C]',fontsize=lfs-4)
				if i >= 3:
					axs[i].set_xlabel(u'Fechas',fontsize=14)
				
				# Se grafíca la validación
				axs[i].set_xticklabels(xlabels, rotation=40)
				axs[i].xaxis.set_major_formatter(myFmt)
				axs[i].set_ylim([-8,8])
				x += 1

			# Se plotea la legenda y se guarda la imágen
			plt.tight_layout()
			if FlagV == True:
				plt.savefig(PathImg + '/Validation/ValH_'+ str(Method) + '_'+ str(Year) + '_' + Name + '_3' +'.png' )
			plt.close('all')

			# Se comienzan a hacer las gráficas
			# Los primeros 6 meses
			fig, axs = plt.subplots(2,3, figsize=(15, 8), facecolor='w', edgecolor='k')
			plt.rcParams.update({'font.size': lfs-6})
			axs = axs.ravel() # Para hacer un loop con los subplots
			myFmt = mdates.DateFormatter('%d/%m/%y') # Para hacer el formato de fechas
			xlabels = ['Ticklabel %i' % i for i in range(10)] # Para rotar los ejes

			
			for i in range(6): # Primeros 6 valores
				# Se obtiene la información de las fechas
				if x < 10:
					Fi = FechaHE.index(str(Year)+ '-0' + str(x) +'-01-0000') # Año incial
					if x == 9:
						Ff = FechaHE.index(str(Year)+ '-' + str(x+1) + '-01-0000') # Año incial
					else:
						Ff = FechaHE.index(str(Year)+ '-0' + str(x+1) + '-01-0000') # Año incial
				else:
					Fi = FechaHE.index(str(Year)+ '-' + str(x) +'-01-0000') # Año incial
					if x == 12: # Mes 12
						Ff = FechaHE.index(str(Year)+ '-' + str(x) + '-31-0000') # Año incial
					else:
						Ff = FechaHE.index(str(Year)+ '-' + str(x+1) + '-01-0000') # Año incial
				
				RecHi = RecH[Fi:Ff]
				ValRi = ValR[Fi:Ff]
				# Se calcula la correlación
				XX, YY, N = utl.NoNaN(RecHi,ValRi)
				CCP,CCS,QQ = cr.CorrC(XX,YY,True,0,0.05)

				# Se extraen los datos
				E_Mi = E_M[Fi:Ff]
				FechaHVi = FechaHVE[Fi:Ff]
				# Se grafican los datos
				axs[i].plot(FechaHVi,E_Mi,'k-', label = u'Reconstruidos')
				axs[i].plot([FechaHVi[0],FechaHVi[len(FechaHVi)-1]],[np.nanmean(E_Mi),np.nanmean(E_Mi)],'k--')
				#axs[i].text(max(E_Mi)-0.5, FechaHVE[1] , u'Correlación de Pearson: \color{blue} '+str(round(CCP,3)), fontsize=15)
				# Se ponen las correlaciones
				axs[i].text(FechaHVi[1],L, r'Correlación de Pearson:', fontsize=12)
				axs[i].text(FechaHVi[1],L-1, r'Correlación de Spearman:', fontsize=12)
				gg = 400
				if CCP >= QQ[0]:
					axs[i].text(FechaHVi[gg],L, r'%s' %(round(CCP,3)), fontsize=12,color='blue')
				else:
					axs[i].text(FechaHVi[gg],L, r'%s' %(round(CCP,3)), fontsize=12,color='red')

				if CCS >= QQ[1]:
					axs[i].text(FechaHVi[gg],L-1, r'%s' %(round(CCS,3)), fontsize=12,color='blue')
				else:
					axs[i].text(FechaHVi[gg],L-1, r'%s' %(round(CCS,3)), fontsize=12,color='red')
				axs[i].set_title(Mes[x-1],fontsize=lfs)

				# Se nombran los ejes
				if i == 0 or i == 3:
					axs[i].set_ylabel(u'Error de Estimación [°C]',fontsize=lfs-4)
				if i >= 3:
					axs[i].set_xlabel(u'Fechas',fontsize=14)
				
				# Se grafíca la validación
				axs[i].set_xticklabels(xlabels, rotation=40)
				axs[i].xaxis.set_major_formatter(myFmt)
				axs[i].set_ylim([-8,8])
				x += 1
			# Se plotea la legenda y se guarda la imágen
			plt.tight_layout()
			if FlagV == True:
				plt.savefig(PathImg + '/Validation/ValH_'+ str(Method) + '_'+ str(Year) + '_' + Name + '_4' +'.png' )
			plt.close('all')

			
			# Se calcula el histograma 
			# Se quitan los valores NaN
			q = ~np.isnan(E_M)
			E_MM = E_M[q]

			fig, axs = plt.subplots(1,2, figsize=(15, 8), facecolor='w', edgecolor='k')
			plt.rcParams.update({'font.size': 14})
			axs = axs.ravel() # Para hacer un loop con los subplots

			# Se calculan los momentos estadísticos
			A = np.nanmean(E_MM)
			B = np.nanstd(E_MM)
			C = st.skew(E_MM)
			D = st.kurtosis(E_MM)

			# the histogram of the data
			n, bins, patches = axs[0].hist(E_MM,bins=30, normed=1, facecolor='blue', alpha=0.5)
			# add a 'best fit' line
			axs[0].set_title('Histograma del Error de Estimación',fontsize=18)
			axs[0].set_xlabel('Error de Estimación [°C]',fontsize=16)
			axs[0].set_ylabel('Probabilidad',fontsize=16)

			# Se incluyen los momentos estadísticos
			axs[0].text(bins[0],max(n)-0.05, r'$\mu=$ %s' %(round(A,3)), fontsize=12)
			axs[0].text(bins[0],max(n)-0.1, r'$\sigma=$ %s' %(round(B,3)), fontsize=12)
			axs[0].text(bins[0],max(n)-0.15, r'$\gamma=$ %s' %(round(C,3)), fontsize=12)
			axs[0].text(bins[0],max(n)-0.2, r'$\kappa=$ %s' %(round(D,3)), fontsize=12)


			axs[1].plot([-4,40], [-4,40], 'k-')
			axs[1].scatter(RecH, ValR, linewidth='0')
			axs[1].set_title('Diagrama de Dispersión',fontsize=18)
			axs[1].set_xlabel('Datos Calculados [°C]',fontsize=16)
			axs[1].set_ylabel('Datos Reales [°C]',fontsize=16)
			axs[1].set_ylim(-3,30)
			axs[1].set_xlim(-3,30)
			plt.savefig(PathImg + '/Validation/ValH_'+ str(Method) + '_'+ str(Year) + '_' + Name + '_5' +'.png' )
			plt.close('all')


			return E_M

	def GraphD(self,RecD,FechaE,FechaVE,Year,PathImg,FlagV=False,ValR=0,Name='',Method=1):
		'''
			DESCRIPTION:
		
		Con esta función se pretende graficas los datos de Reconstrucción horarios
		por año. A su vez pued realizar la validación de los datos.
		_________________________________________________________________________

			INPUT:
		+ RecD: Serie reconstruida.
		+ FechaE: Serie de Fechas Escogidas.
		+ FechaVE: Serie de Fechas para graficar.
		+ PathImg: Ruta para guardar las imágenes, de debe tener una carpeta /Validation/
				   en la ruta para que se guarden ahí.
		+ FlagV: ¿Es Necesario realizar reconstrucción?
		+ ValR: Serie de validación
		+ Name: Nombre de la serie que se reconstruyó.
		_________________________________________________________________________
		
			OUTPUT:
		Este archivo
		- E_M: Errores de medición
		'''
		# --------------------------------------------------
		# Se inicializan las variables
		# --------------------------------------------------
		# Variables de meses
		plt.close('all')
		
		lfs = 22

		# Se comienzan a hacer las gráficas
		# Se grafica todo el año
		fig, axs = plt.subplots(1,1, figsize=(15, 8), facecolor='w', edgecolor='k')
		plt.rcParams.update({'font.size': lfs-6})
		
		myFmt = mdates.DateFormatter('%d/%m/%y') # Para hacer el formato de fechas
		xlabels = ['Ticklabel %i' % i for i in range(10)] # Para rotar los ejes
		i = 0

		RecHi = RecD
		# Se grafican los datos
		plt.gca().set_color_cycle([(1.0,0.3,0.0), (0.4,0.4,1.0)])
		plt.plot(FechaVE,RecHi,'-', label = u'Reconstruidos')
		plt.title(str(Year),fontsize=lfs)
		# Se nombran los ejes
		if i == 0 or i == 3:
			plt.ylabel(u'Temperatura[°C]',fontsize=lfs-4)
		if i >= 3:
			plt.xlabel(u'Fechas',fontsize=14)
		# Se grafíca la validación
		if FlagV == True:
			ValRi = ValR
			plt.plot(FechaVE,ValRi,'-', label = u'Reales')
			plt.legend(loc='best')
		labels = axs.get_xticklabels()
		plt.setp(labels, rotation=40)
		plt.gca().xaxis.set_major_formatter(myFmt)
		
		# Se plotea la legenda y se guarda la imágen
		plt.tight_layout()
		if FlagV == True:
			plt.savefig(PathImg + '/Validation/ValD_'+ str(Method) + '_' + str(Year) + '_' + Name + '_1' +'.png' )
		elif FlagV == False:
			plt.savefig(PathImg + '/Rec/Img/' + str(Year) + '_1' +'.png' )
		plt.close('all')

		

		# Se realiza la segunda parte de la validación
		if FlagV == True:
			# Se calcula el error de medición
			E_M = ValR-RecD # Real - Reconstruido

			# Se comienzan a hacer las gráficas
			# Los primeros 6 meses
			fig, axs = plt.subplots(1,1, figsize=(15, 8), facecolor='w', edgecolor='k')
			plt.rcParams.update({'font.size': lfs-6})
			
			myFmt = mdates.DateFormatter('%d/%m/%y') # Para hacer el formato de fechas
			xlabels = ['Ticklabel %i' % i for i in range(10)] # Para rotar los ejes
			i = 1
			
			# Se calcula la correlación
			XX, YY, N = utl.NoNaN(RecHi,ValRi)
			CCP,CCS,QQ = cr.CorrC(XX,YY,True,0,0.05)

			# Se extraen los datos
			E_Mi = E_M
			
			
			# Se grafican los datos
			plt.plot(FechaVE,E_Mi,'k-', label = u'Reconstruidos')
			plt.plot([FechaVE[0],FechaVE[len(FechaVE)-1]],[np.mean(E_Mi),np.mean(E_Mi)],'k--')
			# Se ponen las correlaciones
			plt.text(FechaVE[1],max(E_Mi)+0.5, r'Correlación de Pearson:', fontsize=12)
			plt.text(FechaVE[1],max(E_Mi), r'Correlación de Spearman:', fontsize=12)
			gg = 100
			if CCP >= QQ[0]:
				plt.text(FechaVE[gg],max(E_Mi)+0.5, r'%s' %(round(CCP,3)), fontsize=12,color='blue')
			else:
				plt.text(FechaVE[gg],max(E_Mi)+0.5, r'%s' %(round(CCP,3)), fontsize=12,color='red')
			if CCS >= QQ[1]:
				plt.text(FechaVE[gg],max(E_Mi), r'%s' %(round(CCS,3)), fontsize=12,color='blue')
			else:
				plt.text(FechaVE[gg],max(E_Mi), r'%s' %(round(CCS,3)), fontsize=12,color='red')
			plt.title(str(Year),fontsize=lfs)

			# Se nombran los ejes
			if i == 0 or i == 3:
				plt.ylabel(u'Error de Estimación [°C]',fontsize=lfs-4)
			if i >= 3:
				plt.xlabel(u'Fechas',fontsize=14)
			
			# Se grafíca la validación
			labels = axs.get_xticklabels()
			plt.setp(labels, rotation=40)
			plt.gca().xaxis.set_major_formatter(myFmt)
			plt.ylim([min(E_M)-1,max(E_M)+1])
			
			# Se plotea la legenda y se guarda la imágen
			plt.tight_layout()
			if FlagV == True:
				plt.savefig(PathImg + '/Validation/ValD_'+ str(Method) + '_'+ str(Year) + '_' + Name + '_3' +'.png' )
			plt.close('all')

			

			
			# Se calcula el histograma 
			# Se quitan los valores NaN
			q = ~np.isnan(E_M)
			E_MM = E_M[q]

			fig, axs = plt.subplots(1,2, figsize=(15, 8), facecolor='w', edgecolor='k')
			plt.rcParams.update({'font.size': 14})
			axs = axs.ravel() # Para hacer un loop con los subplots

			# Se calculan los momentos estadísticos
			A = np.nanmean(E_MM)
			B = np.nanstd(E_MM)
			C = st.skew(E_MM)
			D = st.kurtosis(E_MM)

			# the histogram of the data
			n, bins, patches = axs[0].hist(E_MM,bins=30, normed=1, facecolor='blue', alpha=0.5)
			# add a 'best fit' line
			axs[0].set_title('Histograma del Error de Estimación',fontsize=18)
			axs[0].set_xlabel('Error de Estimación [°C]',fontsize=16)
			axs[0].set_ylabel('Probabilidad',fontsize=16)

			# Se incluyen los momentos estadísticos
			axs[0].text(bins[0],max(n)-0.05, r'$\mu=$ %s' %(round(A,3)), fontsize=12)
			axs[0].text(bins[0],max(n)-0.1, r'$\sigma=$ %s' %(round(B,3)), fontsize=12)
			axs[0].text(bins[0],max(n)-0.15, r'$\gamma=$ %s' %(round(C,3)), fontsize=12)
			axs[0].text(bins[0],max(n)-0.2, r'$\kappa=$ %s' %(round(D,3)), fontsize=12)


			axs[1].plot([0,40], [0,40], 'k-')
			axs[1].scatter(RecD, ValR, linewidth='0')
			axs[1].set_title('Diagrama de Dispersión',fontsize=18)
			axs[1].set_xlabel('Datos Calculados [°C]',fontsize=16)
			axs[1].set_ylabel('Datos Reales [°C]',fontsize=16)
			axs[1].set_ylim([min(ValR)-2,max(ValR)+2])
			axs[1].set_xlim([min(RecD)-2,max(RecD)+2])
			plt.savefig(PathImg + '/Validation/ValD_'+ str(Method) + '_'+ str(Year) + '_' + Name + '_5' +'.png' )
			plt.close('all')


			return E_M

	def ValData(self,TotalH,TotalD,TotalmaxD,TotalminD,Fecha,FechaV,FechaH,FechaHV,ZT,\
		ValRow,Year,Method,CurvasCC,CurvasSS,CurvasGG,CRow,Name,PathImg):
		'''
			DESCRIPTION:
		
		Con esta función se pretende desarrollar la validación de la información
		que se tiene en la hoja de Excel Data_imp.xlsx, se debe tener en cuenta
		que la validación se desarrollará tanto diaria como horaria y que los
		datos con los cuales se va a validar deben tener esta misma resolución.
		_________________________________________________________________________

			INPUT:
		+ TotalD, ...maxD y ...minD: Matriz completa diaria.
		+ ZT: Vector de alturas.
		+ ValRow: Filas en donde se encuentran los datos de validación.
		+ Year: Año de validación.
		+ Method: Método que se usará para validar.
		+ CurvasCC, ...SS, ...GG: Curvas que se usarán para realizar la
								  reconstrucción diaria.
		+ CVRowC: Fila de la curva característica.
		+ CVRowG: Designa cual curva general se utilizará.
		_________________________________________________________________________
		
			OUTPUT:
		Este archivo
		- E_M
		'''

		# --------------------------------------------------
		# Se realizan los PVTs
		# --------------------------------------------------
		start_time = time.time() # Para calcular el tiempo estimado de cálculo
		print(' Se calculan los PVTs sin la estación')
		# Se obtiene el PVT sin los datos de la serie que se va a reconstruir
		TotalH2 = np.delete(TotalH,(ValRow),axis=0)
		TotalD2 = np.delete(TotalD,(ValRow),axis=0)
		TotalmaxD2 = np.delete(TotalmaxD,(ValRow),axis=0)
		TotalminD2 = np.delete(TotalminD,(ValRow),axis=0)
		ZT2 = np.delete(ZT,(ValRow),axis=0)
		# Se extrae el año en cuestion
		# Horaria
		Fi = FechaH.index(str(Year)+'-01-01-0000') # Año incial
		Ff = FechaH.index(str(Year+1)+'-01-01-0000') # Año incial
		TotalHE2 = TotalH2[:,Fi:Ff]

		# Diaria
		Fi = np.where(Fecha == (str(Year)+'-01-01'))[0] # Año incial
		Ff = np.where(Fecha == (str(Year+1)+'-01-01'))[0] # Año final
		TotalDE2 = TotalD2[:,Fi:Ff]
		TotalmaxDE2 = TotalmaxD2[:,Fi:Ff]
		TotalminDE2 = TotalminD2[:,Fi:Ff]


		Fecha2,fTotalH,NH,fTotalD,ND,fTotalmaxD,NmaxD,fTotalminD,NminD,Caso,Hu,ZTH,std_errH,std_errD,std_errDM,std_errDm\
			= self.PVTCalc(ZT2,TotalHE2,TotalDE2,\
			TotalmaxDE2,TotalminDE2,Year,Year,len(TotalH-1))

		print()
		print('----')
		print("--- Timestep: %s seg ---" % (time.time() - start_time))
		# --------------------------------------------------
		# Se realiza la validación
		# --------------------------------------------------
		print('Validación: '+ Name)
		# Se mira el tipo de reconstrucción que se realizará 

		if Method == 1:
			# Se buscan las Fechas en común
			#Fi = np.where(FechaH == (str(Year)+'-01-01-0000'))[0] # Año incial
			#Ff = np.where(FechaH == (str(Year+1)+'-01-01-0000'))[0] # Año final
			Fi = FechaH.index(str(Year)+'-01-01-0000') # Año incial
			Ff = FechaH.index(str(Year+1)+'-01-01-0000') # Año incial

			FechaHVE = FechaHV[Fi:Ff] # Fechas para desarrollar las gráficas
			FechaHE = FechaH[Fi:Ff] # Fechas para desarrollar las gráficas

			# Se extrae la información para la validación
			ValR = TotalH[ValRow,Fi:Ff]
			ZVal = ZT[ValRow]

			# Se realiza la reconstrucción
			fTotalHE = fTotalH
			RecH = self.RecData(fTotalHE,ZVal,False)
			# Se grafican los datos
			E_M = self.GraphH(RecH,FechaHE,FechaHVE,Year,PathImg,True,ValR,Name)

			return E_M
			print('End')
			print("--- Timestep: %s seg ---" % (time.time() - start_time))

		elif Method == 2:
			# Se buscan las Fechas en común
			Fi = FechaH.index(str(Year)+'-01-01-0000') # Año incial
			Ff = FechaH.index(str(Year+1)+'-01-01-0000') # Año incial
			FechaHVE = FechaHV[Fi:Ff] # Fechas para desarrollar las gráficas
			FechaHE = FechaH[Fi:Ff] # Fechas para desarrollar las gráficas
			fTotalHE = fTotalH
			ValRH = TotalH[ValRow,Fi:Ff]

			Fi = np.where(Fecha == (str(Year)+'-01-01'))[0] # Año incial
			Ff = np.where(Fecha == (str(Year+1)+'-01-01'))[0] # Año final

			FechaVE = FechaV[Fi:Ff] # Fechas para desarrollar las gráficas
			FechaE = Fecha[Fi:Ff] # Fechas para desarrollar las gráficas

			# Se extrae la información para la validación
			ValR = TotalD[ValRow,Fi:Ff]
			ZVal = ZT[ValRow]

			# Se realiza la reconstrucción
			fTotalDE = fTotalD
			fTotalmaxDE = fTotalmaxD
			fTotalminDE = fTotalminD
			RecD, RecDH, Delta = self.RecData(fTotalHE,ZVal,True,fTotalDE,fTotalmaxDE,fTotalminDE,CurvasCC[CRow],Year)

			# Se grafican los datos horarios
			E_M = self.GraphH(RecDH,FechaHE,FechaHVE,Year,PathImg,True,ValRH,Name,Method)
			E_MM = self.GraphD(RecD,FechaE,FechaVE,Year,PathImg,True,ValR,Name,Method)
			return E_M
			print('End')
			print("--- Timestep: %s seg ---" % (time.time() - start_time))

		elif Method == 3:
			# Se buscan las Fechas en común
			Fi = FechaH.index(str(Year)+'-01-01-0000') # Año incial
			Ff = FechaH.index(str(Year+1)+'-01-01-0000') # Año incial
			FechaHVE = FechaHV[Fi:Ff] # Fechas para desarrollar las gráficas
			FechaHE = FechaH[Fi:Ff] # Fechas para desarrollar las gráficas
			fTotalHE = fTotalH
			ValRH = TotalH[ValRow,Fi:Ff]

			Fi = np.where(Fecha == (str(Year)+'-01-01'))[0] # Año incial
			Ff = np.where(Fecha == (str(Year+1)+'-01-01'))[0] # Año final

			FechaVE = FechaV[Fi:Ff] # Fechas para desarrollar las gráficas
			FechaE = Fecha[Fi:Ff] # Fechas para desarrollar las gráficas

			# Se escoge la curva
			if CRow == -2:
				Curva = CurvasSS
			elif CRow == -3:
				Curva = CurvasGG


			# Se extrae la información para la validación
			ValR = TotalD[ValRow,Fi:Ff]
			ZVal = ZT[ValRow]

			# Se realiza la reconstrucción
			fTotalDE = fTotalD
			fTotalmaxDE = fTotalmaxD
			fTotalminDE = fTotalminD
			RecD, RecDH, Delta = self.RecData(fTotalHE,ZVal,True,fTotalDE,fTotalmaxDE,fTotalminDE,Curva,Year)

			# Se grafican los datos horarios
			E_M = self.GraphH(RecDH,FechaHE,FechaHVE,Year,PathImg,True,ValRH,Name,Method)
			E_MM = self.GraphD(RecD,FechaE,FechaVE,Year,PathImg,True,ValR,Name,Method)
			return E_M
			print('End')
			print("--- Timestep: %s seg ---" % (time.time() - start_time))

	def Recon(self,fTotalH,fTotalD,fTotalmaxD,fTotalminD,TotalH,TotalD,Fecha,FechaV,FechaH,FechaHV,Z,ZT,\
		FechaC,Caso,CurvasCC,CurvasSS,CurvasGG,Hu,PathImg,Pathout,Name=''):
		'''
			DESCRIPTION:
		
		Con esta función se pretende desarrollar la reconstrucción de la
		información según los casos que se tengan para toda la serie de datos.
		_________________________________________________________________________

			INPUT:
		+ fTotalH, ...D, ...maxD, ...minD: Matrices de regresiones lineales.
		+ TotalD, ...maxD y ...minD: Matriz completa diaria.
		+ ZT: Vector de alturas totales.
		+ Z: Altura de reconstrucción.
		+ ValRow: Filas en donde se encuentran los datos de validación.
		+ Year: Año de validación.
		+ Method: Método que se usará para validar.
		+ CurvasCC, ...SS, ...GG: Curvas que se usarán para realizar la
								  reconstrucción diaria.
		+ CVRowC: Fila de la curva característica.
		+ CVRowG: Designa cual curva general se utilizará.
		_________________________________________________________________________
		
			OUTPUT:
		Este archivo
		- Rec
		'''
		# Se crea el documento 
		Nameout = Pathout + 'Serie_Reconstruida_Z('+str(Z) +').xlsx'
		W = xlsxwl.Workbook(Nameout)
		bold = W.add_format({'bold': True,'align': 'left'})
		boldc = W.add_format({'bold': True,'align': 'center'})
		date_format_str = 'yyyy/mm/dd-HHMM'
		date_format = W.add_format({'num_format': date_format_str,'align': 'left'})
		# Meses
		Mes = ['Enero','Febrero','Marzo','Abril','Mayo','Junio',\
		'Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'];
		RecHH = dict()



		# Se realiza la reconstrucción por años

		Ai = FechaC[0]
		Af = FechaC[1]
		kk = 0
		for An in range(Ai,Af+1):

			# Se extraen los datos de cada año.
			# Horarios:
			Fi = FechaH.index(str(An)+'-01-01-0000') # Año incial
			if An == Af:
				Ff = FechaH.index(str(An)+'-12-31-2300') # Año incial
				FechaHVE = FechaHV[Fi:Ff+1] # Fechas para desarrollar las gráficas
				FechaHE = FechaH[Fi:Ff+1] # Fechas para desarrollar las gráficas
				fTotalHE = fTotalH[Fi:Ff+1,:] # Pendientes e interceptos horarios
			else:
				Ff = FechaH.index(str(An+1)+'-01-01-0000') # Año incial
				FechaHVE = FechaHV[Fi:Ff] # Fechas para desarrollar las gráficas
				FechaHE = FechaH[Fi:Ff] # Fechas para desarrollar las gráficas
				fTotalHE = fTotalH[Fi:Ff,:] # Pendientes e interceptos horarios

			Fi = np.where(Fecha == (str(An)+'-01-01'))[0] # Año incial
			if An == Af:
				Ff = np.where(Fecha == (str(An)+'-12-31'))[0] # Año final
				fTotalDE = fTotalD[Fi:Ff+1,:]
				fTotalmaxDE = fTotalmaxD[Fi:Ff+1,:]
				fTotalminDE = fTotalminD[Fi:Ff+1,:]
				FechaVE = FechaV[Fi:Ff+1] # Fechas para desarrollar las gráficas
				FechaE = Fecha[Fi:Ff+1] # Fechas para desarrollar las gráficas
				# Se obtiene el caso de reconstrucción
				CasoA = Caso[Fi:Ff+1]
			else:
				Ff = np.where(Fecha == (str(An+1)+'-01-01'))[0] # Año final
				fTotalDE = fTotalD[Fi:Ff,:]
				fTotalmaxDE = fTotalmaxD[Fi:Ff,:]
				fTotalminDE = fTotalminD[Fi:Ff,:]
				FechaVE = FechaV[Fi:Ff] # Fechas para desarrollar las gráficas
				FechaE = Fecha[Fi:Ff] # Fechas para desarrollar las gráficas
				# Se obtiene el caso de reconstrucción
				CasoA = Caso[Fi:Ff]

			
			L = len(CasoA)
			S = sum(CasoA)
			
			if S == L: # En este caso se reconstruyen los datos horarios
				RecH = self.RecData(fTotalHE,Z,False)
				E_M = self.GraphH(RecH,FechaHE,FechaHVE,An,PathImg,False)
			elif S == (2*L): # Reconstrucción enteramente diaria
				RecD, RecH, Delta = self.RecData(fTotalHE,Z,True,fTotalDE,fTotalmaxDE,fTotalminDE,CurvasCC,An,True,FechaE,Hu)
				E_M = self.GraphH(RecH,FechaHE,FechaHVE,An,PathImg,False)
			else: # Reconstrucción combinada
				# Reconstrucción Horaria
				# Se realiza toda la reconstrucción horaria y después se cambian los puntos por la reconstrucción diaria
				RecH = self.RecData(fTotalHE,Z,False)
				q = np.where(CasoA == 2)[0]
				# Reconstrucción Diaria
				RecD, RecDH, Delta = self.RecData(fTotalHE,Z,True,fTotalDE,fTotalmaxDE,fTotalminDE,CurvasCC,An,True,FechaE,Hu)
				for P in q:
					RecH[P*24:(P*24)+24] = RecDH[P*24:(P*24)+24]
				E_M = self.GraphH(RecH,FechaHE,FechaHVE,An,PathImg,False)


			# Se realiza la escritura de la información
			worksheet = W.add_worksheet(str(An))

			# Se genera la plantilla 
			# Se escribe la información general
			worksheet.write(0,0, r'Proyecto ISAGEN-EIA',bold) # Titulo
			worksheet.write(2,0, r'Serie Reconstruida',bold) # Serie
			worksheet.write(3,0, r'Nombre:',bold) # Nombre
			worksheet.write(3,1, Name) # Nombre
			worksheet.write(4,0, r'Altitud:',bold) # Altura
			worksheet.write(4,1, str(Z)+'msnm') # Altura
			worksheet.write(5,0, r'Variable:',bold) # Variable
			worksheet.write(5,1, r'Temperatura') # Temperatura
			
			x = 7
			for i in Mes:# Ciclo para los meses
				worksheet.write(x,0, r'Mes:',boldc) # Mes
				worksheet.write(x,1, i) # Mes

				# Dias
				worksheet.write(x+1,0, 'Día/Hora',boldc) # Título
				for j in range(1,32):
					worksheet.write(x+1,j,j,boldc) # Días
				# Horas
				for g in range(24):
					if g < 10:
						worksheet.write(x+2+g,0,'0'+str(g)+':'+'00',boldc) # Horas
					else:
						worksheet.write(x+2+g,0,str(g)+':'+'00',boldc) # Horas
				worksheet.write(x+27,0,'Mín',bold)
				worksheet.write(x+28,0,'Máx',bold)
				worksheet.write(x+29,0,'Promedio',bold)
				x += 31
			x = 9
			xx = 0
			for i in range(1,13): # Meses
				FFi = date(An,i,1)
				if i == 12:
					FFf = date(An+1,1,1)
					DF = FFf-FFi
				else:
					FFf = date(An,i+1,1)
					DF = FFf-FFi

				for j in range(DF.days):
					for g in range(24):
						
						worksheet.write(x+g,j+1,RecH[xx]) # Horas
						xx += 1
					# Datos adicionales
					worksheet.write(x+27-2,j+1,np.min(RecH[xx-24:xx])) # Horas
					worksheet.write(x+28-2,j+1,np.max(RecH[xx-24:xx])) # Horas
					worksheet.write(x+29-2,j+1,np.mean(RecH[xx-24:xx])) # Horas
				x += 31

			RecHH[kk] = RecH
			kk += 1
		
		# Serie de datos
		x = 1
		worksheet = W.add_worksheet('Horarias_Completa')
		worksheet.write(0,0, r'Fecha',bold) # Fechas
		worksheet.write(0,1, r'Temperatura',bold) # Temperatura
		
		for i in range(len(RecHH)):
			for j in RecHH[i]:
				worksheet.write(x,0, FechaHV[x-1],date_format) # Fechas
				worksheet.write(x,1, j) # Fechas
				x += 1

		# Se copian los casos de cada día
		worksheet = W.add_worksheet('Casos')
		worksheet.write(0,0, r'Fecha/Día',bold) # Fechas
		x=1
		xx = 0
		for An in range(Ai,Af+1):
			Astr = str(An)
			for i in range(1,13):
				worksheet.write(x,0,Mes[i-1]+'-' + Astr[2:],bold) # Fechas
				FFi = date(An,i,1)
				if i == 12:
					FFf = date(An+1,1,1)
					DF = FFf-FFi
				else:
					FFf = date(An,i+1,1)
					DF = FFf-FFi

				for j in range(DF.days):
					if An == Ai and i == 1:
						worksheet.write(x-1,j+1,j+1,boldc) # Fechas
					worksheet.write(x,j+1,Caso[xx]) # Fechas
					xx += 1
				x += 1

		return RecHH
