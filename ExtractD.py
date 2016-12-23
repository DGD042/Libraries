# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#						Coded by Daniel González Duque
#						    Last revised 26/02/2016
#______________________________________________________________________________
#______________________________________________________________________________

#______________________________________________________________________________
#
# DESCRIPCIÓN DE LA CLASE:
# 	En esta clase se incluyen las rutinas para tratar datos de diferentes
#	fuentes de información, extraer los datos, completarlos y generar 
#	cambios de escala a la información. Además esta librería también genera
#	archivos en diferentes formatos para el posterior uso de la información.
#
#	Esta libreria es de uso libre y puede ser modificada a su gusto, si tienen
#	algún problema se pueden comunicar con el programador al correo:
# 	dagonzalezdu@unal.edu.co
#______________________________________________________________________________

import numpy as np
import sys
import csv
import xlrd # Para poder abrir archivos de Excel
import xlsxwriter as xlsxwl
import warnings

# Se importan los paquetes para manejo de fechas
from datetime import date, datetime, timedelta

from UtilitiesDGD import UtilitiesDGD
utl = UtilitiesDGD()


class ExtractD:

	def __init__(self):

		'''
			DESCRIPTION:

		Este es el constructor por defecto, no realiza ninguna acción.
		'''

	def ED(self,Tot,flagD=True,sheet=0,Header=True,ni=0,n=1,deli=';',rrr=2):
		'''
			DESCRIPTION:
		
		Con esta función se pretende extraer la información de un archivo, por 
		defecto siempre extrae la primera columna y luego la segunda, si es
		necesario extraer otra columna adicional se debe incluir un vector en n.

		Máximo se extraerán 4 columnas y los datos que se extraen deben estar en
		flias y las variables en columnas.
		_________________________________________________________________________

			INPUT:
		+ Tot: Es la ruta completa del archivo que se va a abrir.
		+ flagD: Para ver que tipo de archivo se abrirá.
				True: Para archivos de Excel.
				False: Para el resto.
		+ sheet: Número de la hoja del documento de excel
		+ Header: Se pregunta si el archivo tiene encabezado.
		+ ni: es la columna inicial en donde se tomarán los datos de tiempo.
		+ n: es la columna o columnas que se van a extraer.
		+ deli: Delimitador que separa las variables en caso que no sean de Excel.
		+ rrr: Fila en donde comienzan los valores
		_________________________________________________________________________
		
			OUTPUT:
		- Tiempo: Es la primera variable que se extrae del documento en formato txt.
		- Hora: Valores de hora, solo si len(ni) > 1.
		- V1: Variable 1 que se extrae como float
		- V2: Variable 2 que se extrae como float, depende de la cantidad de columnas.
		- V3: Variable 3 que se extrae como float, depende de la cantidad de columnas.
		- V4: Variable 4 que se extrae como float, depende de la cantidad de columnas.
		
		'''
		# Se mira cuantas columnas se va a extraer como m
		try:
			mi = len(ni)
		except:
			mi = 1 
		try:
			m = len(n)
		except:
			m = 1 

		# Contador de filas
		if Header == True:
			rr = rrr-1
			Head0 = ["" for k in range(1)] # Variable para los encabezados
			Head1 = ["" for k in range(1)] # Variable para los encabezados
			Head2 = ["" for k in range(1)] # Variable para los encabezados
			Head3 = ["" for k in range(1)] # Variable para los encabezados
			Head4 = ["" for k in range(1)] # Variable para los encabezados
		else:
			rr = 0
			rrr = 0

		# Se inicializa la variable de tiempo
		if mi >= 1 and mi <= 2:
			Tiempo = ["" for k in range(1)]
			Hora = ["" for k in range(1)]
		else:
			utl.ExitError('ED','ExtractD','No se pueden incluir más de dos columans con texto.')

		if flagD == True:

			# Se inicializan las variables que se van a extraer
			if mi == 1:
				Tiempo = []
			elif mi==2:
				Tiempo = []
				Hora = []

			if m >= 1 and m<=4:
				V1 = []
				V2 = []
				V3 = []
				V4 = []
			else:
				utl.ExitError('ED','ExtractD',\
					'No se pueden extraer tantas variables, por favor reduzca el número de variables')

			# --------------------------------------
			# Se abre el archivo de Excel
			# --------------------------------------
			book = xlrd.open_workbook(Tot)
			# Se carga la página en donde se encuentra el información
			S = book.sheet_by_index(sheet)

			# Se extren los encabezados
			if Header == True:
				if mi == 1:
					Head0 = S.cell(rrr-1,ni).value
				else:
					Head0 = S.cell(rrr-1,ni[0]).value # Se toman los encabezados
					Head01 = S.cell(rrr-1,ni[1]).value  # Se toman los encabezados
				if m == 1:
					Head1 = S.cell(rrr-1,n).value
				else:
					Head1 = S.cell(rrr-1,n[0]).value
					try:
						Head2 = S.cell(rrr-1,n[1]).value
					except:
						Head2 = np.nan
					try:
						Head3 = S.cell(rrr-1,n[2]).value
					except:
						Head3 = np.nan
					try:
						Head4 = S.cell(rrr-1,n[3]).value
					except:
						Head4 = np.nan


			# Se genera un ciclo para extraer toda la información
			k = 1
			xx = rrr # Contador de las filas
			while k == 1:
				if mi == 1:
					if m == 1:
						try:
							# Se verifica que se esté sacando una variable correcta
							a = S.cell(xx,ni).value # Tiempo, debería estar completo
							if a != '':
								# Se carga el Tiempo
								Tiempo.append(S.cell(xx,ni).value)
								# Se carga la variable
								V1.append(S.cell(xx,n).value)
								if V1[-1] == '':
									V1[-1] = np.nan
						except IndexError:
							k = 2
						xx += 1

			if Header == True:
			# Se guarda la información
				if mi == 1:
					if m == 1:
						Head = [Head0,Head1]
						return Tiempo, V1, Head
			

		elif flagD == False:

			# Se inicializan las variables dependiendo del número de filas 
			# que se tengan

			if m >= 1 and m<=4:
				V1 = ["" for k in range(1)]
				V2 = ["" for k in range(1)]
				V3 = ["" for k in range(1)]
				V4 = ["" for k in range(1)]
			else:
				utl.ExitError('ED','ExtractD',\
					'No se pueden extraer tantas variables, por favor reduzca el número de variables')

			# Se abre el archivo que se llamó
			f = open(Tot)

			# Se inicializan las variables
			ff = csv.reader(f, dialect='excel', delimiter= deli)

			r = 0 # Contador de filas
			# Ciclo para todas las filas
			for row in ff:
				if Header == True:
					if (rr == rrr-1 and r == rrr-1):
						if mi ==1:
							Head0 = row[ni] # Se toman los encabezados
						else:
							Head0 = row[ni[0]] # Se toman los encabezados
							Head01 = row[ni[1]] # Se toman los encabezados
						if m == 1:
							Head1 = row[n] # Se toman los encabezados
						else:
							Head1 = row[n[0]] # Se toman los encabezados
							try:
								Head2 = row[n[1]] # Se toman los encabezados
							except:
								Head2 = float('nan')
							try:
								Head3 = row[n[2]] # Se toman los encabezados
							except:
								Head3 = float('nan')
							try:
								Head4 = row[n[3]] # Se toman los encabezados
							except:
								Head4 = float('nan')

				if r == rrr:
					if mi ==1:
						Tiempo[0] = row[ni]
					else:
						Tiempo[0] = row[ni[0]]
						Hora[0] = row[ni[1]]

					if m == 1:
						try:
							V1[0] = float(row[n])
						except:
							V1[0] = float('nan')
					else:
						try:
							V1[0] = float(row[n[0]])
						except:
							V1[0] = float('nan')
						try:
							V2[0] = float(row[n[1]])
						except:
							V2[0] = float('nan')
						try:
							V3[0] = float(row[n[2]])
						except:
							V3[0] = float('nan')
						try:
							V4[0] = float(row[n[3]])
						except:
							V4[0] = float('nan')
				
				elif r > rr:
					
					if mi ==1:
						Tiempo.append(row[ni])
					else:
						Tiempo.append(row[ni[0]])
						Hora.append(row[ni[1]])

					if m == 1:
						try:
							
							V1.append(float(row[n]))
						except:
							V1.append(float('nan'))
					else:
						try:
							V1.append(float(row[n[0]]))
						except:
							V1.append(float('nan'))
						try:
							V2.append(float(row[n[1]]))
						except:
							V2.append(float('nan'))
						try:
							V3.append(float(row[n[2]]))
						except:
							V3.append(float('nan'))
						try:
							V4.append(float(row[n[3]]))
						except:
							V4.append(float('nan'))
				r += 1
			f.close()
			if Header == True:
				if mi == 1:
					if m == 1: 
						Head = [Head0,Head1]
						return Tiempo, V1, Head
					elif m == 2:
						Head = [Head0,Head1,Head2]
						return Tiempo, V1, V2, Head
					elif m == 3:
						Head = [Head0,Head1,Head2,Head3]
						return Tiempo, V1, V2, V3, Head
					elif m == 4:
						Head = [Head0,Head1,Head2,Head3,Head4]
						return Tiempo, V1, V2, V3, V4, Head
				elif mi == 2:
					if m == 1: 
						Head = [Head0,Head01,Head1]
						return Tiempo, Hora, V1, Head
					elif m == 2:
						Head = [Head0,Head01,Head1,Head2]
						return Tiempo, Hora, V1, V2, Head
					elif m == 3:
						Head = [Head0,Head01,Head1,Head2,Head3]
						return Tiempo, Hora, V1, V2, V3, Head
					elif m == 4:
						Head = [Head0,Head01,Head1,Head2,Head3,Head4]
						return Tiempo, Hora, V1, V2, V3, V4, Head
			else:
				if mi == 1:
					if m == 1: 
						return Tiempo, V1
					elif m == 2:
						return Tiempo, V1, V2
					elif m == 3:
						return Tiempo, V1, V2, V3
					elif m == 4:
						return Tiempo, V1, V2, V3, V4
				elif mi == 2:
					if m == 1: 
						return Tiempo, Hora, V1
					elif m == 2:
						return Tiempo, Hora, V1, V2
					elif m == 3:
						return Tiempo, Hora, V1, V2, V3
					elif m == 4:
						return Tiempo, Hora, V1, V2, V3, V4

	def EDEIA(self,Tot,Ai,Af):
		'''
			DESCRIPTION:
		
		Con esta función se pretende extraer la información horaria de los
		sensores brindados por la Universidad EIA en el sector del PNN Los 
		Nevados.

		Los sensores Hobo fueron instalados por el grupo de investigación a cargo
		del profesor Daniel Ruiz Carrascal y son de uso libre para proyectos 
		designados al estudio de la climatología de alta montaña.

		Los archivos deben tener la organización específica de la plantilla de la
		EIA, de otra manera la función no arrojará los resultados correctos.
		_________________________________________________________________________

			INPUT:
		+ Tot: Es la ruta completa del archivo que se va a abrir.
		+ Ai: Año incial de descarga no puede ser menor al 2008.
		+ Af: Año final de descarga no puede ser mayor al 2015, se cambiará cuando
			  se tengan los nuevos años.
		_________________________________________________________________________
		
			OUTPUT:
		- Fecha: Vector con todas las fechas descargadas en formato 'Año/Mes/Dia-HH:MM'.
		- V1: Variable que se extrae como float.
		
		
		'''
		# ------------------------------
		# Inicializador de variables
		# ------------------------------

		Fecha = ["" for k in range(1)]
		V1 = ["" for k in range(1)]

		d = timedelta(days=1) # Delta de días

		# ------------------------------
		# Determinación de hojas
		# ------------------------------

		# Se miran la hojas que se van a descargar con la variable Ind
		if Ai == 2008:
			Ind = 3
		elif Ai > 2008:
			Ind = (Ai-2008)+3
		elif Ai <2008 or Af > 2015:
			utl.ExitError('ExtractD','EDEIA','There is no data before 2008, and after 2016.')
		

		# ------------------------------
		# Extracción de datos
		# ------------------------------

		# Se Abre el libro determinado
		book = xlrd.open_workbook(Tot)
		
		# Ciclo para extraer los datos
		for i in range(Ai,Af+1): # Años
			#print(i)
			# Escoge la hoja en donde se encuentran los datos
			S = book.sheet_by_index(Ind)
			x = 12 # filas donde se encuentran los datos
			
			for j in range(1,13): # Meses
				#print(j)
				y = 1 # Columnas donde se encuentran los datos
				# Contador de meses
				Fi = date(i,j,1)
				if j == 12:
					Ff = date(i+1,1,1)
				else:
					Ff = date(i,j+1,1)
				DF = Ff-Fi
				
				for g in range(1,DF.days+1): # Dias
					#print('_'+str(g))
					FR = date(i,j,g)	
					FRR = FR.strftime('%Y'+'/'+'%m'+'/'+'%d')
					xx = x
					for H in range(0,24): # Horas
						if xx == 12 and i ==Ai and g==1:
							# Se toma la fecha completa
							if H < 10:
								Fecha[0] = FRR + '-0' + str(H) + '00'
							else:
								Fecha[0] = FRR + '-' + str(H) + '00'
							# Se extraen los datos
							try:
								V1[0] = float(S.cell(xx,y).value)
							except:
								V1[0] = float('nan')
						else:
							# Se toma la fecha completa
							if H < 10:
								Fecha.append(FRR + '-0' + str(H) + '00')
							else:
								Fecha.append(FRR + '-' + str(H) + '00')
							# Se extraen los datos
							try:
								V1.append(float(S.cell(xx,y).value))
							except:
								V1.append(float('nan'))
						xx += 1
					y += 1
					
				x = x+31

			Ind +=1 # Se cambia de página

		return Fecha, V1

	def EDEST(self,Tot,Aii,Ai,Af,flagMa=True):
		'''
			DESCRIPTION:
		
		Con esta función se pretende extraer la información diaria de las
		estaciones Cenicafé y Brisas dadas por Universidad EIA. El formato debe 
		ser el entregado por Daniel Ruiz Carrascal ya que depende de las hojas 
		del documento de excel.
		_________________________________________________________________________

			INPUT:
		+ Tot: Es la ruta completa del archivo que se va a abrir.
		+ Aii: Año inicial del documento.
		+ Ai: Año incial de descarga no puede ser menor al 2008.
		+ Af: Año final de descarga no puede ser mayor al 2015, se cambiará cuando
			  se tengan los nuevos años.
		+ flagMa: Condicional para saber si se obtienen los datos máximos y mínimos
				  True: Obtiene los datos máximos y mínimos díarios.
				  False: Solamente obtienen los datos diarios
		_________________________________________________________________________
		
			OUTPUT:
		- Fecha: Vector con todas las fechas descargadas en formato 'Año/Mes/Dia'.
		- V1: Variable que se extrae como float.
		- Vmax: Variable para los máximos.
		- Vmin: Variable para los mínimos.
		'''
		# ------------------------------
		# Inicializador de variables
		# ------------------------------

		Fecha = ["" for k in range(1)]
		V1 = ["" for k in range(1)]
		Vmax = ["" for k in range(1)]
		Vmin = ["" for k in range(1)]

		d = timedelta(days=1) # Delta de días

		# ------------------------------
		# Determinación de hojas
		# ------------------------------
		Ind = 6 # Medios darios
		# Se miran la hojas que se van a descargar con la variable Ind
		
		Indmax = 4 # Máximos
		Indmin = 2 # Mínimos
		
		# ------------------------------
		# Extracción de datos
		# ------------------------------

		# Se Abre el libro determinado
		book = xlrd.open_workbook(Tot)
		x = 9 # filas donde se encuentran los datos
		# Ciclo para extraer los datos
		for i in range(Aii,Af+1): # Años
			if i >= Ai:
				#print(i)
				# Escoge la hoja en donde se encuentran los datos
				S = book.sheet_by_index(Ind)
				Smax = book.sheet_by_index(Indmax)
				Smin = book.sheet_by_index(Indmin)
				y = 1 # Columnas
				for j in range(1,13):

					# Contador de meses
					Fi = date(i,j,1)
					if j == 12:
						Ff = date(i+1,1,1)
					else:
						Ff = date(i,j+1,1)
					DF = Ff-Fi
					xx = x
					for g in range(1,DF.days+1): # Dias
						#print('_'+str(g))
						FR = date(i,j,g)	
						FRR = FR.strftime('%Y'+'/'+'%m'+'/'+'%d')
						

						if g == 1 and j == 1 and i == Ai:
							Fecha[0] = FRR
							try:
								V1[0] = float(S.cell(xx,y).value)
							except:
								V1[0] = float('nan')
							try:
								Vmax[0] = float(Smax.cell(xx,y).value)
							except:
								Vmax[0] = float('nan')
							try:
								Vmin[0] = float(Smin.cell(xx,y).value)
							except:
								Vmin[0] = float('nan')
						else:
							Fecha.append(FRR)
							try:
								V1.append(float(S.cell(xx,y).value))
							except:
								V1.append(float('nan'))
							try:
								Vmax.append(float(Smax.cell(xx,y).value))
							except:
								Vmax.append(float('nan'))
							try:
								Vmin.append(float(Smin.cell(xx,y).value))
							except:
								Vmin.append(float('nan'))

						xx += 1
					y += 1
			x = x + 36


		if flagMa == True:
			return Fecha, V1,Vmax,Vmin
		else:
			return Fecha, V1

	def CompD(self,Fecha,V1,flagH=False,Hora='',flagM=False,Min='',dtm=1):
		'''
			DESCRIPTION:
		
		Con esta función se pretende completar la información faltante de un
		docuemnto que no tenga todas las fechas activas, el código llena años
		enteros utilizando escribiendo como 'nan' a los datos faltantes.
		_________________________________________________________________________

			INPUT:
		+ Fecha: Fecha de los datos organizada como 'año/mes/dia' los '/' pueden
				 ser substituidos por cualquier caracter. 
				 Debe ser un vector string.
		+ V1: Variable que se desea rellenar. 
		+ flagH: Para ver si se llenan los datos horarios.
				True: Para llenar datos horarios.
				False: Para llenar datos diarios.
		+ Hora: Es el vector de horas en 24H, debe ser un vector string con dos 
				dígitos, es decir, con un '0' si son números de 1 a 9.
		+ flagM: Para ver si se llenan los datos minutales.
				True: Para llenar datos minutales.
				False: Para llenar datos horarios o diarios.
		+ Min: Son los minutos en caso de ser necesitados, deben tener el mismo
			   formato que las horas.
		+ dtm: delta de tiempo de minutos, por defecto esta cada minuto
		_________________________________________________________________________
		
			OUTPUT:
		- FechaC: Vector con las fechas completas de la forma 'año/mes/dia' si
				  si tiene Horas y minutos quedará como 'año/mes/dia-HH:MM'
		- V1C: Variable con los datos completados.
		- FechaN: Vector de fechas con las variables con el número designado
				  por Python.
		- FechaD: Vector de fechas originales con los datos horarios.
		'''
		# Se inicializan las Fechas
		FechaC = ["" for k in range(1)] # Vector de comparación
		FechaN = ["" for k in range(1)] # Fecha en vector numérico
		FechaD = ["" for k in range(len(Fecha))]
		
		Sep = Fecha[0][4] # Separador de la Fecha

		# Se toma una Fecha completa
		if flagH == True:
			if flagM == True:
				if dtm == 5:
					for i,j in enumerate(Fecha): # Este caso se debe cambiar si el dtm es diferente!

						if (int(Min[i][1]) >=3 and int(Min[i][1])<=6):
							FechaD[i] = (j +'-'+ Hora[i]+Min[i][0]+'5')
						elif int(Min[i][1]) <3:
							FechaD[i] = (j +'-'+ Hora[i]+Min[i][0]+'0')
						elif int(Min[i][1]) >6:
							if int(Min[i][0]) == 5:
								if int(Hora[i]) == 23:
									a = date(int(j[0:4]),int(j[5:7]),int(j[8:10]))
									D = timedelta(days = 1)
									aa = a+D # Se suma in día adicional a la fecha
									B = aa.strftime('%Y'+Sep+'%m'+Sep+'%d')
									FechaD[i] = (B + '-' + '00' + '00')

								else:
									HH = int(Hora[i]) + 1
									if HH < 10:
										FechaD[i] = (j +'-'+ '0' + str(HH) +'00')
									else:
										FechaD[i] = (j +'-' + str(HH) +'00')
							else:
								MM = int(Min[i][0]) + 1
								FechaD[i] = (j +'-'+ Hora[i]+str(MM)+'0')
				elif dtm == 1:
					for i,j in enumerate(Fecha):
						FechaD[i] = (j +'-'+ Hora[i]+Min[i])
				elif dtm == 30:
					for i,j in enumerate(Fecha):
						FechaD[i] = (j +'-'+ Hora[i]+Min[i])
				elif dtm == 15:
					for i,j in enumerate(Fecha):
						FechaD[i] = (j +'-'+ Hora[i]+Min[i])
			elif flagM == False:
				FechaD = [(j +'-'+ Hora[i]+'00') for i,j in enumerate(Fecha)]
		elif flagH == False:
			FechaD = Fecha

		# Se toma la fecha de inicio y la fecha final para hacer el vector completo de fechas
		yeari = Fecha[0][0:4]
		yearf = Fecha[-1][0:4]

		rr = 0
		# Se realiza el vector completo de fechas
		for result in utl.perdelta(date(int(yeari), 1, 1), date(int(yearf)+1, 1, 1), timedelta(days=1)):
			#print(result.strftime('%Y%m%d'))
			FR = result.strftime('%Y'+Sep+'%m'+Sep+'%d')			
			if flagH == True: # Condicional para horario
				if flagM == True: # Condicional para minutal
					for i in range(0,24): # Ciclo para las horas
						for jj in range(0,60,dtm): # Ciclo para los minutos
							if rr == 0:
								FechaN[0] = result
								if i < 10:
									if jj < 10:
										FechaC[rr] = FR + '-0' +str(i)+'0'+str(jj)
									else:
										FechaC[rr] = FR + '-0' +str(i)+str(jj)
								else:
									if jj < 10:
										FechaC[rr] = FR + '-' +str(i)+'0'+str(jj)
									else:
										FechaC[rr] = FR + '-' +str(i)+str(jj)
							else:
								FechaN.append(result)
								if i < 10:
									if jj < 10:
										FechaC.append(FR + '-0' +str(i)+'0'+str(jj))
									else:
										FechaC.append(FR + '-0' +str(i)+str(jj))
								else:
									if jj < 10:
										FechaC.append(FR + '-' +str(i)+'0'+str(jj))
									else:
										FechaC.append(FR + '-' +str(i)+str(jj))
							rr +=1
				elif flagM == False:
					for i in range(0,24):
						if rr == 0:
							FechaN[0] = result
							if i < 10:
								FechaC[rr] = FR + '-0' +str(i)+'00'								
							else:
								FechaC[rr] = FR + '-' +str(i)+'00'
						else:
							FechaN.append(result)
							if i < 10:
								FechaC.append(FR + '-0' +str(i)+'00')
							else:
								FechaC.append(FR + '-' +str(i)+'00')
						rr +=1
			elif flagH == False:
				if rr == 0:
					FechaC[rr] = FR
					FechaN[0] = result
				else:
					FechaC.append(FR)
					FechaN.append(result)
					
				rr +=1
				

		#Se analizan las fechas reales vs las fechas del documento
		V1C = np.empty(len(FechaC))*np.nan # Vector real de valores


		FechaC = np.array(FechaC)
		FechaD = np.array(FechaD)
		V1 = np.array(V1)

		x = FechaC.searchsorted(FechaD)

		V1C[x] = V1


		# # Se llena el vector con los datos reales
		# for i,j in enumerate(FechaD):
		# 	x = FechaC.index(j)
		# 	V1C[x] = V1[i]

		return FechaC, V1C, FechaN,FechaD

	def CompDC(self,FechaC,V1C,ai,af,flagH=False,flagLH=False,dt=60):
		'''
			DESCRIPTION:
		
		Con esta función se pretende extrear la información de unos años 
		específicos y donde no se tenga la información simplemente se completarán
		los datos. Esta función extraer años enteros, si es necesario una resolución
		más exacta se deben hacer modificaciones manuales a la información.
		_________________________________________________________________________

			INPUT:
		+ FechaC: Fecha de los datos organizada como 'año/mes/dia' los '/' pueden
				 ser substituidos por cualquier caracter. 
				 Debe ser un vector string.
		+ V1C: Variable que se desea extraer. 
		+ ai: Año inicial que se desea extraer.
		+ af: Año final al que se desea extraer
		+ flagH: Para ver si se extraen los datos horarios.
				 True: Para extraer datos horarios.
				 False: Para extraer datos diarios.
		+ flagLH: Para extraer datos menores al horario.
		+ dt: Delta de tiempo en minutos.
		_________________________________________________________________________
		
			OUTPUT:
		- FechaC2: Vector con las fechas completas de la forma 'año/mes/dia' si
				  si tiene Horas y minutos quedará como 'año/mes/dia-HH:MM'
		- FechaN2: Vector de fechas con las variables con el número designado
				  por Python.
		- V1C2: Variable con los datos completados.
		
		'''
		
		# -------------------------------------------
		# Inicialización de variables
		# -------------------------------------------
		FechaC2 = ["" for k in range(1)]
		V1C2 = ["" for k in range(1)]
		FechaN2 = ["" for k in range(1)]

		Sep = FechaC[0][4] # Separador de la Fecha
		rr = 0

		# -------------------------------------------
		# Vector de fechas
		# -------------------------------------------

		for result in utl.perdelta(date(int(ai), 1, 1), date(int(af)+1, 1, 1), timedelta(days=1)):
			FR = result.strftime('%Y'+Sep+'%m'+Sep+'%d') # Fecha
			if flagH == True:
				if flagLH == True:
					for i in range(0,24):
						for j in range(0,60,dt):
							if rr == 0:
								FechaN2[0] = result
								if i < 10:
									if j < 10:
										FechaC2[rr] = FR + '-0' +str(i)+'0'+str(j)
									else:
										FechaC2[rr] = FR + '-0' +str(i)+str(j)
								else:
									if j < 10:
										FechaC2[rr] = FR + '-' +str(i)+'0'+str(j)
									else:
										FechaC2[rr] = FR + '-' +str(i)+str(j)
							else:
								FechaN2.append(result)
								if i < 10:
									if j < 10:
										FechaC2.append(FR + '-0' +str(i)+'0'+str(j))
									else:
										FechaC2.append(FR + '-0' +str(i)+str(j))
								else:
									if j < 10:
										FechaC2.append(FR + '-' +str(i)+'0'+str(j))
									else:
										FechaC2.append(FR + '-' +str(i)+str(j))
							rr += 1 # Se suman las filas
				elif flagLH == False:
					for i in range(0,24):
						if rr == 0:
							FechaN2[0] = result
							if i < 10:
								FechaC2[rr] = FR + '-0' +str(i)+'00'
							else:
								FechaC2[rr] = FR + '-' +str(i)+'00'
						else:
							FechaN2.append(result)
							if i < 10:
								FechaC2.append(FR + '-0' +str(i)+'00')
							else:
								FechaC2.append(FR + '-' +str(i)+'00')
						rr += 1 # Se suman las filas
			elif flagH == False:
				if rr == 0:
					FechaN2[0] = result
					FechaC2[rr] = FR
				else:
					FechaN2.append(result)
					FechaC2.append(FR)
				rr += 1
		# -------------------------------------------
		# Se realiza la extracción
		# -------------------------------------------
		#Se analizan las fechas reales vs las fechas del documento
		

		# V1C2 = np.empty(len(FechaC2))*np.nan # Vector real de valores
		# FechaC = np.array(FechaC)
		# FechaC2 = np.array(FechaC2)
		# V1C = np.array(V1C)

		# yeari = FechaC[0][0:4] # Separador de la Fecha
		# if int(yeari) < int(ai):
		# 	x = FechaC.searchsorted(FechaC2)
		# 	V1C2[x] = V1C
		# else:
		# 	x = FechaC2.searchsorted(FechaC)
		# 	print(x)
		# 	V1C2[x] = V1C


		V1C2 = [float('nan') for k in FechaC2] # Vector real de valores
		yeari = FechaC[0][0:4] # Separador de la Fecha
		x = 0 # Contador para las fechas del archivo
		if int(yeari) < int(ai):
			for i,j in enumerate(FechaC):

				if len(FechaC2) == x:
					xx = x
				else:
					if j == FechaC2[x]:	
						#print(j,FechaC2[x])
						V1C2[x] = V1C[i]
						x +=1
		else:
			for i,j in enumerate(FechaC2):

				if len(FechaC) == x:
					xx = x
				else:
					if j == FechaC[x]:							
						V1C2[i] = V1C[x]
						x +=1

		return FechaC2,FechaN2,V1C2

	def Ca_E(self,FechaC,V1C,dt=24,escala=1,op='mean',flagMa=False,flagDF=False):
		'''
			DESCRIPTION:
		
		Con esta función se pretende cambiar de escala temporal los datos,
		agregándolos a diferentes escalas temporales, se deben insertar series
		completas de tiempo.

		Los datos faltantes deben estar como NaN.
		_________________________________________________________________________

			INPUT:
		+ FechaC: Fecha de los datos organizada como 'año/mes/dia - HHMM' 
				  los '/' pueden ser substituidos por cualquier caracter. 
				  Debe ser un vector string y debe tener años enteros.
		+ V1C: Variable que se desea cambiar de escala temporal. 
		+ dt: Delta de tiempo para realizar la agregación, depende de la naturaleza
			  de los datos.
			  Si se necesitan datos mensuales, el valor del dt debe ser 1.
		+ escala: Escala a la cual se quieren pasar los datos:
				0: de minutal o horario.
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
		_________________________________________________________________________
		
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
			utl.ExitError('ExtractD','Ca_E','Todavía no se han programado estas escalas')

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


		# Los años se toman para generar el output de FechasEs
		if escala == 0 or escala == 1: # Para datos horarios o diarios
			for result in utl.perdelta(date(int(yeari), 1, 1), date(int(yearf)+1, 1, 1), timedelta(days=1)):
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
			if escala == 0 or escala == 1: 
				# Ciclo para realizar el agregamiento de los datos
				for i in range(0,len(V1C),dt): 
					dtt = dtt + dt # Se aumenta el número de filas en el contador
					if i == 0:
						q = np.isnan(V1C[i:dtt])
						qq = sum(q)
						if qq > dt/2:
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
						if qq > dt/2:
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
			elif escala == 2: # Agragamiento mensual
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
							q = ~np.isnan(V1C[d:dtt])
							qq = sum(q)
							NNF.append(qq/len(V1C[d:dtt]))
							NF.append(1-NNF[-1])
							if qq > DF.days/2:
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
							q = ~np.isnan(V1C[d:dtt])
							qq = sum(q)
							NNF.append(qq/len(V1C[d:dtt]))
							NF.append(1-NNF[-1])
							if qq > DF.days/2:
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
			if escala == 0 or escala == 1: 
				# Ciclo para realizar el agregamiento de los datos
				for i in range(0,len(V1C),dt): 
					dtt = dtt + dt # Se aumenta el número de filas en el contador
					if i == 0:
						q = np.isnan(V1C[i:dtt])
						qq = sum(q)
						
						if qq > dt/2:
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
						if qq > dt/2:
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
							if qq > DF.days/2:
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
							if qq > DF.days/2:
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
				return FechaEs, FechaNN, VE, VEMax, VEMin, NF,NNF
			else:
				return FechaEs, FechaNN, VE, VEMax, VEMin
		elif flagMa == False:
			if flagDF:
				return FechaEs, FechaNN, VE,NF,NNF
			else:
				return FechaEs, FechaNN, VE

	def EDH(self,Tot,flagD=True,sheet=1,Header=True,n=2):

		'''
			DESCRIPTION:
		
		Con esta función se pretende extraer la información de un archivo con
		información horaria y cualquier fecha, y después organizarlo.
		_________________________________________________________________________

			INPUT:
		+ Tot: Es la ruta completa del archivo que se va a abrir.
		+ flagD: Para ver que tipo de archivo se abrirá.
				True: Para archivos de Excel.
				False: Para el resto.
		+ sheet: Número de la hoja del documento de excel
		+ Header: Se pregunta si el archivo tiene encabezado.
		+ n: es la columan en donde se encuentran los datos que se van a extraer.
		_________________________________________________________________________
		
			OUTPUT:
		- FechaA: Fechas del archivo.
		- HoraA: Extrae la hora si el archivo la contiene.
		- Var: Variable que se va a extraer del archivo.
		- FechaC: Fechas del archivo completas de la forma "Fecha-Hora".
		- FechaR: Fechas reales de la forma "Fecha-Hora".
		- ValR: Variable del archivo con las fechas reales completas.
		- FechaD: Fechas reales en formato fecha de Python.
		'''

		# Contador de filas
		if Header == True:
			r = 1 
		else:
			r = 0

		
		p = 0 # Activador de filas para excel


		# Se inicializan las variables
		FechaA = ["" for k in range(1)]
		HoraA = ["" for k in range(1)]
		Val = ["" for k in range(1)]
		
		FechaR = ["" for k in range(1)]

		# Se abren los archivos dependiendo de la extensión
		if flagD == False:
			
			# Abre el libro determinado
			book = xlrd.open_workbook(Tot)
			# Escoge la hoja en donde se encuentran los datos
			S = book.sheet_by_index(0)

			while p == 0:
				
				# Se mira si tiene encabezado
				if Header == True:
					if r == 1:
						# Se obtiene la fecha
						FechaA[0] = str(int(S.cell(r,0).value))
						
						# Se obtiene la hora
						HH = S.cell(r,1).value[0:2] # Estos valores se deben 
						MM = S.cell(r,1).value[3:5] # cambiar dependiendo del archivo
						# Se mira si es a.m. o p.m.
						AM = S.cell(r,1).value[9]
						if AM == 'p':
							HH = str(int(HH) + 12)

						if int(MM) >= 1:

							if int(HH) <9:
								HH = '0'+str(int(HH)+1)
								MM = '00'
							elif int(HH) == 23:
								HH = '00'
								MM = '00'
								
								yeari = FechaA[0][0:4]
								monthi = FechaA[0][4:6]
								dayi = FechaA[0][6:8]
								
								delta = timedelta(days=1)
								D = date(int(yeari), int(monthi), int(dayi))

								DD = D + delta
								FR = DD.strftime('%Y%m%d')

								FechaA[0] = FR
							elif int(HH) == 24:
								HH = '13'
								MM = '00'
							elif int(HH) == 12:
								HH = '01'
								MM = '00'
							elif int(HH) == 11:
								HH = '24'
								MM = '00'	
							else:
								HH = str(int(HH)+1)
								MM = '00'


						HoraA[0] = str(HH)+str(MM)

						# Se obtiene la variable
						try:
							Val[0] = float(S.cell(r,2).value)
						except:
							Val[0] = float('nan')
						
					else:

						# Se incluyen los siguientes datos hasta que no haya más
						# Se mira si se tiene la última fecha sino se sale del programa
						try:
							FechaA.append(str(int(S.cell(r,0).value)))
						except:
							break

						HH = S.cell(r,1).value[0:2] # Estos valores se deben 
						MM = S.cell(r,1).value[3:5] # cambiar dependiendo del archivo
						# Se mira si es a.m. o p.m.
						AM = S.cell(r,1).value[9]
						if AM == 'p':
							HH = str(int(HH) + 12)

						if int(MM) >= 1:
							if int(HH) <9:
								HH = '0'+str(int(HH)+1)
								MM = '00'
							elif int(HH) == 23:
								HH = '00'
								MM = '00'

								yeari = FechaA[r-1][0:4]
								monthi = FechaA[r-1][4:6]
								dayi = FechaA[r-1][6:8]
								
								delta = timedelta(days=1)
								D = date(int(yeari), int(monthi), int(dayi))

								DD = D + delta
								FR = DD.strftime('%Y%m%d')

								FechaA[r-1] = FR
							elif int(HH) == 24:
								HH = '13'
								MM = '00'
							elif int(HH) == 12:
								HH = '01'
								MM = '00'
							elif int(HH) == 11:
								HH = '24'
								MM = '00'
							else:
								HH = str(int(HH)+1)
								MM = '00'

						HoraA.append(str(HH)+str(MM))

						try:
							Val.append(float(S.cell(r,2).value))
						except:
							Val.append(float('nan'))


				else:

					if r == 0:
						# Se obtiene la fecha
						FechaA[0] = str(int(S.cell(r,0).value))
						
						# Se obtiene la hora
						HH = S.cell(r,1).value[0:2] # Estos valores se deben 
						MM = S.cell(r,1).value[3:5] # cambiar dependiendo del archivo
						# Se mira si es a.m. o p.m.
						AM = S.cell(r,1).value[9]
						if AM == 'p':
							HH = str(int(HH) + 12)

						if int(MM) >= 1:
							if int(HH) <9:
								HH = '0'+str(int(HH)+1)
								MM = '00'
							elif int(HH) == 23:
								HH = '00'
								MM = '00'

								yeari = FechaA[0][0:4]
								monthi = FechaA[0][4:6]
								dayi = FechaA[0][6:8]
								
								delta = timedelta(days=1)
								D = date(int(yeari), int(monthi), int(dayi))

								DD = D + delta
								FR = DD.strftime('%Y%m%d')

								FechaA[0] = FR
							elif int(HH) == 24:
								HH = '13'
								MM = '00'
							elif int(HH) == 12:
								HH = '01'
								MM = '00'
							elif int(HH) == 11:
								HH = '24'
								MM = '00'
							else:
								HH = str(int(HH)+1)
								MM = '00'

						HoraA[0] = str(HH)+str(MM)

						# Se obtiene la variable
						try:
							Val[0] = float(S.cell(r,2).value)
						except:
							Val[0] = float('nan')
						
					else:

						# Se incluyen los siguientes datos hasta que no haya más
						# Se mira si se tiene la última fecha sino se sale del programa
						try:
							FechaA.append(str(int(S.cell(r,0).value)))
						except:
							break
							#return FechaA,HoraA,Var

						HH = S.cell(r,1).value[0:2] # Estos valores se deben 
						MM = S.cell(r,1).value[3:5] # cambiar dependiendo del archivo

						# Se mira si es a.m. o p.m.
						AM = S.cell(r,1).value[9]
						if AM == 'p':
							HH = str(int(HH) + 12)

						if int(MM) >= 1:
							if int(HH) <9:
								HH = '0'+str(int(HH)+1)
								MM = '00'
							elif int(HH) == 23:
								HH = '00'
								MM = '00'
								
								yeari = FechaA[r][0:4]
								monthi = FechaA[r][4:6]
								dayi = FechaA[r][6:8]
								
								delta = timedelta(days=1)
								D = date(int(yeari), int(monthi), int(dayi))

								DD = D + delta
								FR = DD.strftime('%Y%m%d')

								FechaA[r] = FR
							elif int(HH) == 24:
								HH = '13'
								MM = '00'
							elif int(HH) == 12:
								HH = '01'
								MM = '00'
							elif int(HH) == 11:
								HH = '24'
								MM = '00'
							else:
								HH = str(int(HH)+1)
								MM = '00'

						HoraA.append(str(HH)+str(MM))

						try:
							Val.append(float(S.cell(r,2).value))
						except:
							Val.append(float('nan'))


				#if r == 30:
					#p = 1


				# Se suma la fila
				r += 1
			
			# Se arreglan las horas
			for i,j in enumerate(HoraA):
				if HoraA[i] == '1200':
					HoraA[i] = '0000'
				elif HoraA[i] == '2400':
					HoraA[i] = '1200'


			# Se toma una Fecha completa
			FechaC = [(j +'-'+ HoraA[i]) for i,j in enumerate(FechaA)]

			# Se toma la fecha de inicio y la fecha final para hacer el vector completo de fechas
			yeari = FechaA[0][0:4]
			monthi = FechaA[0][4:6]
			dayi = FechaA[0][6:8]

			yearf = FechaA[len(FechaA)-1][0:4]
			monthf = FechaA[len(FechaA)-1][4:6]
			dayf = FechaA[len(FechaA)-1][6:8]

			rr = 0

			FechaD = ["" for k in range(1)]
			# Se realiza el vector completo de fechas
			for result in utl.perdelta(date(int(yeari), 1, 1), date(int(yearf)+1, 1, 1), timedelta(days=1)):
				
				#print(result.strftime('%Y%m%d'))
				FR = result.strftime('%Y%m%d')
				for i in range(0,24):
					if rr == 0:
						FechaD[0] = result
						if i < 10:
							FechaR[rr] = FR + '-0' +str(i)+'00'
						else:
							FechaR[rr] = FR + '-' +str(i)+'00'
					else:
						FechaD.append(result)
						if i < 10:
							FechaR.append(FR + '-0' +str(i)+'00')
						else:
							FechaR.append(FR + '-' +str(i)+'00')


					rr +=1

			#Se analizan las fechas reales vs las fechas del documento
			ValR = [float('nan') for k in FechaR] # Vector real de valores

			x = 0 # Contador para las fechas del archivo
			for i,j in enumerate(FechaR):

				if len(FechaC) == x:
					print('acabe')
				else:
					if j == FechaC[x]:	

						ValR[i] = Val[x]
						x +=1
						# Los archivos pueden problemas con la toma de datos si se repiten los datos
						# se toma el primero de ellos.
						if len(FechaC) == x: 
							print('acabe')
						else:
							if FechaC[x-1] == FechaC[x]:
								x +=1
								

					

			return FechaA,HoraA,Val,FechaC,FechaR,ValR,FechaD

		else:
			# Se abre el archivo que se llamó
			f = open(Tot)

			# Se inicializan las variables
			ff = csv.reader(f, dialect='excel', delimiter=';')

			# Se inicializan las variables


			r = 0
			# Se recorren todas las filas de la matriz
			for row in ff:

				#print(row)
				if r == 1:
					
					# Estos valores se deben cambiar según el documento
					if float(row[2]) < 10:
						
						if float(row[3]) < 10:
							FechaA[0] = str(row[1]) + '0' + str(int(row[2])) + '0' + str(int(row[3]))
						else:
							FechaA[0] = str(row[1]) + '0' + str(int(row[2])) + str(int(row[3]))
					else:
						if float(row[3]) < 10:
							FechaA[0] = str(row[1]) + str(int(row[2])) + '0' + str(int(row[3]))
						else:	
							FechaA[0] = str(row[1]) + str(row[2]) + str(row[3])


					
					if float(row[4]) < 10:
						HoraA[0] = '0' + str(int(row[4])) + '00'
					else:
						HoraA[0] = str(row[4]) + '00'

					# Se obtiene la variable
					try:
						Val[0] = float(row[8])
					except:
						Val[0] = float('nan')
				
				elif r > 0:

					if float(row[2]) < 10:
						
						if float(row[3]) < 10:
							FechaA.append(str(row[1]) + '0' + str(int(row[2])) + '0' + str(int(row[3])))
						else:
							FechaA.append(str(row[1]) + '0' + str(int(row[2])) + str(int(row[3])))
					else:
						if float(row[3]) < 10:
							FechaA.append(str(row[1]) + str(int(row[2])) + '0' + str(int(row[3])))
						else:
							FechaA.append(str(row[1]) + str(row[2]) + str(row[3]))
					

					if float(row[4]) < 10:
						HoraA.append('0' + str(int(row[4])) + '00')
					else:
						HoraA.append(str(row[4]) + '00')

					# Se obtiene la variable
					try:
						Val.append(float(row[8]))
					except:
						Val.append(float('nan'))


				r += 1 # Se suman las filas


			# Se toma una Fecha completa
			FechaC = [(j +'-'+ HoraA[i]) for i,j in enumerate(FechaA)]

			# Se toma la fecha de inicio y la fecha final para hacer el vector completo de fechas
			yeari = FechaA[0][0:4]
			monthi = FechaA[0][4:6]
			dayi = FechaA[0][6:8]

			yearf = FechaA[len(FechaA)-1][0:4]
			monthf = FechaA[len(FechaA)-1][4:6]
			dayf = FechaA[len(FechaA)-1][6:8]

			rr = 0

			FechaD = ["" for k in range(1)]
			# Se realiza el vector completo de fechas
			for result in utl.perdelta(date(int(yeari), 1, 1), date(int(yearf)+1, 1, 1), timedelta(days=1)):
				
				#print(result.strftime('%Y%m%d'))
				FR = result.strftime('%Y%m%d')
				for i in range(0,24):
					if rr == 0:
						FechaD[0] = result
						if i < 10:
							FechaR[rr] = FR + '-0' +str(i)+'00'
						else:
							FechaR[rr] = FR + '-' +str(i)+'00'
					else:
						FechaD.append(result)
						if i < 10:
							FechaR.append(FR + '-0' +str(i)+'00')
						else:
							FechaR.append(FR + '-' +str(i)+'00')


					rr +=1

			#Se analizan las fechas reales vs las fechas del documento
			ValR = [float('nan') for k in FechaR] # Vector real de valores

			x = 0 # Contador para las fechas del archivo
			for i,j in enumerate(FechaR):

				if len(FechaC) == x:
					print('acabe')
				else:
					if j == FechaC[x]:	
						#print(j)
						#print(x)
						ValR[i] = Val[x]
						x +=1
						# Los archivos pueden problemas con la toma de datos si se repiten los datos
						# se toma el primero de ellos.
						if len(FechaC) == x: 
							print('acabe')
						else:
							if FechaC[x-1] == FechaC[x]:
								x +=1


			return FechaA,HoraA,Val,FechaC,FechaR,ValR,FechaD

	def MIA(self,FechasC,Fechas,Data):
		'''
			DESCRIPTION:
		
		Con esta función se pretende encontrar la cantidad de datos faltantes de
		una serie.
		_________________________________________________________________________

			INPUT:
		+ FechaC: Fecha inicial y final de la serie original.
		+ Fechas: Vector de fechas completas de las series.
		+ Data: Vector de fechas completo
		_________________________________________________________________________
		
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

	def WriteD(self,path,fieldnames,data,deli=','):
		'''
			DESCRIPTION:
		
		Esta función fue extraída de la página web: 
		https://dzone.com/articles/python-101-reading-and-writing
		Y los códigos fueron creados por Micheal Driscoll

		Con esta función se pretende guardar los datos en un .csv.
		_________________________________________________________________________

			INPUT:
		+ path: Ruta con nombre del archivo.
		+ fieldnames: Headers del archivo.
		+ data: Lista con todos los datos.
		_________________________________________________________________________
		
			OUTPUT:
		Esta función arroja un archivo .csv con todos los datos.
		'''
		with open(path, "w") as out_file:
			writer = csv.DictWriter(out_file, delimiter=deli, fieldnames=fieldnames)

			writer.writeheader()
			for row in data:
				writer.writerow(row)

	def SaEIA(self,Fecha,V,VD,VDmax,VDmin,Pathout,Name,Var,Var2,Hdt=1):
		'''
			DESCRIPTION:
		
		Con esta función se pretende guardar los datos en una plantilla similar
		a la plantilla de la EIA para los datos de los sensores.
		_________________________________________________________________________

			INPUT:
		+ Fecha: Fecha de las variables en formato yyyy/mm/dd-HHMM.
		+ V: Variable que se quiere guardar en la plantilla
		+ VD: Variable en agregación diaria.
		+ VDmax: Variable en agregación diaria máxima.
		+ VDmin: Variable en agregación diaria mínima.
		+ Pathout: Ruta con nombre del archivo.
		+ Name: Nombre del documento.
		+ Var: Variable.
		+ Hdt: Delta de tiempo.
		_________________________________________________________________________
		
			OUTPUT:
		Esta función arroja un archivo .xlsx con todos los datos.
		'''

		# Se carga el documento de la base de datos
		PathDataBase = '/Users/DGD042/Google Drive/Est_Information/'
		Tot = PathDataBase + 'Data_Base_EIA.xlsx'
		book = xlrd.open_workbook(Tot) # Se carga el archivo
		SS = book.sheet_by_index(0) # Datos para trabajar
		# Datos de las estaciones
		Hoja = int(SS.cell(2,2).value)
		S = book.sheet_by_index(Hoja) # Datos para trabajar
		NEst = int(S.cell(1,0).value) # Número de estaciones y sensores que se van a extraer


		# Se inicializan las variables que se tomarán del documento
		x = 3 # Contador
		ID = []
		Names = []
		ZT = []
		FIns = []
		Lat = []
		Lon = []
		# Se llaman las variables
		for i in range(NEst):
			ID.append(S.cell(x,0).value)
			Names.append(S.cell(x,1).value)
			ZT.append(str(S.cell(x,5).value))
			FIns.append(S.cell(x,7).value)
			Lat.append(S.cell(x,8).value)
			Lon.append(S.cell(x,9).value)
			x += 1
		
		# Se busca la posición del ID
		q = ID.index(Name)

		ZTE = ZT[q]
		FInsE = FIns[q]
		LatE = Lat[q]
		LonE = Lon[q]
		NameE = Name + ' ' + Names[q] + ' ' + Var

		# Se crea el documento 
		Nameout = Pathout + NameE +'.xlsx'
		W = xlsxwl.Workbook(Nameout)
		bold = W.add_format({'bold': True,'align': 'left'})
		boldc = W.add_format({'bold': True,'align': 'center'})
		date_format_str = 'yyyy/mm/dd-HHMM'
		date_format = W.add_format({'num_format': date_format_str,'align': 'left'})
		

		Ai = int(Fecha[0][0:4])
		Af = int(Fecha[len(Fecha)-1][0:4])

		# Se crean las primeras hojas
		worksheet = W.add_worksheet('Min Daily Records')
		self.HeadEIA(NameE,ZTE,FInsE,LatE,LonE,Var2,worksheet,W)
		self.DailyEIA(Ai,Af,VDmin,worksheet,W)

		worksheet = W.add_worksheet('Max Daily Records')
		self.HeadEIA(NameE,ZTE,FInsE,LatE,LonE,Var2,worksheet,W)
		self.DailyEIA(Ai,Af,VDmax,worksheet,W)

		worksheet = W.add_worksheet('AV Daily Records')
		self.HeadEIA(NameE,ZTE,FInsE,LatE,LonE,Var2,worksheet,W)
		self.DailyEIA(Ai,Af,VD,worksheet,W)

		xx = 0 
		# Se realiza la escritura de la información
		for i in range(Ai,Af+1):
			# Se agrega la hoja de cada año
			worksheet = W.add_worksheet(str(i))
			# Se escribe el encabezado
			self.HeadEIA(NameE,ZTE,FInsE,LatE,LonE,Var2,worksheet,W)
			# Se llenan los datos
			xx = self.HourEIA(Ai,i,V,Hdt,xx,worksheet,W)

	def HeadEIA(self,NameE,ZTE,FInsE,LatE,LonE,Var,worksheet,W):
		'''
			DESCRIPTION:
		
		Con esta función se pretende escribir el inicio de los archivos de la
		base de datos de la escuela.
		_________________________________________________________________________

			INPUT:

		_________________________________________________________________________
		
			OUTPUT:
		Esta función arroja un archivo .xlsx con todos los datos.
		'''

		bold = W.add_format({'bold': True,'align': 'left'})
		boldc = W.add_format({'bold': True,'align': 'center'})
		date_format_str = 'yyyy/mm/dd-HHMM'
		date_format = W.add_format({'num_format': date_format_str,'align': 'left'})

		worksheet.write(0,0, r'Programa en Ingeniería Ambiental - Escuela de Ingeniería de Antioquia',bold) # Titulo
		worksheet.write(2,0, r'SENSOR',bold) # Sensor
		worksheet.write(2,1, r'U23-001 HOBO® Temperature/Relative Humidity Data')
		worksheet.write(3,0, r'NAME',bold) # Nombre
		worksheet.write(3,1, NameE) # Nombre
		worksheet.write(4,0, r'LATITUDE',bold) 
		worksheet.write(4,1, LatE)
		worksheet.write(5,0, r'LONGITUDE',bold) 
		worksheet.write(5,1, LonE)
		worksheet.write(6,0, r'ALTITUDE',bold) 
		worksheet.write(6,1, ZTE[0]+',' +ZTE[1:-2] +'m')
		worksheet.write(7,0, r'INSTALLED',bold)
		worksheet.write(7,1, FInsE)
		worksheet.write(8,0, r'VARIABLE',bold)
		worksheet.write(8,1, Var)

	def SaEIANev(self,Fecha,V,VD,VDmax,VDmin,Pathout,Name,Var,Var2,Hdt=1,PathDataBase=''):
		'''
			DESCRIPTION:
		
		Con esta función se pretende guardar los datos en una plantilla similar
		a la plantilla de la EIA para los datos de los sensores.
		_________________________________________________________________________

			INPUT:
		+ Fecha: Fecha de las variables en formato yyyy/mm/dd-HHMM.
		+ V: Variable que se quiere guardar en la plantilla
		+ VD: Variable en agregación diaria.
		+ VDmax: Variable en agregación diaria máxima.
		+ VDmin: Variable en agregación diaria mínima.
		+ Pathout: Ruta con nombre del archivo.
		+ Name: Nombre del documento.
		+ Var: Variable.
		+ Var2: Variable con unidades.
		+ Hdt: Delta de tiempo.
		_________________________________________________________________________
		
			OUTPUT:
		Esta función arroja un archivo .xlsx con todos los datos.
		'''

		# Se carga el documento de la base de datos
		Tot = PathDataBase + 'Data_Imp.xlsx'
		book = xlrd.open_workbook(Tot) # Se carga el archivo
		S = book.sheet_by_index(1) # Datos para trabajar
		NEst = int(S.cell(1,0).value) # Número de estaciones y sensores que se van a extraer


		# Se inicializan las variables que se tomarán del documento
		x = 3 # Contador
		ID = []
		Names = []
		ZT = []
		FIns = []
		Lat = []
		Lon = []
		Marker = []
		# Se llaman las variables
		for i in range(NEst):
			ID.append(S.cell(x,2).value)
			Names.append(S.cell(x,1).value)
			ZT.append(str(S.cell(x,3).value))
			FIns.append(S.cell(x,7).value)
			Lat.append(S.cell(x,4).value)
			Lon.append(S.cell(x,5).value)
			Marker.append(str(S.cell(x,7).value))
			x += 1
		
		# Se busca la posición del ID
		q = ID.index(Name)

		ZTE = ZT[q]
		FInsE = FIns[q]
		LatE = Lat[q]
		LonE = Lon[q]
		NameE = Name + ' ' + Var
		NameEE = Name +' - GPS mark ' + Marker[q]

		# Se crea el documento 
		Nameout = Pathout + NameE +'.xlsx'
		W = xlsxwl.Workbook(Nameout)
		bold = W.add_format({'bold': True,'align': 'left'})
		boldc = W.add_format({'bold': True,'align': 'center'})
		date_format_str = 'yyyy/mm/dd-HHMM'
		date_format = W.add_format({'num_format': date_format_str,'align': 'left'})
		

		Ai = int(Fecha[0][0:4])
		Af = int(Fecha[-1][0:4])

		# Se crean las primeras hojas
		worksheet = W.add_worksheet('Min Daily Records')
		self.HeadEIANev(NameEE,ZTE,FInsE,LatE,LonE,Var2,worksheet,W)
		self.DailyEIA(Ai,Af,VDmin,worksheet,W)

		worksheet = W.add_worksheet('Max Daily Records')
		self.HeadEIANev(NameEE,ZTE,FInsE,LatE,LonE,Var2,worksheet,W)
		self.DailyEIA(Ai,Af,VDmax,worksheet,W)

		worksheet = W.add_worksheet('AV Daily Records')
		self.HeadEIANev(NameEE,ZTE,FInsE,LatE,LonE,Var2,worksheet,W)
		self.DailyEIA(Ai,Af,VD,worksheet,W)

		xx = 0 
		# Se realiza la escritura de la información
		for i in range(Ai,Af+1):
			
			# Se agrega la hoja de cada año
			worksheet = W.add_worksheet(str(i))
			# Se escribe el encabezado
			self.HeadEIANev(NameEE,ZTE,FInsE,LatE,LonE,Var2,worksheet,W)
			# Se llenan los datos
			xx = self.HourEIANev(Ai,i,V,Hdt,xx,worksheet,W)		

	def HeadEIANev(self,NameE,ZTE,FInsE,LatE,LonE,Var,worksheet,W):
		'''
			DESCRIPTION:
		
		Con esta función se pretende escribir el inicio de los archivos de la
		base de datos de la escuela.
		_________________________________________________________________________

			INPUT:

		_________________________________________________________________________
		
			OUTPUT:
		Esta función arroja un archivo .xlsx con todos los datos.
		'''

		bold = W.add_format({'bold': True,'align': 'left'})
		boldc = W.add_format({'bold': True,'align': 'center'})
		date_format_str = 'yyyy/mm/dd-HHMM'
		date_format = W.add_format({'num_format': date_format_str,'align': 'left'})

		worksheet.write(0,0, r'Programa en Ingeniería Ambiental - Escuela de Ingeniería de Antioquia',bold) # Titulo
		worksheet.write(2,0, r'SENSOR',bold) # Sensor
		worksheet.write(2,1, r'U23-001 HOBO® Temperature/Relative Humidity Data')
		worksheet.write(3,0, r'NAME',bold) # Nombre
		worksheet.write(3,1, NameE) # Nombre
		worksheet.write(4,0, r'LATITUDE',bold) 
		worksheet.write(4,1, LatE)
		worksheet.write(5,0, r'LONGITUDE',bold) 
		worksheet.write(5,1, LonE)
		worksheet.write(6,0, r'ALTITUDE',bold) 
		try:
			worksheet.write(6,1, ZTE[0]+',' +ZTE[1:-2] +'m')
		except:
			worksheet.write(6,1, 'NNN')
		worksheet.write(7,0, r'INSTALLED',bold)
		worksheet.write(7,1, FInsE)
		worksheet.write(8,0, r'VARIABLE',bold)
		worksheet.write(8,1, Var)

	def DailyEIA(self,Ai,Af,VD,worksheet,W):
		'''
			DESCRIPTION:
		
		Con esta función se pretende escribir los datos diarios de la plantilla.
		_________________________________________________________________________

			INPUT:

		_________________________________________________________________________
		
			OUTPUT:
		Esta función arroja un archivo .xlsx con todos los datos.
		'''
		bold = W.add_format({'bold': True,'align': 'left'})
		boldc = W.add_format({'bold': True,'align': 'center'})
		date_format_str = 'yyyy/mm/dd-HHMM'
		date_format = W.add_format({'num_format': date_format_str,'align': 'left'})

		MesI = ['J','F','M','A','M','J','J','A','S','O','N','D']

		# Contador de datos
		x = 12 # Donde comienzan los datos
		xx = 0 # Contador de la matriz de los datos
		for i in range(Ai,Af+1):
			# Se escriben los encabezados
			worksheet.write(x-2,0, r'Year',boldc)
			worksheet.write(x-2,1, i)
			worksheet.write(x-1,0, r'Month',boldc)
			for j in range(1,13):
				worksheet.write(x-1,j, MesI[j-1],boldc)
			for g in range(1,32):
				worksheet.write(x+g-1,0, g,boldc)

			worksheet.write(x+32,0, r'Min',boldc)
			worksheet.write(x+33,0, r'Max',boldc)
			worksheet.write(x+34,0, r'Average',boldc)

			# Se incluyen los datos
			for j in range(1,13):
				Fi = date(i,j,1)
				if j == 12:
					Ff = date(i+1,1,1)
				else:
					Ff = date(i,j+1,1)
				DF = Ff-Fi
				VC = [] # Variable para hacer los otros cálculos
				for g in range(DF.days):
					VC.append(VD[xx])
					# Se escriben los datos
					if np.isnan(VD[xx]):
						worksheet.write(x+g,j,'')
					else:	
						worksheet.write(x+g,j,VD[xx])
					xx += 1

				# Se calculan los medios, máximos y mínimos de cada mes
				M = np.nanmean(np.array(VC))
				Mmin = np.nanmin(np.array(VC))
				Mmax = np.nanmax(np.array(VC))

				if np.isnan(Mmin):
					worksheet.write(x+32,j, '',bold)
				else:
					worksheet.write(x+32,j, Mmin,bold)

				if np.isnan(Mmax):
					worksheet.write(x+33,j, '',bold)
				else:
					worksheet.write(x+33,j, Mmax,bold)

				if np.isnan(M):
					worksheet.write(x+34,j, '',bold)
				else:
					worksheet.write(x+34,j, M,bold)
				

			x += 39

	def HourEIA(self,Ai,ii,VD,Hdt,xx,worksheet,W):
		'''
			DESCRIPTION:
		
		Con esta función se pretende escribir los datos horarios de la plantilla.
		_________________________________________________________________________

			INPUT:

		_________________________________________________________________________
		
			OUTPUT:
		Esta función arroja un archivo .xlsx con todos los datos.
		'''
		# Meses
		Mes = ['January','February','March','April','May','June',\
		'July','August','Septiembre','October','November','December']

		bold = W.add_format({'bold': True,'align': 'left'})
		boldc = W.add_format({'bold': True,'align': 'center'})
		date_format_str = 'yyyy/mm/dd-HHMM'
		date_format = W.add_format({'num_format': date_format_str,'align': 'left'})

		# Contador de datos
		x = 12 # Donde comienzan los datos
		if ii == Ai:
			xx = 0 # Contador de la matriz de los datos

		for i in range(1,13):

			# Se escriben los encabezados
			worksheet.write(x-2,0, r'MONTH',boldc)
			worksheet.write(x-2,1, Mes[i-1])
			worksheet.write(x-1,0, r'DAY/HOUR',boldc)
			for j in range(1,32):
				worksheet.write(x-1,j, j,boldc)
			# Se calcula el delta de tiempo
			dt = 24/Hdt
			h = 0 # Contador de horas
			m = 0 # Contador de minutos
			for g in range(0,int(dt)):
				if g == 0:
					MM = str(int(m))
					HH = str(h)
					worksheet.write(x+g,0, HH+':0'+MM,boldc)
				else:
					if m == 60:
						h += 1
						m = 0
					MM = str(int(m))
					HH = str(h)
					if m < 10:
						worksheet.write(x+g,0, HH+':0'+MM,boldc)
					else:
						worksheet.write(x+g,0, HH+':'+MM,boldc)
				m += Hdt*60

			worksheet.write(x+49,0, r'Min',boldc)
			worksheet.write(x+50,0, r'Max',boldc)
			worksheet.write(x+51,0, r'Average',boldc)

			Fi = date(ii,i,1)
			if i == 12:
				Ff = date(ii+1,1,1)
			else:
				Ff = date(ii,i+1,1)
			DF = Ff-Fi

			# Se incluyen los datos
			for j in range(0,DF.days):
				VC = [] # Variable para hacer los otros cálculos
				for g in range(int(dt)):
					VC.append(VD[xx])
					# Se escriben los datos
					if np.isnan(VD[xx]):
						worksheet.write(x+g,j+1,'')
					else:	
						worksheet.write(x+g,j+1,VD[xx])
					xx += 1

				# Se calculan los medios, máximos y mínimos de cada mes
				M = np.nanmean(np.array(VC))
				Mmin = np.nanmin(np.array(VC))
				Mmax = np.nanmax(np.array(VC))

				if np.isnan(Mmin):
					worksheet.write(x+49,j+1, '',bold)
				else:
					worksheet.write(x+49,j+1, Mmin,bold)

				if np.isnan(Mmax):
					worksheet.write(x+50,j+1, '',bold)
				else:
					worksheet.write(x+50,j+1, Mmax,bold)

				if np.isnan(M):
					worksheet.write(x+51,j+1, '',bold)
				else:
					worksheet.write(x+51,j+1, M,bold)

			x += 55

		return xx

	def HourEIANev(self,Ai,ii,VD,Hdt,xx,worksheet,W):
		'''
			DESCRIPTION:
		
		Con esta función se pretende escribir los datos horarios de la plantilla.
		_________________________________________________________________________

			INPUT:

		_________________________________________________________________________
		
			OUTPUT:
		Esta función arroja un archivo .xlsx con todos los datos.
		'''
		# Meses
		Mes = ['January','February','March','April','May','June',\
		'July','August','Septiembre','October','November','December']

		bold = W.add_format({'bold': True,'align': 'left'})
		boldc = W.add_format({'bold': True,'align': 'center'})
		date_format_str = 'yyyy/mm/dd-HHMM'
		date_format = W.add_format({'num_format': date_format_str,'align': 'left'})

		# Contador de datos
		x = 12 # Donde comienzan los datos
		if ii == Ai:
			xx = 0 # Contador de la matriz de los datos

		for i in range(1,13):

			# Se escriben los encabezados
			worksheet.write(x-2,0, r'MONTH',boldc)
			worksheet.write(x-2,1, Mes[i-1])
			worksheet.write(x-1,0, r'DAY/HOUR',boldc)
			for j in range(1,32):
				worksheet.write(x-1,j, j,boldc)
			# Se calcula el delta de tiempo
			dt = 24/Hdt
			h = 0 # Contador de horas
			m = 0 # Contador de minutos
			for g in range(0,int(dt)):
				if g == 0:
					MM = str(int(m))
					HH = str(h)
					worksheet.write(x+g,0, HH+':0'+MM,boldc)
				else:
					if m == 60:
						h += 1
						m = 0
					MM = str(int(m))
					HH = str(h)
					if m < 10:
						worksheet.write(x+g,0, HH+':0'+MM,boldc)
					else:
						worksheet.write(x+g,0, HH+':'+MM,boldc)
				m += Hdt*60

			worksheet.write(x+25,0, r'Min',boldc)
			worksheet.write(x+26,0, r'Max',boldc)
			worksheet.write(x+27,0, r'Average',boldc)

			Fi = date(ii,i,1)
			if i == 12:
				Ff = date(ii+1,1,1)
			else:
				Ff = date(ii,i+1,1)
			DF = Ff-Fi

			# Se incluyen los datos
			for j in range(0,DF.days):
				VC = [] # Variable para hacer los otros cálculos
				for g in range(int(dt)):
					VC.append(VD[xx])
					# Se escriben los datos
					if np.isnan(VD[xx]):
						worksheet.write(x+g,j+1,'')
					else:	
						worksheet.write(x+g,j+1,VD[xx])
					xx += 1

				# Se calculan los medios, máximos y mínimos de cada mes
				M = np.nanmean(np.array(VC))
				Mmin = np.nanmin(np.array(VC))
				Mmax = np.nanmax(np.array(VC))

				if np.isnan(Mmin):
					worksheet.write(x+25,j+1, '',bold)
				else:
					worksheet.write(x+25,j+1, Mmin,bold)

				if np.isnan(Mmax):
					worksheet.write(x+26,j+1, '',bold)
				else:
					worksheet.write(x+26,j+1, Mmax,bold)

				if np.isnan(M):
					worksheet.write(x+27,j+1, '',bold)
				else:
					worksheet.write(x+27,j+1, M,bold)

			x += 32

		return xx

	def CSVEIA(self,fieldnames,FechaCT,TC,DPC,HRC,Pathout,Name):
		'''
			DESCRIPTION:
		
		Con esta función se pretende guardar los datos de la EIA en un archivo .csv.
		
		Esta función guardará las tres variables: Temperatura, Temperatura a punto 
		de rocío y Humedad Relativa.
		_________________________________________________________________________

			INPUT:
			+ filednames: Encabezados de las variables.
			+ FechaCt: Vector de fechas.
			+ TC: Temperatura.
			+ DPC: Temperatura de punto de rocío.
			+ HRC: Humedad Relativa
			+ Pathout: Ruta de salida.
			+ Name: Nombre del archivo.
		_________________________________________________________________________
		
			OUTPUT:
		Esta función arroja un archivo .csv con todos los datos.
		'''

		data1=[] # Se inicializa la variable de los datos sin valores NaN
		# Datos
		for i in range(len(TC)):
			if np.isnan(TC[i]) and np.isnan(DPC[i]) and np.isnan(HRC[i]):
				data1.append((str(FechaCT[i])+'; ; ; ').split(';'))
			elif np.isnan(TC[i]) and np.isnan(DPC[i]):
				data1.append((str(FechaCT[i])+'; ; ;'+str(HRC[i])).split(';'))
			elif np.isnan(TC[i]) and np.isnan(HRC[i]):
				data1.append((str(FechaCT[i])+'; ;'+str(DPC[i])+'; ').split(';'))
			elif np.isnan(DPC[i]) and np.isnan(HRC[i]):
				data1.append((str(FechaCT[i])+';'+str(TC[i])+'; ; ').split(';'))
			elif np.isnan(TC[i]):
				data1.append((str(FechaCT[i])+'; ;'+str(DPC[i])+';'+str(HRC[i])).split(';'))
			elif np.isnan(DPC[i]):
				data1.append((str(FechaCT[i])+';'+str(TC[i])+'; '+';'+str(HRC[i])).split(';'))
			elif np.isnan(HRC[i]):
				data1.append((str(FechaCT[i])+';'+str(TC[i])+';'+str(DPC[i])+'; ').split(';'))
			else:
				data1.append((str(FechaCT[i])+';'+str(TC[i])+';'+str(DPC[i])+';'+str(HRC[i])).split(';'))
			

		my_list1=[]
		# Se juntan los valores
		for values in data1[:]:
			inner_dict = dict(zip(fieldnames, values))
			my_list1.append(inner_dict)

		nameout1 = Pathout + Name + '.csv'

		self.WriteD(nameout1,fieldnames,my_list1)

	def CSVCUASHI(self,Fecha,T,Pathout,Name,VariableCode,V=0):
		'''
			DESCRIPTION:
		
		Con esta función se pretende guardar los datos de la EIA en un archivo .csv
		con la plantilla de CUASHI.

		Si es necesario modificar algo de la plantilla se debe realizar manualmente
		_________________________________________________________________________

			INPUT:
			+ filednames: Encabezados de las variables.
			+ FechaCt: Vector de fechas.
			+ T: Variable que se quiere guardar.
			+ Pathout: Ruta de salida.
			+ Name: Nombre del archivo.
			+ V: Value accuracy conditional. 1: Temperature
											 2: Relative Humidity
		_________________________________________________________________________
		
			OUTPUT:
		Esta función arroja un archivo .csv con todos los datos.
		'''

		# Se guardan los archivos en un .csv por aparte
		# Encabezados
		fieldnames = ['DataValue','ValueAccuracy','LocalDateTime','UTCOffset','DateTimeUTC','SiteCode',\
			'VariableCode','OffsetValue','OffsetTypeCode','CensorCode','QualifierCode','MethodCode',\
			'SourceCode','SampleCode','DerivedFromID','QualityControlLevelCode']
		data1=[] # Se inicializa la variable de los datos sin valores NaN
		TT = ['' for k in range(len(T))]
		# Datos
		for i in range(len(T)):
			if ~np.isnan(T[i]):
				TT[i] = T[i]

		FechaCUASHI=[]
		for i in Fecha:
			Year = i[0:4]
			Month = i[5:7]
			Day = i[8:10]
			HH = i[11:13]
			MM = i[13:15]
			FechaCUASHI.append(Month+'/'+Day+'/'+Year+' '+HH+':'+MM)
		
		LocalDate = FechaCUASHI
		UTCOffset = '-5'
		DateTimeUTC = ''
		SiteCode = Name
		
		OffsetValue = ''
		OffsetTypeCode = ''
		CensorCode = ''
		QualifierCode = ''
		MethodCode = 'HOBO_datalogger'
		SourceCode = 'EIA'
		SampleCode = ''
		DerivedFromID = ''
		QualityControlLevelCode = 0

		my_list1=[]
		for i in range(len(TT)):
			if V == 0: # Para datos de temperatura
				if TT[i] == '':
					ValueAc = ''
				elif TT[i] >= 0:
					ValueAc = 0.21
				elif TT[i] < 0:
					ValueAc = 0.7
			elif V == 1: # Para datos de humedad
				if TT[i] == '':
					ValueAc = ''
				elif TT[i] >= 10 and TT[i] <= 90:
					ValueAc = 2.5
				elif TT[i] > 90:
					ValueAc = 4.5
			else: # No se ha programado
				ValueAc = ''


			# Se juntan los valores
			
			data1 = [str(TT[i]),str(ValueAc),str(LocalDate[i]),str(UTCOffset),str(DateTimeUTC)\
				,str(SiteCode),str(VariableCode),str(OffsetValue),str(OffsetTypeCode),str(CensorCode)\
				,str(QualifierCode),str(MethodCode),str(SourceCode),str(SampleCode),str(DerivedFromID)\
				,str(QualityControlLevelCode)]
			inner_dict = dict(zip(fieldnames, data1))
			my_list1.append(inner_dict)
			
		nameout1 = Pathout+ Name + '.csv'
		self.WriteD(nameout1,fieldnames,my_list1)

	def CrASCIIfile(self,data,xllcorner,yllcorner,cellsize,Nameout):
		'''
			DESCRIPTION:
		
		Con esta función se pretende guardar los datos en formato ASCII.
		_________________________________________________________________________

			INPUT:
			+ data: Matriz de datos.
			+ xllcorner: Left down corner coordinates.
			+ yllcorner: Down left corner coordinates.
			+ cellsize
			+ Nameout: Name out.
		_________________________________________________________________________
		
			OUTPUT:
		Esta función arroja un archivo .asc con todos los datos.
		'''

		# Se guarda la información
		ncols = data.shape[1]
		nrows = data.shape[0]

		# Se abre el documento
		f = open(Nameout, 'w')
		# Encabezado
		f.write('ncols '+str(int(ncols))+'\r\n')
		f.write('nrows '+str(int(nrows))+'\r\n')
		f.write('xllcorner '+str(float(xllcorner))+'\r\n')
		f.write('yllcorner '+str(float(yllcorner))+'\r\n')
		f.write('cellsize '+str(float(cellsize))+'\r\n')
		f.write('nodata_value '+str(float(-9999.0))+'\r\n')
		# Se cargan los datos
		for i in range(len(data)):
			d = [str(j) + ' ' for j in data[i]]
			dd = ''.join(d)
			ddd = dd[:-1]+'\r\n'
			f.write(ddd)
		# Se cierra el archivo
		f.close()
