# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#						Coded by Daniel González Duque
#						    Last revised 29/09/2016
#______________________________________________________________________________
#______________________________________________________________________________

#______________________________________________________________________________
#
# DESCRIPCIÓN DE LA CLASE:
# 	En esta clase se incluyen las rutinas para tratar la información de Anglo
#	en el proyecto Anglo-GOTTA.
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


class Anglo_GOTTA_Li:

	def __init__(self):

		'''
			DESCRIPTION:

		Este es el constructor por defecto, no realiza ninguna acción.
		'''

	def EDAn(self,Tot,sheetname,Ai,Af,flagD=True):
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
		+ sheetname: Nombre de la hoja a extraer.
		+ Ai: Año inicial para extraer la información.
		+ Af: Año final para extraer la información.
		+ flagD: condicional para extraer los datos diarios.
			True: Se extraen los datos diarios.
			False: Se extraen datos horarios.
		_________________________________________________________________________
		
			OUTPUT:
		- FechaP: Fechas que se extrajeron.
		- Hora: Valores de hora, solo si len(ni) > 1.
		- V1: Variable 1 que se extrae como float
		'''

		# Se verifica el tipo de información que se va a extraer
		if flagD:

			

			# --------------------------------------
			# Se abre el archivo de Excel
			# --------------------------------------
			book = xlrd.open_workbook(Tot)
			# Se carga la página en donde se encuentra el información
			S = book.sheet_by_name(sheetname)
			# Se lee toda la segunda columna
			k = 2
			x = 0
			An = []
			while k == 2:
				try: 
					An.append(S.cell(x,1).value)
				except IndexError:
					k = 1
				x += 1
			An = np.array(An)
			# Posicion del inicio de los datos en cada matriz
			q = np.where(An == 'Año')[0]
			q = q+1

			Ai = float(An[q[0]])
			Ai = int(Ai)
			try:
				Af = float(An[q[-1]])
			except:
				try:
					Af = float(An[q[-2]])
				except:
					try:
						Af = float(An[q[-3]])
					except:
						Af = float(An[q[-4]])


			
			Af = int(Af)


			# Se hace el vector de fechas
			Fi = date(Ai,1,1)
			Ff = date(Af,12,31)
			F = Fi
			FechaP = [Fi]
			while True:
				FechaP.append(FechaP[-1]+timedelta(1))
				F = FechaP[-1]
				if F == Ff:
					break

			# Parámetros para tener en cuenta
			xf = q[0]
			xc = np.arange(4,4+12,1) # Columnas en donde se encuentran los meses

			# Se inicia la variable
			V1 = []
			
			# Ciclo para los años
			for ii,A in enumerate(range(Ai,Af+1)):
				xf = q[ii]
				# Ciclo para los meses
				for j in range(1,13):
					x = xf # Filas de cada mes
					if j == 12:
						FFi = date(A,j,1)
						FFf = date(A+1,1,1)
					else:
						FFi = date(A,j,1)
						FFf = date(A,j+1,1)

					Dif = (FFf-FFi).days # Días del mes
					# Ciclo para los días
					for k in range(Dif): 
						V1.append(S.cell(x,xc[j-1]).value) # Valor que se extrajo
						if V1[-1] == '':
							V1[-1] = np.nan
						x += 1
				

			return FechaP, V1

				



