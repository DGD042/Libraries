# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#						Coded by Daniel González Duque
#						    Last revised 18/02/2016
#______________________________________________________________________________
#______________________________________________________________________________

#______________________________________________________________________________
#
# DESCRIPCIÓN DE LA CLASE:
# 	En esta clase se incluyen las rutinas para tratar datos de diferentes
#	fuentes de información, extraer los datos y completarlos y generar 
#	cambios de escala a la información. Además esta librería también genera
#	archivos en diferentes formatos para el posterior uso de la información.
#
#	Esta libreria es de uso libre y puede ser modificada a su gusto, si tienen
#	algún problema se pueden comunicar con el programador al correo:
# 	dagonzalezdu@unal.edu.co
#______________________________________________________________________________

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import sys
import warnings

# Se importan los paquetes para manejo de fechas
from datetime import date, datetime, timedelta

from UtilitiesDGD import UtilitiesDGD
utl = UtilitiesDGD()



class Hydro_Plotter:

	def __init__(self):

		'''
			DESCRIPTION:

		Este es el constructor por defecto, no realiza ninguna acción.
		'''
	def monthlab(self,Date):
		''' 			
			DESCRIPTION:
		Esta función pretende cambiar los ticks de las gráficas de años a 
		mes - año para compartir el mismo formato con las series de Excel.
		__________________________________________________________________
			
			INPUT:
		+ Date: Fechas en formato ordinal.
		__________________________________________________________________

			OUTPUT:	
		- Labels: Labes que se ubicarán en los ejes.
		'''

		Year = [date.fromordinal(int(i)).year for i in Date]
		Month = [date.fromordinal(int(i)).month for i in Date]

		Meses = ['ene.','feb.','mar.','abr.','may.','jun.','jul','ago.','sep.','oct.','nov.','dec.']

		Labels = [Meses[Month[i]-1] + ' - ' + str(Year[i])[2:] for i in range(len(Year))]
		return Labels

	def DalyS(self,Date,Value,Var_LUn,Var='',flagT=True,v='',fH=20,PathImg='',**args):
		'''
			DESCRIPTION:
		
		Esta función permite hacer las gráficas de series temporales parecidas a
		a las presentadas en Excel para el proyecto.

		_________________________________________________________________________

			INPUT:
		+ Date: Vector de fechas en formato date.
		+ Value: Vector de valores de lo que se quiere graficar.
		+ Var_LUn: Label de la variable con unidades, por ejemplo Precipitación (mm).
		+ Var: Nombre de la imagen.
		+ flagT: Flag para saber si se incluye el título.
		+ v: Titulo de la Figura.
		+ fH: Largo de la figura en cm, el ancho lo calcula automáticamente.
		+ PathImg: Ruta donde se quiere guardar el archivo.
		+ **args: Argumentos adicionales para la gráfica, como color o ancho de la línea.
		_________________________________________________________________________
		
			OUTPUT:
		Esta función arroja una gráfica y la guarda en la ruta desada.
		'''
		# Ancho de la Figura
		fV = fH*(2/3)
		# Se crea la carpeta para guardar la imágen
		utl.CrFolder(PathImg)

		# Se genera la gráfica
		F = plt.figure(figsize=utl.cm2inch(fH,fV))
		# Parámetros de la Figura
		plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
			,'font.sans-serif': 'Arial Narrow'\
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
		# Se realiza la figura ***
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
			plt.title(v,fontsize=18)
		plt.ylabel(Var_LUn,fontsize=16)
		# Se arregla el espaciado de la figura
		plt.tight_layout()
		# Se guarda la figura
		plt.savefig(PathImg + Var + '.png',format='png',dpi=300)
		plt.close('all')




