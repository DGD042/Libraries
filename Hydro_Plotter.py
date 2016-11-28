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
# 	En esta clase se incluyen las rutinas para generar gráficos de información
#	hidrológica.
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

		Meses = ['ene.','feb.','mar.','abr.','may.','jun.','jul.','ago.','sep.','oct.','nov.','dec.']

		Labels = [Meses[Month[i]-1] + ' - ' + str(Year[i])[2:] for i in range(len(Year))]
		return Labels

	def DalyS(self,Date,Value,Var_LUn,Var='',flagT=True,v='',PathImg='',**args):
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
		+ PathImg: Ruta donde se quiere guardar el archivo.
		+ **args: Argumentos adicionales para la gráfica, como color o ancho de la línea.
		_________________________________________________________________________
		
			OUTPUT:
		Esta función arroja una gráfica y la guarda en la ruta desada.
		'''
		# Tamaño de la Figura
		fH=20 # Largo de la Figura
		fV = fH*(2/3) # Ancho de la Figura
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
		# Se realiza la figura 
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

	def NaNMGr(self,Date,NNF,NF,Var='',flagT=True,Var_L='',Names='',PathImg=''):
		'''
			DESCRIPTION:
		
		Esta función permite hacer las gráficas de datos faltantes mensuales.

		_________________________________________________________________________

			INPUT:
		+ Date: Vector de fechas en formato date.
		+ Value: Vector de valores de lo que se quiere graficar.
		+ Var: Nombre de la imagen.
		+ flagT: Flag para saber si se incluye el título.
		+ Var_L: Label de la variable sin unidades.
		+ Names: Nombre de la estación para el título.
		+ PathImg: Ruta donde se quiere guardar el archivo.
		+ **args: Argumentos adicionales para la gráfica, como color o ancho de la línea.
		_________________________________________________________________________
		
			OUTPUT:
		Esta función arroja una gráfica y la guarda en la ruta desada.
		'''
		# Tamaño de la Figura
		fH=20 # Largo de la Figura
		fV = fH*(2/3) # Ancho de la Figura
		# Se crea la carpeta para guardar la imágen
		utl.CrFolder(PathImg)

		NN = np.array(NNF)*100
		N = np.array(NF)*100

		# Parámetros de la gráfica
		F = plt.figure(figsize=utl.cm2inch(fH,fV))
		# Parámetros de la Figura
		plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
			,'font.sans-serif': 'Arial Narrow'\
			,'xtick.labelsize': 15,'xtick.major.size': 6,'xtick.minor.size': 4\
			,'xtick.major.width': 1,'xtick.minor.width': 1\
			,'ytick.labelsize': 16,'ytick.major.size': 6,'ytick.minor.size': 4\
			,'ytick.major.width': 1,'ytick.minor.width': 1\
			,'axes.linewidth':1\
			,'grid.alpha':0.1,'grid.linestyle':'-'})
		plt.tick_params(axis='x',which='both',bottom='on',top='off',\
			labelbottom='on',direction='out')
		plt.tick_params(axis='y',which='both',left='on',right='off',\
			labelleft='on')
		plt.tick_params(axis='y',which='major',direction='out') 
		# Se realiza la figura 
		p1 = plt.bar(Date,NN,color='#006600',width=31,edgecolor='none') # Disponibles
		p2 = plt.bar(Date,N,color='#E46C0A',bottom=NN,width=31,edgecolor='none') # Faltantes
		# Se arreglan los ejes
		axes = plt.gca()
		axes.set_ylim([0,100])
		# Se cambia el label de los eje
		xTL = axes.xaxis.get_ticklocs() # List of position in x
		Labels2 = self.monthlab(xTL)
		plt.xticks(xTL, Labels2) # Se cambia el valor de los ejes
		plt.legend((p2[0],p1[0]), ('Faltantes','Disponibles'),loc=4)
		# Labels
		if flagT:
			# Título
			plt.title('Estado de la información de ' + Var_L + ' en la estación ' + Names,fontsize=15 )
		plt.xlabel(u'Mes',fontsize=16)  # Colocamos la etiqueta en el eje x
		plt.ylabel('Porcentaje de datos',fontsize=16)  # Colocamos la etiqueta en el eje y
		plt.tight_layout()
		plt.savefig(PathImg + Var +'_NaN_Mens' + '.png',format='png',dpi=300 )
		plt.close('all')

	def DalyCycle(self,HH,CiDT,ErrT,VarL,VarLL,Name,NameA,PathImg,**args):
		'''
			DESCRIPTION:
		
		Esta función permite hacer las gráficas del ciclo diurno

		_________________________________________________________________________

			INPUT:
		+ HH: Vector de horas.
		+ CiDT: Vector de datos horarios promedio.
		+ ErrT: Barras de error de los datos.
		+ VarL: Label de la variable con unidades, por ejemplo Precipitación (mm).
		+ VarLL: Label de la variable sin unidades.
		+ Name: Nombre de la Estación.
		+ NameA: Nombre del archivo.
		+ PathImg: Ruta donde se quiere guardar el archivo.
		+ **args: Argumentos adicionales para la gráfica, como color o ancho de la línea.
		_________________________________________________________________________
		
			OUTPUT:
		Esta función arroja una gráfica y la guarda en la ruta desada.
		'''
		# Tamaño de la Figura
		fH=20 # Largo de la Figura
		fV = fH*(2/3) # Ancho de la Figura
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
			labelbottom='on',direction='in')
		plt.tick_params(axis='x',which='major',direction='inout')
		plt.tick_params(axis='y',which='both',left='on',right='off',\
			labelleft='on')
		plt.tick_params(axis='y',which='major',direction='inout') 
		plt.grid()
		# Argumentos que se deben incluir color=C,label=VarLL,lw=1.5
		plt.errorbar(HH,CiDT,yerr=ErrT,fmt='-',**args)
		plt.title('Ciclo Diurno de ' + VarLL + ' en ' + Name,fontsize=16 )  # Colocamos el título del gráfico
		plt.ylabel(VarL,fontsize=16)  # Colocamos la etiqueta en el eje x
		plt.xlabel('Horas',fontsize=16)  # Colocamos la etiqueta en el eje y
		ax = plt.gca()
		plt.xlim([0,23])

		# The minor ticks are included
		xTL = ax.xaxis.get_ticklocs() # List of Ticks in x
		MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
		minorLocatorx = MultipleLocator(MxL)
		yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
		MyL = (yTL[1]-yTL[0])/5 # minorLocatory value
		minorLocatory = MultipleLocator(MyL)
		plt.gca().xaxis.set_minor_locator(minorLocatorx)
		plt.gca().yaxis.set_minor_locator(minorLocatory)
		plt.tight_layout()
		plt.savefig(PathImg + 'CTErr_' + NameA+'.png',format='png',dpi=300 )
		plt.close('all')

	def DalyCyclePer(self,HH,zM,zMed,P10,P90,PV90,PV10,VarL,VarLL,Name,NameA,PathImg):
		'''
			DESCRIPTION:
		
		Esta función permite hacer las gráficas del ciclo diurno con área de
		porcentaje.
		_________________________________________________________________________

			INPUT:
		+ HH: Vector de horas.
		+ zM: Media.
		+ zMed: Mediana.
		+ P10: Percentil inferior.
		+ P90: Percentil superior.
		+ PV90: Valor que se quiere tomar para el percentil superrior.
		+ PV10: Valor que se quiere tomar para el percentil inferior.
		+ VarL: Label de la variable con unidades, por ejemplo Precipitación (mm).
		+ VarLL: Label de la variable sin unidades.
		+ Name: Nombre de la Estación.
		+ NameA: Nombre del archivo.
		+ PathImg: Ruta donde se quiere guardar el archivo.
		_________________________________________________________________________
		
			OUTPUT:
		Esta función arroja una gráfica y la guarda en la ruta desada.
		'''
		# Tamaño de la Figura
		fH=20 # Largo de la Figura
		fV = fH*(2/3) # Ancho de la Figura
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
			labelbottom='on',direction='in')
		plt.tick_params(axis='x',which='major',direction='inout')
		plt.tick_params(axis='y',which='both',left='on',right='off',\
			labelleft='on')
		plt.tick_params(axis='y',which='major',direction='inout') 
		plt.grid()
		# Argumentos que se deben incluir color=C,label=VarLL,lw=1.5
		plt.plot(HH,zM,'k-',label='Media',lw=1.5)
		plt.plot(HH,zMed,'k--',label='Mediana',lw=1.5)
		plt.fill_between(HH,P10,P90,color='silver',label=r'P$_{%s}$ a P$_{%s}$' %(PV10,PV90))
		plt.plot(HH,P10,'w-',lw=0.0001)
		plt.legend(loc=0)
		plt.xlim([0,23])
		ax = plt.gca()
		yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
		MyL = (yTL[1]-yTL[0])/5 # minorLocatory value
		plt.ylim([yTL[0]-2*MyL,yTL[-1]+2*MyL])
		plt.title('Ciclo Diurno de ' + VarLL + ' en ' + Name,fontsize=16 )  # Colocamos el título del gráfico
		plt.ylabel(VarL,fontsize=16)  # Colocamos la etiqueta en el eje x
		plt.xlabel('Horas',fontsize=16)  # Colocamos la etiqueta en el eje y
		# The minor ticks are included
		ax = plt.gca()
		xTL = ax.xaxis.get_ticklocs() # List of Ticks in x
		MxL = (xTL[1]-xTL[0])/5 # minorLocatorx value
		minorLocatorx = MultipleLocator(MxL)
		yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
		MyL = (yTL[1]-yTL[0])/5 # minorLocatory value
		minorLocatory = MultipleLocator(MyL)
		plt.gca().xaxis.set_minor_locator(minorLocatorx)
		plt.gca().yaxis.set_minor_locator(minorLocatory)
		plt.tight_layout()
		plt.savefig(PathImg + 'CTPer_' + NameA+'.png',format='png',dpi=300 )
		plt.close('all')

	def AnnualCycle(self,MesM,MesE,VarL,VarLL,Name,NameA,PathImg,AH=False,**args):
		'''
			DESCRIPTION:
		
		Esta función permite hacer las gráficas del ciclo anual

		_________________________________________________________________________

			INPUT:
		+ MesM: Valor medio del mes, debe ir desde enero hasta diciembre.
		+ MesE: Barras de error de los valores.
		+ VarL: Labes de la variable con unidades.
		+ VarLL: Label de la variable sin unidades.
		+ Name: Nombre de la Estación.
		+ NameA: Nombre del archivo.
		+ PathImg: Ruta donde se quiere guardar el archivo.
		+ AH: Este es un flag para saber si se hace el gráfico con el año hidrológico.
		+ **args: Argumentos adicionales para la gráfica, como color o ancho de la línea.
		_________________________________________________________________________
		
			OUTPUT:
		Esta función arroja una gráfica y la guarda en la ruta desada.
		'''
		# Tamaño de la Figura
		fH=20 # Largo de la Figura
		fV = fH*(2/3) # Ancho de la Figura
		# Se crea la carpeta para guardar la imágen
		utl.CrFolder(PathImg)
		# Vector de meses
		Months = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dec']
		if AH:
			Months2 = Months[5:]+Months[:5]
			MesM2 = np.hstack((MesM[5:],MesM[:5]))
			MesE2 = np.hstack((MesE[5:],MesE[:5]))
		else:
			Months2 = Months
			MesM2 = MesM
			MesE2 = MesE

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
			labelbottom='on',direction='in')
		plt.tick_params(axis='x',which='major',direction='inout')
		plt.tick_params(axis='y',which='both',left='on',right='off',\
			labelleft='on')
		plt.tick_params(axis='y',which='major',direction='inout') 
		plt.grid()
		# Argumentos que se deben incluir color=C,label=VarLL,lw=1.5
		plt.errorbar(np.arange(1,13),MesM2,yerr=MesE2,fmt='-',**args)
		plt.title('Ciclo anual de '+ VarLL +' de ' + Name,fontsize=16 )  # Colocamos el título del gráfico
		plt.ylabel(VarL,fontsize=16)  # Colocamos la etiqueta en el eje x
		plt.xlabel('Meses',fontsize=16)  # Colocamos la etiqueta en el eje y
		# The minor ticks are included
		ax = plt.gca()
		yTL = ax.yaxis.get_ticklocs() # List of Ticks in y
		MyL = (yTL[1]-yTL[0])/5 # minorLocatory value
		minorLocatory = MultipleLocator(MyL)
		plt.gca().yaxis.set_minor_locator(minorLocatory)
		ax.set_xlim([0,13])
		plt.xticks(np.arange(1,13), Months2) # Se cambia el valor de los ejes
		plt.tight_layout()
		plt.savefig(PathImg + 'CAErr_' + NameA+'.png',format='png',dpi=300 )
		plt.close('all')


	def GCorr(self,CP,VarL1,VarL2,Names,NameA,PathImg='',**args):
		'''
			DESCRIPTION:
		
		Esta función permite hacer las gráficas de correlación.

		_________________________________________________________________________

			INPUT:
		+ CP: Valores de correlación.
		+ VarL1: Label de la primera variable.
		+ VarL2: Label de la segunda variable.
		+ Names: Nombre de las estaciones.
		+ NamesA: Nombre de la gráfica.
		+ PathImg: Ruta donde se quiere guardar el archivo.
		+ **args: Argumentos adicionales para la gráfica, como color o ancho de la línea.
		_________________________________________________________________________
		
			OUTPUT:
		Esta función arroja una gráfica y la guarda en la ruta desada.
		'''
		# Tamaño de la Figura
		fH=20 # Largo de la Figura
		fV = fH*(2/3) # Ancho de la Figura
		# Se crea la carpeta para guardar la imágen
		utl.CrFolder(PathImg)

		N = len(CP)

		# Parámetros de la gráfica
		F = plt.figure(figsize=utl.cm2inch(fH,fV))
		# Parámetros de la Figura
		plt.rcParams.update({'font.size': 15,'font.family': 'sans-serif'\
			,'font.sans-serif': 'Arial Narrow'\
			,'xtick.labelsize': 15,'xtick.major.size': 6,'xtick.minor.size': 4\
			,'xtick.major.width': 1,'xtick.minor.width': 1\
			,'ytick.labelsize': 16,'ytick.major.size': 6,'ytick.minor.size': 4\
			,'ytick.major.width': 1,'ytick.minor.width': 1\
			,'axes.linewidth':1\
			,'grid.alpha':0.1,'grid.linestyle':'-'})
		plt.tick_params(axis='x',which='both',bottom='on',top='off',\
			labelbottom='on',direction='out')
		plt.tick_params(axis='y',which='both',left='on',right='off',\
			labelleft='on')
		plt.tick_params(axis='y',which='major',direction='out') 
		# Se realiza la figura 
		p1 = plt.bar(np.arange(0,N),CP,width=0.8,**args)
		plt.title('Correlación entre la ' + VarL1 + ' y la ' + VarL2,fontsize=16 )  # Colocamos el título del gráfico
		plt.xlabel(u'Estaciones',fontsize=16)  # Colocamos la etiqueta en el eje x
		plt.ylabel('Correlación',fontsize=16)  # Colocamos la etiqueta en el eje y
		plt.xlim([-0.2,N])
		axes = plt.gca()
		axes.xaxis.set_ticks(np.arange(0.4,N+0.4,1))
		axes.set_xticklabels(Names)
		yTL = axes.yaxis.get_ticklocs() # List of Ticks in y
		MyL = (yTL[1]-yTL[0])/5 # minorLocatory value
		minorLocatory = MultipleLocator(MyL)
		plt.gca().yaxis.set_minor_locator(minorLocatory)
		plt.tight_layout()
		plt.savefig(PathImg + 'Correlaciones_'+ NameA +'.png',format='png',dpi=300 )
		plt.close('all')
