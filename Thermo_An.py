# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#						Coded by Daniel González Duque
#						    Last revised 09/10/2016
#______________________________________________________________________________
#______________________________________________________________________________

#______________________________________________________________________________
#
# DESCRIPCIÓN DE LA CLASE:
# 	This class have all the thermodynamics functions applied to Atmospheric
#	analysis. 
#______________________________________________________________________________

import numpy as np
import sys
import csv
import xlrd 
import xlsxwriter as xlsxwl
import scipy.io as sio 
from scipy import stats as st 
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates 
import matplotlib.mlab as mlab
import time
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import date, datetime, timedelta

import warnings

# Import the created libraries
from UtilitiesDGD import UtilitiesDGD

utl = UtilitiesDGD()

class Thermo_An:

	def __init__(self):

		'''
			DESCRIPTION:

		This is a build up function.
		'''
	def PVTs(self,Z,T,Flagline=True,Zmax=4000,Zmin=1000):
		'''
			DESCRIPTION:
		
		This function calculates the Vertical Temperature Profile (PVT, in spanish).

		Both, Altitude and Temperature vectores must have the same lenght.
		_________________________________________________________________________

			INPUT:
		+ Z: Altitude.
		+ T: Temperature.
		+ Flagline: Flag to know a line is needed for a graph.
		+ Zmax: maximum value of altitude for the line.
		+ Zmin: minimum value of altitude for the line.
		_________________________________________________________________________
		
			OUTPUT:
		- slope: Slope of the regressio.
		- intercept: Intercept of the regression.
		- r_value: Correlation.
		- M0: Values of temperature for the line.
		- M1: Values of altitude for the line.
		'''
		
		XX,YY,NminD = utl.NoNaN(Z,T)
		slope, intercept, r_value, p_value, std_err = st.linregress(XX,YY)
		if Flagline:
			M0 = [(slope*Zmax+intercept),(slope*Zmin+intercept)]
			M1 = [Zmin,Zmax]
			return slope, intercept, r_value, M0, M1
		return slope, intercept, r_value

	def MALR(self,slope,intercept,Zmax=7000,Zmin=1000,LCLp=1800,FlagGraph=False,PathImg='',NameArch='',Name=''):
		'''
			DESCRIPTION:
		
		Script para encontrar la moist adiabatic lapse rate a partir de los datos
		horarios dados, esta se encontrará para todas las diferentes horas,
		presentando una gráfica diferente por hora.

		Este script utilizará las ecuaciones descritas por Del Genio (s.f.) en su
		curso de termodinámica en la universidad de columbia, estas fueron
		programas por Daniel Ruiz en eñ archivo 'Claro River Profiles Final
		Version.xls'. Para mayor información remitirse al documento 'Summary of
		key equations.docx' el cual contiene un resumen de lo expuesto por Del
		Genio (s.f.) en las notas de clase, realizado por Daniel Ruiz.

		El nivel de condensación por elevación (LCL) se supone según estimaciones
		previas, según Cuevas (2015) el LCL se encuentra entre
		2190 -> Secos - 2140 -> Mojados para una región en el PNN Los Nevados.
		Estimaciones previas de Daniel Ruiz (sugerencia personal) sugieren que el
		valor se encuentra a 1800 msnm, hace falta un estudio más profundo para ver
		en donde se encuentra este punto.
		_________________________________________________________________________

			INPUT:
		+ 
		_________________________________________________________________________
		
			OUTPUT:
		- 
		'''
		# En esta sección se colacarán y calcularán algunas de las constantes,
		# estas constantes se encuentran expuestas en el archivo 'Claro River
		# Profiles Final Version.xls', en la hoja 'Constants' realizado por el
		# profesor Daniel Ruiz.

		# Presión de vapor en saturación a 0°C
		e_s0 = 6.11 # mb
		e_sp0 = e_s0*100 # Pa

		# Calor latente, Este valor es a 0°C y puede ser asumido como cte.
		L = 2.5*10**6 # J/kg H_2O 

		# Constante de gases para vapor de agua 
		R_v = 461 # J/K/kg

		# Temperatura 
		T_0 = 273.15 # K
		T = 288.15 # K
		Tc = T-T_0 # °C

		# Presión de vapor de agua en saturación a 15°C
		e_s = e_sp0*np.exp((L/R_v)*((1/T_0)-(1/T))) # Pa
		e_sh = e_s/100 # hPa

		# Epsilon (Buscar ¿qué es?)
		epsilon = 0.622 # R_d/R_v

		# Presión (¿Qué presión?)
		p = 80000 # Pa

		# Saturated Mixing ratio 
		W_s = epsilon*(e_s/p)

		# Gas constant for dry air
		R_d = 287 # J/K/kg
		DivCR = 3.5 # C_p/R_d

		# Specific heat for dry at constant pressure
		C_p = R_d*DivCR # J/K/kg

		# dry adiabatic lapse rate
		Gamma_d = 9.8 # K/km

		# Moist adiabatic lapse rate
		Gamma_m = Gamma_d*((1+((L*W_s)/(R_d*T)))/((1+(((L**2)*W_s)/(C_p*R_v*(T**2))))))

		# Deltas
		PF = 273.15 # K
		DT = T-PF # K
		DZ = DT/Gamma_m # km

		# Datos adicionales
		p_0 = 1013*100 # Pa
		H_s = 8631 # m
		z = -H_s*np.log(p/p_0) # m

		# Freezing level
		FL = (DZ*1000)+ z # m

		# Assumed Freezing level
		AFL = 4900 # m
		ADZ = (AFL-z)/1000 # km

		# Buscar qué es Tao!!
		Tao = DT/ADZ # K/km

		# R_d/C_p
		DivRC = R_d/C_p

		# Presió 0 
		P_00 = 1000 # hPa - mb
		# ------------------------------------------------------------

		# Se crea el vecto de alturas
		Al = np.arange(Zmin,Zmax+50,50) # Vector de alturas cada 50 metros
		Als = len(Al) # Tamaño del vector

		Headers = ['Mean Annual T [°C]', 'Mean Annual T [K]'\
			, 'e_s (T) [Pa]', 'Atmospheric pressure [mbar]'\
			, 'Atmospheric pressure [Pa]', 'W_s', 'Gamma_m [K/km]'\
			, 'Gamma_d [K/km]', 'Profile T [K]-LCL='+str(LCLp),'Profile T dry']

		Hes = len(Headers)

		# Se crea la matriz con todos los valores
		TM = np.zeros((Als,Hes))

		# Se calcula la temperatura a partir de una regresión previamente realizada
		TM[:,0] = slope*Al+intercept

		# Se encuentra el primer valor para el cual T<=0
		x_0 = np.where(TM[:,0] <= 0)[0]
		Al_0 = Al[x_0[0]]

		# Se pasa la temperatura Kelvin
		TM[:,1] = TM[:,0]+273.15

		# Se inicializa el vector del perfil vertical de temperatura
		TM[0,-2] = TM[0,1]
		TM[0,-1] = TM[0,1]

		# Se calculan el resto de los valores
		for ii,i in enumerate(Al):
			# Presión de vapor e_s [Pa]
			TM[ii,2] = e_sp0*np.exp((L/R_v)*((1/T_0)-(1/TM[ii,1])))

			# Presión atmosférica [mbar] -> Se puede cambiar por datos!!
			# Se calcula con la ecuación hidroestática p=1009.28 exp(-z/H)
			# donde H: Scale Height = 8631.

			TM[ii,3] = 1009.28*np.exp(-i/H_s)

			# Presión atmosférica [Pa]
			TM[ii,4] = TM[ii,3]*100

			# Rata de mezcla W_s
			TM[ii,5] = epsilon*(TM[ii,2]/TM[ii,4])

			# Moist adiabatic lapse rate Gamma_m
			TM[ii,6] = Gamma_d*((1+((L*TM[ii,5])/(R_d*TM[ii,1])))/((1+(((L**2)*TM[ii,5])/(C_p*R_v*(TM[ii,1]**2))))))

			# Dry adiabatic lapse rate
			TM[ii,7] = Gamma_d

			# Se genera el perfil vertical de temperatura
			if ii > 0:

				# Perfil de temperatura vertical [k]
				if i <= LCLp:
					# Perfil adiabático seco
					TM[ii,8] = TM[ii-1,8]-TM[ii,7]*((i-Al[ii-1])/1000)
				else:
					# Perfil adiabático húmedo
					TM[ii,8] = TM[ii-1,8]-((TM[ii-1,6]+TM[ii,6])/2)*((i-Al[ii-1])/1000)

				# Dry adiabatic lapse rate profile
				TM[ii,9] = TM[ii-1,9]-TM[ii,7]*((i-Al[ii-1])/1000)



		# Se realiza la gráfica
		if FlagGraph:
			# Se crea la carpeta en donde se guarda la información
			utl.CrFolder(PathImg)
			# Se organizan los valores para graficarlos
			x = np.where(Al <= LCLp)[0]
			xx = np.where(Al > LCLp)[0]
			# Parámetros de la gráfica
			fH = 20
			fV = fH*(2/3)
			minorLocatorx = MultipleLocator(1)
			minorLocatory = MultipleLocator(100)
			F = plt.figure(figsize=utl.cm2inch(fH,fV))
			plt.rcParams.update({'font.size': 14,'font.family': 'sans-serif'\
				,'font.sans-serif': 'Arial'})
			plt.plot(TM[:,0],Al,'k-',label='Gradiente Ambiental')
			plt.plot(TM[x,8]-273.15,Al[x],'r--',label='Gradiente Adiabático Seco')
			plt.plot(TM[:,9]-273.15,Al,'r--')
			plt.plot(TM[xx,8]-273.15,Al[xx],'b--',label='Gradiente Adiabático Húmedo')
			plt.legend(loc=0,fontsize=12)
			plt.title(Name,fontsize=16 )  # Colocamos el título del gráfico
			plt.xlabel(u'Temperatura [°C]',fontsize=14)  # Colocamos la etiqueta en el eje x
			plt.ylabel('Altura [msnm]',fontsize=14)  # Colocamos la etiqueta en el eje y
			plt.gca().set_ylim([900,4500])
			plt.gca().xaxis.set_minor_locator(minorLocatorx)
			plt.gca().yaxis.set_minor_locator(minorLocatory)
			plt.xlim([-5,40])
			plt.savefig(PathImg + NameArch +'.png', format='png',dpi=200)
			plt.close('all')

		# Se reportan los resultados
		return Al,TM


