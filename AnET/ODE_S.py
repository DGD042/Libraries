# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#						Coded by Daniel González Duque
#						    Last revised 14/03/2016
#______________________________________________________________________________
#______________________________________________________________________________

#______________________________________________________________________________
#
# DESCRIPCIÓN DE LA CLASE:
# 	En esta clase se incluyen las rutinas para encontrar las soluciones a
#	ecuaciones diferenciales ordinarias.
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

class ODE_S:

	def __init__(self):

		'''
			DESCRIPTION:

		Este es el constructor por defecto, no realiza ninguna acción.
		'''
	def RK4(self,f,y0,x0,dt=1,n=4):
		'''
			DESCRIPTION:
		
		Con esta función se pretende aplicar la metodología de Runge-Kutta 4 para
		solucionar ecuaciones diferenciales ordianrias.
		_________________________________________________________________________

			INPUT:
		+ f: función
		+ y0: Valor inicial de y, pueden ser un vector con varios valores
		+ x0: Valor incial de x, puede ser un vector con varios valores
		+ x1: Valor que continua en x.
		+ n: Orden del Runge-Kutta.
		_________________________________________________________________________
		
			OUTPUT:
		
		'''
		vx = [0]*(n + 1)
		vy = [0]*(n + 1)
		#h = (x1 - x0)/n
		h = dt
		vx[0] = x = x0
		vy[0] = y = y0[0]
		for i in range(1, n + 1):
			if i >= 1:
				#print(f(x,y))
				k1 = h*f(x, vy)
				k2 = h*f(x + 0.5*h, vy + 0.5*k1)
				k3 = h*f(x + 0.5*h, vy + 0.5*k2)
				k4 = h*f(x + h, vy + k3)
			vx[i] = x = x + h
			vy[i] = vy[i-1] + (k1 + k2 + k2 + k3 + k3 + k4)/6
		return vx, vy

