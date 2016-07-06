# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#						Coded by Daniel González Duque
#						    Last revised 13/06/2016
#______________________________________________________________________________
#______________________________________________________________________________

#______________________________________________________________________________
#
# CLASS DESCRIPTION:
# 	This class pretend to fit different functions in different sets of data.
#______________________________________________________________________________

import numpy as np
import sys
import scipy.io as sio # To use .mat files
from scipy import stats as st # To make linear regressions
from scipy.optimize import curve_fit # To do curve fitting
import os
import warnings

# This classes that are needed
from UtilitiesDGD import UtilitiesDGD

utl = UtilitiesDGD()


# The functions are defined
def LR(x,a,b):
	return a*x+b
def PL(x,a,b):
	return a*x**b

class CFitting:

	def __init__(self):

		'''
			DESCRIPTION:

		This is a default function.
		'''
	def FF(self,xdata,ydata,F=1,func=0):
		'''
			DESCRIPTION:
		
		This function makes 
		_________________________________________________________________________

			INPUT:
		+ xdata: Data in the x axis.
		+ ydata: Data in the y axis.
		+ F: Function that wants to be fitted, by default is 1.
			1: Linear Fitting.
			2: Power Fitting.
			3: Exponential Fitting.
			4: Log Fitting.
			5: Your Function.
		+ func: Is the function that would be fitted, it is not necessary unless
			    you input 5 in the F variable.
		_________________________________________________________________________
		
			OUTPUT:
		
		'''

		# Se remueven los valores NaN
		X,Y = utl.NoNaN(xdata,ydata,False)

		if F == 2:
			Coef, pcov = curve_fit(PL,X,Y)
			ss_res = np.dot((Y - PL(X, *Coef)),(Y - PL(X, *Coef)))

		perr = np.sqrt(np.diag(pcov))



		
		ymean = np.mean(Y)
		ss_tot = np.dot((Y-ymean),(Y-ymean))
		R2 = 1-(ss_res/ss_tot)

		return Coef, perr, R2


