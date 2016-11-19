# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#						Coded by Daniel González Duque
#						    Last revised 14/09/2016
#______________________________________________________________________________
#______________________________________________________________________________

#______________________________________________________________________________
#
# CLASS DESCRIPTION:
# 	This class intend to have all the functions related to Fractals and Chaos
#	Theory.
#______________________________________________________________________________

# We import the packages
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.io as sio # To use .mat files
from scipy import stats as st # To make linear regressions
from scipy.optimize import curve_fit # To do curve fitting
import os
import warnings
from pandas import Series, DataFrame
import random


class FyC_Class:

	def __init__(self):

		'''
			DESCRIPTION:

		This is a default function.
		'''
	def FDim(self,Method,DataM,BS,M):
		'''
			DESCRIPTION:
		
		This Function calculates the fractal dimension using several functions, 
		the user can pick the function using Box Counting.

		The algorithms were created by using the Theiler (1989) article 
		"Estimating fractal dimension" and some codes used by Johnson (2014)
		on his blog entry "Fractal Dimension And Box Counting".
		
		_________________________________________________________________________

			INPUT:
		+ Method: Used method. 0: to Box Counting.
							   1: to Correlation Dimension.
		+ DataM: vector or matrix with the data, the data must be in columns.
		+ BS: Box size, it could be a vector or a value.
		+ M: Values that enclose the figure, if it has more than one dimension
			 please input the as (x,y,...). You have to include start and finish.
		_________________________________________________________________________
		
			OUTPUT:
		- N: Number of boxes.
		'''


		# It descriminates the method to use
		if Method == 0: # Box Counting
			
			# Matrix dimensions
			a = DataM.shape
			q = len(a)
			
			if q == 1: # One dimension

				# It searchs for the Box Size lenght (BN)
				try:
					BN = len(BS)
				except TypeError:
					BN = 0

				if BN == 0:
					
					# Number of boxes
					NN = np.int(np.floor((M[1]-M[0])/BS))
					# Changes the data to a DataFrame
					data = Series( DataM )
					# Initiate the Count variable
					Count = np.empty(NN)
					for i in range(NN):
						# See if the data is inside the box
						cond = (data >= i*BS)&(data < (i+1)*BS)
						# Separate and count the data
						subset = data[ cond ]
						Count[i] = subset.count()
					C = np.where(Count > 0)[0]
					N = len(C)
					return N

				else:
					N = np.empty(BN)
					for ii in range(BN):
						# Number of boxes
						NN = np.int(np.floor((M[1]-M[0])/BS[ii]))
						# Changes the data to a DataFrame
						data = Series( DataM )
						# Initiate the Count variable
						Count = np.empty(NN)
						for i in range(NN):
							# See if the data is inside the box
							cond = (data >= i*BS[ii])&(data < (i+1)*BS[ii])
							# Separate and count the data
							subset = data[ cond ]
							Count[i] = subset.count()
						C = np.where(Count > 0)[0]
						N[ii] = len(C)
					return N
			
			if q == 2: # Tow Dimensions
				
				# It searchs for the Box Size lenght (BN)
				try:
					BN = len(BS)
				except TypeError:
					BN = 0

				if BN == 0:
					
					# Number of boxes
					NN1 = np.int(np.floor((M[0,1]-M[0,0])/BS))
					NN2 = np.int(np.floor((M[1,1]-M[1,0])/BS))
					# Changes the data to a DataFrame
					data = Series( DataM )
					# Initiate the Count variable
					Count = np.empty((NN1,NN2))
					for i in range(NN1):
						for j in range(NN2):
							# See if the data is inside the box
							cond = ((data[0] >= i*BS)&(data[0] < (i+1)*BS)) & \
								((data[1] >= j*BS)&(data[1] < (j+1)*BS))
							# Separate and count the data
							Count[i,j] = sum(cond)
					C = np.where(Count > 0)[0]
					N = len(C)
					return N

				else:
					N = np.empty(BN)
					for ii in range(BN):
						# Number of boxes
						NN1 = np.int(np.floor((M[0,1]-M[0,0])/BS[ii]))
						NN2 = np.int(np.floor((M[1,1]-M[1,0])/BS[ii]))
						# Changes the data to a DataFrame
						data = DataM
						# Initiate the Count variable
						Count = np.empty((NN1,NN2))
						for i in range(NN1):
							for j in range(NN2):
								# See if the data is inside the box
								cond = ((data[0] >= i*BS[ii])&(data[0] < (i+1)*BS[ii])) & \
									((data[1] >= j*BS[ii])&(data[1] < (j+1)*BS[ii]))
								# Separate and count the data
								Count[i,j] = sum(cond)
						C = np.where(Count > 0)[0]
						N[ii] = len(C)
					return N				



	def RegF(self,r,N,PathImg,Title,Name):
		'''
			DESCRIPTION:
		
		This Function makes the regression to calculate the fractal dimension using
		the Box Counting Method.
		
		_________________________________________________________________________

			INPUT:
		+ r: Box sizes
		+ N: Number of boxes
		+ PathImg: Path to the image.
		+ Title: Title of the image.
		+ Name: Name of the Graph.
		_________________________________________________________________________
		
			OUTPUT:
		- A: Intercept
		- Df: Fractal Dimension 
		'''


		# Designated regresion
		def f(x,A,DF):
			# This is the function that is needed to 
			# calculate the Fractal Dimension
			return DF * x + A

		popt, pcov = curve_fit(f,-np.log(r),np.log(N))
		A, Df = popt

		# The graph is made
		
		F = plt.figure(figsize=(15,10))
		plt.rcParams.update({'font.size': 22})
		plt.scatter(r,N,s=15)
		plt.plot(r,np.exp(A)*r**(-Df))
		plt.title(Title,fontsize=26 )  
		plt.xlabel(r'$r$',fontsize=24)  
		plt.ylabel(r'$N$',fontsize=24)  
		ax = plt.gca()
		ax.set_xscale('log')
		ax.set_yscale('log')
		#ax.set_aspect(1)
		plt.grid(which='minor', ls='-', color='0.75')
		plt.grid(which='major', ls='-', color='0.25')
		plt.savefig(PathImg + Name +'.png' )
		plt.close('all')

		return Df, A

	def RandCasGen(self,Dens,P,N,b):
		'''
			DESCRIPTION:
		
		This Function generates the Random cascade based on a p value and a Density  
		using the steps given in Over and Gupta (1994). 

		This function only works in	two dimensions, further progamming would be 
		given for higher or lower dimensions.
		
		_________________________________________________________________________

			INPUT:
		+ Dens: Density of the function.
		+ P: Value of P given by the formula in Over and Gupta (1994).
		+ N: Number of levels.
		+ b: Branching number.
		_________________________________________________________________________
		
			OUTPUT:
		- RCM: Different random cascade results.
		'''

		# We initiate the 

		RCM = dict()
		RCM[0] = Dens

		for n in range(1,N+1):

			# Se genera la matriz de b^(n/2) X b^(n/2)
			A = np.empty((int(b**(n/2)),int(b**(n/2))))*np.nan

			W = np.empty((int(b**(n/2)),int(b**(n/2))))
			# Se incluye un número aleatorio normal
			for i in range(int(b**(n/2))):
				for j in range(int(b**(n/2))):
					p=random.random() 
					if p <= P:
						W[i,j] = 0
					else:
						W[i,j] = 1/(1-P)


			RCM[n] = np.zeros((int(b**(n/2)),int(b**(n/2))))
			if n == 1:
				RCM[n] = Dens*W
			else:
				
				xi = 0
				for i in range(int(b**((n-1)/2))):
					xj = 0	
					for j in range(int(b**((n-1)/2))):

						RCM[n][xi:xi+int(b/2),xj:xj+int(b/2)] = W[xi:xi+int(b/2),xj:xj+int(b/2)]\
							*DensN[i,j]
						
						xj = xj + int(b/2)
					xi = xi + int(b/2)
				
			DensN = RCM[n].copy()

		return RCM

	def LCalc(self,RainT,N,b,d,flagG=True,Title='',Name='',PathImg=''):
		'''
			DESCRIPTION:
		
		This Function generates the Random cascade calculates the regression of the
		relation between the spatial-scale relation and gives the P value.
		
		_________________________________________________________________________

			INPUT:
		+ RainT: Input Matrix.
		+ N: Number of subdivisions of the total matrix.
		+ b: Branching number.
		+ flagG: Flag for the graph.
		+ Title: Title of the graph.
		+ Name: Name of the Image.
		+ PathImg: Path of the Image.
		_________________________________________________________________________
		
			OUTPUT:
		- P: Value of P.
		'''

		# Se realiza el conteo de cajas para la fracción de precipitación
		x = np.where(RainT > 0)[0]

		#NB = np.array([1,4,12,29,73])

		Lambda = [(b**(-n/2)) for n in range(0,N+1)]

		NBB = np.ones(len(Lambda))
		NBB[len(NBB)-1] = len(x)
		# Se encuentra la cantidad de datos a diferentes escalas
		for n in range(N-1,0,-1):
			# Matriz con datos
			R = np.empty((int(b**((n)/2)),int(b**((n)/2))))
			xi = 0
			for i in range(int(b**((n)/2))):
				xj = 0	
				for j in range(int(b**((n)/2))):

					if n == N-1:
						R[i,j] = np.max(RainT[xi:xi+int(b/2),xj:xj+int(b/2)])
					else:
						R[i,j] = np.max(RR[xi:xi+int(b/2),xj:xj+int(b/2)])
					
					xj = xj + int(b/2)
				xi = xi + int(b/2)
			NBB[n] = len(np.where(R > 0)[0])
			RR = R.copy()

		fLambda = NBB/np.array([(b**(n)) for n in range(0,N+1)])

		# Se encuentra la pendiente
		# Designated regresion
		def f(x,A,S):
			# This is the function that is needed to calculate the slope
			return S * x + A

		popt, pcov = curve_fit(f,np.log(Lambda),np.log(fLambda))
		A, S = popt

		if flagG == True:
			# The graph is made
			F = plt.figure(figsize=(15,10))
			plt.rcParams.update({'font.size': 22})
			plt.scatter(np.log(Lambda),np.log(fLambda),s=15)
			plt.plot(np.log(Lambda),np.log(np.exp(A)*Lambda**(S)))
			plt.title(Title,fontsize=26 )  
			plt.ylabel(r'$\log(f(\lambda_n))$',fontsize=24)  
			plt.xlabel(r'$\log(\lambda_n)$',fontsize=24)  
			ax = plt.gca()
			# ax.set_xscale('log')
			# ax.set_yscale('log')
			#ax.set_aspect(1)
			# plt.grid(which='minor', ls='-', color='0.75')
			plt.grid(which='major', ls='-', color='0.25')
			plt.savefig(PathImg + Name +'.png' )
			plt.close('all')

		# Se calcula P con la fórmula dada
		P = 1-(b**(-S/d))

		return P
		




