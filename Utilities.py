# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#						Coded by Daniel González Duque
#						    Last revised 21/05/2017
#______________________________________________________________________________
#______________________________________________________________________________
# ------------------------
# Importing Modules
# ------------------------ 
# Manipulate Data
import numpy as np
# System
import sys
import os
import glob as gl
import re
import operator as op
import warnings

class Utilities(object):
	'''
	____________________________________________________________________________
	
	CLASS DESCRIPTION:
		
		This class allows the user to do some function and data manipulation. 

		This class also have some functions that are needed to use the other
		classes, so it is crucial that you have this class along with the
		the other classes.
	
		This class is of free use and can be modify, if you have some 
		problem please contact the programmer to the following e-mails:
	
		- danielgondu@gmail.com	
		- dagonzalezdu@unal.edu.co
		- daniel.gonzalez17@eia.edu.co
	
		--------------------------------------
		 How to use the library
		--------------------------------------

	____________________________________________________________________________

	'''

	def __init__(self):
		# Operation findings
		self.operations= {'Over':op.gt,'over':op.gt,'>':op.gt,
					 	  'Lower':op.lt,'lower':op.lt,'<':op.lt,
					 	  '>=':op.ge,'<=':op.le,
					 	  'mean':np.nanmean,'sum':np.nansum,
					 	  'std':np.nanstd}

		return
	
	# System
	def ShowError(self,fn,cl,msg):
		'''
		DESCRIPTION:

			This function manages errors, and shows them. 

			Error managment is -1 in all this class.
		_______________________________________________________________________
		INPUT:
			+ fn: Function that produced the error.
			+ cl: Class that produced the error.
			+ msg: Message of the error.
		'''

		print('ERROR: Function <'+fn+'> Class <'+cl+'>: '+msg)
		
		return -1

	def ExitError(self,fn,cl,msg):

		'''
		DESCRIPTION:

			This function stops the ejecution of a code with a given error
			message.
		_______________________________________________________________________
		
		INPUT:
			+ fn: Función que produjo el error
			+ cl: Clase que produjo el error
			+ msg: Mensaje del error
		'''

		print('ERROR: Function <'+fn+'> Class <'+cl+'>: '+msg)
		sys.exit(0)


	def CrFolder(self,Path):
		'''
		DESCRIPTION:
		
			This function creates a folder in the given path, if the path does 
			not exist then it creates the path itself
		_______________________________________________________________________

		INPUT:
			+ Path: Path that needs to be created
		_______________________________________________________________________
		OUTPUT:
			This function create all the given path.
		'''

		# Verify if the path already exists
		if not os.path.exists(Path):
			os.makedirs(Path)

	# Units
	def cm2inch(self,*tupl):
		'''
		DESCRIPTION:
		
			This functions allows to change centimeters to inch so you can 
			denote the size of the figure you want to save.
		_______________________________________________________________________

		INPUT:
			+ *tupl: Tuple variable with centimeter values.
		_______________________________________________________________________
		
		OUTPUT:
			- tuple: Values in inch.
		'''
		inch = 2.54
		if isinstance(tupl[0], tuple):
			return tuple(i/inch for i in tupl[0])
		else:
			return tuple(i/inch for i in tupl)

	# Data Manipulation
	def Oper_Det(self,Oper):
		'''
		DESCRIPTION:
		
			This functions verifies the existence of the operation inside 
			the operations directory and returns the operation.
		_______________________________________________________________________

		INPUT:
			+ Oper: Operation to be verified.
		_______________________________________________________________________
		
		OUTPUT:
			- OperRet: Returned operation.
		'''
		try:
			OperRet = self.operations[Oper]
			return OperRet
		except KeyError:
			return self.ShowError('Oper_Det','Utilities','Operation or comparation not found, verify the given string')

	def NoNaN(self,X,Y,flagN=True):
		'''
		DESCRIPTION:
		
			This function removes the NaN values of two series and count
			all the non NaN values in the vectors.
		_________________________________________________________________________

			INPUT:
		+ X: First vector.
		+ Y: Second vector.
		+ flagN: flag to know if it liberates the non NaN data.
		_________________________________________________________________________
		
			OUTPUT:
		- XX: Fisrt vector without NaN values.
		- YY: Second vector without NaN values.
		'''
		if len(X) != len(Y):
			Er = self.ShowError('NoNaN','Utilities','X and Y are not the same length')
			if flagN == True:
				return Er, Er, Er
			else:
				return Er, Er
		q = ~(np.isnan(X) | np.isnan(Y))
		N = sum(q) 
		XX = X[q]
		YY = Y[q]

		if flagN == True:
			return XX,YY,N
		else:
			return XX,YY





