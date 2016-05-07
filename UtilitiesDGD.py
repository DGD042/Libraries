# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Código realizado por: Daniel González (2015) 
# Con base en un código de: Juan Pablo Rendón Álvarez -> Todo el crédito
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# DESCRIPCIÓN DE LA CLASE:
# En esta clase se incluyen las rutinas varias que ayudan a manipular
# la información y realizar procesos varios.
# ----------------------------------------------------------------------


import numpy as np
import sys

class UtilitiesDGD:

	def __init__(self):

		'''
			DESCRIPTION:

		Este es el constructor por defecto, no realiza ninguna acción.
		'''

	def ExitError(self,fn,cl,msg):

		'''
			DESCRIPTION:
		Este detiene la ejecución de un código por un error que se
		presente que no deje avanzar la ejecución.

			INPUT:
		+ fn: Función que produjo el error
		+ cl: Clase que produjo el error
		+ msg: Mensaje del error
		'''

		print('Error in the function '+fn+' from the class '+cl+': '+msg)
		sys.exit(0)

	def BTS(self,T):

		'''
			DESCRIPTION:
		
		Con esta función se pretende hacer un reordenamiento de una serie utilizando "Boot-straping"
		en este caso se reordenan las dos series.

			INPUT:
		+ T: Serie de datos que se va a reordenar
		
			OUTPUT:
		- TT: Serie reordenada.
		'''
		# Se toma la cantidad de datos
		L = len(T)

		
		b = np.random.uniform(0,1,L)*(L-2)+1
		c = np.round(b)
		c = c.astype(int)
		
		TT = np.copy(T[c])

		return TT


	def BTSS(self,T1,T2):
		'''
			DESCRIPTION:
		
		Con esta función se pretende hacer un reordenamiento de una serie utilizando "Boot-straping"
		en este caso se reordena una de las series.

			INPUT:
		+ T1: Primera serie de datos que se va a reordenar.
		+ T2: Segunda serie de datos que se va a reordenar.
		
			OUTPUT:
		- TT1: Primera serie reordenada.
		- TT2: Segunda serie reordenada.
		'''

		#if len(T1) is not len(T2):
		#	self.ExitError('FT','BTSS','Vectors T1 and T2 must be the same length')

		L = len(T1)

		b = np.random.uniform(0,1,L)*(L-2)+0

		c = np.round(b)
		c = c.astype(int)

		TT1 = np.copy(T1[c])
		TT2 = np.copy(T2[c])

		return TT1,TT2


	def perdelta(self,start, end, delta):
		'''
			DESCRIPTION:
		
		Función extraída de internet que permite realizar vectores de fechas a partir de dos fechas

			INPUT:
		+ start: Fecha de inicio.
		+ end: Fecha de final.
		+ delta: Paso de tiempo
		
		
			OUTPUT:
		
		'''
		curr = start
		while curr < end:
			yield curr
			curr += delta

	def NoNaN(self,X,Y,flagN=True):
		'''
			DESCRIPTION:
		
		Con esta función se pretende remover los valores NaN de dos series y
		contar la cantidad de valores NaN de los vectores.
		_________________________________________________________________________

			INPUT:
		+ X: Primer vector.
		+ Y: Segundo vector.
		+ flagN: ¿Liberar la cantidad de datos NaN?.
		_________________________________________________________________________
		
			OUTPUT:
		- XX: Primer vector sin NaN.
		- YY: Segundo vector sin NaN.
		'''
		# Se miran las posiciones en donde se encuentren los valores NaN
		q = ~(np.isnan(X) | np.isnan(Y))
		N = sum(q) # Se suman la cantidad de datos no NaN

		# Se cambian los vectores los dos vectores sin valores NaN
		XX = X[q]
		YY = Y[q]

		if flagN == True:
			return XX,YY,N
		else:
			return XX,YY

	def Interp(self,Xi,Yi,X,Xf,Yf):
		'''
			DESCRIPTION:
		
		Con esta función se pretende realizar una interpolación lineal de los datos.
		_________________________________________________________________________

			INPUT:
		+ Xi: Primer valor de x.
		+ Yi: Primer valor de y.
		+ X: Valor del punto en x.
		+ Xf: Segundo valor de x.
		+ Yf: Segundo valor de y.
		_________________________________________________________________________
		
			OUTPUT:
		- Y: Valor interpolado de Y
		'''
		Y = Yi+((Yf-Yi)*((X-Xi)/(Xf-Xi)))

		return Y

	def NaNEl(self,V):
		'''
			DESCRIPTION:
		
		Con esta función se pretende interpolar los datos NaN que se encuentran
		cercanos y dejar los demás datos como NaN.
		_________________________________________________________________________

			INPUT:
		+ V: Variable que se quiere interpolar.
		_________________________________________________________________________
		
			OUTPUT:
		- VV: Datos con NaN interpolados.
		'''
		# En esta sección se eliminan o se interpolan los datos faltantes
		q = np.isnan(V)
		qq = np.where(q == True)[0]
		VV = V.copy() # Se crea la otra variable
		for ii,i in enumerate(qq):
			if ii == len(qq)-1:
				VV[i] = V[i] # Se dejan los datos NaN.
			else:
				if i == qq[ii+1]-1:
					VV[i] = V[i] # Se dejan los datos NaN.
				else:
					VV[i] = utl.Interp(1,V[i-1],2,3,V[i+1])

		return VV