# -*- coding: utf-8 -*-

#______________________________________________________________________________
#______________________________________________________________________________
#
#						Coded by Daniel González Duque
#						    Last revised 04/03/2016
#______________________________________________________________________________
#______________________________________________________________________________

# ----------------------------------------------------------------------
# 	DESCRIPCIÓN DE LA CLASE:
# En esta clase se tendran las funciones de análisis de correlación entre
# series de datos.
# ----------------------------------------------------------------------

import numpy as np
from scipy import linalg as la # Eigen valores
import matplotlib.pyplot as plt
from scipy import stats as st

from UtilitiesDGD import UtilitiesDGD
utl = UtilitiesDGD()

class CorrSt:


	def __init__(self):

		'''
			DESCRIPTION:

		Este es el constructor por defecto, no realiza ninguna acción.
		'''


	def CorrC(self,T1,T2,FlagC=True,Met=0,alpha=0.05):
		'''
			DESCRIPTION:
		
		Con esta función se obtienen las correlaciones de Pearson y Spearman.

			INPUT:
		+ T1: Primera serie de datos.
		+ T2: Segunda serie de datos.
		+ Met: Método de Boot-straping.
			0: Solamente reordenando una serie.
			1: Reordenando las dos series.
		+ alpha: Nivel de significancia
		
			OUTPUT:
		- CCP: Correlación de Pearson.
		- CCS: Correlación de Spearman.
		- QQ: Significancia estadística.
		'''

		#if len(T1) is not len(T2):
		#	utl.ExitError('FT','CorrC','Vectors T1 and T2 must be the same length')


		# Se obtiene la correlación
		CCP = st.pearsonr(T1,T2)[0] # Pearson
		CCS = st.spearmanr(T1,T2)[0] # Spearman
		if FlagC == True:

			Val1,Rand1 = np.histogram(T1); [float(i) for i in Val1]
			Val1 = Val1/float(Val1.sum())

			Val2,Rand2 = np.histogram(T2); [float(i) for i in Val2]
			Val2 = Val2/float(Val2.sum())
			
			TRes1 = np.zeros((1000,len(T1)))
			TRes2 = np.zeros((1000,len(T2)))
			CCP1 = np.zeros((1000))
			CCS1 = np.zeros((1000))

			for i in range(1000):
				b,TRes1[i,:] = self.rndval(T1,Val1,Rand1)
				b,TRes2[i,:] = self.rndval(T2,Val2,Rand2)
				CCP1[i] = st.pearsonr(TRes1[i,:],TRes2[i,:])[0]
				CCS1[i] = st.spearmanr(TRes1[i,:],TRes2[i,:])[0]

			QQ = np.percentile(CCP1,1-alpha)
			QQ = np.hstack((QQ,np.percentile(CCS1,1-alpha)))

			# Se encuentra la significacia estadística con boots-strapping
			TRes1 = np.zeros((1000,len(T1)))
			TRes2 = np.zeros((1000,len(T2)))
			CCP1 = np.zeros((1000))
			CCS1 = np.zeros((1000))

			for i in range(1000):
				if Met == 0:
					TRes1[i,:], TRes2[i,:] = utl.BTSS(T1,T2)
				elif Met == 1:
					TRes1[i,:] = utl.BTS(T1)
					TRes2[i,:] = utl.BTS(T2)
				else:
					utl.ExitError('FT','CorrC','Not a method written')
				# Correlations
				CCP1[i] = st.pearsonr(TRes1[i,:],TRes2[i,:])[0]
				CCS1[i] = st.spearmanr(TRes1[i,:],TRes2[i,:])[0]

			QQ = np.hstack((QQ,np.percentile(CCP1,1-alpha)))
			QQ = np.hstack((QQ,np.percentile(CCS1,1-alpha)))
			if CCP >= 0:
				QQ = abs(QQ)

			return CCP,CCS,QQ
		else:
			return CCP,CCS

	def rndval(self,DT,Val,Rand):
		'''
			DESCRIPTION:
		
		Con esta función se obtienen datos aleatorios de una misma
		distribución.

			INPUT:
		+ DT: Variable que se quiere replicar.
		+ Val: Valores del histograma
		+ Rand: Valores de los Bins
		
			OUTPUT:
		- CCP: Correlación de Pearson.
		- CCS: Correlación de Spearman.
		- QQ: Significancia estadística.
		'''
		L = len(DT)
		r = np.random.uniform(0,1,L)
		N = len(Val)
		P=1
		P2=0
		c = np.zeros((N))
		b = np.zeros((N))
		Res = np.zeros((L))
		for i in range(N):
			c[i]=np.sum(Val[0:i+1])
			if i == 0:
				b[i]=sum(r<=c[i])
				q = np.where(r<=c[i])[0]
				if b[i] ==0:
					Re=0
				else:
					Re = np.random.uniform(0,1,b[i])*(Rand[i+1]-Rand[i])+Rand[i]
			else:
				b[i] = -sum(r <= c[i-1])+sum(r<=c[i])
				q = np.where((r>=c[i-1])&(r<=c[i]))[0]
				if b[i] ==0:
					Re=0
				else:
					Re = np.random.uniform(0,1,b[i])*(Rand[i+1]-Rand[i])+Rand[i]

			#P1 = len(Re)
			#P2 = P2+P1

			Res[q] = Re
			#P = P2+1

		return b,Res

		