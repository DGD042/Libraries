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
# En esta clase se tendran las funciones de análisis de datos espaciales
# y temporales, como el Filtro por Fourier y las Funciones Ortogonales 
# Empíricas, entre otros. Adicionalmente se tienen varios graficadores.
# ----------------------------------------------------------------------

import numpy as np
from scipy import linalg as la # Eigen valores
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
from scipy.fftpack import rfft, irfft, fftfreq # Paquete para utilizar las funciones

from UtilitiesDGD import UtilitiesDGD
utl = UtilitiesDGD()

class AnET:


	def __init__(self):

		'''
			DESCRIPTION:

		Este es el constructor por defecto, no realiza ninguna acción.
		'''

	def FT(self,f,t,dt=1):

		'''
			DESCRIPTION:

		Con esta función se obtiene la transformada de Fourier de unos datos.
		____________________________________________________________________________

			INPUT:
		+ f: Los datos a los que se les va a sacar la transformada de Fourier.
		+ t: La serie temporal utilizada para obtener las dfrecuencias.
		+ dt: el paso de tiempo determinado para la serie de tiempo.
		____________________________________________________________________________

			OUTPUT:
		- amplitud: La amplitud de los coeficientes.
		- potencia: la potencia asociada a los coeficientes.
		- total: La suma total de las potencias.
		- var: El porcentaje de varianza explicado por cada potencia.
		- fr: La frecuencia asociada al tiempo.
		- Period: Periodo asociado al tiempo.
		- A: Coeficientes encontrados con la FFT.
		'''
		# Se eliminan los datos faltantes
		# xx = np.where(f == np.isnan(f))
		# ff = f[xx[0]]

		A=np.fft.fft(f)
		fr=np.fft.fftfreq(len(t),dt)
		#if len(f) is not len(t):
		#	utl.ExitError('FT','AnET','Vectors A and fr must be the same length')
		amplitud=np.abs(A)
		potencia=np.abs(A)**2
		total=np.sum(potencia)
		var=potencia*100/total
		Period = 1/fr

		return amplitud, potencia, total, var, fr, Period, A

	def FTr(self,f,t,dt=1):
		'''
			DESCRIPTION:

		Con esta función se obtiene la transformada de Fourier de unos datos.
		____________________________________________________________________________

			INPUT:
		+ f: Los datos a los que se les va a sacar la transformada de Fourier.
		+ t: La serie temporal utilizada para obtener las dfrecuencias.
		+ dt: el paso de tiempo determinado para la serie de tiempo.
		____________________________________________________________________________
		
			OUTPUT:
		- amplitud: La amplitud de los coeficientes.
		- potencia: la potencia asociada a los coeficientes.
		- total: La suma total de las potencias.
		- var: El porcentaje de varianza explicado por cada potencia.
		- fr: La frecuencia asociada al tiempo.
		- Period: Periodo asociado al tiempo.
		- A: Coeficientes encontrados con la FFT.
		'''
		# Se eliminan los datos faltantes
		# xx = np.where(f == np.isnan(f))
		# ff = f[xx[0]]

		A=fftpack.rfft(f)
		fr=fftpack.rfftfreq(len(t),dt)
		#if len(f) is not len(t):
		#	utl.ExitError('FTr','AnET','Vectors f and fr must be the same length')
		amplitud=np.abs(A)
		potencia=np.abs(A)**2
		total=np.sum(potencia)
		var=potencia*100/total
		Period = 1/fr

		return amplitud, potencia, total, var, fr, Period, A

	def graphfft(self,t,amplitud,potencia,var,fr,Period,Pathimg,ii=1,i=1):


		'''
			DESCRIPTION:

		Con esta función se obtiene diferentes gráficas a partir de los datos de la
		Transformada de Fourier.

			INPUT:
		+ amplitud: La amplitud de los coeficientes.
		+ potencia: la potencia asociada a los coeficientes.
		+ total: La suma total de las potencias.
		+ var: El porcentaje de varianza explicado por cada potencia.
		+ fr: La frecuencia asociada al tiempo.
		+ Period: Periodo asociado al tiempo.
		+ A: Coeficientes encontrados con la FFT.

			OUTPUT:
		Gráficas de las diferentes variables.
		'''


		# Código para graficar las cosas por aparte
		plt.figure(figsize=(20,10))
		plt.plot(fr[:len(t)/2-1], amplitud[:len(t)/2-1], 'g-', linewidth = 2, label = u'Amplitud positiva')  # Dibujamos los valores de las parejas ordenadas con una línea contínua
		plt.plot(fr[len(t)/2:], amplitud[len(t)/2:], 'b-', linewidth = 2, label = u'Amplitud negativa')
		plt.title('Amplitud de la fft' )  # Colocamos el título del gráfico
		plt.xlabel(u'Frecuencia [Hz]')  # Colocamos la etiqueta en el eje x
		plt.ylabel('Amplitud')  # Colocamos la etiqueta en el eje y
		plt.legend(loc='best')
		#Guardar la imagen
		plt.savefig(Pathimg + 'fft(Amplitud)'+ str(ii) + '_' + str(i) + '.png')
		

		####gráfica de la frecuencia vs potencia
		plt.figure(figsize=(20,10))
		plt.plot(fr[:len(t)/2-1], potencia[:len(t)/2-1], 'k-', linewidth = 2, label = u'Potencia positiva')  # Dibujamos los valores de las parejas ordenadas con una línea contínua
		# plt.plot(fr[len(t)/2:], potencia[len(t)/2:], 'r-', linewidth = 2, label = u'Potencia negativa')
		plt.title('Potencia espectral vs Frecuencia' )  # Colocamos el título del gráfico
		plt.xlabel(u'Frecuencia [Hz]')  # Colocamos la etiqueta en el eje x
		plt.ylabel(u'Potencia espectral')  # Colocamos la etiqueta en el eje y
		#Guardar la imagen
		plt.savefig(Pathimg + 'Pot_fr'+ str(ii) + '_' + str(i) + '.png')

		####gráfica de la periodo vs potencia
		plt.figure(figsize=(20,10))
		plt.plot(Period[:len(t)/2-1], potencia[:len(t)/2-1], 'r-', linewidth = 2, label = u'Potencia positiva')  # Dibujamos los valores de las parejas ordenadas con una línea contínua
		# plt.plot(fr[len(t)/2:], potencia[len(t)/2:], 'r-', linewidth = 2, label = u'Potencia negativa')
		plt.title('Potencia espectral vs Periodo' )  # Colocamos el título del gráfico
		plt.xlabel(u'Periodo')  # Colocamos la etiqueta en el eje x
		plt.ylabel(u'Potencia espectral')  # Colocamos la etiqueta en el eje y
		#Guardar la imagen
		plt.savefig(Pathimg + 'Pot_Per'+ str(ii) + '_' + str(i) + '.png')
		

		####gráfica de la frecuencia vs varianza
		plt.figure(figsize=(20,10))
		# plt.plot(fr[:len(t)/2-1], var[:len(t)/2-1], 'g-', linewidth = 2, label = u'varianza')  # Dibujamos los valores de las parejas ordenadas con una línea contínua
		# plt.plot(fr[len(t)/2:], var[len(t)/2:], 'g-', linewidth = 2, label = u'varianza')
		plt.plot(fr[:len(t)/2-1], 2*var[:len(t)/2-1], '-', linewidth = 2, label = u'varianza')  # Dibujamos los valores de las parejas ordenadas con una línea contínua

		plt.title('Porcentaje de varianza explicado' )  # Colocamos el título del gráfico
		plt.xlabel(u'Frecuencia [Hz]')  # Colocamos la etiqueta en el eje x
		plt.ylabel(u'Porcentaje de varianza')  # Colocamos la etiqueta en el eje y
		#Guardar la imagen
		plt.savefig(Pathimg + 'Var'+ str(ii) + '_' + str(i) + '.png')

	def FilFFT(self,f,t,FiltHi,FiltHf,dt=1,flag=True,Pathimg='',x1=0,x2=50,V='Pres',tt='minutos',Ett=0,ii=1,ix='pos',DTT='5'):

		'''
			DESCRIPTION:

		Con esta función se obtiene un filtro cuadrado a partir del espectro de
		potencias encontrado con la transformada de Fourier. Lo que hace es tomar
		una hora de comienzo y de finalización y lo que se encuentre por fuera de
		las horas lo vuelve cero.

			INPUT:
		+ f: Los datos a los que se les va a sacar la transformada de Fourier.
		+ t: La serie temporal utilizada para obtener las dfrecuencias.
		+ Filt_Hi: Hora de inicio del filtro (esta asociado al Periodo NO a la frecuencia).
		+ Filt_Hf: Hora final del filtro (esta asociado al Periodo NO a la frecuencia).
		+ dt: el paso de tiempo determinado para la serie de tiempo.
		
		+ flag: Variable booleana que sirve para indicar si se desean gráficas o no.

		Las variables que se presentarán a continuación se darán solo si flag == True.

		+ Pathimg: Ruta para guardar las imágenes.
		+ x1: Primer valor del xlim.
		+ x2: Segundo valor del xlim.
		+ V: String de la variable que se está analizando .
		+ tt: Escala de tiempo que se desea graficar.
		+ Ett: Escala de tiempo que se desean que se pasen los datos para la graficación.
			0: Se dejan en el tiempo principal.
			1: Se pasan a Horas de minutos ó a minutos de segundos.
			2: Se pasan a Horas de escala cinco-minutal.
			3: Se pasan a Horas de escala quince-minutal.
		+ ii: El número de gráfica que se realizará.
		+ ix: Estación que se está analizando.
		+ DTT: Periodo en qué escala temporal.
		

			OUTPUT:
		- Pres_F: Resultado de los datos filtrados completos.
		- Pres_FF: Resultado de los datos filtrados, solamente los datos reales.
		- P_amp: La amplitud de los coeficientes.
		- P_p: la potencia asociada a los coeficientes.
		- P_t: La suma total de las potencias.
		- P_var: El porcentaje de varianza explicado por cada potencia.
		- P_fr: La frecuencia asociada al tiempo.
		- P_Per: Periodo asociado al tiempo.
		- P_A: Coeficientes encontrados con la FFT.
		'''
		# Se encuentra la transformada de Fourier
		P_amp, P_p, P_t, P_var, P_fr, P_Per, P_A = self.FT(f,t,1)
		# -----------------
		# Filtrado
		# -----------------

		# Los periodos de inicio y finalización
		Filt_Per = FiltHf;
		Filt_Peri = FiltHi;
		

		P_AA = P_A.copy() # Se copia el valor de los ceoficientes.

		# Se desarrolla el filtro cuadrado.
		for i in range(len(P_fr)):
		    if P_Per[i]>-Filt_Peri and P_Per[i]<Filt_Peri: 
		        P_AA[i]=0
		    if P_Per[i]<-Filt_Per or P_Per[i]>Filt_Per:
		    	P_AA[i]=0
		P_AA[0] = P_A[0]

		p_fil=np.abs(P_AA)**2

		# Se realiza la transformada inversa
		Pres_F = np.fft.ifft(P_AA)
		Pres_FF = Pres_F.real
		#Pres_FF = np.sqrt((Pres_F.real**2) + (Pres_F.imag**2))

		if flag:

			# Se halla el pico máximo
			x = np.where(P_p[:-1] == max(P_p[:-1]))

			

			# Se halla el pico máximo
			xx = np.where(p_fil[:-1] == max(p_fil[:-1]))

			
			if Ett == 0:

				# Pico máximo
				PM = round(abs(P_Per[x][0]),2)
				# Pico máximo asociado a las 4 horas
				P4 = round(abs(P_Per[xx][0]),2)

			elif Ett == 1:

				# Pico máximo
				PM = round(abs(P_Per[x][0])/60,2)
				# Pico máximo asociado a las 4 horas
				P4 = round(abs(P_Per[xx][0])/60,2)

			elif Ett ==2:

				# Pico máximo
				PM = round(abs(P_Per[x][0])*5/60,2)
				# Pico máximo asociado a las 4 horas
				P4 = round(abs(P_Per[xx][0])*5/60,2)

			elif Ett ==3:

				# Pico máximo
				PM = round(abs(P_Per[x][0])*15/60,2)
				# Pico máximo asociado a las 4 horas
				P4 = round(abs(P_Per[xx][0])*15/60,2)

			else:

				utl.ExitError('FiltFFT','AnET','Change in the time scale not available yet')

			# Formato de las gráficas
			AFon = 18; Tit = 20; Axl = 16

			F = plt.figure(figsize=(15,10))
			plt.plot(P_Per[:len(t)/2-1], P_p[:len(t)/2-1], '-', lw = 1.5)
			plt.plot(abs(P_Per[x]),P_p[x],'ro', ms=10)
			plt.text(50, P_p[x][0], r'Periodo $\sim$ %s %s' %(PM,tt),fontsize=AFon)
			plt.title('Potencia espectral',fontsize=Tit )  # Colocamos el título del gráfico
			plt.xlabel(u'Periodo [cada '+ DTT +' min]',fontsize=AFon)  # Colocamos la etiqueta en el eje x
			plt.ylabel('Potencia espectral',fontsize=AFon)  # Colocamos la etiqueta en el eje y
			plt.savefig(Pathimg + V +'_ET_' + str(ii)+ '_' + ix + '.png')


			

			# gráfica del esprectro filtrado
			plt.figure(figsize=(15,10))
			plt.rcParams.update({'font.size': Axl})
			plt.plot(P_Per[:len(t)/2-1], p_fil[:len(t)/2-1], '-', lw = 1.5)  # Dibujamos los valores de las parejas ordenadas con una línea contínua
			#plt.plot(P_Per[len(t)/2:], p_fil[len(t)/2:], '-', linewidth = 2)  # Dibujamos los valores de las parejas ordenadas con una línea contínua
			plt.plot(abs(P_Per[x]),p_fil[x],'ro', ms=10)
			#plt.text(abs(P_Per[x][0]-18), p_fil[x][0], r'Periodo \sim %s horas' %(P4))
			plt.text(28, p_fil[x][0], r'Periodo $\sim$ %s horas' %(P4),fontsize=AFon)
			plt.title('Potencia espectral filtrada',fontsize=Tit )  # Colocamos el título del gráfico
			plt.xlabel(u'Periodo [cada '+ DTT +' min]',fontsize=AFon)  # Colocamos la etiqueta en el eje x
			plt.ylabel('Potencia espectral',fontsize=AFon)  # Colocamos la etiqueta en el eje y

			# plt.legend(loc='best')
			plt.xlim(x1, x2)
			# Guardar la imagen
			plt.savefig(Pathimg  + ix + '_' +V +'_filt(%s_h)_' %(int(FiltHf)) + str(ii) + '.png' )
			plt.close('all')


			fig, axs = plt.subplots(1,2, figsize=(15, 8), facecolor='w', edgecolor='k')
			plt.rcParams.update({'font.size': Axl})
			axs = axs.ravel() # Para hacer un loop con los subplots

			axs[0].plot(P_Per[:len(t)/2-1], P_p[:len(t)/2-1], '-', lw = 1.5)
			axs[0].set_title('Potencia espectral sin filtrar',fontsize=Tit )
			axs[0].set_xlabel(u'Periodo [cada '+ DTT +' min]',fontsize=AFon)
			axs[0].set_ylabel('Potencia espectral',fontsize=AFon)
			axs[0].set_xlim([1,400])

			axs[1].plot(P_Per[:len(t)/2-1], p_fil[:len(t)/2-1], '-', lw = 1.5)
			axs[1].set_title('Potencia espectral filtrada',fontsize=Tit )
			axs[1].set_xlabel(u'Periodo [cada '+ DTT +' min]',fontsize=AFon)
			#axs[1].set_ylabel('Potencia espectral',fontsize=AFon)
			axs[1].set_xlim([1,400])

			plt.savefig(Pathimg+'Comp/'+ ix + '_' +V +'_filt(%s-%s_m)_' %(int(FiltHi),int(FiltHf)) + str(ii) + '.png' )
			plt.close('all')



		return Pres_F, P_amp, P_p, P_t, P_var, P_fr, P_Per, P_A 

	def FilFFTr(self,f,t,FiltHi,FiltHf,dt=1,flag=True,Pathimg='',x1=0,x2=50,V='Pres',tt='minutos',Ett=0,ii=1,ix='pos'):

		'''
			DESCRIPTION:

		Con esta función se obtiene un filtro cuadrado a partir del espectro de
		potencias encontrado con la transformada de Fourier. Lo que hace es tomar
		una hora de comienzo y de finalización y lo que se encuentre por fuera de
		las horas lo vuelve cero.

			INPUT:
		+ f: Los datos a los que se les va a sacar la transformada de Fourier.
		+ t: La serie temporal utilizada para obtener las dfrecuencias.
		+ Filt_Hi: Hora de inicio del filtro (esta asociado al Periodo NO a la frecuencia).
		+ Filt_Hf: Hora final del filtro (esta asociado al Periodo NO a la frecuencia).
		+ dt: el paso de tiempo determinado para la serie de tiempo.
		
		+ flag: Variable booleana que sirve para indicar si se desean gráficas o no.

		Las variables que se presentarán a continuación se darán solo si flag == True.

		+ Pathimg: Ruta para guardar las imágenes.
		+ x1: Primer valor del xlim.
		+ x2: Segundo valor del xlim.
		+ V: String de la variable que se está analizando .
		+ tt: Escala de tiempo que se desea graficar.
		+ Ett: Escala de tiempo que se desean que se pasen los datos para la graficación.
			0: Se dejan en el tiempo principal.
			1: Se pasan a Horas de minutos ó a minutos de segundos.
			2: Se pasan a Horas de escala cinco-minutal.
			3: Se pasan a Horas de escala quince-minutal.
		+ ii: El número de gráfica que se realizará.
		+ ix: Estación que se está analizando.
		

			OUTPUT:
		- Pres_F: Resultado de los datos filtrados completos.
		- Pres_FF: Resultado de los datos filtrados, solamente los datos reales.
		- P_amp: La amplitud de los coeficientes.
		- P_p: la potencia asociada a los coeficientes.
		- P_t: La suma total de las potencias.
		- P_var: El porcentaje de varianza explicado por cada potencia.
		- P_fr: La frecuencia asociada al tiempo.
		- P_Per: Periodo asociado al tiempo.
		- P_A: Coeficientes encontrados con la FFT.
		'''
		# Se encuentra la transformada de Fourier
		P_amp, P_p, P_t, P_var, P_fr, P_Per, P_A = self.FTr(f,t,1)
		# -----------------
		# Filtrado
		# -----------------

		# Los periodos de inicio y finalización
		Filt_Per = FiltHf;
		Filt_Peri = FiltHi;
		

		P_AA = P_A.copy() # Se copia el valor de los ceoficientes.

		# Se desarrolla el filtro cuadrado.
		for i in range(len(P_fr)):
		    if P_Per[i]>-Filt_Peri and P_Per[i]<Filt_Peri: 
		        P_AA[i]=0
		    if P_Per[i]<-Filt_Per or P_Per[i]>Filt_Per:
		    	P_AA[i]=0


		p_fil=np.abs(P_AA)**2

		# Se realiza la transformada inversa
		Pres_F = fftpack.irfft(P_AA)
		Pres_FF = Pres_F.real
		#Pres_FF = np.sqrt((Pres_F.real**2) + (Pres_F.imag**2))

		if flag:

			# Se halla el pico máximo
			x = np.where(P_p[:-1] == max(P_p[:-1]))

			

			# Se halla el pico máximo
			xx = np.where(p_fil[:-1] == max(p_fil[:-1]))

			
			if Ett == 0:

				# Pico máximo
				PM = round(abs(P_Per[x][0]),2)
				# Pico máximo asociado a las 4 horas
				P4 = round(abs(P_Per[xx][0]),2)

			elif Ett == 1:

				# Pico máximo
				PM = round(abs(P_Per[x][0])/60,2)
				# Pico máximo asociado a las 4 horas
				P4 = round(abs(P_Per[xx][0])/60,2)

			elif Ett ==2:

				# Pico máximo
				PM = round(abs(P_Per[x][0])*5/60,2)
				# Pico máximo asociado a las 4 horas
				P4 = round(abs(P_Per[xx][0])*5/60,2)

			elif Ett ==3:

				# Pico máximo
				PM = round(abs(P_Per[x][0])*15/60,2)
				# Pico máximo asociado a las 4 horas
				P4 = round(abs(P_Per[xx][0])*15/60,2)

			else:

				utl.ExitError('FiltFFT','AnET','Change in the time scale not available yet')

			# Formato de las gráficas
			AFon = 18; Tit = 20; Axl = 16

			F = plt.figure(figsize=(15,10))
			plt.plot(P_Per[:len(t)/2-1], P_p[:len(t)/2-1], '-', lw = 1.5)
			plt.plot(abs(P_Per[x]),P_p[x],'ro', ms=10)
			plt.text(50, P_p[x][0], r'Periodo $\sim$ %s %s' %(PM,tt),fontsize=AFon)
			plt.title('Potencia espectral',fontsize=Tit )  # Colocamos el título del gráfico
			plt.xlabel(u'Periodo',fontsize=AFon)  # Colocamos la etiqueta en el eje x
			plt.ylabel('Potencia espectral',fontsize=AFon)  # Colocamos la etiqueta en el eje y
			plt.savefig(Pathimg + V +'_ET_' + str(ii)+ '_' + ix + '.png')


			

			# gráfica del esprectro filtrado
			plt.figure(figsize=(15,10))
			plt.rcParams.update({'font.size': Axl})
			plt.plot(P_Per[:len(t)/2-1], p_fil[:len(t)/2-1], '-', lw = 1.5)  # Dibujamos los valores de las parejas ordenadas con una línea contínua
			#plt.plot(P_Per[len(t)/2:], p_fil[len(t)/2:], '-', linewidth = 2)  # Dibujamos los valores de las parejas ordenadas con una línea contínua
			plt.plot(abs(P_Per[x]),p_fil[x],'ro', ms=10)
			#plt.text(abs(P_Per[x][0]-18), p_fil[x][0], r'Periodo \sim %s horas' %(P4))
			plt.text(28, p_fil[x][0], r'Periodo $\sim$ %s horas' %(P4),fontsize=AFon)
			plt.title('Potencia espectral filtrada',fontsize=Tit )  # Colocamos el título del gráfico
			plt.xlabel(u'Periodo [5 min]',fontsize=AFon)  # Colocamos la etiqueta en el eje x
			plt.ylabel('Potencia espectral',fontsize=AFon)  # Colocamos la etiqueta en el eje y

			# plt.legend(loc='best')
			plt.xlim(x1, x2)
			# Guardar la imagen
			plt.savefig(Pathimg  + ix + '_' +V +'_filt(%s_h)_' %(FiltHf) + str(ii) + '.png' )
			plt.close('all')


		return Pres_F, Pres_FF, P_amp, P_p, P_t, P_var, P_fr, P_Per, P_A 	

	def EOF(self,M,xll,iss=0,i=0,flag=False,Pathimg='',ii=1):
		
		'''
			DESCRIPTION:

		Con esta función Funciones Ortogonales Empíricas (EOF) -> Buscar bibliografía

			INPUT:
		+ M: Matriz para aplicar EOFs.
		+ iss: Indicativo para hacer covarianza o correlación.
			0: Covarianza.
			1: Correlación.
		+ i: Indicador de cambio de signo del EOF.
			0: Encontrado.
			1: Inverso.
		+ xll: Limite superior de la gráfica de varianza explicada
		
		+ flag: Variable booleana que sirve para indicar si se desean gráficas o no.

		Las variables que se presentarán a continuación se darán solo si flag == True.

		+ Pathimg: Ruta para guardar las imágenes.
		+ ii: El número de gráfica que se realizará.

			OUTPUT:
		- Corr: Matriz de covarianza o de correlación encontrada
		- e_vals: Eigen valores.
		- e_vals: Eigen vectores.
		- sum_evals: Suma de los eigen valores.
		- var_exp: Porcentaje de varianza explicado en cada 
		- pc_mat: Componentes principales.
		'''

		if iss == 0:
			Corr = np.dot(M,M.T)
		elif iss == 1:
			Corr = np.corrcoef(M)
		else:
			utl.ExitError('EOF','AnET','Not a matrix generation option')

		# se obtienen los eigen vectores y eigen valores
		e_vals, e_vecs = la.eig(Corr)

		# Indicativo para saber si se cambia 
		if i==0:
			e_vecs = e_vecs
		elif i == 1:
			e_vecs = -e_vecs
		else:
			utl.ExitError('EOF','AnET','Not an inversion option')


		x = np.argsort(e_vals)[::-1]
		e_vecs = e_vecs[:,x]
		e_vals = np.sort(e_vals)[::-1]
		

		sum_evals = np.sum(e_vals)
		var_exp = (e_vals / sum_evals) * 100

		# Componente principal
		pc_mat = np.dot(e_vecs.T,M)

		if flag:
			# Gráfica Varianza Explicada
			plt.figure(figsize=(20,10))
			plt.rcParams.update({'font.size': 20})
			plt.plot(range(1,len(var_exp)+1),var_exp.real)
			plt.title('Porcentaje de varianza explicado',fontsize=26)  # Colocamos el título del gráfico
			plt.xlabel(u'Componentes',fontsize=24)  # Colocamos la etiqueta en el eje x
			plt.ylabel(u'Porcentaje de varianza',fontsize=24)
			plt.xlim(0,xll)
			plt.savefig(Pathimg + "var_exp"+ str(ii) + ".png")
			plt.close('all')


		return Corr, e_vals, e_vecs, sum_evals, var_exp, pc_mat

	def EEOF(self,M,xll,iss=0,ip=0,flag=True,Pathimg='',ii=1):
		
		'''
			DESCRIPTION:

		Con esta función Funciones Ortogonales Empíricas Extendidas (EEOF) 
		-> Buscar bibliografía

			INPUT:
		+ M: Matriz para aplicar EOFs.
		+ iss: Indicativo para hacer covarianza o correlación.
			0: Covarianza.
			1: Correlación.
		+ i: Indicador de cambio de signo del EOF.
			0: Encontrado.
			1: Inverso.
		+ xll: Limite superior de la gráfica de varianza explicada
		
		+ flag: Variable booleana que sirve para indicar si se desean gráficas o no.

		Las variables que se presentarán a continuación se darán solo si flag == True.

		+ Pathimg: Ruta para guardar las imágenes.
		+ ii: El número de gráfica que se realizará.

			OUTPUT:
		- Corr: Matriz de covarianza o de correlación encontrada
		- e_vals: Eigen valores.
		- e_vals: Eigen vectores.
		- sum_evals: Suma de los eigen valores.
		- var_exp: Porcentaje de varianza explicado en cada 
		- pc_mat: Componentes principales.
		'''

		# Primero se encuentra la matriz para la serie normal
		if iss == 0:
			Corr1 = np.dot(M,M.T)
		elif iss == 1:
			Corr1 = np.corrcoef(M)
		else:
			utl.ExitError('EOF','AnET','Not a matrix generation option')

		# Se obtiene la matriz para un rezago
		Corr2 = np.zeros((len(M),len(M)))
		for i in range(0,len(M)):
			for j in range(0,len(M)):
				MM = np.vstack((M[i,:M.shape[1]-1],M[j,1:M.shape[1]]))
				if iss == 0:
					Corr2[i,j] = np.dot(MM,MM.T)[0,1]
				elif iss == 1:
					Corr2[i,j] = np.corrcoef(MM)[0,1]
				else:
					utl.ExitError('EOF','AnET','Not a matrix generation option')

		# Se arma la matriz final
		Corr11 = np.c_[Corr1,Corr2]
		Corr12 = np.c_[Corr2,Corr1]
		Corr = np.vstack((Corr11,Corr12))
					

		# se obtienen los eigen vectores y eigen valores
		e_vals, e_vecs = la.eig(Corr)

		# Indicativo para saber si se cambia 
		if ip==0:
			e_vecs = e_vecs
		elif ip == 1:
			e_vecs = -e_vecs
		else:
			utl.ExitError('EOF','AnET','Not an inversion option')

		sum_evals = np.sum(e_vals)
		var_exp = (e_vals / sum_evals) * 100

		# Componente principal
		pc_mat = np.dot(e_vecs[:len(M),:len(M)],M)
		pc_matR = np.dot(e_vecs[len(M):,len(M):],M)

		if flag:
			# Gráfica Varianza Explicada
			plt.figure(figsize=(20,10))
			plt.plot(range(1,len(var_exp)+1),var_exp)
			plt.title('Porcentaje de varianza explicado' )  # Colocamos el título del gráfico
			plt.xlabel(u'Componentes')  # Colocamos la etiqueta en el eje x
			plt.ylabel(u'Porcentaje de varianza')
			plt.xlim(0,xll)
			plt.savefig(Pathimg + "var_exp"+ str(ii) + ".png")
			plt.close('all')


		return Corr, e_vals, e_vecs, sum_evals, var_exp, pc_mat,pc_matR

	def SFTr(self,f,t):
		'''
			DESCRIPTION:

		Con esta función se obtiene la transformada de Fourier de unos datos utilizando
		el paquete de scipy.
		____________________________________________________________________________

			INPUT:
		+ f: Los datos a los que se les va a sacar la transformada de Fourier.
		+ t: La serie temporal utilizada para obtener las dfrecuencias.
		____________________________________________________________________________
		
			OUTPUT:
		- amplitud: La amplitud de los coeficientes.
		- potencia: la potencia asociada a los coeficientes.
		- total: La suma total de las potencias.
		- var: El porcentaje de varianza explicado por cada potencia.
		- W: La frecuencia asociada al tiempo.
		- Period: Periodo asociado al tiempo.
		- A: Coeficientes encontrados con la FFT.
		'''
		W = fftfreq(f.size, d=t[1]-t[0]) # Frecuencia
		f_signal = rfft(f) # transformada de Fourier

		Period = 1/W # Se encuentran 

		amplitud=np.abs(f_signal)
		potencia=np.abs(f_signal)**2
		total=np.sum(potencia)
		var=potencia*100/total

		return amplitud, potencia, total, var, W, Period, f_signal


	def SFilFFTr(self,f,t,Filt_Hi,Filt_Hf):

		'''
			DESCRIPTION:

		Con esta función se obtiene un filtro cuadrado a partir del espectro de
		potencias encontrado con la transformada de Fourier de scipy. Lo que hace 
		es tomar una hora de comienzo y de finalización y lo que se encuentre por 
		fuera de las horas lo vuelve cero.
		____________________________________________________________________________

			INPUT:
		+ f: Los datos a los que se les va a sacar la transformada de Fourier.
		+ t: La serie temporal utilizada para obtener las dfrecuencias.
		+ Filt_Hi: Hora de inicio del filtro (esta asociado al Periodo NO a la frecuencia).
		+ Filt_Hf: Hora final del filtro (esta asociado al Periodo NO a la frecuencia).		

			OUTPUT:
		- cur_signal: Resultado de los datos filtrados completos.
		- P_amp: La amplitud de los coeficientes.
		- P_p: la potencia asociada a los coeficientes.
		- P_t: La suma total de las potencias.
		- P_var: El porcentaje de varianza explicado por cada potencia.
		- P_fr: La frecuencia asociada al tiempo.
		- P_Per: Periodo asociado al tiempo.
		- P_A: Coeficientes encontrados con la FFT.
		'''
		# Se realiza la transformada de Fourier
		P_amp, P_p, P_t, P_var, W, P_Per, f_signal = self.SFTr(f,t)
		# -----------------------
		# Se inicial el filtrado
		# -----------------------
		# Bandas de filtrado
		lowcut = 1/Filt_Hi
		highcut = 1/Filt_Hf

		# Se obtiene la serie que se cortará
		cut_f_signal = f_signal.copy()
		# Se realiza el cortado de la serie
		for j,i in enumerate(W):
			if i < highcut:
				cut_f_signal[j] = 0
			elif i > lowcut:
				cut_f_signal[j] = 0
		cut_f_signal[0:1] = f_signal[0:1] # Se escala a los valores de la serie original
		cut_signal = irfft(cut_f_signal) # Se encuentra la serie ya filtrada

		return cut_signal, P_amp, P_p, P_t, P_var, W, P_Per, f_signal
