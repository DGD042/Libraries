# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 08/05/2018
#______________________________________________________________________________
#______________________________________________________________________________

# ----------------------------
# Se importan las librerias
# ----------------------------
# Manejo de datos
import numpy as np
# Importar datos
from scipy import stats as st
# Se importan los paquetes para manejo de fechas
from datetime import date, datetime, timedelta

# ----------------------------
# Librerías personales
# ----------------------------

def Zeq(z,hs): 
    '''
    DESCRIPTION:

        Dimensionless variable for height, extracted from Makarieva et al. 
        (2015).
    _________________________________________________________________________

    INPUT:
        :param z: height in km.
        :param hs: height.
    _________________________________________________________________________

    OUTPUT:
        :return Z: Dimensionless height. 
    '''
    Z = z/hs
    return Z

def zeq(Z,hs): 
    '''
    DESCRIPTION:

        Height, extracted from Makarieva et al. 
        (2015).
    _________________________________________________________________________

    INPUT:
        :param Z: Dimensionless height.
        :param hs: height.
    _________________________________________________________________________

    OUTPUT:
        :return z: Height in km. 
    '''
    z = Z/hs
    return z

def ceq(rho,rhog=34): 
    '''
    DESCRIPTION:

        Dimensionless variable for lapse rate, extracted from Makarieva et al. 
        (2015).
    _________________________________________________________________________

    INPUT:
        :param rho: Adiabatic lapse rate in K/km.
        :param rhog: autoconvective lapse rate in K/km.
    _________________________________________________________________________

    OUTPUT:
        :return c: Dimensionless lapse rate.
    '''
    c = rho/rhog
    return c

def hseq(Ts,rhog=34): 
    '''
    DESCRIPTION:

        Variable for height, extracted from Makarieva et al. (2015).
    _________________________________________________________________________

    INPUT:
        :param Ts: surface temperature in K.
        :param rhog: autoconvective lapse rate in K/km.
    _________________________________________________________________________

    OUTPUT:
        :return hs: height in km.
    '''
    hs = Ts/rhog
    return hs

def Taeq(Ts,c,Z): 
    '''
    DESCRIPTION:

        Variable for height, extracted from Makarieva et al. (2015).
    _________________________________________________________________________

    INPUT:
        :param Ts: surface temperature in K.
        :param c: Dimensionless lapse rate.
        :param Z: Dimensionless height.
    _________________________________________________________________________

    OUTPUT:
        :return Ta: Mean temperature of the atmopsheric column below Z in K.
    '''
    Ta = Ts*(1-c*(1-(Z/(np.exp(Z)-1))))
    return Ta

def dTaeq(Ta,drho,dTs,Ts,Z,rhog=34): 
    '''
    DESCRIPTION:

        Perturbation of mean temp, extracted from Makarieva et al. (2015).
    _________________________________________________________________________

    INPUT:
        :param Ta: Mean temperature of the atmopsheric column below Z in K.
        :param drho: Perturbation of lapse rate in K/km.
        :param dTs: Perturbation of surface temperature in K.
        :param Ts: Surface temperature in K.
        :param Z: Dimensionless height.
        :param rhog: autoconvective lapse rate in K/km.
    _________________________________________________________________________

    OUTPUT:
        :return Ta: Perturbation of mean temperature of the atmopsheric 
        column below Z in K.
    '''
    dc = drho/rhog
    db = dTs/Ts
    dTa = (((1-(Z/(np.exp(Z)-1)))*dc)-db)*-Ta
    return dTa

def deq(dV,V): 
    '''
    DESCRIPTION:

        Dimensionless diferential, extracted from Makarieva et al. (2015).
    _________________________________________________________________________

    INPUT:
        :param dV: Perturbation of variable V.
        :param V: Mean value of Variable V.
    _________________________________________________________________________

    OUTPUT:
        :return d: Dimensionless diferential.
    '''
    d = dV/V
    return d

def dpseq(dTa,Ta,AirDen,ze=6,g=9.8):
    '''
    DESCRIPTION:

        Perturbation of surface pressure from temperature, extracted from 
        Makarieva et al. (2015).
    _________________________________________________________________________

    INPUT:
        :param dTa: Perturbation in mean temperature in K
        :param Ta: Mean temperature of the atmopsheric column below Z in K.
        :param AirDen: Air density in kg/m3.
        :param ze: isobaric height in km.
        :param g: Gravity constant in m/s2
    _________________________________________________________________________

    OUTPUT:
        :return dp: Perturbations in surface pressure in hPa.
    '''
    dps = -AirDen*g*(ze/Ta)*dTa
    return dps

def Zieq(hs,dTs,drho):
    '''
    DESCRIPTION:

        Dimensionless isothermal height, extracted from Makarieva 
        et al. (2015).
    _________________________________________________________________________

    INPUT:
        :param hs: height ...
        :param dTs: Perturbation of surface temperature in K.
        :param drho: Perturbation of lapse rate in K/km.
    _________________________________________________________________________

    OUTPUT:
        :return Zi: Isothermal height in km.
    '''
    Zi = (1/hs)*(dTs/drho)
    return Zi

def Zeeq(Zi,dPs,Ps,dTs,Ts,drho,Oper='min'):
    '''
    DESCRIPTION:

        Dimensionless isobaric height, extracted from Makarieva 
        et al. (2015).
    _________________________________________________________________________

    INPUT:
        :param Zi: Dimensionless isothermal height.
        :param Ps: Surface pressure mean in hPa.
        :param dPs: Perturbation of surface pressure in hPa.
        :param dTs: Perturbation of surface temperature in K.
        :param Ts: Surface temperature in K.
    _________________________________________________________________________

    OUTPUT:
        :return Zi: Isothermal height in km.
    '''
    da = dPs/Ps
    db = dTs/Ts
    if Oper=='min':
        Ze = Zi*(1-np.sqrt(1+(2*da)/Zi*db))
    elif Oper == 'max':
        Ze = Zi*(1+np.sqrt(1+(2*da)/Zi*db))
    return Ze
