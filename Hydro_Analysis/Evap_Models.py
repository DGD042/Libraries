# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 21/02/2017
#______________________________________________________________________________
#______________________________________________________________________________

#______________________________________________________________________________
#
# DESCRIPCIÓN DE LA CLASE:
#   En esta clase se contendrán varias funciones para el cálculo de la
#   evapotranspiración, tanto potencial como real. Puede calcular valores
#   y campos con cada una de las variables.
#
#   Las ecuaciones fueron extraídas de Vélez, Poveda y Mesa (2000), excepto 
#   la ecuación de FAO Penman-Monteith que es de Allen, Pereira, Raes y 
#   Smith (1998).
#
#   ----------------------
#   Ecuaciones contenidas
#   ----------------------
#
#   La clase contiene las siguientes ecuaciones para calcular la
#   evapotranspiración potencial (PET):
#
#   - Ecuación de Thronthwaite.
#   - Ecuación de García y López.
#   - Ecuación de Truc Modificado.
#   - Ecuación de FAO Penman-Monteith.
#
#   Y las siguientes ecuaciones para calcular la evapotranspiración
#   real (AET):
#
#   - Ecuación de Penman-Monteith.
#   - Ecuación de Budyko
#
#   Las variables que necesita cada ecuación se incluyen en la descripción de cada
#   una.
#   ____________________________________
#   
#______________________________________________________________________________

# Se importan los paquetes
import numpy as np


class Evap_Models:

    def __init__(self):

        '''
            DESCRIPTION:

        This is a build up function.
        '''

    def EqThornthwaite(self,TMonthly,TAnnual):
        '''
            DESCRIPTION:
        
        This function calculates the PET with the Thornthwaite equation.
        _________________________________________________________________________

            INPUT:
        + TMonthly: Monthly temperature.
        + TAnnual: Annual Temperature.
        _________________________________________________________________________
        
            OUTPUT:
        - PET: Potential Evaportanspiration [mm/mes].
        '''
        I = 12*((TAnnual/5)**1.514)
        a = (675*10**(-9))*I**3-(771*10**(-7))*I**2+(179*10**(-4))*I+0.492
        PET = 1.6*(10*(TMonthly/I))**a*10
        return PET
    
    def EqGarciaLopez(self,Td,HRd):
        '''
            DESCRIPTION:
        
        This function calculates the PET with the García y López equation.
        _________________________________________________________________________

            INPUT:
        + Td: Daily temperature.
        + HRd: Daily Relative Humidity.
        _________________________________________________________________________
        
            OUTPUT:
        - PET: Potential Evaportanspiration [mm/day].
        '''
        n = (7.45*Td)/(234.7+Td)
        PET = 1.21*10**n*(1-0.01*HRd)+0.21*Td-2.3
        return PET

    def EqTurcM(self,T,HR,Rs,Timescale=0,Month=30,Map=False):
        '''
            DESCRIPTION:
        
        This function calculates the PET with the Turc modified equation.
        _________________________________________________________________________

            INPUT:
        + T: Daily or Monthly Temperature.
        + HR: Daily or Monthly Relative Humidity.
        + Rs: Mean daily or Monthly Solar Radiation.
        + Timescale: Time scale of the equaton. 0: Daily, 1: Monthly
        + Map: True: For several values of HR and T. False: For 1 value of HR and T.
        _________________________________________________________________________
        
            OUTPUT:
        - PET: Potential Evaportanspiration [mm/day].
        '''
        if Timescale == 0:
            K = 0.013
        elif Timescale == 1:
            if Month == 30 or Month == 31:
                K = 0.40
            else:
                K = 0.37
        if Map:
            x = np.where(HR < 50)
            PET = K*(T/(T+15))*(Rs+50)
            try:
                PET[x] = K*(T/(T+15))*(Rs+50)*(1+((50-HR)/70))
            except:
                del(x)
        else:
            if HR >= 50:
                PET = K*(T/(T+15))*(Rs+50)
            else:
                PET = K*(T/(T+15))*(Rs+50)*(1+((50-HR)/70))

        return PET

    def EqFAOPenmanM(self,T,HR,uh,h,p=0,Tmin=0,e_aq=0,Rn=0,Rs=0,J=1,PhiD=0,G=0):
        '''
            DESCRIPTION:
        
        This function calculates the PET with the FAO Penman-Monteith equation,
        and the AET with the Penman-Monteith equation. 

        This procedure was implemented from Zotarelli, Dukes, Romero, Migliaccio
        and Kelly (2013).
        _________________________________________________________________________

            INPUT:
        + T: Daily temperature.
        + HR: Daily Relative Humidity.
        + uh: Daily wind.
        + h: Height.
        + p: Daily pressure.
        + Tmin: Minimum temperature.
        + e_aq: Vapor pressure equation. 0: for equation 1 and 1 for equation 2.
        + Rn: Net radiation.
        + Rs: Mean daily solar radiation.
        + J: Julian Day, Days from 1 January to 31 of December.
        + PhiD: Latitude in decimal degrees.
        + G: Sensible heat flux into the soil.
        _________________________________________________________________________
        
            OUTPUT:
        - PET: Potential Evaportanspiration [mm/day].
        - AET: Actual Evaportanspiration [mm/day].
        '''

        # Velocidad del viento corregida
        u2 = uh*(4.87/(np.log(67.8*h-5.42))) # [m/s]
        
        # Delta
        D = (4098*(0.6108*np.exp((17.27*T)/(T+237.3))))/(T+237.3)**2

        # Presión
        if p == 0:
            p = 101.3*((293-0.0065*h)/293)**(5.26) # [kPa]

        # Constante Psicométrica
        Gamma = 0.000665*p # [kPa/°C]

        # e_s Saturation vapor pressure 
        e_s = 0.6108*np.exp((17.27*T)/(T+237.3)) # [kPa]

        # e_a
        if e_aq == 0:
            e_a = e_s*(HR/100)
        else:
            e_a = 0.6108*np.exp((17.27*Tmin)/(Tmin+237.3))

        Cn = 1600
        Cd = 0.38
        # Latent Heat
        lv = 2.45 # [MJ/kg]
        # Specific heat at constant pressure
        Cp = 1.013 * 10**(-3) # [MJ/g]

        if Rn == 0:
            # Delta Term
            DT = D/(D+Gamma*(1+0.34*u2)) 
            # Psi Term (PT)
            PT = Gamma/(D+Gamma*(1+0.34*u2))
            # Temperature term
            TT = (900/(T+273))*u2
            # Inverse relative distance Earth-Sun (dr)
            dr = 1+0.033*np.cos((2*np.pi/365)*J)
            # Solar declination J = Julian day
            d = 0.409*np.sin((2*np.pi/365)*J - 1.39)
            # Latitud en radianes PhiD = Latitud en grados decimales
            Phi = np.pi/180 * PhiD
            # Sunset hour angle
            Omega_s = np.arccos(-np.tan(Phi)*np.tan(d))
            # Extraterrestial radiation MJ/m^2day
            Gsc = 0.0820 # MJ/m^2min
            Ra = (24*60)/np.pi * Gsc * dr*(Omega_s*np.sin(Phi)*np.sin(d)+\
                (np.cos(Phi)*np.cos(d)*np.sin(Omega_s)))
            # Clear sky solar radiation
            Rso = (0.75 + 2*10**(-5)*h)*Ra # MJ/m^2day
            # Net solar or net shotwave radiation
            Rns = (1-0.23)*Rs
            # Net outgoing long wave radiation
            sigma = 4.903*10**(-9)
            Rnl = sigma*(T**4)*(0.34-0.14*np.sqrt(e_a))*(1.35*(Rs/Rso)-0.35)
            # Net radiation 
            Rn = Rns-Rnl

        # Se calcula la Evapotranspiración Potencial con Penman-Monteith 
        PET = (0.408*D*(Rn-G)+Gamma*(Cn/(T+273))*u2*(e_s-e_a))/(D+Gamma*(1+Cd*u2))
        # Se calcula
        z0 = 0.3
        ra = 4.72*(np.log(2/z0))**2/(1+0.54*uh)
        hc = 0.30
        LAI = 5.5 + 1.5 * hc
        rs = 200/LAI
        # Air density
        R_specific = 287.05
        rho = p/(R_specific*(T+273))
        AET = ((D*(Rn-G)+(rho*Cp/ra)*(e_s-e_a))/(lv*(D+Gamma*(1+(rs/ra)))))

        return PET,AET

    def EqBudyko(self,P,PET):
        '''
            DESCRIPTION:
        
        This function calculates the AET with the Budyko equation.
        _________________________________________________________________________

            INPUT:
        + P: Precipitation.
        + PET: Potential Evapotranspiration.
        _________________________________________________________________________
        
            OUTPUT:
        - AET: Actual Evaportanspiration [mm/day].
        '''
        try: 
            len(P)
            x = np.where(P <= 0.002)
            AET = (PET*P*np.tanh(P/PET)*(1-np.cosh(PET/P)+np.sinh(PET/P)))**(1/2)
            if len(P) == 0:
                AET = PET[x]*np.tanh(P[x]/PET[x])

        except:
            if ~np.isnan(PET):
                AET = (PET*P*np.tanh(P/PET)*(1-np.cosh(PET/P)+np.sinh(PET/P)))**(1/2)
                
                if np.isnan(AET):
                    AET = PET*np.tanh(P/PET)
            else:
                AET = np.nan
        
        return AET
