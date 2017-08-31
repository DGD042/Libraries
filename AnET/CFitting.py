# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 13/06/2016
#______________________________________________________________________________
#______________________________________________________________________________

#______________________________________________________________________________
#
# CLASS DESCRIPTION:
#   This class pretend to fit different functions in different sets of data.
#______________________________________________________________________________
# Manipulate Data
import numpy as np
import scipy.io as sio # To use .mat files
from scipy import stats as st # To make linear regressions
from scipy.optimize import curve_fit # To do curve fitting
# System
import sys
import os
import warnings

# ------------------
# Personal Modules
# ------------------
# Importing Modules
from Utilities import Utilities as utl


# The functions are defined
def LR(x,a,b):
    return a*x+b
def PL(x,a,b):
    return a*x**b
def Par(x,a,b,c):
    return a*x**2+b*x+c
def Exp(x,a,b):
    return a*np.exp(b*x)
def Log(x,a,b,c):
    return a*np.log(b*x)+c


class CFitting:

    def __init__(self):

        self.Adj = {'lineal':LR,'potential':PL,
                    'parabolic':Par,'exponential':Exp,
                    'logaritmic':Log}

        self.AdjF = {'lineal':r'$y=%.4fx+%.4f$','potential':r'$y=%.4fx^{%.4f}$',
                    'parabolic':r'$y=%.4fx^2+%.4fx+%.4f$','exponential':r'$y = %.4fe{%.4fx}$',
                    'logaritmic':r'$y = %.4f \log(%.4fx) + %.4f$'}
        return

    def addfun(self,fun,key='New',funeq=''):
        '''
        DESCRIPTION:
        
            This function add a new function to the directory of functions to
            be fitted.
        _______________________________________________________________________

        INPUT:
            + fun: Function to be added.
            + key: Name of the function.
        _______________________________________________________________________
        
        '''
        self.Adj[key] = fun
        self.AdjF[key] = funeq
        return

    def FF(self,xdata,ydata,F='lineal'):
        '''
        DESCRIPTION:
        
            This function takes x and y data and makes a fitting of the data with
            the function that wants to be added.
        _______________________________________________________________________

        INPUT:
            + xdata: Data in the x axis.
            + ydata: Data in the y axis.
            + F: Function that wants to be fitted, by default is 'lineal'.

                 The functions that are defaulted to fit are the following:

                      Function keys     |   Function fitting
                        'lineal'        |      y = ax + b
                       'potencial'      |       y = ax^b
                       'parabolic'      |   y = ax^2 + bx + c
                      'exponential'     |     y = aEXP^(bx)
                      'logaritmic'      |     y = aLOG(bx) + c
                        '' or 0         |       Best fitting
        _______________________________________________________________________
        
        OUTPUT:
            This function return a dictionary with the following data:
                - Coef: Ceofficients
                - perr: Standard deviation errors of the parameters.
                - R2: Coefficient of determination.
        '''
        # --------------------------------
        # Error Managment and Parameters
        # --------------------------------

        if isinstance(F,str) == False and F != 0:
            Er = utl.ShowError('FF','CFitting','Given parameter F is not on the listed functions.')
            return Er

        keys = list(self.Adj)
        if F == '' or F == 0:
            flagfitbest = True
        else:
            key = F.lower()
            flagfitbest = False
            try:
                fun = self.Adj[key]
            except KeyError:
                Er = utl.ShowError('FF','CFitting','Given parameter F is not on the listed functions.')
                return Er

        # -----------------
        # Calculations
        # -----------------
        X,Y = utl.NoNaN(xdata,ydata,False)
        Results = dict()

        if flagfitbest:
            CoefT = dict()
            perrT = dict()
            keysT = []
            R2T = []

            for ikey,key in enumerate(keys):
                fun = self.Adj[key]
                # Fitting
                Coef, pcov = curve_fit(fun,X,Y)
                # R2 calculations
                ss_res = np.dot((Y - fun(X, *Coef)),(Y - fun(X, *Coef)))
                ymean = np.mean(Y)
                ss_tot = np.dot((Y-ymean),(Y-ymean))
                R2T.append(1-(ss_res/ss_tot))

                perrT[key] = np.sqrt(np.diag(pcov))
                CoefT[key] = Coef

            # Verify the maximum R^2
            x = np.where(np.array(R2T) == np.nanmax(np.array(R2T)))[0]
            if len(x) > 1:
                x = x[0]
            key = np.array(keys)[x][0]
            Results['Coef'] = CoefT[key]
            Results['perr'] = perrT[key]
            Results['R2'] = np.array(R2T)[x][0]
            Results['Functionkey'] = key
            Results['Function'] = self.Adj[key]
            Results['FunctionEq'] = self.AdjF[key]
        else:
            # Fitting
            Coef, pcov = curve_fit(fun,X,Y)
            ss_res = np.dot((Y - fun(X, *Coef)),(Y - fun(X, *Coef)))
            perr = np.sqrt(np.diag(pcov))
            ymean = np.mean(Y)
            ss_tot = np.dot((Y-ymean),(Y-ymean))
            R2 = 1-(ss_res/ss_tot)
            Results['Coef'] = Coef
            Results['perr'] = perr
            Results['R2'] = R2
            Results['Functionkey'] = key
            Results['Function'] = fun
            Results['FunctionEq'] = self.AdjF[key]

        return Results



