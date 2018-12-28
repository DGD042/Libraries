# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 01/09/2017
#______________________________________________________________________________
#______________________________________________________________________________
'''

The package give several functions for data manipulation, including units
and operation manipulation.
'''
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
import subprocess
import platform

operations= {'Over':op.gt,'over':op.gt,'>':op.gt,
      'Lower':op.lt,'lower':op.lt,'<':op.lt,
      '>=':op.ge,'<=':op.le,
      'mean':np.nanmean,'sum':np.nansum,
      'std':np.nanstd}

# Units
def cm2inch(*tupl):
    '''
    DESCRIPTION:
    
        This functions allows to change centimeters to inch so you can 
        denote the size of the figure you want to save.
    _______________________________________________________________________

    INPUT:
        :param *tupl: a tuple, Tuple variable with centimeter values.
    _______________________________________________________________________
    
    OUTPUT:
        :return: A tuple, Values in inch.
    '''
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

# Data Manipulation
def Oper_Det(Oper):
    '''
    DESCRIPTION:
    
        This functions verifies the existence of the operation inside 
        the operations diccionary and returns the operation.
    _______________________________________________________________________

    INPUT:
        :param Oper: a str, Operation to be verified.
    _______________________________________________________________________
    
    OUTPUT:
        :return OperRet: a function, Returned operation.
    '''
    # Operation findings
    operations= {'Over':op.gt,'over':op.gt,'>':op.gt,
                      'Lower':op.lt,'lower':op.lt,'<':op.lt,
                      '>=':op.ge,'<=':op.le,
                      'mean':np.nanmean,'sum':np.nansum,
                      'std':np.nanstd}
    try:
        OperRet = operations[Oper]
        return OperRet
    except KeyError:
        raise KeyError('Operation or comparation not found, verify the given string')

def NoNaN(X,Y,flagN=True):
    '''
    DESCRIPTION:
    
        This function removes the NaN values of two series and count
        all the non NaN values in the vectors.
    _________________________________________________________________________

    INPUT:
        :param X:     A List or ndarray, First vector.
        :param Y:     A List or ndarray, Second vector.
        :param flagN: A boolean, flag to know if it liberates the non NaN 
                                 data.
    _________________________________________________________________________
    
    OUTPUT:
        :return XX: A List or ndarray, First vector without NaN values.
        :return YY: A List or ndarray, Second vector without NaN values.
        :return N:  An int, number of NaN data.
    '''
    if len(X) != len(Y):
        Er = self.ShowError('NoNaN','Data_Man','X and Y are not the same length')
        if flagN == True:
            return Er, Er, Er
        else:
            return Er, Er
    q = ~(np.isnan(X) | np.isnan(Y))
    N = sum(q) 
    XX = X[q]
    YY = Y[q]
    q = ~(np.isinf(XX) | np.isinf(YY))
    XX = XX[q]
    YY = YY[q]

    if flagN == True:
        return XX,YY,N
    else:
        return XX,YY
    
def Interp(Xi,Yi,X,Xf,Yf):
    '''
    DESCRIPTION:
        
        This function makes a linear interpolation, in the future this
        function would move to another package, so be aware that it would
        not be here anymore.
    _________________________________________________________________________

    INPUT:
        :param Xi: A float, Primer valor de x.
        :param Yi: A float, Primer valor de y.
        :param X:  A float, Valor del punto en x.
        :param Xf: A float, Segundo valor de x.
        :param Yf: A float, Segundo valor de y.
    _________________________________________________________________________
    
    OUTPUT:
        :return: A float, Valor interpolado de Y.
    '''
    return Yi+((Yf-Yi)*((X-Xi)/(Xf-Xi)))





