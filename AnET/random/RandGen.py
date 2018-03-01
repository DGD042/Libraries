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
from numpy.random import randint
import scipy.io as sio # To use .mat files
from scipy import stats as st # To make linear regressions
from scipy.optimize import curve_fit # To do curve fitting
from scipy.stats.distributions import t # t distribution
# System
import sys
import os
import warnings
import re

# ------------------
# Personal Modules
# ------------------
# Importing Modules

# ------------------
# Functions
# ------------------

def RandomPosGen(Pmin=0,Pmax=100,Length=100,flagRepeat=False):
    '''
    DESCRIPTION:
    
        This function makes a vector with random position numbers with 
        a defined length.
    _______________________________________________________________________

    INPUT:
        :param Pmin:        A int, min number of positions, set to 0
        :param Pmax:        A int, max number of positions, set to 100
        :param Length:      A int, Length
        :param flagRepeat: A boolean, flag if the position repeats or no
                           set to False.
    _______________________________________________________________________
    
    '''
    if Length > Pmax-Pmin and not(flagRepeat==False):
        raise Exception('Length is bigger than the number of data to not repeat data')
    R = np.empty(Length)*np.nan
    for L in range(Length):
        Num = randint(Pmin,Pmax-1)
        if not(flagRepeat):
            if L != 0:
                z = True
                while z:
                    x = np.where(R == Num)[0]
                    if len(x) != 0:
                        Num = randint(Pmin,Pmax-1)
                    else:
                        break
        R[L] = Num

    R = R.astype(int)
    return R

