# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 01/09/2017
#______________________________________________________________________________
#______________________________________________________________________________
'''

The functions given on this package allow the user to manipulate and create
certain things inside the computer.

Usage::

    >>> from Utilities import Utilities as utl
    >>> # Create a folder
    >>> utl.CrFolder(<Path_inside_computer>)

'''
# ------------------------
# Importing Modules
# ------------------------ 
# System
import sys
import os
import glob as gl
import re
import operator as op
import warnings
import subprocess
import platform

    
# System
def ShowError(fn,cl,msg):
    '''
    DESCRIPTION:

        This function manages errors, and shows them. 
    _______________________________________________________________________
    INPUT:
        :param fn:  A str, Function that produced the error.
        :param cl:  A str, Class that produced the error.
        :param msg: A str, Message of the error.
    _______________________________________________________________________
    OUTPUT:
       :return: An int, Error managment -1. 
    '''

    raise Exception('ERROR: Function <'+fn+'> Class <'+cl+'>: '+msg)

def ExitError(fn,cl,msg):
    '''
    DESCRIPTION:

        This function stops the ejecution of a code with a given error
        message.
    _______________________________________________________________________
    
    INPUT:
        :param fn:  A str, Function that produced the error.
        :param cl:  A str, Class that produced the error.
        :param msg: A str, Message of the error.
    _______________________________________________________________________
    OUTPUT:
       :return: A msg, message with the error.
    '''

    print('ERROR: Function <'+fn+'> Class <'+cl+'>: '+msg)
    raise -1

def CrFolder(path):
    '''
    DESCRIPTION:
    
        This function creates a folder in the given path, if the path does 
        not exist then it creates the path itself
    _______________________________________________________________________

    INPUT:
        :param path: A str, Path that needs to be created.
    _______________________________________________________________________
    OUTPUT:
        :return: This function create all the given path.
    '''
    if path != '':
        # Verify if the path already exists
        if not os.path.exists(path):
            os.makedirs(path)

    return

def GetFolders(path):
    '''
    DESCRIPTION:
    
        This function gets the folders and documents inside a 
        specific folder.
    _______________________________________________________________________

    INPUT:
        :param path: A str, Path where the data would be taken.
    _______________________________________________________________________
    OUTPUT:
        :return R: A List, List with the folders and files inside 
                           the path.
    '''
    R = next(os.walk(path))[1]
    return R

