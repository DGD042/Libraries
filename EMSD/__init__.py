# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#______________________________________________________________________________
#______________________________________________________________________________

'''

    
This class replaces the ExtractD class to make a better and 
efficient way of Extracting Manipulating and Saving Data 
(EMSD). Functions in this class aims to manipulate times series 
information with a preference for climate data, however you can 
extract data from any file.

This class is of free use and can be modify, if you have some 
problem please contact the programmer to the following e-mails:

- danielgondu@gmail.com 
- dagonzalezdu@unal.edu.co
- daniel.gonzalez17@eia.edu.co

--------------------------------------
 How to use the library
--------------------------------------

You can use any function of the class separatley but using the 
function Open_Data first would allow you to have a better 
control of the full class.

____________________________________________________________________________

'''

try:
    from EMSD.EMSD import EMSD
    from EMSD.Data_Man import *
    from EMSD.Extract_Data import *
    from EMSD.Functions import *
    from EMSD.Specific import *
    from EMSD.Write_Data import *
    from EMSD.Dates import *
except ImportError:
    from EMSD import EMSD
    from Data_Man import *
    from Extract_Data import *
    from Functions import *
    from Specific import *
    from Write_Data import *
    from Dates import *

