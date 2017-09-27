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
    from EMSD.Data_Man import Data_Man
except ImportError:
    from EMSD.Data_Man import EMSD

