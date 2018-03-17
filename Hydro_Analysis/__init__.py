# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#______________________________________________________________________________
#______________________________________________________________________________

'''

This class have different routines for hydrological
and thermodynamical analysis in relation with climatology.

This package also graph all the different intervals requiered from the
given data.


____________________________________________________________________________
This class is of free use and can be modify, if you have some 
problem please contact the programmer to the following e-mails:

- danielgondu@gmail.com 
- dagonzalezdu@unal.edu.co
- daniel.gonzalez17@eia.edu.co
____________________________________________________________________________

'''

try:
    from Hydro_Analysis.Hydro_Analysis import Hydro_Analysis 
    from Hydro_Analysis.Hydro_Plotter import Hydro_Plotter
    from Hydro_Analysis.Evap_Models import Evap_Models
    from Hydro_Analysis.Gen_Functions import *
    from Hydro_Analysis.Meteo import *
    from Hydro_Analysis.Climate import *
    from Hydro_Analysis.Models import *
    from Hydro_Analysis.Dates import *
except ImportError:
    from Hydro_Analysis import Hydro_Analysis 
    from Hydro_Plotter import Hydro_Plotter
    from Gen_Functions import *
    from Meteo import *
    from Climate import *
    from Models import *
    from Dates import *

