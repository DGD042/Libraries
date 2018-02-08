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
except ImportError:
    from Hydro_Analysis import Hydro_Analysis 
    from Hydro_Plotter import Hydro_Plotter
    from Thermo_An import Thermo_An
    from Evap_Models import Evap_Models
    import Gen_Functions
    import Meteo 
    import Climate
    from Models import *

