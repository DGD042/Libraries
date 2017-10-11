# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#______________________________________________________________________________
#______________________________________________________________________________

'''

This package has the different Models and equations relation with
atmospheric and ground physics.
____________________________________________________________________________
This class is of free use and can be modify, if you have some 
problem please contact the programmer to the following e-mails:

- danielgondu@gmail.com 
- dagonzalezdu@unal.edu.co
- daniel.gonzalez17@eia.edu.co
____________________________________________________________________________

'''

try:
    import Hydro_Analysis.Models.Radiation
except ImportError:
    import Models.Radiation
except ImportError:
    import Radiation
except ImportError:
    import Radiation

