# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#______________________________________________________________________________
#______________________________________________________________________________

'''

This package allows the user to do some function and data manipulation. 

This package also have some functions that are needed to use the other
classes and functions of this library, so it is crucial that you have 
this class along with the the other classes.

____________________________________________________________________________

This package is of free use and can be modify, if you have some 
problem please contact the programmer to the following e-mails:

- danielgondu@gmail.com 
- dagonzalezdu@unal.edu.co
- daniel.gonzalez17@eia.edu.co
____________________________________________________________________________

USAGE:: 

    >>> import Utilities


'''

try:
    import Utilities
    from Utilities import Utilities
    from Utilities import Data_Man
    from Utilities.DatesUtil import DatesUtil 
except ImportError:
    import Utilities
    import Data_Man
    from DatesUtil import DatesUtil 

