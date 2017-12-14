# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#______________________________________________________________________________
#______________________________________________________________________________


try:
    from AnET.random import *
    from AnET.AnET import AnET
    from AnET.CFitting import CFitting
    from AnET.CorrSt import CorrSt
    from AnET.ODE_S import ODE_S
    from AnET.FyC_Class import FyC_Class
except ImportError:
    from random import *
    from AnET import AnET
    from CFitting import CFitting
    from CorrSt import CorrSt
    from ODE_S import ODE_S
    from FyC_Class import FyC_Class
