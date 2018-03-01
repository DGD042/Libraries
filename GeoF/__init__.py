# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#______________________________________________________________________________
#______________________________________________________________________________

try:
    from GeoTIFF import *
    from NetCDF import *
    from Maps import *
    from Interpolations import *
    from GeoF import *
except ImportError:
    from GeoF import GeoTIFF
    from GeoF import GeoF
    from GeoF import Interpolations
    from GeoF import Maps
    from GeoF import GeoTimeSeries


