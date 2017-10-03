# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 26/04/2017
#______________________________________________________________________________
#______________________________________________________________________________

#______________________________________________________________________________
#
# CLASS DESCRIPTION:
#   This class aims to create code from other codes.
#
#   This class is of free use and can be modified if needed, if the user
#   encounters an issue please contact the programmer Daniel González to
#   the following e-mails:
#
#   danielgondu@gmail.com
#   dagonzalezdu@unal.edu.co
#   daniel.gonzalez17@eia.edu.co
#______________________________________________________________________________

# We import the packages
import sys
import os
import time

# from UtilitiesDGD import UtilitiesDGD
# utl = UtilitiesDGD()



class Code_Creator(object):

    def __init__(self):

        '''
            DESCRIPTION:

        Constructor.
        '''
        # We organize the date of creation
        DateTuple = time.localtime()
        year = DateTuple[0]
        month = DateTuple[1]
        day = DateTuple[2]

        if month < 10:
            if day < 10:
                Date = '0'+str(day)+'/'+'0'+str(month)+'/'+str(year)
            else:
                Date = str(day)+'/'+'0'+str(month)+'/'+str(year)
        else:
            if day < 10:
                Date = '0'+str(day)+'/'+str(month)+'/'+str(year)
            else:
                Date = str(day)+'/'+str(month)+'/'+str(year)

        self.header = ('# -*- coding: utf-8 -*-\n'+\
                    '#______________________________________________________________________________\n'+\
                    '#______________________________________________________________________________\n'+\
                    '#\n'+\
                    '#                      Coded by Daniel González Duque\n'+\
                    '#                          Created %s\n'%(Date)+\
                    '#______________________________________________________________________________\n'+\
                    '#______________________________________________________________________________\n')


    def Test(self,text):
        '''             
            DESCRIPTION:
        This function creates a test code from other codes.
        __________________________________________________________________
            
            INPUT:
        + text: Text to print.
        __________________________________________________________________

            OUTPUT: 
        Code with the name Test.py
        '''
        # Open the code text
        f = open('Test.py','w+',encoding='utf-8')

        f.write(self.header)
        f.write("\nprint('%s')" %text)
        f.write("\n\n# Este código es de prueba")
        f.write("\nfor i in range(2):")
        f.write("\n\tprint(i)")
        f.close()

        return
        
    def TestFromCode(self,text):
        '''             
            DESCRIPTION:
        This function takes other code and modified it.
        __________________________________________________________________
            
            INPUT:
        + text: Text to print.
        __________________________________________________________________

            OUTPUT: 
        Code with the name Test2.py
        '''
        ff = open('Test.py','r',encoding='utf-8')
        c = ff.readlines()
        ff.close()
        f = open('Test2.py','w+',encoding='utf-8')

        f.write(self.header)
        for row,data in enumerate(c):
            if  row == 9:
                f.write("\nprint('%s')" %text)
            elif row >= 10:
                f.write(data)
        f.close()

