'''
# ------------------------------------------------------------------------
# NOTE
# ------------------------------------------------------------------------
# Purpose:
#       define objective functions as class, and calculate the value and the derivatives
# ------------------------------------------------------------------------
'''


import numpy as np
import os,sys,inspect
import copy
import importlib


# ---------------------------------------------------------------------------------
# class
# ---------------------------------------------------------------------------------
class Rosenbrock_function:

    def __init__(self):
        '''set the function'''

        self.n = 2

        return

    def value(self, x):
        assert x.shape == (self.n ,) or x.shape == (self.n,1)
        x = x.reshape( (self.n,) )

        value = 100 * (x[1] - x[0]**2)**2 + (1-x[0])**2 

        if not isinstance(value, float):
            value = value[0]

        return value
        
    def der_1st(self, x):
        assert x.shape == (self.n ,) or x.shape == (self.n,1)
        x = x.reshape( (self.n,) )

        df_dx1 = 400*(x[0]**3) - 400*x[0]*x[1] + 2*x[0] - 2
        df_dx2 = 200*(x[1]-x[0]**2)

        return np.array([df_dx1, df_dx2]).reshape( (2,1) )

    def der_2nd(self, x):
        assert x.shape == (self.n ,) or x.shape == (self.n,1)
        x = x.reshape( (self.n,) )

        element_11 = 1200*(x[0]**2) - 400*x[1] + 2
        element_12 = -400*x[0]
        element_21 = -400*x[0]
        element_22 = 200
        return np.array( [  [element_11, element_12], [element_21, element_22] ] )



class Rosenbrock_function_extended:

    def __init__(self, n):
        '''set the function, define the dimensions'''
        self.n = n

        return

    def value(self, x):
        assert x.shape == (self.n ,) or x.shape == (self.n,1)
        x = x.reshape( (self.n,) )

        value = 0
        for i in range(self.n-1):
            value = value + 100 * (x[i+1] - x[i]**2)**2 + (1-x[i])**2 

        if not isinstance(value, float):
            value = value[0]
        return value
        
    def der_1st(self, x):
        x = x.reshape( (self.n,) )

        df_dx = np.zeros( (self.n, 1) )
        df_dx[i,0] = 1

        for i in range( 1, self.n-1):
            df_dx[i,0] = 400*(x[0]**3) - 400*x[0]*x[1] + 2*x[0] - 2

        df_dx[self.n-1,0] =  1

        return df_dx


    def der_2nd(self, x):
        x = x.reshape( (self.n,) )

        H = np.zeros( (self.n, self.n) )
        
        return H


