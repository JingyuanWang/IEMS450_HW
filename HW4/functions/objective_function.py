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

class tridiagonal_function:

    def __init__(self, n):
        '''set the function, define the dimensions'''
        self.n = n

        return

    def value(self, x):
        assert x.shape == (self.n ,) or x.shape == (self.n,1)
        x = x.reshape( (self.n,) )

        value = 0.5 * (x[0]-1)**2
        for i in range(self.n-1):
            value = value + 0.5 * (x[i] - 2*x[i+1])**4

        return float(value)
        
    def der_1st(self, x):
        assert x.shape == (self.n ,) or x.shape == (self.n,1)
        x = x.reshape( (self.n,) )

        df_dx = np.zeros( (self.n, 1) )
        df_dx[0,0] = (x[0]-1) + _df_dx1(x[0], x[1])

        for i in range( 1, self.n-1):
            df_dx2     = _df_dx2(x[i-1], x[i])
            df_dx1     = _df_dx1(x[i], x[i+1])
            df_dx[i,0] = df_dx1 + df_dx2

        df_dx[self.n-1,0] = _df_dx2( x[self.n-2], x[self.n-1]  )

        return df_dx


    def der_2nd(self, x):
        assert x.shape == (self.n ,) or x.shape == (self.n,1)
        x = x.reshape( (self.n,) )

        H = np.zeros( (self.n, self.n) )
        H[0,0] = 1

        for i in range(self.n-1):

            # get Hession for this element (2-by-2)
            x1 = x[i]
            x2 = x[i+1]
            element_11 = _d2f_dx1dx1(x1, x2)
            element_12 = _d2f_dx1dx2(x1, x2)
            element_21 = _d2f_dx2dx1(x1, x2)
            element_22 = _d2f_dx2dx2(x1, x2)
            mat = np.array( [  [element_11, element_12], [element_21, element_22] ] )

            H[i:i+2, i:i+2] = H[i:i+2, i:i+2] + mat
        
        return H


# ---------------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------------
# about Hessian and der
def _df_dx1(x1, x2):
    df_dx1 = 2 *(x1-2*x2)**3
    return df_dx1

def _df_dx2(x1, x2):
    df_dx2 = -4 *(x1-2*x2)**3
    return df_dx2

def _d2f_dx1dx1(x1, x2):
    return 6*(x1-2*x2)**2

def _d2f_dx1dx2(x1, x2):
    return -12*(x1-2*x2)**2

def _d2f_dx2dx1(x1, x2):
    return -12*(x1-2*x2)**2

def _d2f_dx2dx2(x1, x2):
    return 24*(x1-2*x2)**2



