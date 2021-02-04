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
class f:

    def __init__(self):
        '''set the function'''

        self.n = 2

        return

    def value(self, x):
        '''can input vectors or one point '''

        # if x is a point, reduce its dimension
        if isinstance(x, np.ndarray):
            if x.shape == (self.n,1):
                x = x.reshape( (self.n,) )

        value = 10 * (x[1] - x[0]**2)**2 + (1-x[0])**2 

        return value
        
    def der_1st(self, x):
        '''must be one point '''

        assert x.shape == (self.n ,) or x.shape == (self.n,1)
        x = x.reshape( (self.n,) )

        df_dx1 = 40*(x[0]**3) - 40*x[0]*x[1] + 2*x[0] - 2
        df_dx2 = 20*(x[1]-x[0]**2)

        return np.array([df_dx1, df_dx2]).reshape( (2,1) )

    def der_2nd(self, x):
        '''must be one point '''

        assert x.shape == (self.n ,) or x.shape == (self.n,1)
        x = x.reshape( (self.n,) )

        element_11 = 120*(x[0]**2) - 40*x[1] + 2
        element_12 = -40*x[0]
        element_21 = -40*x[0]
        element_22 = 20
        return np.array( [  [element_11, element_12], [element_21, element_22] ] )

        return 


class m:

    def __init__(self, f, x):
        '''set the function, 
        -- f must be from the above class f
        -- x must be a point. Will do Taylor expansion of f at the point x'''

        self.n       = f.n

        self.x0      = x
        self.f_x0    = f.value(x)
        self.der_x0  = f.der_1st(x)
        self.H_x0    = f.der_2nd(x)

        return

    def value_step(self, d):
        '''can input vectors or one point '''
        assert d.shape[0] == self.n

        # if d is a point, define its dimension to better for matrix operation
        if d.shape == (self.n, ):
            d = d.reshape( (self.n, 1) )

        num_points = d.shape[1]

        value  = self.f_x0 + self.der_x0.T @ d + 0.5 * np.diag(d.T @ self.H_x0 @ d)

        assert value.shape[1] == num_points
        if d.shape[1] == 1:
            value = value.squeeze()

        return value

    def value(self, X, Y):
        '''X and Y are grid points '''
        assert X.shape == Y.shape

        steps_X = X - self.x0[0] 
        steps_Y = Y - self.x0[1]

        num_points = X.shape[0] * X.shape[1] 
        n = X.shape[0]
        X = steps_X.reshape( (num_points, ) )
        Y = steps_Y.reshape( (num_points, ) )
        D = np.row_stack( (X,Y) )

        values = self.value_step(D)

        return values.reshape( (n,n) )

    def der_1st(self, d):

        assert d.shape == (self.n ,) or d.shape == (self.n,1)
        d = d.reshape( (self.n,1) )

        return (self.der_x0 +  self.H_x0 @ d).squeeze()

    def der_2nd(self, d=None):

        return self.H_x0




