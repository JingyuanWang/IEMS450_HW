'''
# ------------------------------------------------------------------------
# NOTE
# ------------------------------------------------------------------------
# Purpose:
#
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

    def __init__(self, A):
        '''set the function'''

        # 1. check
        assert A.shape[0] == A.shape[1]
        assert np.linalg.matrix_rank(A) == A.shape[0]

        # 2. save the function
        self.A = A 
        
        # 3. calculate the Lipschitz_constant
        self.L = self.Lipschitz_constant()

        return

    def value(self, x):
        value = x@self.A@x
        return value 
        
    def der_1st(self, x):
        value = self.A@x
        return value 

    def der_2nd(self, x=None):
        value = self.A
        return value 

    def Lipschitz_constant(self):
        L = self.A.max()
        return L




