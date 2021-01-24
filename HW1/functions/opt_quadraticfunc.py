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
class steepest_descent:

    def __init__(self, function, alpha = None):
        ''' 
        Input:
        - function: must be class objective_function.f, with well defined derivatives
        - alpha is the step_length, if input is None, then will calculate in each step. '''

        # 1. save 
        self.f = function
        self.alpha = alpha



        return

    # MAIN function -------------------------------------
    def run(self, x0, tol = 1e-6, maxiter = 1000):

        
        i = 0
        improve = tol+1
        x = x0
        obj_old = 9999
        while improve > tol and i < maxiter:

            # calculate obj 
            obj = self.f.value(x)
            improve = obj_old - obj

            # print
            if np.round(i/1) == i/1:
                print(f'iter {i}: obj = {obj}, improve = {improve}')

            # update 
            obj_old      = obj
            self.der_1st = self.f.der_1st(x)

            d_k     = self._get_direction()
            alpha_k = self._get_steplength()

            x_old   = x
            x       = x_old + alpha_k * d_k 

            i = i + 1

        return x_old 


    # auxiliary functions -------------------------------------
    def _get_direction(self):

        der_1st = self.der_1st
        length = np.linalg.norm(der_1st)
        direction =  - 1/length * der_1st

        return direction

    def _get_steplength(self):

        if self.alpha is not None:
            return self.alpha
        else:
            der_1st = self.der_1st
            numerator   = der_1st @ der_1st
            denominator = der_1st @ self.f.A @ der_1st

            alpha_k = numerator/denominator

            assert isinstance(alpha_k, float) # should not be matrix
            return alpha_k

            




