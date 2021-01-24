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

    method_list = ['constant step length', 'exact line search', 'Nesterov’s optimal method', 'heavy-ball']

    def __init__(self, function, method = 'exact line search', alpha = None):
        ''' 
        Input:
        - function: must be class objective_function.f, with well defined derivatives
        - alpha is the step_length, if input is None, then will calculate in each step. '''

        # 1. save 
        assert method in self.method_list
        self.method = method
        if method == 'constant step length':
            assert alpha is not None
        self.alpha = alpha

        self.f = function


        return

    # MAIN function -------------------------------------
    def run(self, x0, tol = 1e-6, maxiter = 1000):

        # initialize: i, x0, first obj, and step related 
        i = 0
        improve = tol+1
        x = x0
        obj_old = 9999
        self._init_for_each_method()

        while improve > tol and i <= maxiter:

            # calculate obj 
            obj = self.f.value(x)
            improve = obj_old - obj

            # print
            if np.round(i/5) == i/5:
                print(f'iter {i:4d}: obj = {obj:.2E}, improve = {improve:.6E}')

            # update 
            obj_old      = obj
            self.der_1st = self.f.der_1st(x)

            p_k     = self._get_step()
            x_old   = x
            x       = x_old +  p_k 

            i = i + 1

        print('======================================')
        if i > maxiter:
            print(f'not converge, last iter {i-1 :4d }: obj = {obj:.2E}, improve = {improve:.6E}')
        else:
            print(f'complete iter {i-1 :4d}: obj = {obj_old:.6E}')

        return x_old 


    # auxiliary functions -------------------------------------
    def _init_for_each_method(self):
        if self.method == 'heavy-ball':
            self.p_lastiter = 0

        return 

    def _get_step(self):

        if self.method == 'constant step length':
            
            der_1st    = self.der_1st
            direction  =  - (1/np.linalg.norm(der_1st) ) * der_1st
            p_k        = self.alpha * direction

        elif self.method == 'exact line search':
            der_1st     = self.der_1st
            numerator   = der_1st @ der_1st
            denominator = der_1st @ self.f.A @ der_1st

            alpha_k = numerator/denominator

            assert isinstance(alpha_k, float) # should not be matrix

            p_k     =  - alpha_k  * der_1st

        #elif self.method == 'Nesterov’s optimal method':
            

        #elif self.method == 'heavy-ball':

            #self.p_lastiter = p_k 
        
        return p_k

            




