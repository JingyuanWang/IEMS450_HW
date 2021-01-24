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

    def __init__(self, function, method = 'exact line search', alpha = None, print_every_n_step= None, print_results=True):
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
        self.print_every_n_step = print_every_n_step
        self.print_results = print_results


        return

    # MAIN function -------------------------------------
    def run(self, x0, tol = 1e-6, maxiter = 1000):

        if self.print_results:
            print('======================================')
            print(f'{self.method}:')

        # initialize: i, x0, first obj, and step related 
        i = 0
        improve = tol+1
        x = x0
        self.obj = 9999
        self._init_for_each_method(x0)

        #while improve > tol and i <= maxiter:
        while self.obj > tol and i <= maxiter:

            # calculate obj 
            obj     = self.f.value(x)
            improve = self.obj - obj

            # print
            if self.print_every_n_step is not None:
                if np.round(i/self.print_every_n_step) == i/self.print_every_n_step:
                    print(f'iter {i:4d}: obj = {obj:.6E}, improve = {improve:.6E}')

            # save and update
            self.i       = i
            self.obj     = obj
            self.obj_list= self.obj_list + [obj]
            self.x       = x
            self.der_1st = self.f.der_1st(x)

            p_k     = self._get_step()
            x       = self.x +  p_k 
            i       = i + 1

        if self.print_results:
            print('======================================')
            if i > maxiter:
                print(f'not converge, last iter {self.i}: obj = {self.obj:.6E}, improve = {improve:.6E}')
            else:
                print(f'complete iter {self.i :4d}: obj = {self.obj:.6E}')
            print('======================================')
            print('\n')

        return 


    # auxiliary functions -------------------------------------
    def _init_for_each_method(self, x0):

        self.x  = x0
        self.obj_list = []

        if self.method == 'Nesterov’s optimal method':
            self.t    = 1
            self.y    = self.x

        elif self.method == 'heavy-ball':
            
            self.p_lastiter = 0

            L          = self.f.Lipschitz_constant()
            m          = self.f.min_Hession_eigenvalue()
            self.alpha = 4/(( np.sqrt(L) + np.sqrt(m) )**2)  
            self.beta  = ( np.sqrt(L) - np.sqrt(m) )/( np.sqrt(L) + np.sqrt(m) )  

        return

    def _get_step(self):

        if self.method == 'constant step length':            
            der_1st    = self.der_1st
            #direction  =  - (1/np.linalg.norm(der_1st) ) * der_1st
            #p_k        = self.alpha * direction

            p_k        = - self.alpha * der_1st


        elif self.method == 'exact line search':
            der_1st     = self.der_1st
            numerator   = der_1st @ der_1st
            denominator = der_1st @ self.f.A @ der_1st
            alpha_k = numerator/denominator

            assert isinstance(alpha_k, float) # should not be matrix
            p_k     =  - alpha_k  * der_1st


        elif self.method == 'Nesterov’s optimal method':
            # update
            der_y   = self.f.der_1st(self.y)
            x_k     = self.y - 1 / self.f.L * der_y
            p_k     = x_k - self.x

            # calculate for the next iter
            t_k     = _next_t(self.t)
            self.y  = x_k  + (self.t-1)/t_k * p_k
            self.t  = t_k


        elif self.method == 'heavy-ball':

            p_k             = - self.alpha * self.der_1st + self.beta * self.p_lastiter
            self.p_lastiter = p_k 
        
        return p_k

def _next_t(t):
    return 0.5 * (1+np.sqrt(1+4* (t**2) ))

            




