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


class linesearch:

    method_list = ['steepest descent', 'basic Newton’s method', 'BFGS']

    def __init__(self, function, method = 'steepest descent', rho=0.5, c=1e-4, print_every_n_step= None, print_results=True):
        ''' 
        Input:
        - function: must be class objective_function.f, with well defined derivatives
        - method must be in the method_list. '''

        # 1. save the function
        self.f = function
        self.n = self.f.n

        # 2. save the method
        assert method in self.method_list
        self.method = method

        # 3. save the backtracking parameters
        self.rho = rho 
        self.c   = c

        # 4. save printing parameter
        self.print_every_n_step = print_every_n_step
        self.print_results = print_results

        return

    def run(self, x0, tol=1e-5, maxiter=500, print_every_n_step= None):
        '''main function of the algorithm'''

        # 0. initialize ----------------------------------------------------------
        # check the shape of x0
        assert x0.shape == (self.n,) or x0.shape == (self.n,1)
        if x0.shape == (self.n,):
            x0 = x0.reshape( (self.n,1) )
        if print_every_n_step is not None:
            self.print_every_n_step = print_every_n_step
        print('===========================================')
        print(self.method)
        print('-------------------------------------------')

        # 1. initialize: i, x0, first obj, and step related -----------------------
        self.i  = 0
        self.x = x0
        self.obj = self.f.value(x0)
        self.obj_list = [self.obj]
        self.der_1st = self.f.der_1st(self.x)
        self.grad_norm = np.linalg.norm(self.der_1st)
        # save H if BFGS
        if self.method == 'BFGS':
            self.der_1st_lastiter = 0
            self.H                = np.linalg.inv( self.f.der_2nd(self.x) )
            self.I                = np.eye( max(x0.shape) )

        if self.print_every_n_step is not None:
            self._print_first_iter()

        
        # 2. start the iterations -------------------------------------------------
        while self.grad_norm > tol and self.i <= maxiter:

            # A find the next step, and calculate the objective function at the next x
            # (1) find direction
            p_k, alpha = self.get_direction()  
            # (2) find step length, and output the last evaluated f value.
            alpha_k, x_new, obj, n_evaluation = self.backtracking(p_k, alpha) 

            # B save the updated x and obj
            self.i         = self.i + 1
            self.obj       = obj
            self.obj_list  = self.obj_list + [obj]
            self.x         = x_new
            if self.method == 'BFGS':
                self.der_1st_lastiter = self.der_1st
                self.s_k   = alpha_k * p_k
            self.der_1st   = self.f.der_1st(self.x)
            self.grad_norm = np.linalg.norm(self.der_1st)
            # double check shape
            assert isinstance(self.obj, float)
            assert isinstance(alpha_k, float)
            assert self.x.shape == (self.n,1)
            assert self.der_1st.shape == (self.n,1)
            assert p_k.shape == (self.n,1)

            # C print
            if self.print_every_n_step is not None:
                if np.round(self.i/self.print_every_n_step) == self.i/self.print_every_n_step:
                    self._print_each_iter(p_k, alpha_k, n_evaluation)


        # 3. print results ----------------------------------------------------------
        if self.print_results:
            print('===========================================')
            if self.i > maxiter:
                print(f'not converge, last iter {self.i}: obj = {self.obj:.6E}')
            else:
                print(f'complete in {self.i:4d} iter: obj = {self.obj:.6E}')
            print('===========================================')
            print('\n')

        return


    # --------------------------------------------------
    # functions called in each iteration
    # --------------------------------------------------

    def get_direction(self):
        '''In each iteraction, find the direction to go '''
        if self.method == 'steepest descent':
            p_k   = -self.der_1st
            alpha = 1.0

        elif self.method == 'basic Newton’s method':
            B     = self.f.der_2nd(self.x)
            p_k   = - np.linalg.solve( B, self.der_1st)
            alpha = 1.0

        elif self.method == 'BFGS':
            if self.i == 0:
                H       = self.H
            if self.i > 0 :
                y_k     = self.der_1st - self.der_1st_lastiter
                s_k     = self.s_k
                assert y_k.shape == s_k.shape
                assert s_k.shape == (self.n, 1)
                rho_k   = 1/(y_k.T@s_k)
                assert rho_k.shape == (1,1)
                rho_k   = rho_k[0]
                I       = self.I

                # BFGS
                H       = (I- rho_k*s_k@y_k.T) @ self.H @ (I- rho_k*y_k@s_k.T) + rho_k*s_k@s_k.T

            p_k    = - H @ self.der_1st 
            self.H = H
            alpha = 1.0

        return  p_k, alpha


    def backtracking(self, p_k, alpha_0):

        # 1. save fixed values
        unit_step = self.der_1st.T @ p_k

        # 2. initialize
        alpha                = alpha_0
        x_new                = self.x   + alpha*p_k
        max_acceptable_value = self.obj + self.c * alpha * unit_step
        actual_value         = self.f.value(x_new)
        j                    = 1

        # 3. iteration
        while (actual_value>max_acceptable_value) and (j<100):

            # update
            alpha   = self.rho * alpha

            # evaluate
            x_new                = self.x   + alpha*p_k
            max_acceptable_value = self.obj + self.c * alpha * unit_step
            actual_value         = self.f.value(x_new)

            # number of evalueations
            j                    = j+1 


        n_evaluation = j
        obj          = actual_value

        return alpha, x_new, obj, n_evaluation



    # --------------------------------------------------
    # auxiliary functions
    # --------------------------------------------------
    def _print_first_iter(self):
        print(' iter ' 
            + '           f ' 
            + '      ||p_k|| '
            + '       alpha '
            + '  #func '
            + '  ||grad_f|| ' )

        step_norm        = 0
        alpha            = 0
        N_func_evaluated = 0
        grad_norm        = self.grad_norm

        print(f'{self.i:4d}  '
            + f'   {self.obj:.4E} ' 
            + f'   {step_norm:.4E}' 
            + f'   {alpha:.4E}' 
            + f' {N_func_evaluated:6d} '
            + f'   {grad_norm:.4E}')

        return 

    def _print_each_iter(self, p_k, alpha_k, n_evaluation):

        step_norm        = np.linalg.norm(p_k)
        alpha            = alpha_k
        N_func_evaluated = n_evaluation
        grad_norm        = self.grad_norm

        print(f'{self.i:4d}  '
            + f'   {self.obj:.4E} ' 
            + f'   {step_norm:.4E}' 
            + f'   {alpha:.4E}' 
            + f' {N_func_evaluated:6d} '
            + f'   {grad_norm:.4E}')

        return 






            




