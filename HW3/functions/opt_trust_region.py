'''
# ------------------------------------------------------------------------
# NOTE
# ------------------------------------------------------------------------
# Purpose:
# define line search class
# ------------------------------------------------------------------------
'''


import numpy as np
import os,sys,inspect
import copy
import importlib



# ---------------------------------------------------------------------------------
# class, trust region
# ---------------------------------------------------------------------------------
class trust_region:


    def __init__(self, function, eta, delta_hat ):
        ''' 
        Input:
        - function: must be class objective_function.f, with well defined derivatives
        - method must be in the method_list. '''

        # 1. save the function
        self.f = function
        self.n = self.f.n
        # 2. save parameters
        assert delta_hat > 0
        self.delta_hat = delta_hat
        assert eta  >= 0 and eta < 0.25
        self.eta = eta 

        return

    def run(self, x0, delta0, tol=1e-6, maxiter=500, print_every_n_step= False, print_results=False):
        '''main function of the algorithm'''

        # 0. initialize ----------------------------------------------------------
        # check the shape of x0
        assert x0.shape == (self.n,) or x0.shape == (self.n,1)
        if x0.shape == (self.n,):
            x0 = x0.reshape( (self.n,1) )
        self.print_every_n_step = print_every_n_step
        self.print_results = print_results
        if self.print_results:
            print('===========================================')
            print('trust region newton-CG')
            print('-------------------------------------------') 
        # 1. initialize: i, x0, first obj, and step related -----------------------
        # initialize
        self.i          = 0
        self.x          = x0
        self.obj        = self.f.value(self.x)
        self.m          = m(self.x, self.f) 
        self.grad_norm  = np.linalg.norm(self.m.der_x0)
        delta = delta0
        # print
        if self.print_every_n_step is not False:
            self._print_first_iter()

        
        # 2. start the iterations -------------------------------------------------
        while self.grad_norm > tol and self.i <= maxiter:

            # A. find the next step, and calculate the objective function at the next x
            # (1) find the next step
            p = self.get_step(delta) 

            # (2) update delta 
            rho = self.get_reduction_rate(p)
            if rho < 0.25:
                delta = delta / 4
            elif rho > 0.75 and np.linalg.norm(p) == delta:
                delta = np.min( (2*delta, self.delta_hat) )

            # (3) update x
            if rho > self.eta:
                self.x     = self.x + p          # update x 
                self.obj   = self.obj_new        # update obj value
                self.m     = m(self.x, self.f)   # update quadratic approx of f at new x
                self.grad_norm  = np.linalg.norm(self.m.der_x0)

            # double check mat shape, update i 
            assert isinstance(self.obj, float)
            self.i = self.i + 1

            # B. print
            if self.print_every_n_step is not False:
                if np.round(self.i/self.print_every_n_step) == self.i/self.print_every_n_step:
                    self._print_each_iter(p)


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
    # functions in each iteration
    # --------------------------------------------------
    def get_reduction_rate(self, p):

        self.obj_new        = self.f.value(self.x+p)
        actual_reduction    = self.obj - self.obj_new  
        predicted_reduction = self.m.value_step(np.zeros( (self.n, ) )) - self.m.value_step(p)
        return actual_reduction/predicted_reduction

    def get_step(self, delta):

        A        = self.m.H_x0
        b        = -self.m.der_x0
        self.eps = np.min( (0.3, np.sqrt(self.grad_norm) ) ) * self.grad_norm
        assert self.grad_norm == np.linalg.norm(b)

        p, self.i_inner = CG_direction(A=A, b=b,delta=delta, tol=self.eps)
        return p 

    # --------------------------------------------------
    # auxiliary functions
    # --------------------------------------------------
    def _print_first_iter(self):
        print(' iter ' 
            + '           f ' 
            + '      ||p_k|| '
            + '         eps '
            + '    #CG  '
            + '  ||grad_f|| ' )

        step_norm       = 0
        eps             = 0
        N_CG            = 0

        print(f'{self.i:4d}  '
            + f'   {self.obj:.4E} ' 
            + f'   {step_norm:.4E}' 
            + f'   {eps:.4E}' 
            + f' {N_CG:6d} '
            + f'   {self.grad_norm:.4E}')

        return 

    def _print_each_iter(self, p_k):

        step_norm        = np.linalg.norm(p_k)

        print(f'{self.i:4d}  '
            + f'   {self.obj:.4E} ' 
            + f'   {step_norm:.4E}' 
            + f'   {self.eps:.4E}' 
            + f' {self.i_inner:6d} '
            + f'   {self.grad_norm:.4E}')

        return 


# ---------------------------------------------------------------------------------
# function: inner loop
# ---------------------------------------------------------------------------------
def CG_direction(A, b, delta, tol):
    itermax_innerloop = 200

    z  = np.zeros( (A.shape[0], 1) )
    r  = -b
    d  = -r # in this context, this is the steepest descent direction

    i_inner = 0
    while (np.linalg.norm(r) > tol) and (i_inner<itermax_innerloop):

        # 1. descenting direction? 
        if float(d.T @ A @ d) <0:
            return   _p(z, d, delta), i_inner
        # 2. CG update 
        res_sq_last  = (r.T @ r)
        alpha        = float(res_sq_last / (d.T @ A @ d))
        z_new        = z + alpha * d
        if np.linalg.norm(z_new) >= delta:
            return   _p(z, d, delta), i_inner
        z            = z_new
        r            = r + alpha * A @ d
        # update d 
        beta         = float((r.T @ r) / res_sq_last)
        d            = -r + beta * d
        i_inner      = i_inner + 1

    p = z 

    return p, i_inner

def _p(z, d, delta):

    a = np.linalg.norm(d)**2
    b = float(2 * z.T @ d )
    c = np.linalg.norm(z)**2 - delta**2

    D = np.sqrt(b**2-4*a*c)
    tau1 = (-b+D) / (2*a)
    tau2 = (-b-D) / (2*a)
    assert tau2 < 0 and tau1 > 0

    p = z + tau1 * d   # must be in the same direction as d

    return p 


# ---------------------------------------------------------------------------------
# class, quadratic approximation of f at point x 
# ---------------------------------------------------------------------------------
class m:

    def __init__(self, x, f=None, f_x=None, grad=None, H = None):
        '''set the function, 
        -- in put the function f, or input the gradient and hession 
        -- x must be a point. Will do Taylor expansion of f at the point x'''

        self.n           = f.n
        self.x0          = x
        # obj value 
        if f_x is None:
            self.f_x0    = f.value(x)
        else:
            self.f_x0    = f_x
        # gradient
        if grad is None:
            self.der_x0  = f.der_1st(x)
        else:
            self.der_x0  = grad
        # hessian
        if H is None:
            self.H_x0    = f.der_2nd(x)
        else:
            self.H_x0    = H
        return

    def value_step(self, d):
        assert d.shape == (self.n ,) or d.shape == (self.n,1)
        d = d.reshape( (self.n,) )

        value  = self.f_x0 + self.der_x0.T @ d + 0.5 * d.T @ self.H_x0 @ d

        return float(value)


            




