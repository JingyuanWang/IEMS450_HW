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
# class: the objective function and constraints
# ---------------------------------------------------------------------------------

class obj_function:

    def __init__(self):

        return

    def value(self, x):
        assert x.shape == (5, ) or x.shape == (5, 1)
        x = x.flatten()
        
        value =np.exp(np.prod(x)) - 0.5 * ( x[0]**3 + x[1]**3 + 1 )**2

        return float(value)

    def der_1st(self, x):
        assert x.shape == (5, ) or x.shape == (5, 1)
        x = x.flatten()

        f0 = np.exp(np.prod(x)) * x[1]*x[2]*x[3]*x[4] 
        f1 = np.exp(np.prod(x)) * x[0]*x[2]*x[3]*x[4] 
        f2 = np.exp(np.prod(x)) * x[0]*x[1]*x[3]*x[4] 
        f3 = np.exp(np.prod(x)) * x[0]*x[1]*x[2]*x[4] 
        f4 = np.exp(np.prod(x)) * x[0]*x[1]*x[2]*x[3] 
        # can be written in a loop: np.exp(np.prod(x)) * np.prod(x) / x[i]

        f_1st = np.array( [f0, f1, f2, f3, f4] )

        g0 = - ( x[0]**3 + x[1]**3 + 1 ) * 3 * x[0]**2
        g1 = - ( x[0]**3 + x[1]**3 + 1 ) * 3 * x[1]**2
        g_1st = np.array( [g0, g1, 0, 0, 0] )

        return f_1st
        #der_1st = f_1st + g_1st
        #return der_1st

    def der_2nd(self, x):

        # first part
        f_2nd = np.zeros( (5,5) )
        # (1) diagnal elements
        for i in range(5):
            f_2nd[i,i] = (np.prod(x)/x[i])**2

        # (2) off-diagnal 
        for i in range(5):
            for j in range(5):
                if j != i:
                    f_2nd[i,j] = ( (np.prod(x)/x[i])*(np.prod(x)/x[j]) + np.prod(x)/(x[i]*x[j]) )

        f_2nd = np.exp(np.prod(x)) * f_2nd

        # second part
        g_2nd = np.zeros( (5,5) )
        # ... 

        return f_2nd
        # der_2nd = f_2nd + g_2nd
        # return der_2nd

        
class constraints:

    def __init__(self):
        self.m = 3
        return

    def value(self, x):
        x1 = c1.value(x)
        x2 = c2.value(x)
        x3 = c3.value(x)
        return np.array( [x1, x2, x3] )

    def A(self,x):
        x1 = c1.der_1st(x)
        x2 = c2.der_1st(x)
        x3 = c3.der_1st(x)
        return np.vstack(  [x1, x2, x3] )

    def SOC(self, x, l):
        '''l is lagrangian multiplier '''

        x1 = c1.der_2nd(x)
        x2 = c2.der_2nd(x)
        x3 = c3.der_2nd(x)
        return x1*l[0] + x2*l[1] + x3*l[2]


class c1:

    def __init__(self):
        return

    #@classmethod
    def value(x):
        assert x.shape == (5, ) or x.shape == (5, 1)
        x = x.flatten()
        
        value = np.linalg.norm(x)**2 - 10
        return float(value)

    def der_1st(x):
        assert x.shape == (5, ) or x.shape == (5, 1)
        x = x.flatten()

        der_1st = 2*x
        return der_1st
    def der_2nd(x):
        assert x.shape == (5, ) or x.shape == (5, 1)
        x = x.flatten()
        return np.eye(5) * 2


class c2:

    def __init__(self):
        return

    def value( x):
        assert x.shape == (5, ) or x.shape == (5, 1)
        x = x.flatten()
        
        value = x[1] * x[2] - 5*x[3]*x[4]
        return float(value)

    def der_1st( x):
        assert x.shape == (5, ) or x.shape == (5, 1)
        x = x.flatten()

        der_1st = np.array(  [0, x[2], x[1], -5*x[4], -5*x[3]] )
        return der_1st
    def der_2nd(x):
        assert x.shape == (5, ) or x.shape == (5, 1)
        x = x.flatten()

        der_2nd = np.zeros( (5,5) )
        der_2nd[2,1] = 1
        der_2nd[1,2] = 1
        der_2nd[3,4] = -5
        der_2nd[4,3] = -5
        return der_2nd

class c3:

    def __init__(self):
        return

    def value( x):
        assert x.shape == (5, ) or x.shape == (5, 1)
        x = x.flatten()
        
        value = x[1] * x[2] - 5*x[3]*x[4]
        return float(value)

    def der_1st( x):
        assert x.shape == (5, ) or x.shape == (5, 1)
        x = x.flatten()

        der_1st = np.array(  [3*x[0]**2, 3*x[1]**2, 0, 0, 0 ] )
        return der_1st
    def der_2nd(x):
        assert x.shape == (5, ) or x.shape == (5, 1)
        x = x.flatten()

        der_2nd = np.zeros( (5,5) )
        der_2nd[0,0] = 6*x[0]
        der_2nd[1,1] = 6*x[1]
        return der_2nd

# ---------------------------------------------------------------------------------
# class
# ---------------------------------------------------------------------------------
class SQP:

    def __init__(self, function, constraint, print_every_n_step= None, print_results=True):
        ''' 
        Input:
        - function: must be class objective_function.f, with well defined derivatives
        - alpha is the step_length, if input is None, then will calculate in each step. '''

        # 1. save 
        self.f = function
        self.c = constraint
        self.print_every_n_step = print_every_n_step
        self.print_results = print_results


        return

    # MAIN function -------------------------------------
    def run(self, x0, lambda_0,  tol = 1e-6, maxiter = 2000):

        # initialize: i, x0, first obj, and step related 
        m,n = 3,5
        i = 0
        x = x0
        l = lambda_0

        g       = self.f.der_1st(x) # n-by-   
        A       = self.c.A(x)       # m-by-n
        assert np.linalg.matrix_rank(A) == m, 'LICQ fails'
        FOC     = g - l @ A
        self.grad_norm = np.linalg.norm(FOC)

        #while improve > tol and i <= maxiter:
        while self.grad_norm > tol and i <= maxiter:

            # calculate obj and the local problem
            f = self.f.value(x)   # scalar
            G = self.f.der_2nd(x) # n-by-n
            L = G - self.c.SOC(x, l) # n-by-n
            c = self.c.value(x)   # m-by-

            # solve the local problem
            KKT_mat = np.block( [[L, -A.T], [A, np.zeros( (m,m) )] ] )
            RHS     = -np.hstack( (g, c) )
            sol     = np.linalg.solve(KKT_mat, RHS)

            # update 
            p       = sol[:n]
            x       = x + p  
            l       = sol[-m:]

            # calculate new norm, and prepare for the next local function
            g       = self.f.der_1st(x) # n-by-   
            A       = self.c.A(x)       # m-by-n
            assert np.linalg.matrix_rank(A) == m, 'LICQ fails'
            FOC     = g - l @ A
            self.grad_norm = np.linalg.norm(FOC)  

            i = i+1

        return x
