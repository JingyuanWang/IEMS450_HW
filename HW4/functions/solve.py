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
def graphical_solution():
    import matplotlib.pyplot as plt
    # figure
    fig = plt.figure(figsize=(8,8))
    ax = fig.gca()

    # obj
    circle1 = plt.Circle((3, 2), 0.5, facecolor='white', edgecolor='orange', fill=False)
    circle2 = plt.Circle((3, 2), 1 , facecolor='white', edgecolor='orange', fill=False)
    circle3 = plt.Circle((3, 2), np.sqrt(2), facecolor='white', edgecolor='orange', fill=False)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)

    # C1
    x1 = np.linspace( -.5,6,200 )
    x2 = 3 - x1
    plt.plot(x1, x2 ,  color='purple')

    # fill 
    # from 0 to 3
    ax.fill_between(x1[ (x1>=0) * ( x1<=3) ] , 0, x2[ (x1>=0) * ( x1<=3) ], color='lightgrey' )

    # optimal:
    plt.scatter( 2,1, color='red' , marker='o', s=60)

    # labels and axis
    plt.vlines(0, -3, 5, color='black', linestyles='dashed')
    plt.hlines(0, -3, 5, color='black', linestyles='dashed')
    plt.xlim( [-.5,5] )
    plt.ylim( [-.5,5] )
    plt.grid(linestyle='dashed')
    plt.show()
    
    return

# ---------------------------------------------------------------------------------
# class: the objective function
# ---------------------------------------------------------------------------------
            

class obj_function:

    def __init__(self):

        self.G = np.eye(2) * 2
        self.c = np.array( [-6, -4] )
        # function = 1/2 x G x + x c

        return

    def value(self, x):
        assert x.shape == (2, ) or x.shape == (2, 1)
        
        value = 0.5 * x.T @ self.G @ x + x.T @ self.c + 13

        return float(value)

    def der_1st(self, x):

        der_1st = self.G @ x + self.c 

        return der_1st

# ---------------------------------------------------------------------------------
# class active_set
# ---------------------------------------------------------------------------------
class active_set:


    def __init__(self, function, lin_constr_A, lin_constr_b):
        ''' 
        Input:
        - function: must be class obj_function, with well defined derivatives'''

        # 1. save the function
        self.f = function
        self.A = lin_constr_A
        self.b = lin_constr_b
        assert self.A.shape[0] == self.b.shape[0]

        # 2. save values
        self.constraint_list = list(range(self.A.shape[0]))

        return

    def run(self, x0, maxiter=500, print_every_n_step= False, print_results=False):
        '''main function of the algorithm'''

        # 0. initialize ----------------------------------------------------------
        # check the shape of x0
        assert x0.shape == (2,) or x0.shape == (2,1)
        tol             = 1e-10

        # 1. initialize: i, x0, first obj, and step related -----------------------
        # initialize
        self.i          = 0
        self.x          = x0
        lambdas         = -999
        p               = np.ones( (2, ))
        assert len(list(np.where(self.A@x0 -self.b > 0)[0])) == 0, 'not feasible, please change an initial value'
        self.list_active_constraints = list(np.where(self.A@x0 - self.b == 0)[0])
        if print_every_n_step:
            self._print_first_iter()
        
        # 2. start the iterations -------------------------------------------------
        while (np.min(lambdas) < 0  or np.linalg.norm(p) > tol) and self.i <= maxiter:

            # (1) find the next step
            p = self.get_direction()

            # for print:
            alpha = 0
            if np.linalg.norm(p) != 0:
                alpha = self.get_steplength(p)
            lambdas = self.get_lambdas()   # this is unnecessary for all cases, can be moved under if np.linalg.norm(p)==0. 
                                           # calculate for all cases just for printing purpose

            # print 
            if print_every_n_step is not False:
                if np.round(self.i/print_every_n_step) == self.i/print_every_n_step:
                    self._print_each_iter(p, alpha, lambdas)

            # (2) update constraint 
            if np.linalg.norm(p)==0:
                min_val = np.min(lambdas)
                if min_val < 0:
                    i = np.where(lambdas==min_val)[0][0]
                    self.list_active_constraints.remove(i)

            # (3) update x
            else:
                self.x = self.x + alpha * p
                self.list_active_constraints = list(np.where(self.A@self.x - self.b >= -tol)[0])
                self.list_active_constraints.sort()

            self.i = self.i + 1

        return


    def get_direction(self):

        if len(self.list_active_constraints ) == 2 :
            # two constraints, no where to go
            A = self.A[ self.list_active_constraints, : ]
            assert  np.linalg.matrix_rank(A) == 2 
            # not true in general, but should be true in this specific question

            p = np.zeros( (2, ) )

        elif len(self.list_active_constraints ) == 1 :
            # 1 constraint, can go along the null space vector, + or -

            A = self.A[ self.list_active_constraints, : ]
            A = np.hstack( (A.T ,np.zeros( (2,1) )) )
            Q, R = np.linalg.qr(A)
            p = Q[:,1]
            assert np.max(A.T @ p) <= 1e-10

            der_1st = self.f.der_1st(self.x)

            alpha = - float(der_1st@p / ( p@self.f.G@p ) )

            p = alpha * p
            # now p is argmin of the local problem

        else:
            assert len(self.list_active_constraints ) == 0
            # no active constraint, unconstrained problem
            x_opt =  np.linalg.solve(self.f.G, -self.f.c)
            p = x_opt - self.x

        return p

    def get_steplength(self, p):

        other_constraints = [i for i in self.constraint_list if i not in self.list_active_constraints]
        critical_values = [1]
        for i in other_constraints:
            
            a = self.A[i, :]
            b = self.b[i]

            if a@p > 0:
                critical = (b - a@self.x) / (a@p)
                critical_values = critical_values + [ float(critical) ]

        alpha = min( critical_values )

        return alpha


    def get_lambdas(self):

        lambdas = np.zeros( (3, ) )
        if len(self.list_active_constraints) == 2:
            A = self.A[ self.list_active_constraints, : ]
            g = self.f.der_1st(self.x)
            lambdas[self.list_active_constraints] = np.linalg.solve(A.T, -g)
        elif len(self.list_active_constraints) == 1:

            A = self.A[ self.list_active_constraints, : ].flatten()
            g = self.f.der_1st(self.x)
            # solve: A * lambda = g

            if g[0] != 0:
                lambdas[self.list_active_constraints] = -A[0]/g[0]
            elif g[1] != 0:
                lambdas[self.list_active_constraints] = -A[1]/g[1]

        return lambdas


    def _print_first_iter(self):
        #np.set_printoptions(precision=2,suppress=True)
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})


        print(' iter ' 
            + '         f ' 
            + '           x   '
            + '    Active C   ' 
            + '      lambdas      ' 
            + '      p_(k)    ' 
            + '   alpha_(k)'
            )

        return 

    def _print_each_iter(self, p_k, alpha, lambdas):

        step_norm        = np.linalg.norm(p_k)
        obj = self.f.value(self.x)

        print(f'{self.i:4d}  '
            + f'   {obj:.2E} ' 
            + f'  {self.x}' 
            + f'   {self.list_active_constraints}   '
            + f'   {lambdas}  ' 
            + f'   {p_k}' 
            + f'   {alpha:.2E}' 
            )

        return 
