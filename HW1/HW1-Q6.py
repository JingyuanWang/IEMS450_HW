# Author: Jingyuan Wang
# python HW1-Q6.py


import numpy as np
import os,sys,inspect
import importlib

randseed = 13344

# import my functions:
dirpath = os.getcwd()
i = 0
while(os.path.basename(dirpath) != "IEMS450_HW") and i <10:
    dirpath = os.path.dirname(dirpath) 
    i = i + 1
targetdir = dirpath + '/HW1/functions'
if targetdir not in sys.path:
    sys.path.insert(0,targetdir)

import objective_function as obj
importlib.reload(obj)
import opt_quadraticfunc as opt_q
importlib.reload(opt_q)


# # Initialize =============================================================

# set up the problem
np.random.seed(randseed)
m, L = 0.01, 1
D = 10 ** np.random.rand(100)
D = (D - np.min(D)) / (np.max(D) - np.min(D)) 
A = np.diag(m + D * (L - m))

# save the function
f = obj.f(A)



# Part a: Perform 10 trial runs, varying the staringpoint, and report the averaged the number of iterations required for convergence.

# ---- I. starting point
np.random.seed(randseed)
x_0 = {}
for i in range(10):
    x_0[i] = np.random.rand(100)

# ---- II. run optimization
required_steps = {'constant step length': [],
                  'exact line search'   : [],
                  'Nesterov’s optimal method' : [],
                  'heavy-ball' : []}

# initialize the methods
importlib.reload(opt_q)
model1 = opt_q.steepest_descent(f, method = 'constant step length', alpha=1/(f.L), print_results=False)
model2 = opt_q.steepest_descent(f, method = 'exact line search'   , print_results=False)
model3 = opt_q.steepest_descent(f, method = 'Nesterov’s optimal method', print_results=False)
model4 = opt_q.steepest_descent(f, method = 'heavy-ball', print_results=False)
methods        = {'constant step length': model1,
                  'exact line search'   : model2,
                  'Nesterov’s optimal method' : model3,
                  'heavy-ball' : model4}

# run for different x0
for j in range(10):
    for method in opt_q.steepest_descent.method_list:
        methods[method].run(x0 = x_0[j])
        num_iter = methods[method].i
        # save convergent steps
        required_steps[method] = required_steps[method] + [num_iter]


# ---- III. print
for method in opt_q.steepest_descent.method_list:
    steps = np.array(required_steps[method]).mean()
    print(f'{method} - fixed steps : {steps :4.1f} ')


# Part b: Draw a plot of the convergence behavior on a typical run ===============================================
import matplotlib.pyplot as plt

burn_steps = 0
stop_step  = 500

# draw the result of the last run
plt.figure(figsize=(9, 3))
plt.plot( np.log10( model1.obj_list[burn_steps:stop_step] ), label = model1.method)
plt.plot( np.log10( model2.obj_list[burn_steps:stop_step] ), label = model2.method)
plt.plot( np.log10( model3.obj_list[burn_steps:stop_step] ), label = model3.method)
plt.plot( np.log10( model4.obj_list[burn_steps:stop_step] ), label = model4.method)
plt.legend(loc='best')





