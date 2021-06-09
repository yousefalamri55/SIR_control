# The code here was to test the reduction of SIR to one ODE
# (using the invariant).  It is correct, but when \sigma
# is allowed to vary then we lose the invariant so the ODE
# doesn't make sense any more.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp
from scipy.special import lambertw, expit

def SIR_forward_augmented(qfun=None, beta=0.3, gamma=0.1, x0=0.99, y0=0.01, T=100):
    """ Model the current outbreak using the SIR model."""

    du = np.zeros(3)
    u0 = np.zeros(3)
    
    def f(t,u):
        qval = qfun(t,u)
        du[0] = -(1-qval)*beta*u[1]*u[0]
        du[1] = (1-qval)*beta*u[1]*u[0] - gamma*u[1]
        du[2] = -(1-qval)*beta*u[0]*(y0+x0-u[0]) - gamma*u[0]*np.log(u[0]/x0)
        return du

    # Initial values
    u0[1] = y0 # Initial infected
    u0[0] = x0
    u0[2] = x0

    times = np.linspace(0,T,10000)
    solution = solve_ivp(f,[0,T],u0,t_eval=times,method='RK23',max_step=0.1)
    x = solution.y[0,:]
    y = solution.y[1,:]
    x2 = solution.y[2,:]
    t = solution.t
    
    return x, y, x2, t
