"""
Forward modeling for SYR-type models with a single
susceptible class and multiple infected classes.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp


def SYR_forward(b, alpha, V, s0, y0, T=100):
    """ 
    Forward model for SYR-type models with a single susceptible class and
    n infected classes.

    Inputs:

        - b: n-vector of infectiousness of each infected class
        - alpha: n-vector of probability of a new infected going into each
              infected class.  Entries sum to 1.
        - V: (n x n) matrix of transitions between infected classes plus removals.
    """
    n = len(y0)

    du = np.zeros(n+1)
    u0 = np.zeros(n+1)
    u0[0] = s0
    u0[1:] = y0
    
    def f(t,u):
        s = u[0]
        y = u[1:]
        force = np.dot(y,b)  # Force of infection
        du[0] = - s*force
        du[1:] = s*force*alpha - np.dot(V,y)
        return du

    times = np.linspace(0,T,10000)
    solution = solve_ivp(f,[0,T],u0,t_eval=times,method='RK23',max_step=0.1)
    s = solution.y[0,:]
    y = solution.y[1:,:]
    t = solution.t
    
    return s, y, t


def plot_timeline(x,y,control,t):
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    ax.plot(t,x)
    ax.plot(t,y)
    ax.plot(t,control)
    ax.legend(['x','y','$\sigma/\sigma_0$']);
    plt.xlabel('t');
    ax.autoscale(enable=True, axis='x', tight=True)
    return fig

def plot_timelines(xs,ys,controls,ts,labels):
    palette = plt.get_cmap('tab10')
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    for i in range(len(xs)):
        #ax.plot(ts[i],xs[i])
        ax.plot(ts[i],ys[i],color=palette(i+1),label=labels[i])
        ax.plot(ts[i],controls[i],'--',color=palette(i+1))
    plt.legend()
    plt.xlabel('t');
    ax.autoscale(enable=True, axis='x', tight=True)
    return fig

def plot_phaseplane(xs=None,ys=None,beta=0.3,gamma=0.1,color=None,labels=None):
    sigma0 = beta/gamma
    N1 = 10; N2=5
    Y, X = np.mgrid[0:1:100j, 0:1:100j]
    U = -beta*X*Y
    V = beta*X*Y - gamma*Y
    x_points = list(np.linspace(0,1,N1)) + list(np.linspace(1./sigma0,1,N2))
    y_points = list(1.-np.linspace(0,1,N1)) + [1.e-6]*N2
    seed_points = np.array([x_points, y_points])

    plt.figure(figsize=(6,6))
    plt.streamplot(X, Y, U, V, start_points=seed_points.T,integration_direction='forward',maxlength=1000,
                   broken_streamlines=False,linewidth=1)
    plt.plot([0,1],[1,0],'-k',alpha=0.5)
    if xs is not None:
        i = -1
        for x, y in zip(xs,ys):
            i += 1
            if color:
                plt.plot(x,y,'-',color=color)
            elif labels:
                plt.plot(x,y,'-',label=labels[i])
            else:
                plt.plot(x,y,'-')
    plt.plot([gamma/beta, gamma/beta],[0,1-gamma/beta],'--k',alpha=0.5)
    plt.xlim(0,1); plt.ylim(0,1);
    plt.xlabel('x'); plt.ylabel('y');
    fig = plt.gcf()
    return fig
