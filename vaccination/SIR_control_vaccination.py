import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp

beta_default = np.array([[0.2, 0.2],[0.2,0.2]])  # Contact rates
gamma_default = 0.1
x0_default = [0.6, 0.3]  # Initial susceptibles
y0_default = [0.01, 0.01] # Initial infected
r_default = [0.001,0.01]  # Risk of death for each cohort

def SIR_forward(beta=beta_default, gamma=gamma_default, x0=x0_default, y0=y0_default, T=365):
    """ Model the current outbreak using the SIR model."""

    n_cohorts = beta.shape[0]
    assert(n_cohorts == 2)

    du = np.zeros(2*n_cohorts)
    u0 = np.zeros(2*n_cohorts)
    
    def f(t,u):
        # x = u[:n_cohorts]
        # y = u[n_cohorts:]

        du[0] = -u[0]*(beta[0,0]*u[2] + beta[0,1]*u[3])
        du[1] = -u[1]*(beta[1,0]*u[2] + beta[1,1]*u[3])

        du[2] = u[0]*(beta[0,0]*u[2] + beta[0,1]*u[3]) - gamma*u[2]
        du[3] = u[1]*(beta[1,0]*u[2] + beta[1,1]*u[3]) - gamma*u[3]

        return du

    # Initial values
    u0[0] = x0[0]
    u0[1] = x0[1]
    u0[2] = y0[0]
    u0[3] = y0[1]

    times = np.linspace(0,T,10000)
    solution = solve_ivp(f,[0,T],u0,t_eval=times,method='RK23',max_step=0.1)
    x_young = solution.y[0,:]
    x_old   = solution.y[1,:]
    y_young = solution.y[2,:]
    y_old   = solution.y[3,:]
    t = solution.t
    
    return x_young, x_old, y_young, y_old, t


def SIR_adjoint(xx, yy, tt, beta=beta_default, gamma=gamma_default, r=r_default, T=365):
    
    n_cohorts = beta.shape[0]
    assert(n_cohorts == 2)

    lam0 = np.zeros(2*n_cohorts)
    dlam = np.zeros(2*n_cohorts)
    x = np.zeros(n_cohorts)
    y = np.zeros(n_cohorts)
    
    def f_adjoint(t,lam):
        # lambda = lam[:n_cohorts]
        # mu = lam[n_cohorts:]

        x[0] = np.interp(t, tt, xx[0,:])
        x[1] = np.interp(t, tt, xx[1,:])
        y[0] = np.interp(t, tt, yy[0,:])
        y[1] = np.interp(t, tt, yy[1,:])
        
        dlam[0] = (lam[0] - lam[2])*(beta[0,0]*y[0] + beta[0,1]*y[1])
        dlam[1] = (lam[1] - lam[3])*(beta[1,0]*y[0] + beta[1,1]*y[1])

        dlam[2] = (lam[0]-lam[2])*beta[0,0]*x[0] + (lam[1]-lam[3])*beta[1,0]*x[1] + gamma*lam[0]
        dlam[3] = (lam[0]-lam[2])*beta[0,1]*x[0] + (lam[1]-lam[3])*beta[1,1]*x[1] + gamma*lam[1]

        return dlam

    # Final values
    lam0[0] = -r[0]
    lam0[1] = -r[1]
    lam0[2] = 0
    lam0[3] = 0

    times = np.linspace(T,0,10000)
    solution = solve_ivp(f_adjoint,[T,0],lam0,t_eval=times,method='RK23',max_step=0.1)

    return solution.y, solution.t



def SIR_forward_with_vaccination(afun, lam, tt, r=r_default, beta=beta_default, gamma=gamma_default, 
                                 x0=x0_default, y0=y0_default, T=365):
    """ Model the current outbreak using the SIR model."""

    n_cohorts = beta.shape[0]
    assert(n_cohorts == 2)

    dw = np.zeros(3*n_cohorts)
    w0 = np.zeros(3*n_cohorts)
    u = np.zeros(n_cohorts)
    lamda = np.zeros(n_cohorts)

    def f(t,w):
        # x = u[:n_cohorts]
        # y = u[n_cohorts:]
        a = afun(t)
        doses = a - w[4] - w[5]
        if doses>0:
            lamda[0] = np.interp(t, tt, lam[0,:])
            lamda[1] = np.interp(t, tt, lam[1,:])
            
            u[:] = 0.

            if lamda[0]+r[0] > lamda[1]+r[1]:
                #print(t,'young')
                if doses < w[0]:
                    u[0] = doses
                elif doses < w[1]:
                    u[1] = doses
            else:
                #print(t,'old')
                if doses < w[1]:
                    u[1] = doses
                elif doses < w[0]:
                    u[0] = doses
                

        dw[0] = -w[0]*(beta[0,0]*w[2] + beta[0,1]*w[3]) - u[0]
        dw[1] = -w[1]*(beta[1,0]*w[2] + beta[1,1]*w[3]) - u[1]

        dw[2] = w[0]*(beta[0,0]*w[2] + beta[0,1]*w[3]) - gamma*w[2]
        dw[3] = w[1]*(beta[1,0]*w[2] + beta[1,1]*w[3]) - gamma*w[3]

        dw[4] = u[0]
        dw[5] = u[1]

        return dw

    # Initial values
    w0[0] = x0[0]
    w0[1] = x0[1]
    w0[2] = y0[0]
    w0[3] = y0[1]
    w0[4] = 0.  # Vaccine used so far
    w0[5] = 0.  # Vaccine used so far

    times = np.linspace(0,T,10000)
    solution = solve_ivp(f,[0,T],w0,t_eval=times,method='RK23',max_step=0.1)
    x_young = solution.y[0,:]
    x_old   = solution.y[1,:]
    y_young = solution.y[2,:]
    y_old   = solution.y[3,:]
    u_young = solution.y[4,:]
    u_old   = solution.y[5,:]
    t = solution.t
    
    return x_young, x_old, y_young, y_old, u_young, u_old, t












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
