import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from ipywidgets import interact, widgets
import matplotlib.dates as mdates
from scipy.integrate import solve_ivp, solve_bvp
from IPython.display import Image
plt.style.use('seaborn-poster')
matplotlib.rcParams['figure.figsize'] = (10., 6.)
from scipy.special import lambertw, expit
import sys
sys.path.append('../covid_forecasting')
import covid_forecast as cf
import data
import deconvolution
from datetime import datetime
palette = plt.get_cmap('tab10')

# $x_1$: high-risk population  
# $x_2$: low-risk population

dgamma = cf.default_gamma
dbeta = cf.default_beta
today = int(mdates.date2num(datetime.today()))
pop_split = 50

def get_ifrs(region,which='mean'):
    """
    Calculate average infection fatality ratio for a population,
    given age-group-specific fatality ratios and assuming a uniform
    attack rate across all age groups.
    """
    N = data.get_population(region)
    age_data = data.age_distribution
    if not region in age_data.keys():
        raise Exception
    if which == 'mean':
        ifr_values = data.ifr
    elif which == 'low':
        ifr_values = data.ifr_low
    elif which == 'high':
        ifr_values = data.ifr_high
    pa = age_data[region]
    pop_decades = np.zeros(9)
    for decade in range(8):
        pop_decades[decade] = pa[decade*2]+pa[decade*2+1]
    ifr_young = np.dot(pop_decades[:pop_split//10],ifr_values[:pop_split//10])/np.sum(pop_decades[:pop_split//10])
    ifr_old = np.dot(pop_decades[pop_split//10:],ifr_values[pop_split//10:])/np.sum(pop_decades[pop_split//10:])
    return ifr_young, ifr_old


def x_inf_u(u,sigma):
    return x_inf(u[0]+u[1],u[2]+u[3],sigma)

def x_inf(x,y,sigma):
    return -1./sigma * np.real(lambertw(-x*sigma*np.exp(-sigma*(x+y))))

def mu(x,y,sigma):
    return x*np.exp(-sigma*(x+y))

def dxinf_dy(x,y,sigma):
    xinf = x_inf(x,y,sigma)
    return -sigma*xinf/(1-sigma*xinf)

def dxinf_dx(x,y,sigma):
    xinf = x_inf(x,y,sigma)
    return (xinf/x)*(1-sigma*x)/(1-sigma*xinf)

def dJ_dx1(x1,x2,y1,y2,sigma,c1,c2):
    x = x1+x2; y = y1+y2
    xinf = x_inf(x,y,sigma)
    return -dxinf_dx(x,y,sigma)*(c1*x1+c2*x2)/x - xinf*(c1-c2)*x2/x**2

def dJ_dx2(x1,x2,y1,y2,sigma,c1,c2):
    x = x1+x2; y = y1+y2
    xinf = x_inf(x,y,sigma)
    return -dxinf_dx(x,y,sigma)*(c1*x1+c2*x2)/x - xinf*(c2-c1)*x1/x**2

def dJ_dy1(x1,x2,y1,y2,sigma,c1,c2):
    x = x1+x2; y = y1+y2
    return -dxinf_dy(x,y,sigma)*(c1*x1+c2*x2)/x

def dJ_dy2(x1,x2,y1,y2,sigma,c1,c2):  # Same as dJ_dy1
    x = x1+x2; y = y1+y2
    return -dxinf_dy(x,y,sigma)*(c1*x1+c2*x2)/x

def optimal_intervention(x10, x20, y10, y20, c1, c2, c3, c4, c5, beta=dbeta, gamma=dgamma,
                         T=200, ymax=0.04, multipliers=(500,150,50,30,20,10,8,5,3,2,1.4,1.2,1.1),
                         verbose=True,Nt=10000):
    """
    Solve the optimal control problem by relaxing it to an easy version and
    then gradually improving the initial guess.
    """
    print(c1,c2,c3,c4,c5)
    if verbose: print(multipliers[0])
    x1, x2, y1, y2, q1star, q2star, t, newguess = \
            solve_pmp(beta=beta,gamma=gamma,x10=x10,x20=x20,y10=y10,y20=y20,
                      c1=c1,c2=c2,c3=multipliers[0]*c3,c4=c4,c5=c5,
                      ymax=ymax,T=T,guess=None,Nt=Nt,tol=1e-3)
    for mult in multipliers[1:]:
        if verbose: print(mult, len(t))
        x1, x2, y1, y2, q1star, q2star, t, newguess = \
                solve_pmp(beta=beta,gamma=gamma,x10=x10,x20=x20,y10=y10,y20=y20,
                          c1=c1,c2=c2,c3=mult*c3,c4=c4,c5=c5,
                          ymax=ymax,T=T,guess=newguess,Nt=Nt,tol=1e-1)
    if verbose: print('1')
    x1, x2, y1, y2, q1star, q2star, t, newguess = \
            solve_pmp(beta=beta,gamma=gamma,x10=x10,x20=x20,y10=y10,y20=y20,
                      c1=c1,c2=c2,c3=c3,c4=c4,c5=c5,
                      ymax=ymax,T=T,guess=newguess,Nt=Nt,tol=1e-3,max_nodes=2000000)
    return x1, x2, y1, y2, q1star, q2star, t


def solve_pmp(beta=0.3, gamma=0.1,
              x10=0.48, x20=0.48, y10=0.01, y20=0.01,
              c1=0.5,c2=1.0,c3=10, c4=0., c5=0., T=100., qmax=1., ymax=0.04,
              guess=None, loadsol=False, usejac=False, Nt=1000, tol=1e-6, max_nodes=300000):

    sigma0 = beta/gamma

    def rhs(t, u):
        # Variables: x1, x2, y1, y2, lambda_1, lambda_2, lambda_3, lambda_4
        du = np.zeros((8,len(t)))
        x1 = u[0,:]; x2 = u[1,:]; x=x1+x2
        y1 = u[2,:]; y2 = u[3,:]; y=y1+y2
        alpha = expit(10*(y-ymax))*(y-ymax)  # Would be better to account for different hosp. rates by age here
        lam1 = u[4,:]; lam2 = u[5,:]; lam3 = u[6,:]; lam4 = u[7,:]

        q1star = (lam3-lam1)*beta*x1*y/(2*c3)
        q1star = np.maximum(0,np.minimum(1,q1star))

        q2star = (lam4-lam2)*beta*x2*y/(2*c3)
        q2star = np.maximum(0,np.minimum(1,q2star))

        du[0,:] = -(1-q1star)*beta*y*x1
        du[1,:] = -(1-q2star)*beta*y*x2
        du[2,:] = (1-q1star)*beta*y*x1 - gamma*y1
        du[3,:] = (1-q2star)*beta*y*x2 - gamma*y2

        du[4,:] = (lam1-lam3)*(1-q1star)*beta*y # - dL/dx1
        du[5,:] = (lam2-lam4)*(1-q2star)*beta*y # - dL/dx2
        du[6,:] = (lam1-lam3)*(1-q1star)*beta*x1 + (lam2-lam4)*(1-q2star)*beta*x2 + lam3*gamma - c4*alpha
        du[7,:] = (lam1-lam3)*(1-q1star)*beta*x1 + (lam2-lam4)*(1-q2star)*beta*x2 + lam4*gamma - c5*alpha

        return du

    def jac(t, u):
        m = u.shape[1]
        M = np.zeros((8,8,m))

        x1 = u[0,:]; x2 = u[1,:]; x=x1+x2
        y1 = u[2,:]; y2 = u[3,:]; y=y1+y2
        lam1 = u[4,:]; lam2 = u[5,:]; lam3 = u[6,:]; lam4 = u[7,:]

        q1 = (lam3-lam1)*beta*x1*y/(2*c3)
        q1 = np.maximum(0,np.minimum(1,q1))

        q2 = (lam4-lam2)*beta*x2*y/(2*c3)
        q2 = np.maximum(0,np.minimum(1,q2))

        M[0,0,:] = -(1-q1)*y*beta
        M[0,1,:] = 0.
        M[0,2,:] = -(1-q1)*beta*x1
        M[0,3,:] = -(1-q1)*beta*x1

        M[1,0,:] = 0.
        M[1,1,:] = -(1-q2)*y*beta
        M[1,2,:] = -(1-q2)*beta*x2
        M[1,3,:] = -(1-q2)*beta*x2

        M[2,0,:] = (1-q1)*y*beta
        M[2,1,:] = 0.
        M[2,2,:] = (1-q1)*beta*x1 - gamma
        M[2,3,:] = (1-q1)*beta*x1

        M[3,0,:] = 0.
        M[3,1,:] = (1-q2)*y*beta
        M[3,2,:] = (1-q2)*beta*x2
        M[3,3,:] = (1-q2)*beta*x2 - gamma

        M[4,2,:] = (lam1-lam3)*(1-q1)*beta
        M[4,3,:] = (lam1-lam3)*(1-q1)*beta
        M[4,4,:] = (1-q1)*beta*y
        M[4,6,:] = -(1-q1)*beta*y

        M[5,2,:] = (lam2-lam4)*(1-q2)*beta
        M[5,3,:] = (lam2-lam4)*(1-q2)*beta
        M[5,5,:] = (1-q2)*beta*y
        M[5,7,:] = -(1-q2)*beta*y

        M[6,0,:] = (lam1-lam3)*(1-q1)*beta
        M[6,1,:] = (lam2-lam4)*(1-q2)*beta
        M[6,4,:] = (1-q1)*beta*x1
        M[6,5,:] = (1-q2)*beta*x2
        M[6,6,:] = -(1-q1)*beta*x1 + gamma
        M[6,7,:] = -(1-q2)*beta*x2

        M[7,0,:] = (lam1-lam3)*(1-q1)*beta
        M[7,1,:] = (lam2-lam4)*(1-q2)*beta
        M[7,4,:] = (1-q1)*beta*x1
        M[7,5,:] = (1-q2)*beta*x2
        M[7,6,:] = -(1-q1)*beta*x1
        M[7,7,:] = -(1-q2)*beta*x2 + gamma

        return M

    x0 = x10+x20; y0 = y10+y20
    assert(x10+x20+y10+y20<=1)

    def bc(ua, ub):
        x1T = ub[0]; x2T=ub[1]
        y1T = ub[2]; y2T=ub[3]
        lam1T = dJ_dx1(x1T,x2T,y1T,y2T,sigma0,c1,c2)
        lam2T = dJ_dx2(x1T,x2T,y1T,y2T,sigma0,c1,c2)
        lam3T = dJ_dy1(x1T,x2T,y1T,y2T,sigma0,c1,c2)
        lam4T = dJ_dy2(x1T,x2T,y1T,y2T,sigma0,c1,c2)

        return np.array([ua[0]-x10, ua[1]-x20, ua[2]-y10, ua[3]-y20,
                         ub[4]-lam1T, ub[5]-lam2T, ub[6]-lam3T, ub[7]-lam4T])

    tt = np.linspace(0,T,Nt+1)
    uu = np.zeros((8,Nt+1))
    x1T = (1./sigma0 + 0.05) * x10/x0
    x2T = (1./sigma0 + 0.05) * x20/x0
    yT = 0.

    if guess is not None:
        uu = guess
        #tt = guess[1]
    elif loadsol:
        uu = np.loadtxt('guess1.txt')
    else:
        uu[0,:] = np.exp(-(beta-gamma)*tt/6)/2.
        uu[1,:] = np.exp(-(beta-gamma)*tt/6)/2.
        uu[2,:] = 0.5*np.exp(-1.e-3*(tt-15)**2)/2.
        uu[3,:] = 0.5*np.exp(-1.e-3*(tt-15)**2)/2.
        uu[4,:] = dJ_dx1(x10,x20,yT,yT,sigma0,c1,c2)
        uu[5,:] = dJ_dx2(x10,x20,yT,yT,sigma0,c1,c2)
        uu[6,:] = dJ_dy1(x10,x20,yT,yT,sigma0,c1,c2)
        uu[7,:] = dJ_dy2(x10,x20,yT,yT,sigma0,c1,c2)

    if usejac: jf=jac
    else: jf = None
    result = solve_bvp(rhs, bc, tt, uu, max_nodes=max_nodes, tol=tol, verbose=2, fun_jac=jf)
    x1, x2, y1, y2 = result.y[0,:], result.y[1,:], result.y[2,:], result.y[3,:]
    x = x1+x2
    y = y1+y2
    lam1, lam2, lam3, lam4 = result.y[4,:], result.y[5,:], result.y[6,:], result.y[7,:]
    t = result.x

    q1star = (lam3-lam1)*beta*x1*y/(2*c3)
    q1star = np.maximum(0,np.minimum(1,q1star))

    q2star = (lam4-lam2)*beta*x2*y/(2*c3)
    q2star = np.maximum(0,np.minimum(1,q2star))

    #t = result.x
    print(result.message)
    return x1, x2, y1, y2, q1star, q2star, t, result.sol(tt)

def plot_timeline(x1,x2,y1,y2,q1,q2,t):
    palette = plt.get_cmap('tab10')
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    ax.plot(t,y1,color=palette(0))
    ax.plot(t,y2,color=palette(1))
    ax.plot(t,q1,'--',color=palette(0))
    ax.plot(t,q2,'--',color=palette(1))
    ax.legend(['y1','y2','q1','q2']);
    plt.xlabel('t');
    ax.autoscale(enable=True, axis='x', tight=True)
    return fig

def opt_params(N,ifr_young,ifr_old,eta,d,eps):
    # ifr_young, ifr_old: fatality ratio for young and old groups
    # eta: Increase in IFR when no medical care is given
    # d: 1e4 Days left of life for average victim
    # eps: 0.2  Fraction of value of a day of life that is lost due to intervention
    c1 = ifr_young
    c2 = ifr_old
    c3 = eps/d
    c4 = eta*ifr_young
    c5 = eta*ifr_old
    return c1/c1, c2/c1, c3/c1, c4/c1, c5/c1

def approx_deaths(N,x1,x2,y1,y2,sigma0,ifr_young,ifr_old,ymax,gamma,T):
    x10 = x1[0]; x20=x2[0]
    x = x1+x2; y = y1+y2
    xinf = x_inf(x[-1],y[-1],sigma0)
    x1inf = xinf*x1[-1]/x[-1]
    x2inf = xinf*x2[-1]/x[-1]
    d1 = N*(ifr_young*(x10-x1inf) + ifr_old*(x20-x2inf))  # Deaths from infection
    d2 = N*(ifr_young*np.sum(np.maximum(0,y1-ymax)) + ifr_old*np.sum(np.maximum(0,y2-ymax)))*gamma*(T/len(y)) # Deaths from inadequate care
    return d1, d2


def model_control(region,params={}):
    T = params.get('T',200)
    ymax = params.get('ymax',0.04)
    gamma = params.get('gamma',cf.default_gamma)
    sigma0 = params.get('sigma0',cf.default_beta/cf.default_gamma)
    beta = sigma0*gamma
    pdf = deconvolution.generate_pdf(8.,1./gamma/8.)

    N = data.get_population(region)
    ifr = cf.avg_ifr(region)
    data_dates, cum_cases, cum_deaths = data.load_time_series(region,smooth=True)
    data_start = mdates.date2num(data_dates[0])  # First day for which we have data

    u0, offset, inf_dates, I, R, delta_I = \
            cf.infer_initial_data(cum_deaths,data_start,ifr,gamma,N,extended_output=1)

    qfun, offset = cf.assess_intervention_effectiveness(cum_deaths,N,ifr,data_dates,fit_type='constant')

    S_rec, I_rec, R_rec = cf.SIR(u0,beta=beta,gamma=gamma,N=N,T=offset,q=qfun,
                                 intervention_start=0,intervention_length=offset)

    u0 = np.array([S_rec[-1],I_rec[-1],R_rec[-1]])

    pa = data.age_distribution[region]
    Nyoung = np.sum(pa[:pop_split//5])
    Nold = np.sum(pa[pop_split//5:])
    ifr_young, ifr_old = get_ifrs(region)

    x10 = u0[0]*Nyoung/N**2
    x20 = u0[0]*Nold/N**2
    y10 = u0[1]*Nyoung/N**2
    y20 = u0[1]*Nold/N**2

    # Moderate intervention scenario
    eta = 2
    d = 1e4
    eps = 0.2
    c1, c2, c3, c4, c5 = opt_params(N,ifr_young,ifr_old,eta,d,eps)
    x1_1, x2_1, y1_1, y2_1, q1_1, q2_1, t_1 = optimal_intervention(x10, x20, y10, y20, c1, c2, c3, c4, c5, ymax=ymax,
                                                                   beta=beta, gamma=gamma, T=T)

    # Strict intervention scenario
    eps = 0.01
    c1, c2, c3, c4, c5 = opt_params(N,ifr_young,ifr_old,eta,d,eps)
    x1_2, x2_2, y1_2, y2_2, q1_2, q2_2, t_2 = optimal_intervention(x10, x20, y10, y20, c1, c2, c3, c4, c5, ymax=ymax,
                                                                   beta=beta, gamma=gamma, T=T)

    I[-offset-1:] = I_rec
    I = I/N

    plt.plot(mdates.num2date(today+np.arange(len(I))-len(I)),I,'-k')
    x_no, y_no, z_no = cf.SIR(u0,beta=beta,gamma=gamma,N=N,T=T,q=0,
                              intervention_start=0,intervention_length=0)

    # Should probably split these figures
    x1 = x_no*Nyoung/N**2; x2 = x_no*Nold/N**2
    y1 = y_no*Nyoung/N**2; y2 = y_no*Nold/N**2
    d1, d2 = approx_deaths(N,x1,x2,y1,y2,sigma0,ifr_young,ifr_old,ymax,gamma,T)
    print('Estimated death toll with no further intervention: '+str(int(d1))+' + '+str(int(d2)))
    #label = 'No intervention ('+str(int(d1))+' + '+str(int(d2))+')'
    #plt.plot_date(mdates.num2date(today+np.arange(T+1)),y1,'-',color=palette(0),label=label)
    #plt.plot_date(mdates.num2date(today+np.arange(T+1)),y2,'--',color=palette(0),label=label)

    d1, d2 = approx_deaths(N,x1_1,x2_1,y1_1,y2_1,sigma0,ifr_young,ifr_old,ymax,gamma,T)
    print('Estimated death toll with optimal moderate intervention: '+str(int(d1))+' + '+str(int(d2)))
    label = 'Moderate intervention ('+str(int(d1))+' + '+str(int(d2))+')'
    plt.plot_date(mdates.num2date(t_1+today),y1_1,'-',color=palette(1),label=label)
    plt.plot_date(mdates.num2date(t_1+today),y2_1,'--',color=palette(1),label=label)

    d1, d2 = approx_deaths(N,x1_2,x2_2,y1_2,y2_2,sigma0,ifr_young,ifr_old,ymax,gamma,T)
    print('Estimated death toll with optimal srict intervention: '+str(int(d1))+' + '+str(int(d2)))
    label = 'Strict intervention ('+str(int(d1))+' + '+str(int(d2))+')'
    plt.plot_date(mdates.num2date(t_2+today),y1_2,'-',color=palette(2),label=label)
    plt.plot_date(mdates.num2date(t_2+today),y2_2,'--',color=palette(2),label=label)

    plt.plot_date(mdates.num2date([today,today+T]),[ymax,ymax],'-.k',alpha=0.5)
    plt.ylabel('Infected fraction')
    ax = plt.gca()
    plt.title(region);
    plt.legend();
    #ax2 = ax.twinx()
    #ax2.set_ylabel('Intervention')
    #ax2.plot_date(mdates.num2date(t1+today),1-sigma1/sigma0,'--',color=palette(1))
    #ax2.plot_date(mdates.num2date(t2+today),1-sigma2/sigma0,'--',color=palette(2))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.autoscale(enable=True, axis='x', tight=True)

