import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys
sys.path.append('../covid_forecasting')
import covid_forecast as cf
import data
import SIR_control
import deconvolution
plt.style.use('seaborn-poster')
palette = plt.get_cmap('tab10')

verbose = False

dgamma = cf.default_gamma
dbeta = cf.default_beta
today = int(mdates.date2num(datetime.today()))

def opt_params(N,ifr,eta,d,eps):
    # alpha: IFR
    # eta: Increase in IFR when no medical care is given
    # d: 1e4 Days left of life for average victim
    # eps: 0.2  Fraction of value of a day of life that is lost due to intervention
    c1 = ifr
    c2 = eps/d
    c3 = eta
    return c1/c1, c2/c1, c3/c1

def approx_deaths(N,x,y,sigma0,ifr,ymax,gamma,T):
    d1 = ifr*N*(1-SIR_control.x_inf(x[-1],y[-1],sigma0)) # Deaths from infection
    d2 = ifr*N*np.sum(np.maximum(0,y-ymax))*gamma*(T/len(y)) # Deaths from inadequate care
    return d1, d2

def optimal_intervention(x0, y0, c1, c2, c3, beta=dbeta, gamma=dgamma, ymax=0.04,
                         T=200, multipliers=(500,50,20,10,8,5,2,1.4,1.2,1.1)):
    """
    Solve the optimal control problem by relaxing it to an easy version and
    then gradually improving the initial guess.
    """
    if verbose: print(multipliers[0])
    x, y, sigma, t, newguess = SIR_control.solve_pmp(beta=beta,gamma=gamma,c1=c1,c2=multipliers[0]*c2,c3=c3,
                                                     ymax=ymax,T=T,guess=None,x0=x0,y0=y0,N=10000)
    for mult in multipliers[1:]:
        if verbose: print(mult)
        x, y, sigma, t, newguess = SIR_control.solve_pmp(beta=beta,gamma=gamma,c1=c1,c2=mult*c2,c3=c3,
                                                         ymax=ymax,T=T,guess=newguess,x0=x0,y0=y0,N=10000)
    if verbose: print('1')
    x, y, sigma, t, newguess = SIR_control.solve_pmp(beta=beta,gamma=gamma,c1=c1,c2=c2,c3=c3,
                                                     ymax=ymax,T=T,guess=newguess,x0=x0,y0=y0,N=10000)
    return x, y, sigma, t


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
    x0, y0, z0 = u0
    x0 = x0/N; y0 = y0/N

    # Moderate intervention scenario
    eta = 2*ifr
    d = 1e4
    eps = 0.2
    c1, c2, c3 = opt_params(N,ifr,eta,d,eps)
    x1, y1, sigma1, t1 = optimal_intervention(x0, y0, c1, c2, c3, ymax=ymax,
                                              beta=beta, gamma=gamma, T=T)

    # Strict intervention scenario
    eps = 0.01
    c1, c2, c3 = opt_params(N,ifr,eta,d,eps)
    x2, y2, sigma2, t2 = optimal_intervention(x0, y0, c1, c2, c3, ymax=ymax,
                                              beta=beta, gamma=gamma, T=T)

    I[-offset-1:] = I_rec
    I = I/N

    plt.plot(mdates.num2date(today+np.arange(len(I))-len(I)),I,'-k')
    x_no, y_no, z_no = cf.SIR(u0,beta=beta,gamma=gamma,N=N,T=T,q=0,
                              intervention_start=0,intervention_length=0)

    d1, d2 = approx_deaths(N,x_no/N,y_no/N,sigma0,ifr,ymax,gamma,T)
    print('Estimated death toll with no further intervention: '+str(int(d1))+' + '+str(int(d2)))
    label = 'No intervention ('+str(int(d1))+' + '+str(int(d2))+')'
    plt.plot_date(mdates.num2date(today+np.arange(T+1)),y_no/N,'-',color=palette(0),label=label)

    d1, d2 = approx_deaths(N,x1,y1,sigma0,ifr,ymax,gamma,T)
    print('Estimated death toll with optimal moderate intervention: '+str(int(d1))+' + '+str(int(d2)))
    label = 'Moderate intervention ('+str(int(d1))+' + '+str(int(d2))+')'
    plt.plot_date(mdates.num2date(t1+today),y1,'-',color=palette(1),label=label)

    d1, d2 = approx_deaths(N,x2,y2,sigma0,ifr,ymax,gamma,T)
    print('Estimated death toll with optimal srict intervention: '+str(int(d1))+' + '+str(int(d2)))
    label = 'Strict intervention ('+str(int(d1))+' + '+str(int(d2))+')'
    plt.plot_date(mdates.num2date(t2+today),y2,'-',color=palette(2),label=label)

    plt.plot_date(mdates.num2date([today,today+T]),[ymax,ymax],'-.k',alpha=0.5)
    ax = plt.gca()
    plt.title(region);
    plt.legend();
    ax2 = ax.twinx()
    ax2.set_ylabel('Intervention')
    ax2.plot_date(mdates.num2date(t1+today),1-sigma1/sigma0,'--',color=palette(1))
    ax2.plot_date(mdates.num2date(t2+today),1-sigma2/sigma0,'--',color=palette(2))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.autoscale(enable=True, axis='x', tight=True)

