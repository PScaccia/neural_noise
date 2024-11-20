#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:10:10 2024

@author: paoloscaccia
"""

import matplotlib.pyplot as plt
import numpy as np
from network import NeuralSystem, generate_tuning_curve, THETA
import matplotlib.collections as mcoll
import sys


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_simulation( system, stimolus, plot_gradient = False, E = None, 
                    outdir = None, color_label = ''):
    
    markers = ['o','+','s','^','*','v']
        
    fig, axs = plt.subplots(1,2,figsize = (14,6))
    
    # Plot Trials per Stimolous
    for i,t in enumerate(stimolus):
        data = system.neurons(t)
        j = i%len(markers)
        axs[0].scatter(data[:,0],data[:,1],
                    alpha=0.5,
                    marker=markers[ j ],
                    label = r"$\theta$ = {:.1f}°".format( np.rad2deg(t) ) 
                    )
        axs[0].scatter(data[:,0].mean(), data[:,1].mean(),
                       marker = 'x',c='red',s=55, zorder = 13)
        if plot_gradient:
            # Plot Derivatives
            x,y  = [ f(t) for f in system.mu]
            grad = [ f(t) for f in system.grad]
            #grad /= np.linalg.norm( grad )
            dx, dy = grad
            axs[0].arrow( x,y, dx, dy, color = 'black', zorder = 13, head_width = .15)
                     
    # Plot Signal Manifold
    mu1 = list( map( system.mu[0], THETA))
    mu2 = list( map( system.mu[1], THETA))
    
    if E is not None:
        cmap='coolwarm'
        # Emin = E.min()
        # Emax = np.percentile(E,90)

        Emin = -7
        Emax =  7
        
        norm=plt.Normalize(Emin, Emax)
        segments = make_segments(mu1, mu2)
        lc = mcoll.LineCollection(segments, array=E, cmap=cmap , norm=norm,
                                  linewidth=5, alpha=1)

        axs[0].add_collection(lc)
        cm = fig.colorbar(lc,ax=axs[0])
        cm.set_label(color_label, size = 10, weight = 'bold')

        """
        for i,mu1,mu2 in zip( range(theta.size), mu1, list( map( system.mu[1], theta)) ):
            d=axs[0].scatter( mu1, 
                             mu2,
                             color = c[i],
                             s = 5 )
        cb = fig.colorbar(d)
        cb.
        """
    else:
        axs[0].plot( list( map( system.mu[0], THETA)), 
                     list( map( system.mu[1], THETA)), 
                     color = 'black',
                     lw = 3 )
    
    try:
        s = np.sqrt(system.V)
    except:
        s = system.beta*system.A
        
    if len(stimolus) > 0:
        n_sigma = 2
        xmax = max(n_sigma*s,  max(mu1))
        ymax = max(n_sigma*s,  max(mu2))
        xmin = min(-n_sigma*s//4, min(mu1))
        ymin = min(-n_sigma*s//4, min(mu2))
    else:
        xmax = max(mu1) + 5
        xmin = min(mu1) - 5
        ymax = max(mu2) + 5
        ymin = min(mu2) - 5
    
    axs[0].set_xlim(xmin, xmax)
    axs[0].set_ylim(ymin, ymax)
    axs[0].set_xlabel('Response 1', weight='bold',size=18)
    axs[0].set_ylabel('Response 2', weight='bold',size=18)
    
    if len(stimolus) > 0:  axs[0].legend( title = r'Neural Response at', prop={'size':6})
    axs[0].set_title(r"$N_{sampling}$" + f": {system.N_trial}",size=12)
    plot_tuning_curves(*system.mu, ax = axs[1])
            
    if outdir is not None:
        plt.savefig(outdir + '/simulation.png',bbox_inches='tight',dpi=300)
        print("Simulation plot saved in ",outdir + "/simulation.png")
    else:
        plt.show()
    
    
    return 


def plot_simulation_single_panel( system, stimolus, plot_gradient = False, E = None, 
                    outdir = None, color_label = ''):
    
    markers = ['o','+','s','^','*','v']
        
    fig, ax = plt.subplots(1,1,figsize = (12,12))
    
    # Plot Trials per Stimolous
    for i,t in enumerate(stimolus):
        data = system.neurons(t)
        j = i%len(markers)
        ax.scatter(data[:,0],data[:,1],
                    alpha=0.5,
                    marker=markers[ j ],
                    label = r"$\theta$ = {:.1f}°".format( np.rad2deg(t) ) 
                    )
        ax.scatter(data[:,0].mean(), data[:,1].mean(),
                       marker = 'x',c='red',s=55, zorder = 13)
        if plot_gradient:
            # Plot Derivatives
            x,y  = [ f(t) for f in system.mu]
            grad = [ f(t) for f in system.grad]
            #grad /= np.linalg.norm( grad )
            dx, dy = grad
            ax.arrow( x,y, dx, dy, color = 'black', zorder = 13, head_width = .15)
                     
    # Plot Signal Manifold
    mu1 = list( map( system.mu[0], THETA))
    mu2 = list( map( system.mu[1], THETA))
    
    if E is not None:
        cmap='coolwarm'
        # Emin = E.min()
        # Emax = np.percentile(E,90)

        Emin = -7
        Emax =  7
        
        norm=plt.Normalize(Emin, Emax)
        segments = make_segments(mu1, mu2)
        lc = mcoll.LineCollection(segments, array=E, cmap=cmap , norm=norm,
                                  linewidth=5, alpha=1)

        ax.add_collection(lc)
        cm = fig.colorbar(lc,ax=ax)
        cm.set_label(color_label, size = 10, weight = 'bold')

        """
        for i,mu1,mu2 in zip( range(theta.size), mu1, list( map( system.mu[1], theta)) ):
            d=axs[0].scatter( mu1, 
                             mu2,
                             color = c[i],
                             s = 5 )
        cb = fig.colorbar(d)
        cb.
        """
    else:
        ax.plot( list( map( system.mu[0], THETA)), 
                 list( map( system.mu[1], THETA)), 
                     color = 'black',
                     lw = 3 )
    
    try:
        s = np.sqrt(system.V)
    except:
        s = system.beta*system.A
        
    if len(stimolus) > 0:
        n_sigma = 2
        xmax = max(n_sigma*s,  max(mu1))
        ymax = max(n_sigma*s,  max(mu2))
        xmin = min(-n_sigma*s//4, min(mu1))
        ymin = min(-n_sigma*s//4, min(mu2))
    else:
        xmax = max(mu1) + 5
        xmin = min(mu1) - 5
        ymax = max(mu2) + 5
        ymin = min(mu2) - 5
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Response 1', weight='bold',size=18)
    ax.set_ylabel('Response 2', weight='bold',size=18)
    
    if len(stimolus) > 0:  ax.legend( title = r'Neural Response at', prop={'size':6})
    ax.set_title(r"$N_{sampling}$" + f": {system.N_trial}",size=12)
            
    if outdir is not None:
        plt.savefig(outdir + '/simulation.png',bbox_inches='tight',dpi=300)
        print("Simulation plot saved in ",outdir + "/simulation.png")
    else:
        plt.show()
       
    return 

def plot_tuning_curves(mu1, mu2, ax = None):
    
    if ax is None:
        plt.plot(THETA, mu1(THETA), label = 'Neuron 1')
        plt.plot(THETA, mu2(THETA), label = 'Neuron 2')
        plt.xlabel(r"$\mathbf{\theta} [\pi]$",size=15)
        plt.ylabel("Activity",size=15,weight='bold')
    else:
        ax.plot(THETA, mu1(THETA), label = 'Neuron 1')
        ax.plot(THETA, mu2(THETA), label = 'Neuron 2')
        ax.set_xlabel(r"$\mathbf{\theta}$ (rad)",size=15)
        ax.set_ylabel("Activity",size=15,weight='bold')
    return

def plot_theta_error(theta, theta_sampling, MSE , title = ' ', 
                     outdir = None, FI = None, bias = None, filename = 'MSE.png'):
    
    from scipy.misc import derivative
    from scipy.stats import circmean
    from decoder import circular_moving_average
    from scipy.interpolate import splrep, BSpline
    
    fig, axs = plt.subplots(2,1,figsize = (10,10))
    axs[0].plot(theta, MSE, c='blue', label = 'Decoding\nError',marker='o',ms=3)
    axs[0].set_xlabel(r'$\mathbf{\theta}$',size = 15)    
    axs[0].set_ylabel( "Mean Squared Error [rad]", weight = 'bold', size = 15)    
    axs[0].set_title(title)
    axs[0].legend(loc='upper left')
    
    av_theta = theta_sampling.mean( axis = 1 )
    bias = av_theta - theta
    
    error = np.array([ x - theta for x in theta_sampling.transpose()]) 
    tan_error = np.arctan2( np.sin(error), np.cos(error))
    MSE = np.sqrt(circmean(tan_error**2,axis=0))
    MSE = circular_moving_average(THETA,MSE,7) 
    
    f  = lambda x : np.interp(x, theta, bias)
    db = lambda x : derivative(f, x, dx=1e-3)
    
    bias_prime = circular_moving_average(THETA, db(THETA), 5)
    
    N = (1 + bias_prime)**2
    Biased_CRB = (N/FI) + (bias)**2
    Biased_CRB = circular_moving_average(theta, Biased_CRB, 3)
    #Biased_CRB = circular_moving_average(theta, Biased_CRB, 11)
    
    ymin = min(MSE.min() - 0.05, min(1/FI) - 0.05, min(Biased_CRB) - 0.05)
    ymin = max(ymin,0)
    ymax = MSE.max() + 0.1
    ymax = 0.1
    
    if FI is not None:
        axs[0].plot(theta, 1/FI, label = 'Unbiased CRB',c='grey', ls ='--', zorder = 1,alpha=0.7)
        axs[0].plot(theta, Biased_CRB , label = 'Biased CRB',c='black',alpha=0.7, zorder = 1)
        axs[0].set_ylim(ymin, ymax)

    axs[0].legend(loc = 'upper center')
    axs[0].set_ylim(ymin, ymax)
        
    axs[1].plot(theta, bias, c = 'red')
    axs[1].set_xlabel(r'$\mathbf{\theta}$', size = 15)
    axs[1].set_ylabel("Bias",weight='bold', size = 15)
    
    #axs[1].set_title(title)
    
    if outdir is not None:
        plt.savefig(outdir + f'/{filename}',bbox_inches='tight',dpi=300)
        print("MSE plot saved in in ",outdir+f"/{filename}")
        plt.clf()
    else:
        plt.show()
    
    return

def plot_theta_ext_prob(theta, theta_sampling, outdir = './'):
    
    nbins = 25
    dx = (theta[-1] - theta[0])/nbins
    bins = np.linspace(theta[0],theta[-1]+dx,nbins)
    for i,t in enumerate(theta):
        plt.hist(theta_sampling[i,:],bins=bins,label=r"$\theta:$ {:.2f}".format(t))
        plt.xlabel(r'$\mathbf{\theta}$', size = 15)
        plt.legend(loc="upper left")
        plt.savefig(outdir+f'/prob_{i:03}.png')
        plt.clf()
        
    print("Saved PDFs in ",outdir)
    
    return

def plot_gradient_space(system, outfile = None):
    
        
    def draw_oriented_ellipse(cov_matrix, center, ax):
    
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        width, height = 2*np.sqrt(eigenvalues)  # Doppio per coprire il 95% dei dati
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * (180 / np.pi)
    
        ell = plt.matplotlib.patches.Ellipse(center, width, height, angle=angle, color='blue', alpha=0.5)
        ax.add_patch(ell)
        return width/2, height/2, np.deg2rad(angle)
    
    fig,ax = plt.subplots(figsize=(10,10))
    if callable(system.rho):
        print("Skipped gradient space plot for noise dependent correlations.")
        return
    
    s = np.sqrt(system.V)

    a,b, alpha = draw_oriented_ellipse(system.sigma(0),[0,0], ax)
    # alpha += np.pi/2
    plt.plot(s*np.cos(THETA),s*np.sin(THETA),ls='--',c='grey')

    mu_grad1 = system.grad[0](THETA)
    mu_grad2 = system.grad[1](THETA)

    xmax = max(2*s, mu_grad1.max())
    ymax = max(2*s, mu_grad2.max())
    xmin = min(-2*s, mu_grad1.min())
    ymin = min(-2*s, mu_grad2.min())

    line = np.linspace(-xmax*2,xmax*2,1000)
    
    angles = system.compute_beneficial_angles(3)
    for i,phi in enumerate(angles):
        y = np.tan(phi)*line
        plt.plot(line, y,c='grey',lw = 1)
        
        if i == 1:
            color = ['green','red'][i%2]
            plt.fill_between(line, np.tan(angles[i-1])*line,y,color=color,alpha=0.05, label = 'Detrimental' )
           
    
    plt.plot( mu_grad1, mu_grad2, c='black', label = "Signal Manifold")
    
    
    plt.xlim(xmin,xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(r"Derivative 1 $\mathbf{\mu}'_{1}(\theta)$",size = 18, weight = 'bold')
    plt.ylabel(r"Derivative 2 $\mathbf{\mu}'_{2}(\theta)$",size = 18, weight = 'bold')

    plt.legend(prop={'size' : 15})

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, bbox_inches = 'tight', dpi=300)
        print("Saved plot of gradients' space in ",outfile)
    
    return

def plot_improvement(theta, R, angles = None, outdir = None, raw_MSE1 = None, raw_MSE2 = None):
    plt.plot(theta, R, c = 'black', label='Smoothed')
    
    if angles is not None:
        for angle in angles:
            plt.axvline(angle, ls ='--', c='red')
    plt.xlabel(r'$\mathbf{\theta}$ [rad]', size = 15,weight = 'bold')
    plt.ylabel('Decoding Improvement [%]', size = 15,weight='bold')
    
    plt.axhline(0,lw=0.5,c='grey')
    
    if raw_MSE1 is not None and raw_MSE2 is not None:
        from decoder import moving_average
        x,y     = moving_average(theta, raw_MSE1, 7)
        x_2,y_2 = moving_average(theta, raw_MSE2, 7)
        raw_R = (1 - y/y_2)*100
        plt.plot(x, raw_R,label='Unfiltered',lw = 0.4,zorder = 1,c='red')
        plt.legend(prop={'size':17})
        
    if outdir is not None:
        plt.savefig(outdir+'/improvement.png')
        print(f"Saved Deciding Improvement in {outdir}/{'improvement.png'}")
        plt.clf()
    else:
        plt.show()
    return

def draw_circle_and_arrows(delta_angle, outfile  = None):
    
    if delta_angle not in [0,90,180]:
        sys.exit("Delta angle not in [0, 90, 180]")
        
    fig, ax = plt.subplots( figsize = (10,10))
    plt.plot(np.cos(THETA), np.sin(THETA), lw = 2, c='black', zorder = 1, alpha = 0.9)
    
    # First Arrow
    plt.arrow(0, 0, 0, 0.95, color = 'darkblue', label = 'Neuron 1', lw = 8, zorder = 13)
    if delta_angle == 90:
        plt.arrow(0, 0, 0.95, 0, color = 'tab:green', label = 'Neuron 2', lw = 8, zorder = 13)
    else:
        sys.exit("Not implemented yet!")
    plt.xticks([])
    plt.yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.legend()
    
    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, bbox_inches='tight', dpi = 300)
        print("Saved picture in ",outfile)
    
    return

def plot_heatmap(theta_sampling, theta_sampling_control,
                 nbins = THETA.size , outfile = None, cmap = 'hot', vmax = 80):
    plt.clf()
    plt.figure()
    plt.title("CORRELATED SYSTEM",size = 20)

    VMAX = vmax
    
    N = theta_sampling.shape[1]
    bins = np.linspace(0, 2*np.pi,nbins+1)

    heatmap = np.zeros((nbins,nbins))
    
    for i,sample in enumerate(theta_sampling.transpose()):    
        heatmap += np.histogram2d(THETA, sample, bins = bins)[0]
    heatmap /= N
    norm = heatmap.sum(axis=0)
    for i,n in enumerate(norm):
        heatmap[:,i] /= n
            
    plt.imshow(heatmap*100,cmap=cmap,vmin=0,vmax=VMAX)
    cb=plt.colorbar()
    cb.set_label(label = 'Bin Accuracy [%]', size = 13)
    
    nticks = 6
    step = nbins//nticks

    ticks = np.arange(nbins+1)[::step]-0.5
    tick_labels = [ str(int(x))+'°' for x in (ticks+0.5)*360/nbins ]
    plt.yticks(ticks,labels=tick_labels)
    plt.xticks(ticks,labels=tick_labels)
    plt.xlim(-0.5,nbins-0.5)

    plt.ylim(-0.5,nbins-0.5)

    plt.xlabel(r"Input Stimulus $\mathbf{\theta^{*}}$", weight = 'bold',size=15)
    plt.ylabel(r'Decoded Stimulus $\mathbf{\hat{\theta}}$', weight = 'bold' , size =15)
    
    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile,bbox_inches='tight',dpi=300)
        print("Heatmap saved in ",outfile)


    heatmap_control = np.zeros((nbins,nbins))
    for i,sample in enumerate(theta_sampling_control.transpose()):    
        heatmap_control += np.histogram2d(THETA, sample, bins = bins)[0]
    heatmap_control /= N
    norm = heatmap_control.sum(axis=0)
    for i,n in enumerate(norm):
        heatmap_control[:,i] /= n
        
        
    # Plot Heatmap Control
    plt.clf()
    plt.figure()
    plt.title("INDEPENDENT SYSTEM",size = 20)
    plt.imshow(heatmap_control*100,cmap=cmap,vmin=0,vmax=VMAX)
    cb=plt.colorbar()
    cb.set_label(label = 'Bin Accuracy [%]', size = 13)
    tick_labels = [ str(int(x))+'°' for x in (ticks+0.5)*360/nbins ]
    plt.yticks(ticks,labels=tick_labels)
    plt.xticks(ticks,labels=tick_labels)
    plt.xlim(-0.5,nbins-0.5)
    plt.ylim(-0.5,nbins-0.5)
    plt.xlabel(r"Input Stimulus $\mathbf{\theta^{*}}$", weight = 'bold',size=15)
    plt.ylabel(r'Decoded Stimulus $\mathbf{\hat{\theta}}$', weight = 'bold' , size =15)
    if outfile is None:
        plt.show()
    else:
        control_outfile = outfile.replace('.png','_control.png')
        plt.savefig(control_outfile,bbox_inches='tight',dpi=300)
        print("Control Heatmap saved in ",control_outfile)

        
    plt.clf()
    plt.figure()
    plt.yticks(ticks,labels=tick_labels)
    plt.xticks(ticks,labels=tick_labels)
    plt.xlim(-0.5,nbins-0.5)
    IMP_MAX = 5
    IMP_MIN = -IMP_MAX
    plt.ylim(-0.5,nbins-0.5)
    IMP = (heatmap - heatmap_control)
    plt.imshow(IMP*100,cmap='seismic',vmin=IMP_MIN,vmax=IMP_MAX)
    plt.xlabel(r"Input Stimulus $\mathbf{\theta^{*}}$", weight = 'bold',size=15)
    plt.ylabel(r'Decoded Stimulus $\mathbf{\hat{\theta}}$', weight = 'bold' , size =15)
    line = np.linspace(0,nbins+1,1000)
    plt.plot(line,line,c='black')
    cb=plt.colorbar()
    cb.set_label(label = 'Accuracy Difference [%]', size = 13)
    if outfile is None:
        plt.show()
    else:
        imp_outfile = outfile.replace('.png','_diff.png')
        plt.savefig(imp_outfile,bbox_inches='tight',dpi=300)
        print("Difference heatmap saved in ",imp_outfile)

    
    return heatmap, IMP



def plot_fisher_experiment(systemA, systemB, samplingA, samplingB, MSEA,MSEB):
    
    from scipy.misc import derivative
    
    FIXED_ANGLE = 2.3291
    
    fig,axs = plt.subplots(3,1,sharex=True)
    FIA = np.array( list(map(systemA.linear_fisher, THETA) ))
    FIB = np.array( list(map(systemB.linear_fisher, THETA) ))
    
    biasA = samplingA.mean(axis=1) - THETA
    biasB = samplingB.mean(axis=1) - THETA
    
    fA = lambda x : np.interp(x,THETA, biasA, left = np.nan, right=np.nan)
    fB = lambda x : np.interp(x,THETA, biasB, left = np.nan, right=np.nan)

    dA = lambda x: derivative(fA,x,dx=0.001)
    dB = lambda x: derivative(fB,x,dx=0.001)

    FIA_corrected = ((1+dA(THETA))**2/FIA) + biasA**2
    FIB_corrected = ((1+dB(THETA))**2/FIB) + biasB**2

    axs[0].plot(THETA, 1/FIA, label='A', c='blue')    
    axs[0].plot(THETA, 1/FIB, label='B', c='red')    

    axs[1].plot(THETA, FIA_corrected, label='A', c='blue')    
    axs[1].plot(THETA, FIB_corrected, label='B', c='red')    

    axs[2].plot(THETA, 2*MSEA, label='A', c='blue')    
    axs[2].plot(THETA, 2*MSEB, label='B', c='red')    
    axs[2].plot(THETA, FIA_corrected, label='CRB A', c='blue',alpha=0.3,ls='--')    
    axs[2].plot(THETA, FIB_corrected, label='CRB B', c='red',alpha=0.3,ls='--')    

    axs[1].set_ylim(0,0.01)
    axs[0].set_ylim(0,0.01)
    axs[2].set_ylim(0,0.1)

    axs[0].axvline(FIXED_ANGLE, c = 'black')
    axs[1].axvline(FIXED_ANGLE, c = 'black')
    axs[2].axvline(FIXED_ANGLE, c = 'black')

    plt.legend()
    plt.show()
    
    
    
    return

def plot_all_cases(system, R_list, outfile = None):
    
    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    
    #theta = np.linspace(0,2*np.pi,360)
    theta = THETA
    
    # Plot Signal Manifold
    mu1 = list( map( system.mu[0], theta))
    mu2 = list( map( system.mu[1], theta))
    
    cmap='coolwarm'

    Emin = -7
    Emax =  7
    
    norm=plt.Normalize(Emin, Emax)
    segments = make_segments(mu1, mu2)

    for R,ax in zip(R_list, axs.flatten()):
        
        lc = mcoll.LineCollection(segments, array=R, cmap=cmap , norm=norm,
                              linewidth=4, alpha=1)

        ax.add_collection(lc)
        cm = fig.colorbar(lc,ax=ax)
        cm.set_label("Improvement [%]", size = 10, weight = 'bold')    
        ax.plot()
    
    if outfile is None:
        plt.show()
    else:
        plt.savefig( outfile , dpi = 200)
        print("Saved plot ",outfile)
        
        
        for i,R in enumerate(R_list):
            single_file = outfile.replace('.png','_{}.png'.format(i+1))

            fig, ax = plt.subplots(1,1,figsize=(10,10))
            lc = mcoll.LineCollection(segments, array=R, cmap=cmap , norm=norm,
                                  linewidth=4, alpha=1)
    
            ax.add_collection(lc)
            cm = fig.colorbar(lc,ax=ax)
            cm.set_label("Improvement [%]", size = 10, weight = 'bold')    
            ax.plot()
            
            plt.savefig( single_file , dpi = 200)
            print("Saved single plot ", single_file)

    return

def plot_error_distr( error , title = None, outfile = None, bins = 100):
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    N = bins
    hist,edges = np.histogram(error, bins = N)
    
    bottom = 0
    max_height = 100


    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    width = (2*np.pi) / N
    
    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta, hist, width=width, bottom=bottom)
    
    # Use custom colors and opacity
    for r, bar in zip(hist, bars):
        bar.set_facecolor("maroon")
        bar.set_alpha(0.8)
    
    plt.show()

    """
    plt.hist(error, bins = 100)
    if title:
        plt.title(title)
        
    if outfile:
        plt.savefig(outfile)
        print("Saved plot ",outfile)
    else:
        plt.show()
    """
    return
