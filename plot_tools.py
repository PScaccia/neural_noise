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
                    outdir = None, color_label = '', cmap = 'coolwarm'):
    
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
        cmap=cmap
        # Emin = E.min()
        # Emax = np.percentile(E,90)

        Emin = -3
        Emax =  3
        
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
                    outdir = None, color_label = '', Emin = -5, Emax = 5 , cmap = 'coolwarm',
                    fig_size = (12,12), xmax = 18, theta_support = THETA):
    
    markers = ['o','+','s','^','*','v']
        
    fig, ax = plt.subplots(1,1,figsize = fig_size)
    
    # Plot Trials per Stimolous
    for i,t in enumerate(stimolus):
        data = system.neurons(t)
        j = i%len(markers)
        ax.scatter( data[:,0],data[:,1],
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
    mu1 = list( map( system.mu[0], theta_support))
    mu2 = list( map( system.mu[1], theta_support))
    
    if E is not None:
        # Emin = E.min()
        # Emax = np.percentile(E,90)

        Emin =  Emin
        Emax =  Emax
            

        norm=plt.Normalize(Emin, Emax)
        segments = make_segments(mu1, mu2)
        if E.size != len(segments)+1: 
            print("Warning! Color array has different size then segments' array")
        
        lc = mcoll.LineCollection(segments, array=E, cmap=cmap , norm=norm,
                                  linewidth=20, alpha=1,antialiaseds=True)

        ax.add_collection(lc)
        cm = fig.colorbar(lc, ax = ax)
        cm.set_label(color_label, size = 30, weight = 'bold')
        cm.ax.tick_params(labelsize=30)
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
        ax.plot( list( map( system.mu[0], theta_support)), 
                 list( map( system.mu[1], theta_support)), 
                     color = 'black',
                     lw = 10 )
    
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
        # xmax = xmax
        # xmin = 0
        # ymax = xmax
        # ymin = 0

    # Tmp    
    xmax = 22
    xmin = 0
    ymax = 22
    ymin = 0

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('RESPONSE A', weight='bold',size=40)
    ax.set_ylabel('RESPONSE B', weight='bold',size=40)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(8)
    ax.spines['bottom'].set_linewidth(8)
    ax.tick_params(width=5,size=20)
    plt.xlim(0,xmax)
    plt.ylim(0,ymax)
    plt.xticks(size=42)
    plt.yticks(size=42)
    
    # if len(stimolus) > 0:  ax.legend( title = r'Neural Response at', prop={'size':18})
    # ax.set_title(r"$N_{sampling}$" + f": {system.N_trial}",size=12)
            
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
    bias = theta - av_theta
    
    error = np.array([ x - theta for x in theta_sampling.transpose()]) 
    tan_error = np.arctan2( np.sin(error), np.cos(error))
    MSE = np.sqrt(circmean(tan_error**2,axis=0))
    MSE = circular_moving_average(THETA,MSE,3) 
    
    f  = lambda x : np.interp(x, theta, bias)
    db = lambda x : derivative(f, x, dx=1e-3)
    
    bias_prime = circular_moving_average(THETA, db(THETA), 3)
    
    
    ymin = 0
    ymax = 0.125
    
    if FI is not None:
        N = (1 - bias_prime)**2
        Biased_CRB = (N/FI) + (bias)**2
        Biased_CRB = circular_moving_average(theta, Biased_CRB, 3)
        Biased_CRB = circular_moving_average(theta, Biased_CRB, 3)


        axs[0].plot(theta, 1/FI, label = 'Unbiased CRB',c='grey', ls ='--', zorder = 1,alpha=0.7)
        axs[0].plot(theta, Biased_CRB , label = 'Biased CRB',c='black',alpha=0.7, zorder = 1)
        axs[0].set_ylim(ymin, ymax)

    axs[0].legend(loc = 'upper right')
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
    
        
    def draw_oriented_ellipse(cov_matrix, center, ax, color, label = ''):
    
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        width, height = np.sqrt(eigenvalues)  
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * (180 / np.pi)
    
        ell = plt.matplotlib.patches.Ellipse(center, width*7, height*7, angle=angle,alpha=0.9,edgecolor=color,facecolor='none',ls='-',lw = 5)
        ax.add_patch(ell)
        return width/2, height/2, np.deg2rad(angle)    
    
    fig,ax = plt.subplots(figsize=(10,10))
    if callable(system.rho):
        print("Skipped gradient space plot for noise dependent correlations.")
        return
    
    s = np.sqrt(system.V)

    a,b, alpha = draw_oriented_ellipse(system.sigma(0),[0,0], ax, 'blue')
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
        from decoder import circular_moving_average
        y     = circular_moving_average(theta, raw_MSE1, 7)
        y_2   = circular_moving_average(theta, raw_MSE2, 7)
        raw_R = (1 - y/y_2)*100
        plt.plot(THETA, raw_R,label='Unfiltered',lw = 0.4,zorder = 1,c='red')
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
            
    plt.imshow(heatmap*100,cmap=cmap,vmin=0,vmax=VMAX, origin = 'upper')
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
    plt.imshow(heatmap_control*100,cmap=cmap,vmin=0,vmax=VMAX, origin = 'upper')
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
    IMP_MAX = 3
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
                                  linewidth=8, alpha=1)
    
            ax.add_collection(lc)
            cm = fig.colorbar(lc,ax=ax)
            cm.set_label("Improvement [%]", size = 14, weight = 'bold')    
            ax.plot()
            
            plt.savefig( single_file , dpi = 200)
            print("Saved single plot ", single_file)

    return

def plot_error_distr( error, other_errors = [] , label = '', other_labels = [],
                      title = None, outfile = None, Nbins = 100, logscale = False,
                      xmax = 2*np.pi, xmin = -2*np.pi, xticks = None, fig_size = (10,10)):
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1,1,figsize = fig_size)

    
    bins = np.linspace(xmin, xmax, Nbins)
    plt.hist(error, bins = bins, label = label, weights=np.ones(error.size)/error.size)
    
    for i,e in enumerate(other_errors):
        try:
            label = other_labels[i]
        except:
            pass
        plt.hist(e, bins = bins, label = label,alpha = 0.6, weights=np.ones(error.size)/error.size)

    plt.axvline(0,c='black', ls ='--')
    plt.xlabel("DECODING ERROR [°]",weight = 'bold',size = 42)
    plt.ylabel("PDF",weight = 'bold',size = 42)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if xticks:
        plt.xticks(xticks,size=40)

    #plt.yticks([],size=40)
    leg = plt.legend( frameon=False, 
                      prop = {'size' : 30},
                      bbox_to_anchor = (1.01, 1.0))

    ax.spines['left'].set_linewidth(8)
    ax.spines['bottom'].set_linewidth(8)
    ax.set_yticklabels([])
    
    if logscale:
        plt.yticks(xticks,size=40)
        plt.yscale('log')
    ax.tick_params(width=5,size=20)
        
    plt.xlim(xmin,xmax)
    colors = ['tab:blue','tab:orange']
    for i, (line, text) in enumerate(zip(leg.get_lines(), leg.get_texts())):
        text.set_color(colors[i])
    
    if outfile:
        plt.savefig(outfile, dpi=300,bbox_inches='tight')
        print("Saved plot ",outfile)
    else:
        plt.show()
    return



def paper_plot_figure_2( system, improvement_list = [], file = None, 
                         OUTDIR = "/home/paolos/Pictures/decoding/paper",
                         skip_frame = False, skip_cases = False, skip_errors = False):
    
    if not skip_frame:
        plt.figure(figsize=(12,12))
        ax = plt.subplot(111)
        plt.xlabel("SNR", weight = 'bold', size = 40)
        plt.ylabel("NOISE CORRELATION", weight = 'bold', size = 40)
        ax.spines['left'].set_linewidth(25)
        ax.spines['bottom'].set_linewidth(25)
        ax.tick_params(width=5,size=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.xticks([])
        plt.yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    
        DX = 2
        
        plt.plot( [0-DX,0-DX],         [1.4, 2.5], marker="",ms = 200,c='black', linewidth = 8)
        plt.plot( [1.4 - DX, 2.5], [0,   0],   marker="",ms = 200,c='black', linewidth = 8)
        
        
        plt.scatter([0],[2.5],marker="^", s = 500,c='black', ls='', linewidth = 600, zorder = 13)
        plt.scatter([2.5],[0],marker=">", s = 500,c='black', ls = '',linewidth = 600, zorder = 13)
        
        # Horizontal Line
        plt.text(0.9 - DX, -0.08,"LOW", size=28)
        plt.text(2.65,-0.08,"HIGH", size=28)
        
        # Vertical Line
        plt.text(-0.08,0.87,"LOW", size=28,rotation = 90)
        plt.text(-0.08,2.75,"HIGH", size=28,rotation = 90)
        
        plt.xlim(-0.2-DX,3.5)
        plt.ylim(-0.2-DX,3.5)
    
        OUTFIG = f"{OUTDIR}/figure2_frame.png"
        plt.savefig(OUTFIG, dpi = 300, bbox_inches = 'tight')
        print("Saved plot ", OUTFIG)
    else:
        pass
    
    if not skip_cases:
        config = {  'rho'           : 'poisson',
                    'alpha'         : 0.1,
                    'beta'          : 2.0,
                    'V'             : 2,
                    'function'      : 'fvm',
                    'A'             : 0.17340510276921817,
                    'width'         : 0.2140327993142855,
                    'flatness'      : 0.6585904840291591,
                    'b'             : 1.2731732385019432,
                    'center'        : 2.9616211149125977 - 0.21264734641020677,
                    'center_shift'  : np.pi/4,
                    'N'             : 1000,
                    'int_step'      : 100
                    }
        system = NeuralSystem(config)
        if len(improvement_list) == 0:
            sys.exit("Computatation of improvement not implemented yet!")
        else:
            for i, color_gradient in enumerate(improvement_list):
                TARGET_STIM = THETA[52] if i >= 2 else THETA[90]

                plot_simulation_single_panel(system, [], E = color_gradient, 
                                             color_label = "DECODING IMPROVEMENT [%]",
                                             fig_size = (10,10))
                plt.scatter(system.mu[0](TARGET_STIM), 
                            system.mu[1](TARGET_STIM),
                            s = 2500, c = 'black',
                            marker='*', zorder = 13)
                plt.xticks(np.linspace(0,25,5).astype(int))
                plt.yticks(np.linspace(0,25,5).astype(int))
                
                plt.xlim(0,22)
                plt.ylim(0,22)
                
                OUTFIG = f"{OUTDIR}/figure2_simulation_{i+1}.pdf"
                plt.savefig( OUTFIG,dpi=300,bbox_inches='tight')
                print("Saved plot ",OUTFIG)
                    
        
        if file is None:
            plt.show()
        else:
            plt.savefig( file , dpi = 300, bbox_inches = 'tight')
    else:
        pass
        
    if not skip_errors:
        plt.clf()
        
        # Plot Distributions
        data = dict(np.load("./data/poisson_selected.npz"))
        data2 = dict(np.load("./data/poisson_selected_paper.npz"))
        
        
        # CASE1
        OUTFIG = f"{OUTDIR}/figure2_errors_1.png"
        theta_sampling = np.hstack((data['a_0.10_b_2.00'][0], data2['a_0.10_b_2.00'][0]))
        theta_sampling_control = np.hstack((data['a_0.10_b_2.00'][1], data2['a_0.10_b_2.00'][1]))
        diff = THETA[88] - theta_sampling[88,:]
        diff_control = THETA[88] - theta_sampling_control[88,:]
        
        diff_control[ diff_control > np.pi] = np.pi - diff_control[ diff_control > np.pi]
        diff_control[ diff_control < -np.pi] = np.pi + diff_control[ diff_control < -np.pi]
        diff[ diff > np.pi] = np.pi - diff[ diff > np.pi]
        diff[ diff < -np.pi] = np.pi + diff[diff < -np.pi]
        plot_error_distr( np.rad2deg(diff), other_errors = [np.rad2deg(diff_control)] , 
                          # label = 'CORRELATED', other_labels= ['INDEPENDENT'],                          
                          Nbins = 70,  xmin = -120, xmax = 120,
                          xticks = [-120,-60,0,60,120], logscale = True,
                          outfile = OUTFIG)
        plt.clf()

        # CASE2
        OUTFIG = f"{OUTDIR}/figure2_errors_2.png"
        theta_sampling = np.hstack((data['a_0.10_b_0.10'][0], data2['a_0.10_b_0.10'][0]))
        theta_sampling_control = np.hstack((data['a_0.10_b_0.10'][1], data2['a_0.10_b_0.10'][1]))
        diff = THETA[90] - theta_sampling[90,:]
        diff_control = THETA[90] - theta_sampling_control[90,:]
        
        diff_control[ diff_control > np.pi] = np.pi - diff_control[ diff_control > np.pi]
        diff_control[ diff_control < -np.pi] = np.pi + diff_control[ diff_control < -np.pi]
        diff[ diff > np.pi] = np.pi - diff[ diff > np.pi]
        diff[ diff < -np.pi] = np.pi + diff[diff < -np.pi]
        
        plot_error_distr( np.rad2deg(diff), other_errors = [np.rad2deg(diff_control)] , 
                          # label = 'CORRELATED', other_labels= ['INDEPENDENT'], 
                          Nbins = 50,  xmin = -30, xmax = 30,
                          xticks = [-30,-15,0,15,30], logscale = True,
                          outfile = OUTFIG)
        plt.clf()

        # CASE3
        OUTFIG = f"{OUTDIR}/figure2_errors_3.png"
        theta_sampling = data['a_0.90_b_2.00'][0]
        theta_sampling_control = data['a_0.90_b_2.00'][1]
        diff = THETA[52] - theta_sampling[52,:]
        diff_control = THETA[52] - theta_sampling_control[52,:]
        plot_error_distr( np.rad2deg(diff), other_errors = [np.rad2deg(diff_control)] , 
                          # label = 'CORRELATED', other_labels= ['INDEPENDENT'], 
                          Nbins = 50, xmax = 45, xmin = -45,
                          xticks = [-45,-30,-15,0,15,30,45],
                          outfile = OUTFIG)
        plt.clf()

        # CASE4
        OUTFIG = f"{OUTDIR}/figure2_errors_4.png"
        theta_sampling = theta_sampling = data['a_0.95_b_0.10'][0]
        theta_sampling_control = data['a_0.95_b_0.10'][1]
        diff = THETA[52] - theta_sampling[52,:]
        diff_control = THETA[52] - theta_sampling_control[52,:]
        plot_error_distr( np.rad2deg(diff), other_errors = [np.rad2deg(diff_control)] , 
                          # label = 'CORRELATED', other_labels= ['INDEPENDENT'], 
                          Nbins = 70, xmax = 10, xmin = -10,
                          xticks = [-10,-5,0,5,10],
                          outfile = OUTFIG)
        plt.clf()
        
        
    else:
        pass

    
    
    
    return
