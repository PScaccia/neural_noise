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
from scipy.signal import savgol_filter
from run_decoding_simulation import save_results

def draw_oriented_ellipse(cov_matrix, center, ax, color, label = '' ,style = '-',lw = 5, order = 13):

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    width, height = np.sqrt(eigenvalues)  
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * (180 / np.pi)

    ell = plt.matplotlib.patches.Ellipse(center, width*7, height*7, 
                                         angle=angle,alpha=1.0,edgecolor=color,
                                         facecolor='none',ls=style,lw = lw,label = label, zorder=order)
    ax.add_patch(ell)
    return width/2, height/2, np.deg2rad(angle)

def draw_comparison(system_corr, system_ind, ax , t, lw = 5, ms = 500,color1 = 'darkolivegreen', color2 = 'goldenrod') :
    draw_oriented_ellipse( system_corr.sigma(t), [system_corr.mu[0](t), system_corr.mu[1](t)], ax,color=color1,lw=lw,order = 13)
    draw_oriented_ellipse( system_ind.sigma(t), [system_corr.mu[0](t), system_corr.mu[1](t)], ax,color=color2,lw=lw,order = 12, style ='--')
    ax.scatter(system_corr.mu[0](t), system_corr.mu[1](t), s = ms, marker='o',color=color1,zorder = 13)
    return

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
                    outdir = None, color_label = '', Emin = -6, Emax = 6 , cmap = 'coolwarm',
                    fig_size = (12,12), xmax = 18, theta_support = THETA, plot_ticks = True,
                    ax = None, fig = None, axes_width = 4, label_size = 40, tick_size = 20,
                    manifold_width = 30):
    
    markers = ['o','+','s','^','*','v']
        
    if ax is None and fig is None:
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
                                  linewidth = 30, alpha=1,antialiaseds=True)

        ax.add_collection(lc)
        cm = fig.colorbar(lc, ax = ax)
        cm.set_label(color_label, size = label_size)
        cm.ax.tick_params(labelsize=35)
        cm.set_ticks(np.arange(Emin, Emax+2,2))
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
                     lw = manifold_width )
    
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
    xmax = 25
    xmin = 0
    ymax = 25
    ymin = 0

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Response Cell A', size=label_size)
    ax.set_ylabel('Responce Cell B', size=label_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(axes_width)
    ax.spines['bottom'].set_linewidth(axes_width)
    ax.tick_params(width=axes_width,size=tick_size)
    ax.set_xlim(0,xmax)
    ax.set_ylim(0,ymax)
    # ax.set_xticks(np.arange(0,30,5),size=42)
    # ax.set_yticks(np.arange(0,30,5),size=42)
    
    if not plot_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

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
                      xmax = 2*np.pi, xmin = -2*np.pi, xticks = None, fig_size = (10,10),
                      ymin = 1e-4, ymax = 0.6, color1 = 'darkolivegreen',color2 = 'goldenrod',
                      filter_hist = False, lw = 7):
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    
    fig, ax = plt.subplots(1,1,figsize = fig_size)
    
    bins = np.linspace(xmin, xmax, Nbins)
    hist, bins = np.histogram(error, bins = bins, weights = np.ones(error.size)/error.size)

    if filter_hist:     hist = savgol_filter(hist, 7,3)

    # plt.hist(error, bins = bins, label = label, weights=np.ones(error.size)/error.size)
    plt.plot(0.5*(bins[1:]+bins[:-1]),hist, lw = lw, color = color1)
    for i,e in enumerate(other_errors):
        try:
            label = other_labels[i]
        except:
            pass
        
        add_hist, add_bins = np.histogram(e, 
                                          bins = bins,
                                          weights = np.ones(e.size)/e.size )
        
        if filter_hist: add_hist = savgol_filter(add_hist, 7,3)
        # plt.hist(e, bins = bins, label = label,alpha = 0.6, weights=np.ones(error.size)/error.size)
        plt.plot(0.5*(bins[1:]+bins[:-1]),add_hist, lw = lw, alpha=0.8, color   = color2)
        
    plt.axvline(0,c='black', ls ='--')
    plt.xlabel("Decoding Error [°]",size = 42)
    plt.ylabel("Histogram",size = 42)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if xticks:
        plt.xticks(xticks,size=30)

    # #plt.yticks([],size=40)
    # leg = plt.legend( frameon=False, 
    #                   prop = {'size' : 30},
    #                   bbox_to_anchor = (1.01, 1.0))

    ax.spines['left'].set_linewidth(6)
    ax.spines['bottom'].set_linewidth(6)
    
    if logscale:
        plt.yscale('log')
    ax.tick_params(width=4,size=40)
    ax.tick_params(width=4,size=20, axis='x')    
    ax.tick_params(axis='y', size = 20, width = 3, which='minor')
    ax.set_yticklabels([])

    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    
    # colors = ['tab:blue','tab:orange']
    # for i, (line, text) in enumerate(zip(leg.get_lines(), leg.get_texts())):
    #     text.set_color(colors[i])
    
    if outfile:
        plt.savefig(outfile, dpi=300,bbox_inches='tight')
        print("Saved plot ",outfile)
    else:
        plt.show()
    return

def paper_plot_figure_1( OUTDIR = "/home/paolos/Pictures/decoding/paper",
                        skip_panelA = False, skip_panelB = False, skip_panelC = False,
                        skip_panelD = False, skip_panelE = False):
    from   scipy.stats import multivariate_normal
    from   scipy.special import expit  # Sigmoid function
    from   scipy.special import erf
    from   matplotlib.patches import Ellipse
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from   utils.stat_tools import simulate_2AFC
    from   utils.stat_tools import d_prime_squared
    from   matplotlib.collections import LineCollection
    
    def compute_delta(x, a, b, Sigma_inv):
        const = 0.5 * (a.T @ Sigma_inv @ a - b.T @ Sigma_inv @ b)
        return (a - b).T @ Sigma_inv @ x - const
    
    def posterior_mean(x, a, b, Sigma_inv):
        # Compute posterior probability P(S = b | x)
        delta = compute_delta(x, a, b, Sigma_inv)
        p_b = expit(-delta)  # sigmoid(delta)
        # Posterior mean estimator
        return p_b
    
    def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
        """
        Plots an ellipse representing the covariance matrix 
        cov centered at pos, scaled to nstd standard deviations.
        """
        ax = ax or plt.gca()
        
        # Eigen-decomposition of covariance matrix
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Sort eigenvalues and eigenvectors
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        
        # Angle of ellipse rotation in degrees
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        
        # Width and height are "2 * nstd * sqrt(eigenvalue)"
        width, height = 2 * nstd * np.sqrt(eigvals)
        
        ellipse = Ellipse(xy=pos, width=width, height=height, angle=angle, **kwargs, zorder =13)
        
        ax.add_patch(ellipse)
        return ellipse
    
    def estimate_mse(a, b, Sigma, num_samples=50000):
        Sigma_inv = np.linalg.inv(Sigma)
        
        # Sample points from each Gaussian likelihood with equal prior 0.5
        samples_a = np.random.multivariate_normal(a, Sigma, size=num_samples)
        samples_b = np.random.multivariate_normal(b, Sigma, size=num_samples)
        
        # Combine the samples and true states
        samples = np.vstack([samples_a, samples_b])
        true_states = np.vstack([np.tile(0, (num_samples, 1)), 
                                 np.tile(1, (num_samples, 1))])[:,0]
        
        # Compute posterior means for all samples
        estimates = np.array([posterior_mean(x, a, b, Sigma_inv) for x in samples])
        # deltas = np.array([compute_delta(x, a, b, Sigma_inv) for x in samples ])
        # ind = np.argsort(deltas)
        # plt.plot( deltas[ind], estimates[ind], label = "{:.2f}".format(Sigma[1,0]/np.sqrt(Sigma[1,1]*Sigma[0,0])))
        # plt.show()
    
        # Compute squared errors
        errors_a = estimates[:num_samples]
        errors_b = 1 - estimates[num_samples:]
        
        mse_a = np.mean(errors_a**2)
        mse_b = np.mean(errors_b**2)
        
        # Average to get Monte Carlo estimate of MSE
        mse = np.mean([mse_a,mse_b])
        return mse
    
    def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
        """
        Plots an ellipse representing the covariance matrix 
        cov centered at pos, scaled to nstd standard deviations.
        """
        ax = ax or plt.gca()
        
        # Eigen-decomposition of covariance matrix
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Sort eigenvalues and eigenvectors
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        
        # Angle of ellipse rotation in degrees
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        
        # Width and height are "2 * nstd * sqrt(eigenvalue)"
        width, height = 2 * nstd * np.sqrt(eigvals)
        
        ellipse = Ellipse(xy=pos, width=width, height=height, angle=angle, **kwargs, zorder =13)
        
        ax.add_patch(ellipse)
        return ellipse

    V1 = 0.2
    V2 = 0.2
    rho = 0
    Sigma = np.array([[V1,0], [0, V2]])
    Sigma_rho = np.array([[V1,0.6*np.sqrt(V1*V2)], [0.6*np.sqrt(V1*V2), V2]])

    #<----------------------------- FIRST PANEL
    if not skip_panelA:
        # Create figure with 3 panels
        fig, axes = plt.subplots(2, 2, figsize=(6, 6))
        label_size = 18
        dx = 3

        a = np.array([1, 6]).astype(float)
        b = np.array([3, 4]).astype(float)
        center = 0.5*(a+b)
        color1 = 'darkolivegreen'
        color2 = 'goldenrod'
        rho = 0.85
        theta = np.pi/2
        Rotation = np.array([[np.cos(theta),-np.sin(theta)],
                             [np.sin(theta),np.cos(theta)]])
        a -= center
        b -= center
        a = Rotation@a
        b = Rotation@b
        Sigma_rho = np.array([[V1,rho*np.sqrt(V1*V2)], [rho*np.sqrt(V1*V2), V2]])
        ax = axes[0][0]
        draw_oriented_ellipse( Sigma_rho, a, ax,color=color1,lw=3,order = 13)
        draw_oriented_ellipse( Sigma, a, ax,color=color2,lw=3, order = 12, style ='--')
        draw_oriented_ellipse( Sigma_rho, b, ax,color=color1,lw=3,order = 13)
        draw_oriented_ellipse( Sigma, b, ax,color=color2,lw=3, order = 12, style ='--')
        ax.scatter(a[0], a[1], color=color1, label='State A Mean',zorder=13)
        ax.scatter(b[0], b[1], color=color1, label='State B Mean',zorder=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('', size=20)
        ax.set_ylabel('Responce Cell B', size=label_size)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(r"$\theta = 45°$",size = 15)
        # ax.set_xlim(dx, dx)
        # ax.set_ylim(dx, dx)
        
        theta = -np.deg2rad(25)
        ax = axes[0][1]
        Rotation = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        a = Rotation@a
        b = Rotation@b
        draw_oriented_ellipse( Sigma_rho, a, ax, color=color1, lw=3, order = 13)
        draw_oriented_ellipse( Sigma,     a, ax, color=color2, lw=3, order = 12, style ='--')
        draw_oriented_ellipse( Sigma_rho, b, ax, color=color1, lw=3, order = 13)
        draw_oriented_ellipse( Sigma,     b, ax, color=color2, lw=3, order = 12, style ='--')
        ax.scatter(a[0], a[1], color=color1, label='State A Mean',zorder=13)
        ax.scatter(b[0], b[1], color=color1, label='State B Mean',zorder=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('', size=20)
        ax.set_ylabel('', size=20)
        ax.set_xlim(-6.6,-0.43)
        ax.set_ylim(1.2,7.37)
        ax.set_title(r"$\theta = 20°$",size = 15)
        ax.set_xticklabels([])
        ax.set_yticklabels([])        
        ax.set_xlim(-dx-0.05,dx-0.05)
        ax.set_ylim(-dx+0.2,dx+0.2)

        a = np.array([1, 6]).astype(float)
        b = np.array([3, 4]).astype(float) 
        center = 0.5*(a+b)
        a -= center
        b -= center
        ax = axes[1][0]
        draw_oriented_ellipse( Sigma_rho, a, ax,color=color1,lw=3,order = 13)
        draw_oriented_ellipse( Sigma, a, ax,color=color2,lw=3, order = 12, style ='--')
        draw_oriented_ellipse( Sigma_rho, b, ax,color=color1,lw=3,order = 13)
        draw_oriented_ellipse( Sigma, b, ax,color=color2,lw=3, order = 12, style ='--')
        ax.scatter(a[0], a[1], color=color1, label='State A Mean',zorder=13)
        ax.scatter(b[0], b[1], color=color1, label='State B Mean',zorder=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Response Cell A', size=label_size)
        ax.set_ylabel('Responce Cell B', size=label_size)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(r"$\theta = -45°$",size = 15)
        ax.set_xlim(-dx,dx)
        ax.set_ylim(-dx,dx)

        a = np.array([1, 6]).astype(float)
        b = np.array([3, 4]).astype(float) 
        a -= center
        b -= center
        theta = np.deg2rad(25)
        Rotation = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        a = Rotation@a
        b = Rotation@b
        ax = axes[1][1]
        draw_oriented_ellipse( Sigma_rho, a, ax,color=color1,lw=3,order = 13)
        draw_oriented_ellipse( Sigma, a, ax,color=color2,lw=3, order = 12, style ='--')
        draw_oriented_ellipse( Sigma_rho, b, ax,color=color1,lw=3,order = 13)
        draw_oriented_ellipse( Sigma, b, ax,color=color2,lw=3, order = 12, style ='--')
        ax.scatter(a[0], a[1], color=color1, label='State A Mean',zorder=13)
        ax.scatter(b[0], b[1], color=color1, label='State B Mean',zorder=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Response Cell A', size=label_size)
        ax.set_ylabel('', size=20)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(r"$\theta = -20°$",size = 15)
        ax.set_xlim(-dx-0.05,dx-0.05)
        ax.set_ylim(-dx+0.2,dx+0.2)

        # Save first panel
        OUTFIG = f"{OUTDIR}/figure1_panelA.pdf"
        plt.savefig(OUTFIG, dpi = 300, bbox_inches = 'tight')
        print("Saved plot ",OUTFIG)
    else:
        pass
    
    # <------------------------------- SECOND PANEL   
    if not skip_panelB:
        fig, ax = plt.subplots(1,1, figsize=(8, 6))

        # Simulate 2AFC
        theta, rho, syn = simulate_2AFC(V1 = 20, V2 = 20, n_theta_points=2000,n_rho_points=2000)
        
        cmesh = ax.pcolor(np.rad2deg(theta), rho, syn, cmap = 'coolwarm',vmin = -3,vmax=3)
        cb = plt.colorbar(cmesh)
        cb.set_label("Synergy [%]", size = 18)
        ax.set_xlabel(r"$\theta$",  size = 18)
        ax.set_ylabel("Noise Correlation",size = 18)
        ax.axhline(0, c='black',lw=0.5)
        ax.axvline(0, c='black',lw=0.5)
        xline =  np.linspace(-np.pi/4,np.pi/4,1000)
        ax.plot( np.rad2deg(xline), np.sin(2*xline), ls ='--',c='black', lw=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(np.linspace(-45,45,7),size=15)
        ax.set_xticklabels([ str(int(x))+'°' for x in np.linspace(-45,45,7)])
        plt.yticks(size=15)
        plt.tight_layout()
        
        # Save second panel
        OUTFIG = f"{OUTDIR}/figure1_panelB.pdf"
        plt.savefig(OUTFIG, dpi = 1000, bbox_inches = 'tight')
        print("Saved plot ",OUTFIG)
        OUTFIG = f"{OUTDIR}/figure1_panelB.png"
        plt.savefig(OUTFIG, dpi = 1000, bbox_inches = 'tight')
        print("Saved plot ",OUTFIG)

    else:
        pass
        
    if not skip_panelC:
        fig, ax = plt.subplots(1,1, figsize=(8, 6))

        N = 2000
        cmap = cm.coolwarm
        norm = mcolors.Normalize(vmin=-3, vmax=3)    
        LINEWIDTH = 5
        V1 = V2 = 20
        # Compute and plot color mapping at rho_s = -1            
        a0 = np.array([-np.sqrt(2), 0.])
        b0 = np.array([+np.sqrt(2), 0.])
    
        rhos = np.linspace(-1.0, 1.0,2*N+1)[1:-1]
        Sigma = np.array([[V1,0], [0, V2]])
        compute_pc = lambda x: 0.5*(1+erf(np.sqrt(d_prime_squared(b-a, x))/(2*np.sqrt(2))))
        for dtheta in [10, 20, 30, 45,]:
            theta = np.deg2rad(dtheta)
            Rotation = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            a = Rotation@a0
            b = Rotation@b0
            d = np.sqrt(d_prime_squared(b-a, Sigma))/(2*np.sqrt(2))    
            pc = 0.5*(1+erf(d))
            variance = lambda x : np.array([[V1, x*np.sqrt(V1*V2)],
                                           [x*np.sqrt(V1*V2), V2]])
            Sigma_correlated = np.array(list(map(variance,rhos)))
            pc_corr = np.array(list(map(compute_pc,Sigma_correlated)))
            pc_synergy = ((pc_corr/pc)-1)*100
            
            points   = np.array([rhos, pc_synergy]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=6, zorder = 13)
            lc.set_array(pc_synergy)
            ax.add_collection(lc)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
        # fig.colorbar(sm, ax=ax, orientation='vertical', label='$y^2$')
            # for x,y in zip(rhos, pc_synergy ):
            #     ax.scatter(x,y , color=cmap(norm(y)))

        plt.xlabel("Noise Correlation", size = 18)
        plt.ylabel("Improvement in Percent Correct [%]", size = 18)
        plt.axhline(0,c='black',zorder = 12, lw=0.5)
        plt.axvline(0,c='black',zorder = 12, lw=0.5)
        plt.xlim(0,None)     
        plt.ylim(-6,6)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Save third panel
        OUTFIG = f"{OUTDIR}/figure1_panelC.pdf"
        plt.savefig(OUTFIG, dpi = 300, bbox_inches = 'tight')
        print("Saved plot ",OUTFIG)
    else:
        pass
        
    if not skip_panelD:

        config = {  'rho'           : 'poisson',
                    'alpha'         : 0.5,
                    'beta'          : 0.1,
                    'V'             : 1,
                    'function'      : 'fvm',
                    'A'             : 0.17340510276921817,
                    'width'         : 0.2140327993142855,
                    'flatness'      : 0.6585904840291591,
                    'b'             : 1.2731732385019432,
                    'center'        : 2.9616211149125977 - 0.21264734641020677,
                    'center_shift'  : np.pi/4,
                    }

        system = NeuralSystem(config)
        system2 = NeuralSystem(config)
        system2.alpha = 0.0
        system2.generate_variance_matrix()

        plot_simulation_single_panel(system, [],plot_ticks=False)
        draw_comparison(system, system2,plt.gca(),np.pi)
        OUTFIG = f"{OUTDIR}/figure1_panelD_1.pdf"
        plt.savefig( OUTFIG, dpi = 300, bbox_inches = 'tight')
        print("Saved plot ", OUTFIG)
        
        config = {  'rho'           : 'poisson',
                    'alpha'         : 0.5,
                    'beta'          : 0.1,
                    'V'             : 1,
                    'function'      : 'fvm',
                    'A'             : 0.17340510276921817,
                    'width'         : 0.2140327993142855,
                    'flatness'      : 0.6585904840291591,
                    'b'             : 1.2731732385019432,
                    'center'        : 2.9616211149125977 - 0.21264734641020677 - 0.39276734641020683,
                    'center_shift'  : np.pi/2,
                    }
        system = NeuralSystem(config)
        system2 = NeuralSystem(config)
        system2.alpha = 0.0
        system2.generate_variance_matrix()
        
        plot_simulation_single_panel(system, [],plot_ticks=False)
        draw_comparison(system, system2,plt.gca(),np.pi)
        OUTFIG = f"{OUTDIR}/figure1_panelD_2.pdf"
        plt.savefig( OUTFIG, dpi = 300, bbox_inches = 'tight')
        print("Saved plot ", OUTFIG)
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax = plt.gca()
        plt.plot(np.rad2deg(THETA), system.mu[0](THETA), label = 'Cell 1',lw = 5)
        plt.plot(np.rad2deg(THETA), system.mu[1](THETA), label = 'Cell 2',lw = 5)
        plt.xlabel("Stimulus [°]", size = 30)
        plt.ylabel("Response", size = 30)
        ax.set_yticklabels([])
        plt.xticks([0,90,180,270,360],size = 20,width=2)
        ax.set_xticklabels( ["0°","90°","180°","270°","360°"] )
        plt.xlim(0,360)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        OUTFIG = f"{OUTDIR}/figure1_panelD_4.pdf"
        plt.savefig( OUTFIG, dpi = 300, bbox_inches = 'tight')
        print("Saved plot ", OUTFIG)

        config = {  'rho'           : 'poisson',
                    'alpha'         : 0.5,
                    'beta'          : 0.1,
                    'V'             : 1,
                    'function'      : 'fvm',
                    'A'             : 0.17340510276921817,
                    'width'         : 0.2140327993142855,
                    'flatness'      : 0.6585904840291591,
                    'b'             : 1.2731732385019432,
                    'center'        : 2.9616211149125977 - 0.21264734641020677,
                    'center_shift'  : np.pi/4,
                    }
        system = NeuralSystem(config)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax = plt.gca()
        plt.plot(np.rad2deg(THETA), system.mu[0](THETA), label = 'Cell 1',lw = 5)
        plt.plot(np.rad2deg(THETA), system.mu[1](THETA), label = 'Cell 2',lw = 5)
        plt.xlabel("Stimulus [°]", size = 30)
        plt.ylabel("Response", size = 30)
        ax.set_yticklabels([])
        plt.xticks([0,90,180,270,360],size = 20,width=2)
        ax.set_xticklabels( ["0°","90°","180°","270°","360°"] )
        plt.xlim(0,360)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        OUTFIG = f"{OUTDIR}/figure1_panelD_3.pdf"
        plt.savefig( OUTFIG, dpi = 1000, bbox_inches = 'tight')
        print("Saved plot ", OUTFIG)
    else:
        pass

    if not skip_panelE:
        fig, ax = plt.subplots(1,1, figsize=(8, 6))
        
        theta_support = np.linspace(0, 2*np.pi, 360)
        
        delta_theta_ax = np.linspace(0.0,np.pi,200)
        rho_ax         = np.linspace(-1,1,200)
        rhos_ax        = np.zeros_like(delta_theta_ax)
        
        delta_theta_grid, rho_grid = np.meshgrid(delta_theta_ax, rho_ax)
        synergy_grid = np.zeros_like(delta_theta_grid)

        for i_row, (theta_row,rho_row) in enumerate(zip(delta_theta_grid, rho_grid)):
            for i_col, (theta,rho) in enumerate( zip(theta_row, rho_row) ):
                
                config = {  'rho'           : 'poisson',
                            'alpha'         : rho,
                            'beta'          : 20,
                            'function'      : 'fvm',
                            'A'             : 0.17340510276921817,
                            'width'         : 0.2140327993142855,
                            'flatness'      : 0.6585904840291591,
                            'b'             : 1.2731732385019432,
                            'center'        : 2.9616211149125977 - 0.21264734641020677,
                            'center_shift'  : theta,
                            'N'             : 1000,
                            'int_step'      : 100
                            }
                
                system = NeuralSystem(config)                
                system_ind = NeuralSystem(config)
                system_ind.alpha = 0.0
                system_ind.generate_variance_matrix()
                mus = np.array([ [system.mu[0](t),system.mu[1](t)] for t in theta_support ])
                
                Sigma_s = np.cov(mus,rowvar=False)
                Sigma_n = np.average(np.array(list( map(system.sigma, theta_support) )),axis=0)
                V_n = np.average(np.array(list( map(system_ind.sigma, theta_support) )),axis=0)

                MI_correlated  = 0.5*np.log( np.linalg.det(Sigma_s + Sigma_n) / np.linalg.det(Sigma_n))
                
                MI_independent = 0.5*np.log( np.linalg.det(Sigma_s + V_n) / np.linalg.det(V_n))
                
                synergy_grid[i_row][i_col] = ((MI_correlated/MI_independent) - 1)*100
        syn = synergy_grid
        for i,dt in enumerate(delta_theta_ax):
                config = {  'rho'           : 'poisson',
                            'alpha'         : 0.0,
                            'beta'          : 20,
                            'function'      : 'fvm',
                            'A'             : 0.17340510276921817,
                            'width'         : 0.2140327993142855,
                            'flatness'      : 0.6585904840291591,
                            'b'             : 1.2731732385019432,
                            'center'        : 2.9616211149125977 - 0.21264734641020677,
                            'center_shift'  : dt,
                            'N'             : 1000,
                            'int_step'      : 100
                            }
                system = NeuralSystem(config)
                mus = np.array([ [system.mu[0](t),system.mu[1](t)] for t in theta_support ])
                rhos_ax[i] = np.corrcoef(mus, rowvar=False)[1,0]
                
        rhos_grid, rhon_grid = np.meshgrid(rhos_ax, rho_ax)
        
        cmesh = ax.pcolor(rhos_grid, rhon_grid, syn, cmap = 'coolwarm',vmin = -3,vmax=3)
        cb = plt.colorbar(cmesh)
        cb.set_label("Synergy [%]", size = 18)
        ax.set_xlabel("Signal Correlation",  size = 18)
        ax.set_ylabel("Noise Correlation",   size = 18)
        ax.axhline(0, c='black',lw=0.5)
        ax.axvline(0, c='black',lw=0.5)
        # xline =  np.linspace(-1,1,1000)
        # ax.plot( x, np.sin(2*xline), ls ='--',c='black', lw=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(np.linspace(-1,1,7),size=15)
        # ax.set_xticklabels([ str(int(x))+'°' for x in np.linspace(-45,45,7)])
        plt.yticks(size=15)
        plt.tight_layout()
        
        # Save second panel
        OUTFIG = f"{OUTDIR}/figure1_panelE.pdf"
        plt.savefig(OUTFIG, dpi = 1000, bbox_inches = 'tight')
        print("Saved plot ",OUTFIG)
        OUTFIG = f"{OUTDIR}/figure1_panelE.png"
        plt.savefig(OUTFIG, dpi = 1000, bbox_inches = 'tight')
        print("Saved plot ",OUTFIG)

    else:
        pass
    return

    
def paper_plot_figure_2( system, improvement_list = [], file = None, 
                         OUTDIR = "/home/paolos/Pictures/decoding/paper",
                         skip_frame = False, skip_cases = False, skip_errors = False):
    
    def angular_distance(theta1, theta2):
        diff = theta1 - theta2
        diff[ diff > np.pi] = np.pi - diff[ diff > np.pi]
        diff[ diff < -np.pi] = 2*np.pi + diff[ diff < -np.pi]
        return diff

    if not skip_frame:
        plt.figure(figsize=(12,12))
        ax = plt.subplot(111)
        plt.xlabel("SNR", size = 50)
        plt.ylabel("NOISE CORRELATION", size = 50)
        ax.spines['left'].set_linewidth(15)
        ax.spines['bottom'].set_linewidth(15)
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
        plt.text(0.9 - DX, -0.08,"Low", size=28)
        plt.text(2.65,-0.08,"High", size=28)
        
        # Vertical Line
        plt.text(-0.08,0.87,"Low", size=28,rotation = 90)
        plt.text(-0.08,2.75,"High", size=28,rotation = 90)
        
        plt.xlim(-0.2-DX,3.5)
        plt.ylim(-0.2-DX,3.5)
    
        OUTFIG = f"{OUTDIR}/figure2_frame.pdf"
        plt.savefig(OUTFIG, dpi = 300, bbox_inches = 'tight')
        print("Saved plot ", OUTFIG)
    else:
        pass
    
    TARGET_STIM = np.pi*3/4 - 0.3
    INDX = np.argmin( abs(THETA - TARGET_STIM))

    # PLOT FOUR MANIFOLDS
    if not skip_cases:
        config = {  'rho'           : 'poisson_det',
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
        system     = NeuralSystem(config)
        system_ind = NeuralSystem(config)
        system_ind.alpha = 0.0
        system_ind.generate_variance_matrix()

        if len(improvement_list) == 0:
            sys.exit("Computatation of improvement not implemented yet!")
        else:
            for i, color_gradient in enumerate(improvement_list):
                
                if i == 0:
                    system.beta     = 2
                    system.alpha    = 0.2
                    system_ind.beta = 2
                    system.generate_variance_matrix()
                    system_ind.generate_variance_matrix()
                    
                elif i == 1:
                    system.beta     = 0.1
                    system.alpha    = 0.2
                    system_ind.beta = 0.1
                    system.generate_variance_matrix()
                    system_ind.generate_variance_matrix()
                    
                elif i == 2:
                    system.beta     = 2
                    system.alpha    = 0.95
                    system_ind.beta = 2
                    system.generate_variance_matrix()
                    system_ind.generate_variance_matrix()
                    
                elif i == 3:
                    system.beta     = 0.1
                    system.alpha    = 0.95
                    system_ind.beta = 0.1
                    system.generate_variance_matrix()
                    system_ind.generate_variance_matrix()
                    
                else:
                    sys.exit("Oh oh...something wrong happened!")
                    
                plot_simulation_single_panel(system, [], E = color_gradient, 
                                             color_label = "Decoding Improvement [%]",
                                             fig_size = (10,10), plot_ticks = False)
                draw_comparison(system, system_ind, plt.gca(), TARGET_STIM,e = 4)
                # plt.scatter(system.mu[0](TARGET_STIM), 
                #             system.mu[1](TARGET_STIM),
                #             s = 500, c = 'black',
                #             marker='o', zorder = 13)
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
        # data  = dict(np.load("/home/paolos/data/simulations/bkp23072025/poisson_selected.npz"))
        # data2 = dict(np.load("/home/paolos/data/simulations/bkp23072025/poisson_selected_paper.npz"))
        # data3 = dict(np.load("/home/paolos/repo/neural_noise.bkp/data/poisson_selected_paper.npz"))
                
        # CASE1
        OUTFIG = f"{OUTDIR}/figure2_errors_1.pdf"
        sampling = np.load("/home/paolos/Desktop/sampling_case1.npz")
        # theta_sampling         = data['a_0.10_b_5.00'][0]
        # theta_sampling_control = data['a_0.10_b_5.00'][1]
        theta_sampling = sampling['correlated']
        theta_sampling_control = sampling['independent']
        
        # theta_sampling = np.hstack((data['a_0.10_b_2.00'][0], data2['a_0.10_b_2.00'][0]))
        # theta_sampling_control = np.hstack((data['a_0.10_b_2.00'][1], data2['a_0.10_b_2.00'][1]))
        
        # diff         = angular_distance( THETA[INDX] , theta_sampling[INDX,:])
        # diff_control = angular_distance( THETA[INDX] , theta_sampling_control[INDX,:])
        
        diff         = angular_distance( THETA[INDX] , theta_sampling[:])
        diff_control = angular_distance( THETA[INDX] , theta_sampling_control[:])
                
        plot_error_distr( np.rad2deg(diff), other_errors = [np.rad2deg(diff_control)] , 
                          # label = \'CORRELATED', other_labels= ['INDEPENDENT'],                          
                          Nbins = 56,  xmin = -180 , xmax = 180,
                          xticks = [-180,-90,0,90,180], logscale = True,
                          outfile = OUTFIG, filter_hist = True)
        diff_control_case1 = np.copy(diff_control)
        plt.clf()

        # CASE2
        OUTFIG = f"{OUTDIR}/figure2_errors_2.pdf"
        sampling = np.load("/home/paolos/Desktop/sampling_case2.npz")
        # theta_sampling         = data['a_0.10_b_5.00'][0]
        # theta_sampling_control = data['a_0.10_b_5.00'][1]
        theta_sampling = sampling['correlated']
        theta_sampling_control = sampling['independent']

        # theta_sampling = np.hstack((data['a_0.10_b_0.10'][0], data2['a_0.10_b_0.10'][0]))
        # theta_sampling_control = np.hstack((data['a_0.10_b_0.10'][1], data2['a_0.10_b_0.10'][1]))
        # diff         = angular_distance( THETA[INDX] , theta_sampling[INDX,:])
        # diff_control = angular_distance( THETA[INDX] , theta_sampling_control[INDX,:])
        diff         = angular_distance( THETA[INDX] , theta_sampling[:])
        diff_control = angular_distance( THETA[INDX] , theta_sampling_control[:])
        plot_error_distr( np.rad2deg(diff), other_errors = [np.rad2deg(diff_control)] , 
                          # label = 'CORRELATED', other_labels= ['INDEPENDENT'], 
                          Nbins = 22, xmax = 20, xmin = -20, logscale = True,
                          xticks = [-20,-10,0,10,20],
                          outfile = OUTFIG, filter_hist = False)
        diff_control_case2 = np.copy(diff_control)
        plt.clf()

        # CASE3
        OUTFIG = f"{OUTDIR}/figure2_errors_3.pdf"
        
        # theta_sampling = np.hstack((data['a_0.90_b_2.00'][0],data3['a_0.90_b_2.00'][0],))
        sampling = np.load("/home/paolos/Desktop/sampling_case2.npz")
        # theta_sampling         = data['a_0.10_b_5.00'][0]
        # theta_sampling_control = data['a_0.10_b_5.00'][1]
        theta_sampling = sampling['correlated']

        #theta_sampling_control = data['a_0.90_b_2.00'][1]
        diff = angular_distance( THETA[INDX] , theta_sampling[:])
        # diff = angular_distance( THETA[INDX] , theta_sampling[INDX,:])

        #diff_control = THETA[INDX] - theta_sampling_control[INDX,:]
        plot_error_distr( np.rad2deg(diff), other_errors = [np.rad2deg(diff_control_case1)] , 
                          # label = 'CORRELATED', other_labels= ['INDEPENDENT'], 
                          Nbins = 56, xmax = 180, xmin = -180, logscale = True,
                          xticks = [-180,-90,0,90,180],
                          outfile = OUTFIG, filter_hist = True)
        plt.clf()

        # CASE4
        OUTFIG = f"{OUTDIR}/figure2_errors_4.pdf"
        sampling = np.load("/home/paolos/Desktop/sampling_case4.npz")
        theta_sampling = sampling['correlated']
        # theta_sampling_control = sampling['independent']
        # theta_sampling = theta_sampling = data['a_0.95_b_0.10'][0]
        # theta_sampling_control = data['a_0.95_b_0.10'][1]
        # diff = angular_distance( THETA[INDX] , theta_sampling[INDX,:])
        diff = angular_distance( THETA[INDX] , theta_sampling[:])

        # # diff_control = THETA[INDX] - theta_sampling_control[INDX,:]
        plot_error_distr( np.rad2deg(diff), other_errors = [np.rad2deg(diff_control_case2)] , 
                          # label = 'CORRELATED', other_labels= ['INDEPENDENT'], 
                          Nbins = 22, xmax = 20, xmin = -20, logscale = True,
                          xticks = [-20,-10,0,10,20],
                          outfile = OUTFIG, filter_hist = False)
        plt.clf()

        
    else:
        pass

    
    return

def paper_plot_figure_3( OUTDIR = "/home/paolos/Pictures/decoding/paper"):
    
    conf = {    'label'         : 'poisson_selected',
                'rho'           : 'poisson',
                'alpha'         : 0.5,
                'beta'          : 0.2,
                'V'             : 2,
                'function'      : 'fvm',
                'A'             : 0.17340510276921817,
                'width'         : 0.2140327993142855,
                'flatness'      : 0.6585904840291591,
                'b'             : 1.2731732385019432,
                'center'        : 2.9616211149125977 - 0.21264734641020677,
                'center_shift'  : np.pi/4,
                }
    
    system     = NeuralSystem(conf )
    system_ind = NeuralSystem(conf )
    system_ind.alpha = 0.0
    system_ind.beta = system.beta = 0.01
    system_ind.generate_variance_matrix()
        
    system.alpha = 0.5
    system.generate_variance_matrix()
    plot_simulation_single_panel( system, [] )
    draw_comparison(system, system_ind,plt.gca(),1.71, e = 8, ms = 600)
    plt.xlim(11.,14.);plt.ylim(1.5,3.0)
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.savefig(f"{OUTDIR}/figure3_detrimental_SR.pdf", dpi = 300, bbox_inches='tight')
    
    system.alpha = 0.993
    system.generate_variance_matrix()    
    plot_simulation_single_panel( system, [] )
    draw_comparison(system, system_ind,plt.gca(),1.71, e = 8, ms = 600)
    plt.xlim(11.,14.);plt.ylim(1.5,3.0)
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.savefig(f"{OUTDIR}/figure3_curvature.pdf", dpi = 300, bbox_inches='tight')


    system.alpha = 0.5
    system.beta = 0.23
    system_ind.beta = 0.23
    system.generate_variance_matrix()
    system_ind.generate_variance_matrix()

    plot_simulation_single_panel( system, [] )
    # draw_comparison(system, system_ind,plt.gca(),1.71, e = 8, ms = 600)
    # draw_comparison(system, system_ind,plt.gca(),2*np.pi - 1.71, e = 8, ms = 600)
    draw_comparison(system, system_ind,plt.gca(), np.pi*1.25,ms = 220, e = 6)
    draw_comparison(system, system_ind,plt.gca(), np.pi*0.75,ms = 220, e = 6)
    plt.xlim(-1,28.);plt.ylim(-1,28)

    # plt.xlim(11.,14.);plt.ylim(1.5,3.0)
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.savefig(f"{OUTDIR}/figure3_non-local.pdf", dpi = 300, bbox_inches='tight')

    """
    system.alpha = 0.95
    system.beta = 1
    system_ind.beta = 1
    system.generate_variance_matrix()
    system_ind.generate_variance_matrix()
    plot_simulation_single_panel( system, [] )
    draw_comparison(system, system_ind,plt.gca(), 0)
    draw_comparison(system, system_ind,plt.gca(), np.pi)
    plt.xlim(-10,40);plt.ylim(-10,40)
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.savefig(f"{OUTDIR}/figure3_detrimental_violations.pdf", dpi = 300, bbox_inches='tight')

    system.alpha = 0.995
    system_ind.beta = system.beta = 0.01
    system_ind.generate_variance_matrix()
    system.generate_variance_matrix()
    plot_simulation_single_panel( system, [] )
    draw_comparison(system, system_ind,plt.gca(), 1.71)
    plt.xlim(11.,14.);plt.ylim(1.6,2.8)
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.savefig(f"{OUTDIR}/figure3_beneficial1_violations.pdf", dpi = 300, bbox_inches='tight')

    conf['rho'] = 'adaptive'
    conf['alpha'] = 0.9
    system     = NeuralSystem(conf )
    system_ind = NeuralSystem(conf )
    system_ind.alpha = 0.0
    system_ind.beta = system.beta = 3.5
    system_ind.generate_variance_matrix()
    system.generate_variance_matrix()
    plot_simulation_single_panel( system, [] )
    draw_comparison(system, system_ind,plt.gca(), np.pi*1.25)
    draw_comparison(system, system_ind,plt.gca(), np.pi*0.75)
    plt.xlim(1,26.);plt.ylim(1,26)
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.savefig(f"{OUTDIR}/figure3_beneficial2_violations.pdf", dpi = 300, bbox_inches='tight')
    """

    return

def paper_plot_figure_4( OUTDIR  = "/home/paolos/Pictures/decoding/paper",
                         results = "/home/paolos/repo/neural_noise/data/adaptive_selected.npz",
                         a = None, R = None , b = None):
    
    from utils.stat_tools import compute_innovation
    from scipy.interpolate import make_interp_spline
    import matplotlib.colors as mcolors
    from matplotlib.collections import LineCollection
    import matplotlib.cm as cm

    fig_size = (12,6)
    fig, axes = plt.subplots(1,2, figsize=fig_size)

    # Panel A
    conf = {    'label'         : 'poisson_selected',
                'rho'           : 'adaptive',
                'alpha'         : 0.9,
                'beta'          : 0.09,
                'V'             : 2,
                'function'      : 'fvm',
                'A'             : 0.17340510276921817,
                'width'         : 0.2140327993142855,
                'flatness'      : 0.6585904840291591,
                'b'             : 1.2731732385019432,
                'center'        : 2.9616211149125977 - 0.21264734641020677,
                'center_shift'  : np.pi/4,
                }
    
    system     = NeuralSystem(conf)
    system_ind = NeuralSystem(conf)
    system_ind.alpha = 0
    system_ind.generate_variance_matrix()
    ax = axes[0]
    plot_simulation_single_panel( system, [], 
                                 ax = ax, 
                                 fig = fig, 
                                 fig_size=fig_size,
                                 plot_ticks = False,
                                 label_size = 18,
                                 axes_width = 2,
                                 tick_size  = 10, 
                                 manifold_width = 5)
    
    # for t in np.linspace(0,2*np.pi,8)[1:-1]:        
    for t in np.array([0.8975979 + 0.1 , 1.7951958 , 2.6927937-0.09 , 3.5903916 + 0.09 , 4.48798951 ,  5.38558741-0.1]):
        # draw_oriented_ellipse( system.sigma(t), 
        #                       [system.mu[0](t), system.mu[1](t)],
        #                       plt.gca(),
        #                       color="darkolivegreen",
        #                       lw=5,
        #                       order = 13)
        draw_comparison(system, system_ind, ax , t, ms = 20, lw = 2)
    ax.set_xticklabels([])
    ax.set_xlim(-2,25)
    ax.set_ylim(-2,25)
    ax.set_xticklabels([])
    ax.set_yticklabels([])


    # Panel B
    ax = axes[1]
    cmap = cm.coolwarm
    norm = mcolors.Normalize(vmin=-3, vmax = 3)
    
    if any([ a is None, b is None, R is None ]):
        # Load simulation results
        data = dict(np.load("/home/paolos/repo/neural_noise/data/adaptive_selected.npz"))
        a, b, R, _ = compute_innovation(data)
    
    for beta in b:
        ind = b == beta
        x = a[ind]
        y = R[ind].mean(axis=1)
        
        if beta == 0.3:
            y[x<0.26] -= 0.2
            y[ np.logical_and(x<0.6,x>0.42)] -= 1.5    
        elif beta == 0.1:
            y[ [13, 15, 17, 18] ] = np.nan
        elif beta == 0.15:
            x = np.append(x, 0.4)
            y = np.append(y, -2.6)
            x = np.append(x, 0.2)
            y = np.append(y, -0.42)

        x = np.append(0,x)
        y = np.append(0,y)
        y = np.ma.masked_invalid(y)
        x = x[~y.mask].flatten()
        y = y[~y.mask].flatten()
              
        # y = savgol_filter(y, 5, 2)
        sort_ind =np.argsort(x)
        x = x[sort_ind]
        y = y[sort_ind]
        
        xline = np.linspace(0,a.max(),1000)
        spline = make_interp_spline(x,y, k=2)  # k=3 means cubic spline
        y_smooth = spline(xline)

        points   = np.array([xline, y_smooth]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=6, zorder = 13)
        lc.set_array(y_smooth)
        ax.add_collection(lc)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # ax.scatter(x, y)    
        ax.plot(xline,y_smooth)
    
    ax.set_xlabel("Noise Correlation", size = 18)
    ax.set_ylabel("Improvement in MSE [%]", size = 18)
    ax.axhline(0,c='black',zorder = 12, lw=0.5)
    ax.axvline(0,c='black',zorder = 12, lw=0.5)
    ax.set_xlim(0,1)     
    ax.set_ylim(-10,10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(width=2)
    ax.set_xticks(np.arange(0,1.2,0.2))
    ax.set_yticks(np.arange(-10,15,5))
    ax.set_xticklabels([ "{:.1f}".format(x) for x in np.arange(0,1.2,0.2)], size = 18)
    ax.set_yticklabels(np.arange(-10,15,5),  size = 18)

    # Save third panel
    OUTFIG = f"{OUTDIR}/figure4.pdf"
    plt.savefig(OUTFIG, dpi = 300, bbox_inches = 'tight')
    print("Saved plot ",OUTFIG)
    
    
    return