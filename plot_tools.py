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
                    label = r"{:.3f}".format(  t ) 
                    )
        
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
        cmap='jet'
        # Emin = E.min()
        # Emax = np.percentile(E,90)

        Emin = -10
        Emax = 10
        
        norm=plt.Normalize(Emin, Emax)
        #c = plt.cm.jet( (E - Emin)/(Emax - Emin) )
        segments = make_segments(mu1, mu2)
        lc = mcoll.LineCollection(segments, array=E, cmap=cmap , norm=norm,
                                  linewidth=2, alpha=1)

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
                     lw = 1 )
    
    s = np.sqrt(system.V)
    axs[0].set_xlim(min(mu1) - 2*s, max(mu1) + 2*s)
    axs[0].set_ylim(min(mu2) - 2*s, max(mu2) + 2*s)
    axs[0].set_xlabel('Response 1', weight='bold',size=18)
    axs[0].set_ylabel('Response 2', weight='bold',size=18)
    
    if len(stimolus) > 0:  axs[0].legend( title = r'Theta')
    axs[0].set_title(f"N: {system.N_trial}",size=12)
    plot_tuning_curves(*system.mu, ax = axs[1])
            
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
                     outdir = None, FI = None, filename = 'MSE.png'):
    
    fig, axs = plt.subplots(2,1,figsize = (10,10))
    axs[0].plot(theta, MSE, c='blue', label = 'Decoding Error',marker='o',ms=3)
    axs[0].set_xlabel(r'$\mathbf{\theta}$',size = 15)    
    axs[0].set_ylabel( "MSE", weight = 'bold', size = 15)    
    axs[0].set_title(title)
    axs[0].legend(loc='upper left')
    if FI is not None:
        ax_twin = axs[0].twinx()
        ax_twin.plot(theta, 1/FI, label = 'Inv. Fisher Information',c='black')
    plt.legend(loc = 'upper right')
    
    av_theta = theta_sampling.mean( axis = 1 )
    
    axs[1].plot( theta, av_theta, c = 'red')
    axs[1].set_xlabel(r'$\mathbf{\theta}$', size = 15)
    axs[1].set_ylabel(r"$\mathbf{ \theta - \langle \hat{\theta} } \rangle $", size = 15)
    
    axs[1].set_title(title)
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
    s = np.sqrt(system.V)

    a,b, alpha = draw_oriented_ellipse(system.sigma,[0,0], ax)
    # alpha += np.pi/2
    plt.plot(s*np.cos(THETA),s*np.sin(THETA),ls='--',c='grey')

    mu_grad1 = system.grad[0](THETA)
    mu_grad2 = system.grad[1](THETA)

    xmax = max(2*s, mu_grad1.max())
    ymax = max(2*s, mu_grad2.max())
    xmin = min(-2*s, mu_grad1.min())
    ymin = min(-2*s, mu_grad2.min())

    line = np.linspace(-xmax*2,xmax*2,1000)
    
    angles = system.compute_beneficial_angles()
    for i,phi in enumerate(angles):
        y = np.tan(phi)*line
        plt.plot(line, y,c='grey',lw = 1)
        
        if i > 0:
            color = ['green','red'][i%2]
            plt.fill_between(line, np.tan(angles[i-1])*line,y,color=color,alpha=0.05 )
           
    
    plt.plot( mu_grad1, mu_grad2, c='black' )
    
    
    plt.xlim(xmin,xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(r"$\mathbf{\mu}'_{1}(\theta)$",size = 15)
    plt.ylabel(r"$\mathbf{\mu}'_{2}(\theta)$",size = 15)

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, bbox_inches = 'tight', dpi=300)
        print("Saved plot of gradients' space in ",outfile)
    
    return

def plot_improvement(theta, R, angles = None, outdir = None):
    plt.plot(theta, R, c = 'black')
    
    if angles is not None:
        for angle in angles:
            plt.axvline(angle, ls ='--', c='red')
    plt.xlabel(r'$\mathbf{\theta}$', size = 15)
    plt.ylabel('Decoding Improvement [%]', size = 15,weight='bold')
    
    plt.axhline(0,lw=0.5,c='grey')
    
    if outdir is not None:
        plt.savefig(outdir+'/improvement.png')
        print(f"Saved Deciding Improvement in {outdir}/{'improvement.png'}")
        plt.clf()
    else:
        plt.show()
    return

