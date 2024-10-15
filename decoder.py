#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:59:34 2024

@author: paolos
"""

import numpy as np
import sys
from network import NeuralSystem, THETA

def moving_average(x,w):
    return np.convolve(x, np.ones(w), 'valid') / w


def bayesian_decoder(system, r):
    from scipy.optimize import minimize_scalar
    
    # Define Integral Parameters
    N_step = 200
    theta_support = np.linspace(THETA[0], THETA[-1],N_step)
    dtheta = (THETA[-1] - THETA[0])/N_step    
    sqrt_det = np.sqrt( np.linalg.det(system.sigma))

    InvSigma = system.inv_sigma
    mu_vect          = lambda x :  np.array([system.mu[0](x),system.mu[1](x)]) 
    cost_function    = lambda x : (  r - mu_vect(x) ).transpose().dot( InvSigma ).dot( r - mu_vect(x) )    
    P_post           = lambda x : np.exp(-0.5*cost_function(x))
    
    # Define Integrand Functions
    P_post_sin = lambda x : P_post(x)*np.sin(x)
    P_post_cos = lambda x : P_post(x)*np.cos(x)
    
    # Integrate 
    sin = np.sum( list(map(P_post_sin, theta_support)))
    cos = np.sum( list(map(P_post_cos, theta_support)))
    norm_factor = dtheta / (2*np.pi*sqrt_det)
    sin *= norm_factor
    cos *= norm_factor
    
    return np.arctan2(sin, cos)


def MAP_decoder(system, r):
    from scipy.optimize import minimize_scalar
        
    # Define cost function
    InvSigma = system.inv_sigma
    mu_vect       = lambda x :  np.array([system.mu[0](x), system.mu[1](x)]) 
    cost_function = lambda x : (  r - mu_vect(x) ).transpose().dot( InvSigma ).dot( r - mu_vect(x) )
    
    # Return minimum
    return minimize_scalar(cost_function, bounds = [THETA[0], THETA[-1]]).x


def sample_theta_ext(system, theta_array, decoder = 'bayesian'):
    import progressbar
    print("DECODER: ", decoder)
    theta_ext_sampling = np.empty( ( len(theta_array), system.N_trial ) )*np.nan
        
    bar = progressbar.ProgressBar(max_size = len(theta_array) )
    
    for i, theta in enumerate(theta_array):
        
        r_sampling = system.neurons(theta)
        if decoder == 'bayesian':
            theta_ext_sampling[i,:] = [ bayesian_decoder(system, r) for r in r_sampling ]
        elif decoder == 'MAP':
            theta_ext_sampling[i,:] = [ MAP_decoder(system, r) for r in r_sampling ]
        else:
            sys.exit("Not implemented yet!")
            
        bar.update(i)
        
    return theta_ext_sampling


if __name__ == '__main__':
    from plot_tools import plot_simulation, plot_theta_error, plot_theta_ext_prob
    
    parameters = { 'A'              : 5,
                   'width'          : .2,
                   'center'         : 2.8,
                   'function'       : 'fvm',
                   'flatness'       : 0.1,
                   'b'              : 0.0,
                   'center_shift'   : 0.5,
                   'V'              : 5e-2,
                   'rho'            : 0.05 }
    
    # Define plot
    stimuli = np.pi*np.array( [ 2/3 ])
    #    stimuli = np.pi*np.array( [0, 1/3, 0.5, 2/3, 0.8,1,1.2,1+1/3,1.6666])

    # Define Neural System
    system = NeuralSystem( parameters, N_trial=200)

    # Compute Decoder MSE
    theta_sampling = sample_theta_ext( system, THETA )
    
    error = np.array([  1 - np.cos(t - theta_sampling[i,:] )  for i,t in enumerate(THETA) ])

    MSE = np.mean( error, axis = 1)
    
    # Plot MSE along Signal Manifold    
    plot_simulation( system, stimuli, E = MSE)
    
    # Plot Decoder Error Analysis
    plot_theta_error(THETA, theta_sampling, MSE, title = f'N: {system.N_trial}')
    
    # Plot Histograms
    plot_theta_ext_prob(THETA, theta_sampling, outdir = '/home/paolos/Pictures/decoding')
    

            