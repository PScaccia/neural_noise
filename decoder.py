#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:59:34 2024

@author: paolos
"""

import numpy as np
import sys
from network import NeuralSystem

def moving_average(x,w):
    return np.convolve(x, np.ones(w), 'valid') / w

def get_opt_ext( system, r, first_guess, eps = 1e-12, max_iter = 100, debug = False):
    
    x = np.copy( first_guess )
    grad_L = [ 999, 999 ]

    InvSigma = np.linalg.inv(  system.sigma )

    debug_out = []    
        
    counter = 1
    while np.linalg.norm( grad_L ) > eps and counter < max_iter:
        
        mu      = np.array( [ f(s) for f,s in zip( system.mu, x) ])
        grad_mu = np.array( [ f(s) for f,s in zip( system.grad, x) ] )
        grad_L  = grad_mu.transpose().dot( InvSigma ).dot( r - mu )
                
        H = grad_mu.transpose().dot( InvSigma ).dot( grad_mu )

        alpha = 1/H

        x = x + alpha * grad_L

        # D-dimensional
        #  alpha = np.linalg.inv(H)
        #x = x + alpha.dot( grad_L )        
        
        if debug:
            debug_out.append( abs(grad_L) )
            print("Iteration: {} gradL: {:e} L: {:.5f}".format(counter, debug_out[-1], -(r - mu )@InvSigma.dot(r-mu)))
        counter += 1
        
    if debug:
        return x, debug_out
    else:
        return x
    
    
    return 

def bayesian_decoder(system, r):
    from scipy.optimize import minimize_scalar
    InvSigma = np.linalg.inv( system.sigma )
    mu_vect = lambda x :  np.array([system.mu[0](x),system.mu[1](x)]) 
    logL = lambda x : (  r - mu_vect(x) ).transpose().dot( InvSigma ).dot( r - mu_vect(x) )
    return minimize_scalar(logL, bounds = [0, 2*np.pi])


def compute_theta_error(system, theta_array, decoder = 'optimal'):
    
    E = np.empty_like( theta_array )*np.nan

    for i, theta in enumerate(theta_array):
        print(f"{i+1}/{len(theta_array)}")

        r_sampling = system.neurons(theta)
        if decoder == 'optimal':
            theta_ext_sampling = [ bayesian_decoder(system, r) for r in r_sampling ]
        else:
            sys.exit("Not implemented yet!")
            
        theta_ext_sampling = r_sampling.mean(axis=0)
        
        E[i] = np.mean( (theta_ext_sampling - theta )**2)
        
    return E

if __name__ == '__main__':
    from plot_tools import plot_simulation, plot_theta_error
    
    parameters = { 'A'              : 5,
                   'width'          : .2,
                   'center'         : 2.8,
                   'function'       : 'fvm',
                   'flatness'       : 0.1,
                   'b'              : 0.0,
                   'center_shift'   : 0.5,
                   'V'              : 5e-1,
                   'rho'            : 0.5 }
    
    # Define plot
    stimuli = np.pi*np.array( [ 2/3 ])
    #    stimuli = np.pi*np.array( [0, 1/3, 0.5, 2/3, 0.8,1,1.2,1+1/3,1.6666])

    # Define Neural System
    system = NeuralSystem( parameters )

    # Compute Decoder MSE
    theta = np.linspace(0,2*np.pi,1000)
    MSE = compute_theta_error( system, theta )

    # Plot MSE along Signal Manifold    
    plot_simulation( system, stimuli, E = MSE)
    
    # Plot Decoder Error Analysis
    plot_theta_error(theta, MSE)
            