#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:59:34 2024

@author: paolos
"""

import numpy as np


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
    
    
def compute_theta_error(system, theta_array):
    
    E = np.empty_like( theta_array )*np.nan

    for i, theta in enumerate(theta_array):
        print(f"{i+1}/{len(theta_array)}")

        r_sampling = system.neurons(theta)
        # theta_ext_sampling = [ get_opt_ext(system, r, [0.5,0.9]) for r in r_sampling ]
        theta_ext_sampling = r_sampling.mean(axis=0)
        
        E[i] = np.mean( (theta_ext_sampling - theta )**2)
        
    return E
