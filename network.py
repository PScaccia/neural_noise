#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:09:15 2024

@author: paoloscaccia
"""
import numpy as np
import sys

def generate_tuning_curve( function = 'vm', 
                           center = 2, A = 2, width = 1.6, flatness = 0.1, b = 0.1):
    # Reference: https://swindale.ecc.ubc.ca/wp-content/uploads/2020/09/orientation_tuning_curves.pdf
    
    if function == 'vm':
        return lambda x : A*( ( (np.exp( np.cos(x - center)/width) - np.exp(-1/width) ) /  (np.exp(1/width) - np.exp(-1/width)) ) + b )
    elif function == 'fvm':
        return lambda x: A*np.exp( (np.cos((x - center - abs(flatness)*np.sin(2*(x - center)))) - 1)/width )
        # return lambda x:  A*np.exp(  width*( flatness*np.sin( x - center ) - 1 )  )
    else:
        sys.exit("Unknown function")

def generate_variance_matrix( V, rho, D = 2):
    if D != 2:
        sys.exit("Not implemented yet!")
    
    return V*np.array( [[1, rho], [rho, 1]])

class NeuralSystem(object):
    
    def __init__( self, tuning_curves, covariance_matrix, N_trial = 1000):
        
        def neural_dynamics(t, mu_functions, cov_matrix, N_trial):
            return np.random.multivariate_normal(  [ f(t) for f in mu_functions ], # Average Vector
                                                   cov_matrix,                     # Covariance Matrix
                                                   size = N_trial )
        #.astype(int)
        self.mu         = tuning_curves
        self.covariance = covariance_matrix
        self.N_trial    = N_trial
        
        self.neurons    = lambda x: neural_dynamics(x, self.mu, self.covariance, self.N_trial)

        return
        