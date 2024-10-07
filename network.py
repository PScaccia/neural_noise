#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:09:15 2024

@author: paoloscaccia
"""
import numpy as np



def generate_tuning_curve( center = 0.3, w = 0.5, a = 0.4, b = 0.1, scale = 1):
    return lambda x : ( ( (np.exp( np.cos(x - center)/w) - np.exp(-1/w) ) /  (np.exp(1/w) - np.exp(-1/w)) ) + b )*scale


class NeuralSystem(object):
    
    def __init__( self, tuning_curves, covariance_matrix, N_trial = 1000):
        
        def neural_dynamics(x, mu_functions, cov_matrix, N_trial):
            return np.random.multivariate_normal(  [ f(x) for f in mu_functions ], # Average Vector
                                                   cov_matrix,                     # Covariance Matrix
                                                   size = N_trial )
        #.astype(int)
                        
        self.mu         = tuning_curves
        self.covariance = covariance_matrix
        self.N_trial    = N_trial
        
        self.neurons    = lambda x: neural_dynamics(x, self.mu, self.covariance, self.N_trial)

        return
        