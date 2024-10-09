#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:09:15 2024

@author: paoloscaccia
"""
import numpy as np
import sys
import progressbar

def generate_tuning_curve( function = 'vm', 
                           center = 2.8, A = 1, width = 0.2, flatness = 0.25, b = 0.1):
    
    if function == 'vm':
        # Reference: https://www.biorxiv.org/content/10.1101/2024.06.26.600826.abstract
        return lambda x : A*( ( (np.exp( np.cos(x - center)/width) - np.exp(-1/width) ) /  (np.exp(1/width) - np.exp(-1/width)) )) + b 
    
    elif function == 'fvm':
        # Reference: https://swindale.ecc.ubc.ca/wp-content/uploads/2020/09/orientation_tuning_curves.pdf
        return lambda x: A*np.exp( (  np.cos(( x - center - abs(flatness)*np.sin(2*(x - center)) )) - 1)/width )  + b 
    
    else:
        sys.exit("Unknown function")

def compute_grad( function = 'vm', 
                  center = 2.8, A = 1, width = 0.2, flatness = 0.25, b = 0.1):
    
    if function == 'vm':
        sys.exit("Not implemented yet!")
    
    elif function == 'fvm':
        f = generate_tuning_curve(A=A,width=width,flatness=flatness,b=b,center=center,function=function)
        k = 1/width
        ni = -abs(flatness)
        return lambda x : -f(x)*k*np.sin( (x - center ) + ni*np.sin(2*(x - center))  )*(1 + 2*ni*np.cos(2*(x - center)))
    
    else:
        sys.exit("Unknown function")   
    
    return 

def generate_variance_matrix( V, rho, D = 2):
    if D != 2:
        sys.exit("Not implemented yet!")
    
    return V*np.array( [[1, rho], [rho, 1]])

class NeuralSystem(object):
    
    def __init__( self, parameters, N_trial = 1e4):
        
        def neural_dynamics(t, mu_functions, cov_matrix, N_trial):
            return np.random.multivariate_normal(  [ f(t) for f in mu_functions ], # Average Vector
                                                   cov_matrix,                     # Covariance Matrix
                                                   size = int(N_trial) )
        
        # Read and Store parameters
        for k,i in parameters.items():
            self.__dict__[k] = i
        
        # Define tuning curves and their gradients
        self.mu         = [ generate_tuning_curve( width  = self.width,
                                                   center = self.center,
                                                   A = self.A, function = self.function,
                                                   flatness = self.flatness),
                           generate_tuning_curve( width  = self.width,
                                                  center = self.center + self.center_shift,
                                                  A = self.A, function = self.function,
                                                  flatness = self.flatness) ]
        self.grad       = [ compute_grad( width  = self.width,
                                          center = self.center,
                                          A = self.A, function = self.function,
                                          flatness = self.flatness),
                            compute_grad( width  = self.width,
                                          center = self.center + self.center_shift,
                                          A = self.A, function = self.function,
                                          flatness = self.flatness) ]
        
        # Define Covariance Matrix
        self.sigma = generate_variance_matrix(self.V, self.rho)
        self.N_trial    = int(N_trial)
        
        # Define neurons
        self.neurons    = lambda x: neural_dynamics(x, self.mu, self.sigma, self.N_trial)

        return
    
     
if __name__ == '__main__':
    from plot_tools import plot_simulation
    
    parameters = { 'A'              : 5,
                   'width'          : .2,
                   'center'         : 2.8,
                   'function'       : 'fvm',
                   'flatness'       : 0.1,
                   'b'              : 0.0,
                   'center_shift'   : 0.5,
                   'V'              : 5e-1,
                   'rho'            : 0.5 }
    
    stimuli = np.pi*np.array( [0, 1/3, 0.5, 2/3, 0.8,1,1.2,1+1/3,1.6666])
    
    system = NeuralSystem( parameters )
    plot_simulation(system, stimuli)