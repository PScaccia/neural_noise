#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:09:15 2024

@author: paoloscaccia
"""
import numpy as np
import sys
import progressbar

THETA = np.linspace(0, 2*np.pi, 360)


def generate_tuning_curve_from_fit(fit):    
    # Parametri in ordine di apparenza: 
    #     area (A)
    #     direzione preferita (mu)
    #     parametro di concentrazione (s)
    #     baseline (b)
    #     parametro di flatness (g).
    #
    xp = np.linspace(0, 2*np.pi, 180)
    return  lambda x : np.interp(x, xp, fit, left = np.nan, right = np.nan)


def generate_tuning_curve( function = 'fvm_from_fits', 
                           center = 2.8, A = 1, width = 0.2, flatness = 0.25, b = 0.1):
    
    if function == 'vm':
        # Reference: https://www.biorxiv.org/content/10.1101/2024.06.26.600826.abstract
        return lambda x : A*( ( (np.exp( np.cos(x - center)/width) - np.exp(-1/width) ) /  (np.exp(1/width) - np.exp(-1/width)) )) + b 
    elif function == 'fvm':
        flatness = abs(flatness)
        # Readapted according to Carlo's fit
        return lambda x: A*np.exp( (  np.cos(( x - center - flatness*np.sin( x - center) )) )/width )  + b 
    # elif function == 'fvm_from_fit':
    #     # Z = 118271476277.5487
    #     f = lambda stimulus, amp, x_peak, width, fact : \
    #             amp* np.exp((np.cos( stimulus-x_peak-fact*np.sin( stimulus-x_peak)))/width)
    #     return lambda x : f( x, A, center, width, flatness) + b
    else:
        sys.exit("Unknown function for generating the tuning curves.")


def FTVM(th, a, th_0, s, baseline, g):           #indexed with 'ftvm'
    '''Modified Von Mises function\n Parameters Area, mean, concentration, baseline, flatness\n g = 0 gives Von Mises function'''
    delta_th = th[1]-th[0]
    f = lambda stimulus, amp, x_peak, width, fact : \
            amp* np.exp((np.cos(np.deg2rad(stimulus-x_peak-fact*np.sin(np.deg2rad(stimulus-x_peak)))))/width)
    ftvm_f = f(th, a, th_0, s, g)
    return (ftvm_f / np.sum(ftvm_f/a/(len(th)/36)) + baseline)*(1 + any(ftvm_f / np.sum(ftvm_f/a) + baseline < 0)*1e10)


def compute_grad( function = 'vm', 
                  center = 2.8, A = 1, width = 0.2, flatness = 0.25, b = 0.1):
    
    if function == 'vm':
        sys.exit("Not implemented yet!")
    
    elif function in ['fvm','fvm_from_fit']:
        f = generate_tuning_curve(A=A,width=width,flatness=flatness,b=b,center=center,function=function)
        k = 1/width
        ni = -abs(flatness)
        return lambda x : -f(x)*k*np.sin( (x - center ) + ni*np.sin( (x - center) )  )*(1 + ni*np.cos((x - center)))
    
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
        self.N_trial    = int(N_trial)
        
        # Define tuning curves and their gradients
        self.mu         = [ generate_tuning_curve( width  = self.width,
                                                   center = self.center,b = self.b,
                                                   A = self.A, function = self.function,
                                                   flatness = self.flatness),
                           generate_tuning_curve( width  = self.width, b = self.b,
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
        self.inv_sigma = np.linalg.inv( self.sigma )
        
        # Define Linear Fisher
        self.linear_fisher = lambda x: self.grad_vector(x).transpose().dot(self.inv_sigma).dot(self.grad_vector(x))
        
        # Define neurons
        self.neurons    = lambda x: neural_dynamics(x, self.mu, self.sigma, self.N_trial)

        return
    
    def grad_vector(self, x):
        return np.array([self.grad[0](x), self.grad[1](x)])
    
     
if __name__ == '__main__':
    from plot_tools import plot_simulation
    from cases import *

    
    stimuli = np.pi*np.array( [0, 1/3, 0.5, 2/3, 0.8,1,1.2,1+1/3,1.6666])
    
    system = NeuralSystem( CASE_2, N_trial = 500)
    plot_simulation(system, stimuli)