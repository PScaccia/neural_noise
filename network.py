#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:09:15 2024

@author: paoloscaccia
"""
import numpy as np
import sys
import progressbar
from scipy.misc import derivative

THETA = np.linspace(0, 2*np.pi, 180)


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
        #flatness = abs(flatness)
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


def compute_grad( mu, function = 'vm', 
                  center = 2.8, A = 1, width = 0.2, flatness = 0.25, b = 0.1):
    
    if function == 'vm':
        sys.exit("Not implemented yet!")
    
    elif function in ['fvm','fvm_from_fit']:
        return lambda x : derivative(mu,x, dx=1e-3)
        
        # f = generate_tuning_curve(A=A,width=width,flatness=flatness,b=b,center=center,function=function)
        # k = 1/width
        # ni = -abs(flatness)
        # return lambda x : -(A*np.exp( (  np.cos(( x - center - flatness*np.sin( x - center) )) )/width )  + b )*k*np.sin( (x - center ) + ni*np.sin( (x - center) )  )*(1 + ni*np.cos((x - center)))
    
    else:
        sys.exit("Unknown function")   
    
    return 

def convert_fit_param( params ):
    a,th_0,s,baseline,g=params
    f = lambda stimulus, amp, x_peak, width, fact : \
            amp* np.exp((np.cos(np.deg2rad(stimulus-x_peak-fact*np.sin(np.deg2rad(stimulus-x_peak)))))/width)
    
    ftvm_f = lambda x : f(x, a, th_0, s, g)
    x_bin = np.linspace(0,360,36)
    Z = np.sum(ftvm_f(x_bin)/a/(len(x_bin)/36))
    
    out = {}
    out['A'] = a/Z
    out['width'] = s
    if th_0 < 0: th_0 = 360 + th_0
    out['center'] = np.deg2rad(th_0)
    out['flatness'] = np.deg2rad(g)
    out['b'] = baseline
        
    return out

def compute_corr_matrix( vector , rho):
    w = np.array([-vector[1],vector[0]])
    v = vector
    return np.eye(vector.size) + rho*(  np.outer(v,v)/v.dot(v) - np.outer(w,w)/w.dot(w) )

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
        self.var_stim_dependence  = True if self.rho in ['adaptive','poisson'] else False
        self.corr_stim_dependence = True if self.rho == 'adaptive' else False
        
        # Define tuning curves and their gradients
        self.mu         = [ generate_tuning_curve( width  = self.width,
                                                   center = self.center,b = self.b,
                                                   A = self.A, function = self.function,
                                                   flatness = self.flatness),
                           generate_tuning_curve( width  = self.width, b = self.b,
                                                  center = self.center + self.center_shift,
                                                  A = self.A, function = self.function,
                                                  flatness = self.flatness) ]
        self.grad       = [ compute_grad( self.mu[0], width  = self.width,
                                          center = self.center,
                                          A = self.A, function = self.function,
                                          flatness = self.flatness),
                            compute_grad( self.mu[1], width  = self.width,
                                          center = self.center + self.center_shift,
                                          A = self.A, function = self.function,
                                          flatness = self.flatness) ]

        if self.rho in ['adaptive','poisson']:
            self.update_rho( self.rho )
            
        # Define Covariance Matrix
        self.generate_variance_matrix()
        
        # Define Linear Fisher
        self.linear_fisher = lambda x: self.grad_vector(x).transpose().dot(self.inv_sigma(x)).dot(self.grad_vector(x))
        
        # Define Fisher Information
        # self.fisher = ...
        
        # Define neurons
        self.neurons    = lambda x: neural_dynamics(x, self.mu, self.sigma(x), self.N_trial)

        return
    
    def grad_vector(self, x):
        return np.array([self.grad[0](x), self.grad[1](x)])
    
    def generate_variance_matrix(self, D = 2):
        if D != 2:
            sys.exit("Not implemented yet!")
        
        if self.var_stim_dependence:
            V = lambda x : np.array( [[np.sqrt(self.beta*self.mu[0](x)), 0], 
                                      [0, np.sqrt(self.beta*self.mu[1](x))]])
            
            if self.corr_stim_dependence:
                # Case with stim-dependent Variance AND correation
                C = lambda x : compute_corr_matrix(np.array( [self.grad[0](x), self.grad[1](x)]), self.alpha)
            else:
                # Case with stim-dependent Variance BUT fixed correation
                C = lambda x : compute_corr_matrix(np.ones(D), self.alpha)
                
            self.sigma = lambda x : V(x).dot(C(x)).dot(V(x))
            
        else:
            # Case with stim-independent Matrix
            self.sigma = lambda x : self.V*np.array( [[1, self.rho], [self.rho, 1]])
            
        self.inv_sigma = lambda x : np.linalg.inv( self.sigma(x))
        return            
    
    def update_rho(self, mode):
        if mode == 'adaptive':
            self.rho = lambda x : self.alpha + x - x
        elif mode == 'poisson':
            self.rho = lambda x : self.alpha + x - x
        else:
            sys.exit("Unknown correlation type!")
        
    def compute_beneficial_angles(self, theta):
        
        eigenvalues, eigenvectors = np.linalg.eig(self.sigma(theta))
        width, height = 2*np.sqrt(eigenvalues)  # Doppio per coprire il 95% dei dati
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) 
        a,b = width/2, height/2
        X = np.sqrt(  (self.V - b*b) / (1 - (b/a)**2) )
        Y = b*np.sqrt(1-(X/a)**2)
        rot_matrix = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])

        xint1,xint2   = X, -X
        yint1 , yint2 = Y, -Y
        
        points = np.array([ [ xint1, yint1 ],
                            [ xint1, yint2],
                            [ xint2, yint1],
                            [ xint2, yint2]
                           ]).transpose()
        points = rot_matrix.dot(points)
        points = points.transpose()

        angles = np.array(  [ np.arctan2(*p) for p in points ])
        #angles[angles < 0] = np.pi/2 - angles[angles < 0]
        
        return angles

    def compute_signal_corr(self):
         theta = np.linspace(0,2*np.pi,500)
         Z = 1/len(theta)
         avg_f = lambda x : np.sum(x(theta))*Z
         
         diff = np.array([ self.mu[0](theta) - avg_f(self.mu[0]), self.mu[1](theta) - avg_f(self.mu[1]) ])
         variances = np.sum(diff**2,axis=1)
         covariance = diff[0,:].dot(diff[1,:])
                  
         return covariance/np.sqrt(variances[0]*variances[1])
     
    def compute_rho_signal(self):
         n = 1000
         x = np.linspace(0,2*np.pi,1000)
         
         mu1bar = np.sum(system.mu[0](x))/n
         mu2bar = np.sum(system.mu[1](x))/n
         
         mu1_std = np.sqrt( np.sum( (system.mu[0](x) - mu1bar)**2 )/n)
         mu2_std = np.sqrt( np.sum( (system.mu[1](x) - mu2bar)**2 )/n)
         covar   = np.sum( (  system.mu[0](x) - mu1bar )*( system.mu[1](x) - mu2bar)) / n 
         
         return covar/(mu1_std*mu2_std)

if __name__ == '__main__':
    from plot_tools import plot_simulation
    from cases import *

    
    stimuli = np.pi*np.array( [0, 1/3, 0.5, 2/3, 0.8,1,1.2,1+1/3,1.6666])
    
    system = NeuralSystem( CASE_2, N_trial = 500)
    plot_simulation(system, stimuli)