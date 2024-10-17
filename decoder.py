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
    
    points = list(map(cost_function, THETA))
    return THETA[np.argmin(points)]
   
    # Return minimum
#    return (minimize_scalar(cost_function, bounds = [THETA[0], THETA[-1]]).x)


def sample_theta_ext(system, theta_array, decoder = 'bayesian'):
    from tqdm import tqdm
    theta_ext_sampling = np.empty( ( len(theta_array), system.N_trial ) )*np.nan
            
    for i,theta in zip(tqdm(range(len(theta_array)), desc = 'Computing decoding error: ' ), theta_array):
    # for i, theta in enumerate(theta_array):
        
        r_sampling = system.neurons(theta)
        if decoder == 'bayesian':
            theta_ext_sampling[i,:] = [ bayesian_decoder(system, r) for r in r_sampling ]
        elif decoder == 'MAP':
            theta_ext_sampling[i,:] = [ MAP_decoder(system, r) for r in r_sampling ]
        else:
            sys.exit("Not implemented yet!")
                    
    return theta_ext_sampling

def compute_MSE(theta_sampling):
    error = np.array([  1 - np.cos(t - theta_sampling[i,:] )  for i,t in enumerate(THETA) ])
    return np.mean( error, axis = 1)
    
if __name__ == '__main__':
    from plot_tools import plot_simulation, plot_theta_error, plot_theta_ext_prob
    from network import generate_tuning_curve_from_fit
    from cases import *
    import os
    # import argparse

    # parser = argparse.ArgumentParser()
    
    outdir = '/home/paolos/Pictures/decoding/case1/'
    save_theta_sampling = True
    DECODER = 'bayesian'
    CASE    = CASE_1

    print()    
    print("CASE:     1")    
    print("DECODER: ", DECODER)
    print()    

    
    # parser.add_argument('-c','--case',type=str,required=True, choices = ['case1', 'case2', 'case3', 'case4'],
    #                     help="Simulation's scenario")
    # parser.add_argument('-o','--outdir',type=str,required=False,default="./")    
    # args = parser.parse_args()
    
    # data = np.load('/home/paolos/data/quad_fits.npz')    
    # fit = data['quad_ftvm_fits'][:]
    
    # Define plot
    stimuli = np.array( [ np.pi ])
    #    stimuli = np.pi*np.array( [0, 1/3, 0.5, 2/3, 0.8,1,1.2,1+1/3,1.6666])

    # Define Neural System
    system = NeuralSystem( CASE, N_trial=500)
    # system.mu = [ generate_tuning_curve_from_fit(fit[1] - data['quad_ftvm_fit_parameters'][1,-2]),
    #               generate_tuning_curve_from_fit(fit[2]*( 55.62/73.80) - data['quad_ftvm_fit_parameters'][2,-2])  ]
    # del(data, fit)
    
    # Compute Decoder MSE
    theta_sampling = sample_theta_ext( system, THETA, decoder = DECODER )
    if save_theta_sampling:
        theta_outfile = './theta_sampling.txt'
        with open(theta_outfile,"w") as ofile:
            for k,v in CASE.items():
                print(f"# {k} : {v}", file = ofile)
            print(f"# DECODER {DECODER}", file = ofile)
            np.savetxt(ofile, theta_sampling)
            print("Saved theta sampling in ", theta_outfile)
    
    error = np.array([  1 - np.cos(t - theta_sampling[i,:] )  for i,t in enumerate(THETA) ])

    MSE = np.mean( error, axis = 1)
    FI=np.array(list(map(system.linear_fisher, THETA)))
    
    # Plot MSE along Signal Manifold    
    plot_simulation( system, stimuli, E = MSE, outdir = outdir)
    
    # Plot Decoder Error Analysis
    plot_theta_error(THETA, theta_sampling, MSE, FI = FI, title = f'N: {system.N_trial}', outdir = outdir)
    
    # Plot Histograms
    plot_theta_ext_prob(THETA, theta_sampling, outdir = outdir)
    os.system(f"convert -delay 5 -loop 0 {outdir}/prob_*.png {outdir}/decoding_prob.gif")
    os.system(f"rm {outdir}/prob_*.png")
    print(f"Created animation in {outdir}/decoding_prob.gif")

            