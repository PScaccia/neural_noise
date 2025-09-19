#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:59:34 2024

@author: paolos
"""

import numpy as np
import sys, os
import subprocess
from   network import NeuralSystem, THETA
from   tqdm import tqdm
import scipy
from   utils.stat_tools import compute_MSE
import warnings
warnings.filterwarnings("ignore")

HOMEDIR = subprocess.check_output("echo $HOME", shell=True, text=True).replace('\n','')

def moving_average(x,y,w):
    if w == 1:
        return x,y
    n = w//2
    if w%2 == 0:
        return 0.5*(x[n-1:len(x)-n] + x[n:len(x)-(n-1)]), np.convolve(y, np.ones(w), 'valid') / w
    else:
        return x[n:-n], np.convolve(y, np.ones(w), 'valid') / w
        
def circular_moving_average(x,y,w):
    if w == 1:
        return x,y
    
    n = w//2

    if w%2 == 0:
        y = np.concatenate( ( y[len(x)-n:], y, y[:n]) )

        out = 0.5*(x[n-1:len(x)-n] + x[n:len(x)-(n-1)]), np.convolve(y, np.ones(w), 'valid') / w
    else:
        y = np.concatenate( ( y[len(x)-n:], y, y[:n]) )
        out = x[n:-n], np.convolve(y, np.ones(w), 'valid') / w
    
    return out[1]
    
def simmetrize_curve( curve, center, copy_right = True):    
    return  np.append( curve[91:-1][::-1], curve[88:])
    
def bayesian_decoder(r, mu, inv_sigma, det, theta_support):
    
    # Define Integral Parameters
    # dtheta = (THETA[-1] - THETA[0])/N_step    

    # if not callable(system.rho):    
    #     sqrt_det = np.sqrt( np.linalg.det(system.sigma(theta)))
    #     norm_factor = dtheta / (2*np.pi*sqrt_det)
    # else:
    #     norm_factor = dtheta / (2*np.pi)
        
    # mu_vect          = lambda x :  np.array([system.mu[0](x),system.mu[1](x)]) 
    
    # if not callable(system.rho):
    #     cost_function    = lambda x : (  r - mu_vect(x) ).transpose().dot( system.inv_sigma(x) ).dot( r - mu_vect(x) )
    # else:
    #     cost_function    = lambda x : (  r - mu_vect(x) ).transpose().dot( system.inv_sigma(x) ).dot( r - mu_vect(x) ) + abs(np.log(np.linalg.det(system.sigma(x))))   

    # P_post           = lambda x : np.exp(-0.5*cost_function(x))
    
    delta_r = r.reshape(r.shape[0],-1) - mu
    cost_function    = np.array([ dr.T@inv_matrix@dr + np.log(d) for dr, inv_matrix, d in zip(delta_r.transpose(),inv_sigma,det) ])
    P_post           = np.exp(-0.5*cost_function)

    # Define Integrand Functions
    P_post_sin = P_post*np.sin(theta_support)
    P_post_cos = P_post*np.cos(theta_support)

    # Integrate Naiv Algorithm
    # sin = np.sum( list(map(P_post_sin, theta_support)))
    # cos = np.sum( list(map(P_post_cos, theta_support)))    
    # sin *= norm_factor
    # cos *= norm_factor

    # Integrate via Simpson's rule
    sin = scipy.integrate.simpson(P_post_sin, x = theta_support)
    cos = scipy.integrate.simpson(P_post_cos, x = theta_support)
    
    # Integrate via Quadrate Method
    # sin,_ = scipy.integrate.quad(P_post_sin, 0, 2*np.pi)
    # cos,_ = scipy.integrate.quad(P_post_cos, 0, 2*np.pi)
    
    extimate = np.arctan2(sin, cos)

    return extimate if extimate >= 0 else extimate + 2*np.pi

def MAP_decoder(system, r):
    from scipy.optimize import minimize_scalar
        
    # Define cost function
    InvSigma      = lambda x: system.inv_sigma(x)
    mu_vect       = lambda x :  np.array([system.mu[0](x), system.mu[1](x)]) 
    cost_function = lambda x : (  r - mu_vect(x) ).transpose().dot( InvSigma(x) ).dot( r - mu_vect(x) )
    
    points = list(map(cost_function, THETA))
    return THETA[np.argmin(points)]
   
    # Return minimum
#    return (minimize_scalar(cost_function, bounds = [THETA[0], THETA[-1]]).x)


def sample_theta_ext(system, theta_array, decoder = 'bayesian', N_step = 500, 
                     multi_thread = False, num_threads = 5):

    theta_ext_sampling = np.empty( ( len(theta_array), system.N_trial ) )*np.nan
    theta_support = np.linspace(THETA[0], THETA[-1], N_step)
    mu = np.array([ list(map(system.mu[0],theta_support)),list(map(system.mu[1],theta_support))])
    inv_sigma = np.array(list(map(system.inv_sigma,theta_support)))
    det = np.array(list(map(lambda x: np.linalg.det(system.sigma(x)),theta_support)))

    if multi_thread is False:   
        # Single Thread         
        for i,theta in zip(tqdm(range(len(theta_array)), desc = 'Computing decoding error: ' ), theta_array):
            r_sampling = system.neurons(theta)
            if decoder == 'bayesian':
                decoder_f = lambda r: bayesian_decoder(r, mu, inv_sigma, det, theta_support)
            elif decoder == 'MAP':
                decoder_f = lambda r: MAP_decoder(system, r)
            else:
                sys.exit("Not implemented yet!")
            theta_ext_sampling[i,:] = list(map(decoder_f,r_sampling))
    else:
        # Multi Thread !!!
        from multiprocess import Pool

        if system.N_trial % num_threads != 0:
            sys.exit(f"Num. worker should be a divisor of the sampling size ({system.N_trial})")
            
        def get_single_theta_sample( input_array, n):
            n = int(n)
            _tmp = np.empty( ( len(input_array), n ) )*np.nan
            for i,theta in zip(tqdm(range(len(input_array)), desc = 'Computing decoding error: ' ), input_array):
                r_sampling = system.neurons(theta)[:n,:]

                if decoder == 'bayesian':
                    decoder_f = lambda r: bayesian_decoder(r, mu, inv_sigma, det, theta_support)
                elif decoder == 'MAP':
                    decoder_f = lambda r: MAP_decoder(system, r)
                else:
                    sys.exit("Not implemented yet!")
                _tmp[i,:] = list(map(decoder_f,r_sampling))
            return _tmp
        N = system.N_trial//num_threads

        # with Pool(num_threads) as pool:
        with Pool(num_threads) as pool:
            for i,results in enumerate(pool.starmap( 
                                        get_single_theta_sample,\
                                        zip( theta_array.repeat(num_threads).reshape(theta_array.size,num_threads).transpose(),
                                             np.ones(num_threads)*N ) 
                                          )):
               if i==0:
                   theta_ext_sampling = results
               else:
                   theta_ext_sampling = np.hstack( (theta_ext_sampling,results ))
                          
    return theta_ext_sampling


def save_sampling(system, theta_sampling, case, outfile, decoder = 'bayesian'):
    with open(outfile,"w") as ofile:
        for k,v in case.items():
            print(f"# {k} : {v}", file = ofile)
        print(f"# DECODER {decoder}", file = ofile)
        print(f"# M_sampling {system.N_trial}", file = ofile)
        
        np.savetxt(ofile, theta_sampling)
        print("Saved theta sampling in ", outfile)
    
    return

def get_filtered_R(theta, MSE1, MSE2,w1 = 3, w2 = 3, w3 = 3):
    # from scipy.interpolate import splrep, BSpline
    """
    y = circular_moving_average(theta,MSE1, 3)
    y = circular_moving_average(theta,y, 5)

    y_control = circular_moving_average(THETA, MSE2,5)
    y_control = circular_moving_average(THETA, y_control,5)
    
    R = (1 - (y/y_control))*100
    """
    y = circular_moving_average(theta,MSE1, w1) if w1 != 1 else MSE1
    y_control = circular_moving_average(THETA, MSE2,w2) if w2 != 1 else MSE2

    R = (1- (y/y_control))*100
    
    R = circular_moving_average(THETA, R,w3) if w3 != 1 else R
    # tck_s = splrep(x, R, s=len(x)-10)
    # R=BSpline(*tck_s)(theta)
    return R
    # circular_moving_average(THETA, R , 7)


def parser():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--decoder',type=str,required=False,choices=['bayesian','MAP'],default='bayesian')
    parser.add_argument('--case','-c',type=int,required=False,default=1,help="System Configuration")
    parser.add_argument('--noise','-n',type=str,required=False,default='constant',choices=['constant','adaptive','poisson','experimental'],help="System Configuration")
    parser.add_argument('-N','--n_trials',type=int,required=False,default=1000,help="Numerosity of the sampling")
    parser.add_argument('--n_proc',type=int,required=False,default=5,help="Number of workers")
    parser.add_argument('--multi_thread',action='store_true', default=False,help="Multi-threading")
    parser.add_argument('-N_step','--integration_step',type=int,required=False,default=360,help="Number of integration step for the Bayesian Extimation")
    return parser.parse_args()

if __name__ == '__main__':
    from plot_tools import  *
    from cases import CASES, A_CASES, P_CASES, E_CASES
    import os
    
    # Parse Arguments
    args = parser()
    
    # Init Simulation
    save_theta_sampling = True
    DECODER = args.decoder
    CASE    = args.case
    
    outdir  = f'{HOMEDIR}/Pictures/decoding/case{CASE}/'
    N_trial = args.n_trials
    N_step  = args.integration_step
    file_ext = '.txt'
    if args.noise != 'constant':
        if args.noise == 'adaptive':
             CASES = A_CASES 
        elif args.noise == 'experimental' :
            CASES = E_CASES
        else:
            CASES = P_CASES
        outdir = outdir.replace('case',f'{args.noise.upper()}_case')
        file_ext = f'_{args.noise.upper()}' +'.txt'
    sampling_file         = outdir + f'/theta_sampling_case{CASE}' + file_ext
    control_sampling_file = sampling_file.replace('.txt','_control') + file_ext

    print()    
    print("CASE:         ",CASE)    
    print("DECODER:      ", DECODER)
    print("NOISE CORR.:  ",args.noise)
    print("OUTDIR:       ",outdir)
    print("N TRIALS:     ",N_trial)
    print("MULTI-THREAD: ",args.multi_thread)
    print()

    if not os.path.isdir(outdir):
        os.system(f"mkdir -p {outdir}")
    
    # Define plot
    stimuli = np.array( [ np.pi ])

    # Define Neural System
    system = NeuralSystem( CASES[CASE], N_trial=N_trial)
    
    # Compute Decoder MSE (Correlated system)
    if os.path.isfile(sampling_file):
        theta_sampling = np.loadtxt( sampling_file )
        print("Loaded theta sampling from ",sampling_file)
    else:
        theta_sampling = sample_theta_ext( system, THETA, decoder = DECODER, N_step = N_step,  multi_thread = args.multi_thread, num_threads=args.n_proc)
        if save_theta_sampling: save_sampling(system, theta_sampling,  CASES[CASE], sampling_file)
        
    MSE = compute_MSE(theta_sampling)
    FI  = np.array(list(map(system.linear_fisher, THETA)))
    
    # Plot Decoder Error Analysis (Correlated system)
    title = r"$N_{sampling}$" + f': {system.N_trial} ' + r"$\rho_{N}: $" + f"{system.rho:.2}" if not callable(system.rho) else \
            r"$N_{sampling}$" + f': {system.N_trial} ' + r"$\rho_{N}: $" + f"{system.alpha:.2}"
    plot_theta_error(THETA, theta_sampling, 2*MSE, FI = FI if args.noise != 'poisson' else None, title = title, outdir = outdir)
    
    # Simulate decorrelated system
    if os.path.isfile(control_sampling_file):
        control_theta_sampling = np.loadtxt( control_sampling_file )
        print("Loaded control theta sampling from ",control_sampling_file)
    else:
        system.rho   = 0.0 if not callable(system.rho) else lambda x : 0.0
        system.alpha = 0.0
        #system.corr_stim_dependence = False
        system.generate_variance_matrix()
        
        print()
        print("Simulating Indipendent System")
        
        control_theta_sampling = sample_theta_ext( system, THETA, decoder = DECODER , N_step = N_step, multi_thread = args.multi_thread, num_threads=args.n_proc)
        if save_theta_sampling: save_sampling(system, control_theta_sampling,  CASES[CASE], control_sampling_file)

    # Compute Decoder MSE (Indipendent system)        
    control_MSE = compute_MSE(control_theta_sampling)
    control_FI  = np.array(list(map(system.linear_fisher, THETA)))
    
    # Plot Decoder Error Analysis (Indipendent system)
    plot_theta_error(THETA, control_theta_sampling, 2*control_MSE, FI = control_FI if args.noise != 'poisson' else None,
                     title = r"$N_{sampling}$" + f': {system.N_trial}', outdir = outdir, filename = 'control_MSE.png')

    # Reload system params
    system = NeuralSystem( CASES[CASE], N_trial=N_trial)
    
    # Plot Improvement
    R = get_filtered_R(THETA, MSE, control_MSE)
    plot_improvement(THETA, R, outdir = outdir, raw_MSE1=MSE, raw_MSE2=control_MSE)

    # Plot MSE along Signal Manifold    
    plot_simulation( system, [], E = R, outdir = outdir, color_label = 'Improvement [%]', cmap = 'coolwarm' if args.noise else 'coolwarm')
    
    # Plot Gradient Space
    plot_gradient_space(system, outfile = outdir +'/gradient_space.png')
    
    # Plot Decoding Accuracy (Heatmap)
    # case 1 nbins = 90, vmax = 1.8, IMP = 1.8 
    # case 2 nbins = 90, vmax = 20, IMP = 1.8

    if CASE == 1:
        plot_heatmap(theta_sampling, control_theta_sampling, nbins = 90, vmax = 10, cmap = 'jet',outfile = outdir + '/heatmap.png')
    elif CASE == 2:
        plot_heatmap(theta_sampling, control_theta_sampling, nbins = 90, vmax = 30, cmap = 'jet',outfile = outdir + '/heatmap.png')
    elif CASE == 3:
        plot_heatmap(theta_sampling, control_theta_sampling, nbins = 90, vmax = 10, cmap = 'jet',outfile = outdir + '/heatmap.png')
    elif CASE == 4:
        plot_heatmap(theta_sampling, control_theta_sampling, nbins = 90, vmax = 30, cmap = 'jet',outfile = outdir + '/heatmap.png')
        
