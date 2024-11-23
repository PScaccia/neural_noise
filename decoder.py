#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:59:34 2024

@author: paolos
"""

import numpy as np
import sys, os
import subprocess
from network import NeuralSystem, THETA


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
    

def bayesian_decoder(system, r, theta, N_step = 600):
    from scipy.optimize import minimize_scalar
    
    # Define Integral Parameters
    theta_support = np.linspace(THETA[0], THETA[-1],N_step)
    dtheta = (THETA[-1] - THETA[0])/N_step    

    if not callable(system.rho):    
        sqrt_det = np.sqrt( np.linalg.det(system.sigma(theta)))
        norm_factor = dtheta / (2*np.pi*sqrt_det)
    else:
        norm_factor = dtheta / (2*np.pi)
        
    mu_vect          = lambda x :  np.array([system.mu[0](x),system.mu[1](x)]) 
    cost_function    = lambda x : (  r - mu_vect(x) ).transpose().dot( system.inv_sigma(x) ).dot( r - mu_vect(x) )    
    
    if not callable(system.rho):
        P_post           = lambda x : np.exp(-0.5*cost_function(x))
    else:
        P_post           = lambda x : np.exp(-0.5*cost_function(x)) / np.sqrt(np.linalg.det(system.sigma(x)))
    
    # Define Integrand Functions
    P_post_sin = lambda x : P_post(x)*np.sin(x)
    P_post_cos = lambda x : P_post(x)*np.cos(x)
    
    # Integrate 
    sin = np.sum( list(map(P_post_sin, theta_support)))
    cos = np.sum( list(map(P_post_cos, theta_support)))

    sin *= norm_factor
    cos *= norm_factor
    
    extimate = np.arctan2(sin, cos)
    
    return extimate if extimate >= 0 else extimate + 2*np.pi

def MAP_decoder(system, r):
    from scipy.optimize import minimize_scalar
        
    # Define cost function
    InvSigma = lambda x: system.inv_sigma(x)
    mu_vect       = lambda x :  np.array([system.mu[0](x), system.mu[1](x)]) 
    cost_function = lambda x : (  r - mu_vect(x) ).transpose().dot( InvSigma(x) ).dot( r - mu_vect(x) )
    
    points = list(map(cost_function, THETA))
    return THETA[np.argmin(points)]
   
    # Return minimum
#    return (minimize_scalar(cost_function, bounds = [THETA[0], THETA[-1]]).x)


def sample_theta_ext(system, theta_array, decoder = 'bayesian', N_step = 600, 
                     multi_thread = False, num_thread = 5):
    from tqdm import tqdm

    theta_ext_sampling = np.empty( ( len(theta_array), system.N_trial ) )*np.nan

    if multi_thread is False:            
        for i,theta in zip(tqdm(range(len(theta_array)), desc = 'Computing decoding error: ' ), theta_array):
            r_sampling = system.neurons(theta)
            if decoder == 'bayesian':
                decoder_f = lambda r: bayesian_decoder(system, r, theta, N_step = N_step)
            elif decoder == 'MAP':
                decoder_f = lambda r: MAP_decoder(system, r)
            else:
                sys.exit("Not implemented yet!")
            theta_ext_sampling[i,:] = list(map(decoder_f,r_sampling))
    else:
        # Multi Thread !!!
        import concurrent.futures
        num_threads = num_thread

        if len(theta_array)%num_threads != 0:
            sys.exit(f"Num. worker should be a divisor of the input size ({len(theta_array)})")

        def get_sample(input_array):
            _tmp = np.empty( ( len(input_array), system.N_trial ) )*np.nan
            for i,theta in zip(tqdm(range(len(input_array)), desc = 'Computing decoding error: ' ), input_array):
                r_sampling = system.neurons(theta)    
                if decoder == 'bayesian':
                    decoder_f = lambda r: bayesian_decoder(system, r, theta, N_step = N_step)
                elif decoder == 'MAP':
                    decoder_f = lambda r: MAP_decoder(system, r)
                else:
                    sys.exit("Not implemented yet!")
                _tmp[i,:] = list(map(decoder_f,r_sampling))
            return _tmp
        
        inputs = theta_array.reshape(num_thread, len(theta_array)//num_thread)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Esegui la funzione f() in parallelo con input diversi
            futures = {executor.submit(get_sample, x): x for x in inputs}
                
            # Gather results
            results = [ (future.result(), futures[future]) for future in concurrent.futures.as_completed(futures)]
            t
            # Order results
            sorted_results = list(map( lambda x: x[0], sorted(results, key=lambda x: x[1][0])))
 
            theta_ext_sampling = np.vstack(np.array(sorted_results))
            
    return theta_ext_sampling

def compute_MSE(theta_sampling, mode = 'cos', theta = THETA):
    if mode == 'cos':
        error = np.array([  1 - np.cos(t - theta_sampling[i,:] )  for i,t in enumerate(theta) ])
        return np.mean( error, axis = 1)
    elif mode == 'mse':
        from scipy.stats import circmean
        error = np.array([ x - theta for x in theta_sampling.transpose()]) 
        tan_error = np.arctan2( np.sin(error), np.cos(error))
        MSE = np.sqrt(circmean(tan_error**2,axis=0))
        return MSE

def save_sampling(system, theta_sampling, case, outfile, decoder = 'bayesian'):
    with open(outfile,"w") as ofile:
        for k,v in case.items():
            print(f"# {k} : {v}", file = ofile)
        print(f"# DECODER {decoder}", file = ofile)
        np.savetxt(ofile, theta_sampling)
        print("Saved theta sampling in ", outfile)
    
    return

def get_filtered_R(theta, MSE1, MSE2):
    # from scipy.interpolate import splrep, BSpline
    y = circular_moving_average(theta,MSE1, 5)
    y = circular_moving_average(theta,y, 5)

    y_control = circular_moving_average(THETA, MSE2,5)
    y_control = circular_moving_average(THETA, y_control,5)
    
    R = (1 - (y/y_control))*100
    # tck_s = splrep(x, R, s=len(x)-10)
    # R=BSpline(*tck_s)(theta)
    return R
    # circular_moving_average(THETA, R , 7)


def parser():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--decoder',type=str,required=False,choices=['bayesian','MAP'],default='bayesian')
    parser.add_argument('--case','-c',type=int,required=False,default=1,help="System Configuration")
    parser.add_argument('--noise','-n',type=str,required=False,default='constant',choices=['constant','adaptive','poisson'],help="System Configuration")
    parser.add_argument('-N','--n_trials',type=int,required=False,default=1000,help="Numerosity of the sampling")
    parser.add_argument('--n_proc',type=int,required=False,default=5,help="Number of workers")
    parser.add_argument('--multi_thread',action='store_true', default=False,help="Multi-threading")
    parser.add_argument('-N_step','--integration_step',type=int,required=False,default=360,help="Number of integration step for the Bayesian Extimation")
    return parser.parse_args()

if __name__ == '__main__':
    from plot_tools import  *
    from cases import CASES, A_CASES, P_CASES
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
        CASES = A_CASES if args.noise == 'adaptive' else P_CASES
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
        theta_sampling = sample_theta_ext( system, THETA, decoder = DECODER, N_step = N_step,  multi_thread = args.multi_thread, num_thread=args.n_proc)
        if save_theta_sampling: save_sampling(system, theta_sampling,  CASES[CASE], sampling_file)
        
    MSE = compute_MSE(theta_sampling)
    FI  = np.array(list(map(system.linear_fisher, THETA)))
    
    # Plot Decoder Error Analysis (Correlated system)
    title = r"$N_{sampling}$" + f': {system.N_trial} ' + r"$\rho_{N}: $" + f"{system.rho:.2}" if not callable(system.rho) else \
            r"$N_{sampling}$" + f': {system.N_trial} ' + r"$\rho_{N}: $" + "STIM. DEPENDENT" 
    plot_theta_error(THETA, theta_sampling, 2*MSE, FI = FI, title = title, outdir = outdir)
    
    # Plot Histograms
    # plot_theta_ext_prob(THETA, theta_sampling, outdir = outdir)
    # os.system(f"convert -delay 5 -loop 0 {outdir}/prob_*.png {outdir}/decoding_prob.gif")
    # os.system(f"rm {outdir}/prob_*.png")
    # print(f"Created animation in {outdir}/decoding_prob.gif")

    # Simulate decorrelated system
    if os.path.isfile(control_sampling_file):
        control_theta_sampling = np.loadtxt( control_sampling_file )
        print("Loaded control theta sampling from ",control_sampling_file)
    else:
        system.rho = 0.0 if not callable(system.rho) else lambda x : 0.0
        system.alpha = 0.0
        system.generate_variance_matrix()
        print()
        print("Simulating Indipendent System")
        
        control_theta_sampling = sample_theta_ext( system, THETA, decoder = DECODER , N_step = N_step, multi_thread = args.multi_thread, num_thread=args.n_proc)
        if save_theta_sampling: save_sampling(system, control_theta_sampling,  CASES[CASE], control_sampling_file)

    # Compute Decoder MSE (Indipendent system)        
    control_MSE = compute_MSE(control_theta_sampling)
    control_FI  = np.array(list(map(system.linear_fisher, THETA)))
    
    # Plot Decoder Error Analysis (Indipendent system)
    plot_theta_error(THETA, control_theta_sampling, 2*control_MSE, FI = control_FI,
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
        