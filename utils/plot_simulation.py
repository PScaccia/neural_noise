#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:34:36 2025

@author: paoloscaccia
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.stat_tools import compute_MSE
from decoder import circular_moving_average
from network import THETA
from scipy.signal import savgol_filter

def parser():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data','-d',type=str,required=True,help="Simulation File")
    parser.add_argument('--mode','-m',type=str,required=False,default='R_vs_rho',choices=['R_vs_rho'],help="Plot Mode")
    parser.add_argument('--outdir','-o',type=str,required=False,default='./',choices=['R_vs_rho'],help="Plot Mode")

    return parser.parse_args()

def compute_innovation(results, errors = False):
    alphas = []
    betas  = []
    R      = []
    R_err  = []
    
    print("Reading results...")
    for i,(label, sampling) in enumerate(results.items()):
                if 'config' in label: continue

                a,b=label.split('_')[1::2]
                try:
                    alphas.append(float(a))
                    betas.append(float(b))
                except ValueError:
                    continue
                
                # Compute MSE (default = MSE)                
                MSE, MSE_error               = compute_MSE( np.array( sampling[0,:,:]),  THETA, mode = 'mse', errors = errors)
                MSE_control, MSE_control_err = compute_MSE( np.array( sampling[1,:,:]),  THETA, mode = 'mse', errors = errors)
                    
                # Smooth with Savitzky-Golay filter 
                # MSE = circular_moving_average(THETA, MSE,15)
                # MSE_control = circular_moving_average(THETA, MSE_control,15)

                # MSE         = savgol_filter(MSE,         window_length=11, polyorder=3)
                # MSE_control = savgol_filter(MSE_control, window_length=11,  polyorder=3)
                                
                impr = (1 - (MSE/MSE_control))*100 
                R.append(impr)

                if errors:
                    impr_err = (100/np.abs(MSE_control))*np.sqrt( (MSE_error**2 + (MSE*MSE_control_err)**2) )
                    R_err.append( impr_err )

    return np.array(alphas), np.array(betas), np.array(R), np.array(R_err)

def plot_R_vs_rho(alphas, betas, R, filename = None ):
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    import numpy as np
    from   scipy.interpolate import CubicSpline

    plt.figure(figsize=(8,6))
    symbol = ['^','o','+']

    
    for i,b in enumerate(np.unique(betas)):
        ind = betas == b
        #plt.scatter( alphas[ ind ], R[ind].mean(axis = 1), label = str(b))
        sigma = np.sqrt((R[ind].std(axis=1)))
        label = "HIGH " if b > 0.5 else "LOW " 
        plt.errorbar( alphas[ ind ],  R[ind].mean(axis = 1), 
                      #yerr = sigma,  
                      ls='', label = label +  "("+str(b)+")",
                      marker="o",     alpha = 0.8, ms = 10)
    
    plt.axhline(0,c='black',lw=0.5)
    plt.axvline(0,c='black',lw=0.5)
    
    plt.legend(title = r"NOISE ($\beta$)", prop={'size' : 20})
    plt.xlabel("NOISE CORRELATION", weight='bold',size=16)
    plt.ylabel("IMPROVEMENT IN DECODING [%]",weight='bold',size=16)
    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=300)
        print("Saved plot in ",filename)

    return

def plot_improvement_single_panel( results, alpha, beta, debug = False):
    """
    Plot Signal Manifold colored with Percent Improvement after decoding.

    Parameters
    ----------
    results : dict
        DESCRIPTION.
    alpha : float
        DESCRIPTION.
    beta : float
        DESCRIPTION.

    Returns
    -------
    None.

    """
    import sys
    from decoder    import compute_MSE, circular_moving_average
    from plot_tools import plot_simulation_single_panel
    from network    import NeuralSystem 
    from config.special import config
    
    case_label = f"a_{alpha:.2f}_b_{beta:.2f}"
    print(case_label)
    if case_label not in results: sys.exit(f"Case with alpha {alpha} and beta {beta} not present in results.")

    theta_sampling         = results[case_label][0,:,:]
    theta_sampling_control = results[case_label][1,:,:]
    
    MSE         = compute_MSE( theta_sampling, theta = THETA )
    MSE_control = compute_MSE( theta_sampling_control , theta = THETA)
    
    MSE_raw = np.copy(MSE)
    MSE_control_raw = np.copy(MSE_control)
    
    MSE = circular_moving_average(THETA, MSE, 5)
    MSE_control = circular_moving_average(THETA, MSE_control, 5)
    MSE_averaged         = savgol_filter(MSE,window_length=5,polyorder=4)
    MSE_control_averaged = savgol_filter(MSE_control,window_length=5,polyorder=4)

    # MSE_averaged = circular_moving_average(THETA, MSE_averaged, 3)
    # MSE_control_averaged = circular_moving_average(THETA, MSE_control_averaged, 3)
        
    abs_err = -MSE_averaged + MSE_control_averaged
    filtered_abs_err = savgol_filter(abs_err,window_length=11,polyorder=7)
    filtered_MSE_control = savgol_filter(MSE_control_averaged,window_length=11,polyorder=7)

    if debug:
        #plt.plot(THETA, abs_err)
        #plt.plot(THETA, filtered_abs_err,c='blue')
        #plt.plot(THETA, MSE_control - MSE,c='black')
        plt.figure()
        plt.plot(THETA, MSE, c ='tab:blue', label = 'corr')
        plt.plot(THETA, MSE_control,  c = 'tab:orange',label = 'control')
        plt.plot(THETA, MSE_raw, c ='blue',marker='o',label='corr raw')
        plt.plot(THETA, MSE_control_raw, marker='o',c='orange', label = 'control raw')
        plt.legend()
        plt.show()
        
    
    imp = filtered_abs_err*100/filtered_MSE_control
    imp[:imp.size//2] = imp[imp.size//2:][::-1]
    config['alpha'] = alpha
    config['beta']  = beta

    system = NeuralSystem(config, N_trial = 10000)
    plot_simulation_single_panel( system, [], E = imp, color_label = "IMPROVEMENT IN DECODING [%]" )

    return

if __name__ == '__main__':
    
    args = parser()
    
    results = np.load(args.data)
    
    alphas, betas, R = compute_innovation(results)

        
    
    