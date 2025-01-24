#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:34:36 2025

@author: paoloscaccia
"""

import numpy as np
import matplotlib.pyplot as plt
from decoder import compute_MSE, circular_moving_average
from network import THETA
def parser():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data','-d',type=str,required=True,help="Simulation File")
    parser.add_argument('--mode','-m',type=str,required=False,default='R_vs_rho',choices=['R_vs_rho'],help="Plot Mode")
    parser.add_argument('--outdir','-o',type=str,required=False,default='./',choices=['R_vs_rho'],help="Plot Mode")

    return parser.parse_args()

def compute_innovation(results):
    
    alphas = []
    betas  = []
    R      = []
    
    print("Reading results...")
    for i,(label, sampling) in enumerate(results.items()):
                if 'config' in label: continue

                a,b=label.split('_')[1::2]
                try:
                    alphas.append(float(a))
                    betas.append(float(b))
                except ValueError:
                    continue
                
                MSE = compute_MSE( np.array( sampling[0,:,:]) , theta = THETA[:2])
                MSE_control = compute_MSE( np.array(sampling[1,:,:]), theta = THETA[:2])

                R.append( circular_moving_average(THETA, (1-MSE/MSE_control)*100,11) )

    return np.array(alphas), np.array(betas), np.array(R)

def plot_R_vs_rho(alphas, betas, R):
    
    
    for b in np.unique(betas):
        ind = betas == b
        plt.scatter( alphas[ ind ], R[ind].mean(axis = 1), label = str(b))
    plt.legend()
    plt.show()
    
    
    return

if __name__ == '__main__':
    
    args = parser()
    
    results = np.load(args.data)
    
    alphas, betas, R = compute_innovation(results)

        
    
    