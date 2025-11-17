#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 11:57:38 2025

@author: paolo
"""

from   network import NeuralSystem, THETA
from   config.landscape import config as cf
import numpy as np
from   utils.stat_tools import compute_signal_corr_from_shift

def parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_contrast_points', type=int, required=False, default = 20,   help="Number of contrast conditions")
    parser.add_argument('--n_sample', type=int, required=False, default = int(cf['N']),   help="Sampling size")
    parser.add_argument('--min_amp', type=float, required=False, default = cf['A'] ,  help="Minimum amplitude for the tuning curves")
    parser.add_argument('--max_amp', type=float, required=False, default = cf['A'] + 0.1,  help="Maximum amplitude for the tuning curves")
    parser.add_argument('--alpha', type=float, required=False, default = 0.5,  help="Fixed noise correlation")   
    
    parser.add_argument('--beta', type=float, required=False, default = cf['beta'],  help="Neuron Fano factor")    
    parser.add_argument('--shift', type=float, required=False, default = np.pi/4,  help="Tnuning curve's centers shift")    
    return parser.parse_args()


def simulate_system(args):
    Amax = args.max_amp
    Amin = args.min_amp
    n_contrast_points = args.n_contrast_points
    system_config = cf
    system_config['alpha'] = args.alpha
    system_config['beta']  = args.beta
    system_config['center_shift'] = args.shift
    
    responses = np.zeros( (  n_contrast_points, THETA.size, args.n_sample, 2 ) )
    rho_s     = np.zeros( (  n_contrast_points ) )
    tuning_curves = np.zeros( ( n_contrast_points, THETA.size, 2))

    for i,A in enumerate(np.linspace(Amin, Amax, n_contrast_points)):
        system_config['A'] = A
        system =  NeuralSystem(system_config, N_trial=args.n_sample)
        tuning_curves[i, :, 0] = system.mu[0](THETA)
        tuning_curves[i, :, 1] = system.mu[1](THETA)
    
        rho_s[i] = compute_signal_corr_from_shift([ system_config['center_shift']], system_config['alpha'])[0]
        for t, theta in enumerate(THETA):
            tmp = system.neurons(theta)
            responses[i, t,:,:] = tmp
            
    return rho_s, tuning_curves, responses


def plot_system( tuning_curves, responses):
    import matplotlib.pyplot as plt
    from   matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    cmap = cm.get_cmap('gist_yarg')

    norm = Normalize(vmin=22, vmax=31)
    
    for tc in tuning_curves:    
        C = tc.max()
        plt.plot(np.rad2deg(THETA), tc[:,0], label = 'Neuron 1',      c = cmap(norm(C)), lw = 1.8)
        plt.plot(np.rad2deg(THETA), tc[:,1], label = 'Neuron 2',      c = cmap(norm(C)) , lw = 1.8)
        plt.xlabel(r"Direction (deg)",size=15)
        plt.ylabel("Cell Activity",size=15)
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax = plt.gca())
    cbar.set_label('Contrast', fontsize=16)
    cbar.set_ticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlim(0,360)
    plt.show()
    
    for tc,r  in zip(tuning_curves, responses):
        C = tc.max()
        plt.plot(tc[:,0],tc[:,1],  c = cmap(norm(C)), lw = 2.5, zorder = 13)
        plt.xlabel("Activity Cell A",size=15)
        plt.ylabel("Activity Cell B",size=15)
        for r_t in r:
            plt.scatter(r_t[:,0], r_t[:,1], c = 'burlywood', alpha=0.4, zorder = 1, s=4)
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax = plt.gca())
    cbar.set_label('Contrast', fontsize=16)
    cbar.set_ticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()
    
    return

def compute_stats(tuning_curves, responses):

    
    rho_s_A = np.corrcoef(tuning_curves.reshape((-1,2)), rowvar=False)[0]
    rho_s_B = np.corrcoef(tuning_curves[0,:,:], rowvar=False)
    
    
    return

if __name__ == '__main__':
    
    args = parser()
    
    rho_s, tuning_curves, responses = simulate_system(args)
    
    
    
    
    
    