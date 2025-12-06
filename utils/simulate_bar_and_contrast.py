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
from   scipy.special    import expit
import sys

def parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_contrast_points', type=int, required=False, default = 100,   help="Number of contrast conditions")
    parser.add_argument('--n_sample', type=int,   required=False, default = int(cf['N']),   help="Sampling size")
    parser.add_argument('--min_amp', type=float,  required=False, default = cf['A'] ,  help="Minimum amplitude for the tuning curves")
    parser.add_argument('--max_amp', type=float,  required=False, default = cf['A'] + 0.1,  help="Maximum amplitude for the tuning curves")
    parser.add_argument('--alpha',   type=float,  required=False, default = 0.5,  help="Fixed noise correlation")   
    parser.add_argument('--beta',    type=float,  required=False, default = cf['beta'],  help="Neuron Fano factor")    
    parser.add_argument('--shift',   type=float,  required=False, default = np.pi/4,  help="Tnuning curve's centers shift")    
    parser.add_argument('--mode',   type=str,     required=False, default = 'sigmoid', choices=['linear','sigmoid'], help="Tnuning curve's centers shift")
    return parser.parse_args()


def simulate_system(args):
    Amax = args.max_amp
    Amin = args.min_amp
    mode = args.mode
    
    n_contrast_points = args.n_contrast_points
    system_config = cf
    system_config['alpha'] = args.alpha
    system_config['beta']  = args.beta
    system_config['center_shift'] = args.shift
    
    responses = np.zeros( (  n_contrast_points, THETA.size, args.n_sample, 2 ) )
    rho_s     = np.zeros( (  n_contrast_points ) )
    tuning_curves = np.zeros( ( n_contrast_points, THETA.size, 2))

    amplitutes = np.linspace(Amin, Amax, n_contrast_points) if mode == 'linear' else (Amax - Amin)*expit(np.linspace(-10,10,n_contrast_points)) + Amin

    for i,A in enumerate(amplitutes):
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

    norm = Normalize(vmin=int(tuning_curves.max(axis=1).min())+1, vmax=int(tuning_curves.max(axis=1).max()))
    
    for tc in tuning_curves:    
        C = tc.max()
        plt.plot(np.rad2deg(THETA), tc[:,0], label = 'Neuron 1',      c = cmap(norm(C)), lw = 1.8)
        plt.plot(np.rad2deg(THETA), tc[:,1], label = 'Neuron 2',      c = cmap(norm(C)) , lw = 1.8)
        plt.xlabel(r"Direction (deg)",size=15)
        plt.ylabel("Cell Response",size=15)
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
        plt.xlabel("Response Cell A",size=15)
        plt.ylabel("Response Cell B",size=15)
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
    
    plt.plot(tuning_curves[:,80,0], lw = 5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylabel("Response Cell A",size=15)
    plt.gca().set_xticklabels([])
    plt.show()
    
    return


def compute_stats(tuning_curves, responses, mode = 'rho'):
    n_contrast_points, n_angles = tuning_curves.shape[:-1]

    # SIGNAL CORRELATIONS    
    normalization = np.sqrt(  np.prod( np.var( responses.reshape(-1,2) ,axis = 0) ) )
                            
    # Net + BG | FEATURES = theta + contrast
    normalizationA = np.sqrt( np.prod( np.var(tuning_curves.reshape(-1,2), axis=0))) if mode == 'rho' else normalization
    rho_s_A =  np.cov( tuning_curves.reshape(-1,2), rowvar=False)[1,0] / normalizationA

    # Net | FEATURE = theta
    normalizationB =  np.sqrt( np.prod(np.var(  responses.mean( axis = (0,2)), axis=0) )) if mode == 'rho' else normalization
    rho_s_B        =  np.cov( responses.mean( axis = (0,2))  , rowvar = False )[1,0]/normalizationB
    
    # FEATURE = contrast
    normalizationC = np.sqrt( np.prod(np.var( responses.mean( axis = (1,2)), axis=0) )) if mode == 'rho' else normalization
    rho_s_C =  np.cov( responses.mean( axis = (1,2)), rowvar = False )[1,0]/ normalizationC

    # NOISE CORRELATIONS

    # Net + BG | FEATURES = theta + contrast
    # eta       = responses - tuning_curves[:,:,None,:]
    cov = lambda x : np.cov( x , rowvar=False)[1,0]
    var = lambda x : np.var(x , axis=0)
    _gathered_cov = np.array(list(map(cov, responses.reshape(-1,responses.shape[-2],2))))
    _gathered_var = np.array(list(map(var, responses.reshape(-1,responses.shape[-2],2)))).mean(axis=0)

    normalizationA = np.sqrt( np.prod( _gathered_var)) if mode == 'rho' else normalization
    rho_n_A = _gathered_cov.mean()/normalizationA

    # Net | FEATURE = theta
    normalizationB = np.sqrt(np.prod( np.mean([ np.var(responses[:,t,...].reshape(-1,2), axis=0)  for t in range(n_angles)], axis = 0))      )
    rho_n_B = np.mean([ np.cov(responses[:,t,...].reshape(-1,2), rowvar = False)[1,0]  for t in range(n_angles) ])/normalizationB
        
    # FEATURE = contrast
    normalizationC = np.sqrt(np.prod( np.mean([ np.var(responses[c,...].reshape(-1,2), axis=0)  for c in range(n_contrast_points)], axis = 0))      )
    rho_n_C = np.mean([ np.cov(responses[c,...].reshape(-1,2), rowvar = False)[1,0]  for c in range(n_contrast_points) ])/normalizationC
    
    return np.array(( rho_s_A, rho_n_A)), np.array(( rho_s_B, rho_n_B)), np.array(( rho_s_C, rho_n_C))

def plot_correlations( pointA, pointB, pointC, mode = 'rho'):
    import matplotlib.pyplot as plt
    from   matplotlib.colors import Normalize
    import matplotlib.cm as cm

    plt.scatter(*pointA)
    plt.scatter(*pointB)
    plt.scatter(*pointC)    
    
    shift = np.array([-0.02, 0.002])
    plt.text(*pointA + shift, 'CONTRAST\n       +\nDIRECTION')
    plt.text(*pointB + shift, 'DIRECTION')
    plt.text(*pointC + shift, 'CONTRAST')

    if mode != 'rho':
        plt.xlabel(r'$r^s$', size = 20)
        plt.ylabel(r'$r^n$', size = 20)
    else:
        plt.xlabel(r'$\rho^s$', size = 20)
        plt.ylabel(r'$\rho^n$', size = 20)
        
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlim(0.6, 1.02)

    plt.show()
    
    # cmap = cm.get_cmap('gist_yarg')
    # norm = Normalize(vmin=0, vmax=pointsB.shape[1])
    # plt.axhline(0,c='black')
    # plt.axvline(0,c='black')
    
    # plt.scatter(*pointA, zorder = 13, c = 'maroon', s = 80, marker = 's')
    # for i,point in enumerate(pointsB.transpose()):
    #     sm = plt.scatter(point[0], point[1], c=cmap(norm(i)))
    # for i,point in enumerate(pointsC.transpose()):
    #     plt.scatter(point[0], point[1], c='orange')

    # cbar = plt.colorbar(sm, cmap = 'gist_yard')
    # cbar.set_label('Contrast', fontsize=16)
    # cbar.set_ticks([])
    # plt.ylabel('Noise correlation', size = 15)
    # plt.xlabel('Stimulus correlation',size = 15)

    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # # plt.xlim(0.4,0.8)
    # # plt.ylim( pointA[1] - 0.00005, pointA[1] + 0.00005)
    
    # plt.show()
    
    return

if __name__ == '__main__':
    
    args = parser()
    
    rho_s, tuning_curves, responses = simulate_system(args)
    
    
    
    
    
    