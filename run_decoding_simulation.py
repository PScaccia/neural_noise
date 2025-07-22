#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:51:51 2025

@author: paoloscaccia
"""
import numpy as np
import os, sys
from   network import NeuralSystem, THETA
from   decoder import sample_theta_ext

def parser():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_proc',type=int,required=False, default=20,help="Number of workers")
    parser.add_argument('--multi_thread',action='store_true', default=True,help="Multi-threading")
    parser.add_argument('--config','-c',type=str,required=False,default='config/special.pkl',help="Configuration File")
    return parser.parse_args()

def read_conf( file ):
    
    if not os.path.isfile(file):
        sys.exit(f"Configuration file {file} not found!")
    elif '.pkl' not in file and 'default.py' not in file:
        sys.exit("Configuration file should be a pickle file!")
    elif 'default.py' in file:
        from config.default import config
    else:
        import pickle    
        with open(file,"rb") as handle:
            config = pickle.load(handle)
    return config

def main_cicle( config, args):
    import copy
    
    print("RUNNING SIMULATION WITH CONFIG FILE: ",args.config)
    print()
    
    alpha_grid, beta_grid = np.meshgrid( config['alpha'], config['beta'] )    
    case = copy.deepcopy(config)
    results = {}
    outfile = "./data/" + config['label'] + '.npz'
    
    for a, b in zip(  alpha_grid.flatten(), 
                      beta_grid.flatten()   ):

        print("Running simulation with alpha: {} beta: {}".format(a,b))
        
        case['alpha'] = a
        case['beta']  = b

        # Run single simulation with alpha = a and beta = b
        results['a_{:.2f}_b_{:.2f}'.format(a,b)] = simulation(case, args)
        
        # Save results
        save_results(results, outfile )
            
    return

def simulation( config, args ):
    
    # Define neural system
    system = NeuralSystem( config, N_trial = config['N'] )

    # Sample theta extimate
    theta_sampling = sample_theta_ext(system, THETA, 
                                      decoder = 'bayesian', N_step = config['int_step'], 
                                      multi_thread = args.multi_thread, 
                                      num_threads = args.n_proc )
    
    # Defin independent system
    system.alpha = 0.0
    system.rho = lambda x : 0.0
    system.generate_variance_matrix()
    
    # Sample theta extimate with the independent system
    theta_sampling_control = sample_theta_ext(system, THETA, 
                                              decoder = 'bayesian', N_step = config['int_step'], 
                                              multi_thread = args.multi_thread, 
                                              num_threads = args.n_proc )

    return [ theta_sampling, theta_sampling_control ]

def save_results(results, file ):
    if os.path.isfile(file):
        saved_data = dict(np.load(file))
        for key, result in results.items():
            if key not in saved_data:
                saved_data[key] = result
        np.savez_compressed(file, **saved_data)
        print("Updated file ",file)
    else:
        np.savez_compressed(file, **results)
        print("Created file ",file)
    return

if __name__ == '__main__':
    
    # Parse inline arguments
    args = parser()
    
    # Parse config dictionary from config file
    config = read_conf(args.config)

    # Run Main Cicle
    main_cicle( config, args )
    
    