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
from   utils.io_tools import save_results, save_results_hdf5

def parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_proc',type=int,required=False, default=20,help="Number of workers")
    parser.add_argument('--multi_thread',action='store_true', default=True,help="Multi-threading")
    parser.add_argument('--hdf5',action='store_true', default=False,help="Save in HDF5 format")
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

def print_intro(args, message = ''):
    print()
    print(message)
    for key, value in sorted(vars(args).items()):
        print(f"{key:<15} : {value}")
    print()
    return


def main( config, args):
    import copy

    # Print All command-line arguments    
    print_intro(args, message = "Running decoding simulation with fixed tuning curves.")
    
    alpha_grid, beta_grid = np.meshgrid( config['alpha'], config['beta'] )    
    case = copy.deepcopy(config)
    results = {}
    outfile = "./data/" + config['label']
    if args.hdf5: outfile += '.hdf5'
    else:  outfile += '.npz'
    
    # Function to check if the independent case has already been simulated
    check_simulated_noise = lambda x : len([ k for k in results.keys() if '_b_{:.2f}'.format(x) in k]) != 0

    # Iterate on parameter grid
    for a, b in zip(  alpha_grid.flatten(), 
                      beta_grid.flatten()   ):

        print("Running simulation with alpha: {} beta: {}".format(a,b))       
        case['alpha'] = a
        case['beta']  = b
        key = 'a_{:.2f}_b_{:.2f}'.format(a,b)

        # Run single simulation with alpha = a and beta = b
        if check_simulated_noise(b):
            # If it has already ran the independent case for that noise take that
            old_case = [ x for x in results.keys() if '_b_{:.2f}'.format(b) in x][-1]
            results[key] = [ simulation(case, args,skip_independent = True)[0], 
                                                         results[old_case][1]   ]

        else:
            # Otherwise, simulate both cases
            results[key] = simulation(case, args)
        
        # Save results
        if args.hdf5: save_results_hdf5({ key : results[key] }, outfile)
        else: save_results({ key : results[key] }, outfile )
            
    return

def simulation( config, args, skip_independent = False ):
    
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
    if skip_independent:
        return [theta_sampling, None]
    else:
        # Sample theta extimate with the independent system
        theta_sampling_control = sample_theta_ext(system, THETA, 
                                                  decoder = 'bayesian', N_step = config['int_step'], 
                                                  multi_thread = args.multi_thread, 
                                                  num_threads = args.n_proc )
    
        return [ theta_sampling, theta_sampling_control ]


if __name__ == '__main__':
    
    # Parse inline arguments
    args = parser()
    
    # Parse config dictionary from config file
    config = read_conf(args.config)

    # Run Main Cicle
    main( config, args )