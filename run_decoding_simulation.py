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

def main( config, args):
    import copy
    
    print("RUNNING SIMULATION WITH CONFIG FILE: ",args.config)
    print(args.multi_thread)
    print()
    
    alpha_grid, beta_grid = np.meshgrid( config['alpha'], config['beta'] )    
    case = copy.deepcopy(config)
    results = {}
    outfile = "./data/" + config['label'] + '.npz'
    
    # Function to check if the independent case has already been simulated
    check_simulated_noise = lambda x : len([ k for k in results.keys() if '_b_{:.2f}'.format(x) in k]) != 0

    # Iterate on parameter grid
    for a, b in zip(  alpha_grid.flatten(), 
                      beta_grid.flatten()   ):

        print("Running simulation with alpha: {} beta: {}".format(a,b))       
        case['alpha'] = a
        case['beta']  = b

        # Run single simulation with alpha = a and beta = b
        if check_simulated_noise(b):
            # If it has already ran the independent case for that noise take that
            old_case = [ x for x in results.keys() if '_b_{:.2f}'.format(b) in x][-1]
            results['a_{:.2f}_b_{:.2f}'.format(a,b)] = [ simulation(case, args,skip_independent = True)[0], 
                                                         results[old_case][1]   ]

        else:
            # Otherwise, simulate both cases
            results['a_{:.2f}_b_{:.2f}'.format(a,b)] = simulation(case, args)
        
        # Save results
        save_results(results, outfile )
            
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

def save_results(results, file ):
    
    if os.path.isfile(file):
        # If file already exists, update it with new results
        saved_data = dict(np.load(file))
        for key, result in results.items():
            if key not in saved_data.keys():
                # Add new case...
                saved_data[key] = result

        np.savez_compressed(file, **saved_data)
        print("Updated file ",file)
    else:
        # If file doesn't exist, create it
        np.savez_compressed(file, **results)
        print("Created file ",file)
    return

def remove_case_from_results(file, a = None, b = None):
    
    if a is None and b is None: sys.exit("Specify alpha or beta for the case to be removed")

    os.system(f"mv {file} {file}.bkp")    
    print(f"Created bkp file {file}.bkp.")

    a_keys = []
    b_keys = []
    
    data = dict(np.load(file+'.bkp'))
    if a is not None and b is not None:
        sys.exit("Not implemented yet!")

    elif a is not None:
        keys = [k for k in data.keys() if "a_{:.2f}_".format(a) in k ]

    elif b is not None:
        keys = [k for k in data.keys() if "b_{:.2f}".format(b) in k] 

    for k in keys:
        del(data[k])
        print(f"Deleted case {k}!")
        
    save_results(data, file)
    del(data)        
    
    return

if __name__ == '__main__':
    
    # Parse inline arguments
    args = parser()
    
    # Parse config dictionary from config file
    config = read_conf(args.config)

    # Run Main Cicle
    main( config, args )