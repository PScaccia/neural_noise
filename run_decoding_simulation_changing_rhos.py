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
from   run_decoding_simulation import parser, read_conf, simulation, print_intro
from   utils.io_tools import save_results, save_results_hdf5

def main( config, args):
    """
    Re-adapted (from script run_decoding_simulation.py) for different shift in the tuning curves
    """
    import copy
    
    # Print All command-line arguments    
    print_intro(args, message = "Running decoding simulation changing tuning curves.")
    
    alpha_grid, delta_theta = np.meshgrid( config['alpha'], config['center_shift'] )    
    case = copy.deepcopy(config)
    results = {}
    outfile = "./data/" + config['label'] + '.hdf5' if args.hdf5 else '.npz'
    
    # Function to check if the independent case has already been simulated
    check_simulated = lambda x : len([ k for k in results.keys() if '_dtheta_{:.2f}'.format(x) in k]) != 0
    
    b = config['beta']
    
    # Iterate on parameter grid
    for a, dtheta in zip(  alpha_grid.flatten(), 
                           delta_theta.flatten()   ):
        
        # TMP EDIT
        results = {}

        print("Running simulation with alpha: {} dtheta: {}".format(a,dtheta))       
        case['alpha']         = a
        case['center_shift']  = dtheta
        case['beta']          = b
        key = 'a_{:.2f}_dtheta_{:.2f}'.format(a,dtheta)
        
        # Run single simulation with alpha = a and center_shift = dtheta
        if check_simulated(dtheta):
            # If it has already ran the independent case for that noise take that
            old_case = [ x for x in results.keys() if '_dtheta_{:.2f}'.format(dtheta) in x][-1]
            results[key] = [ simulation(case, args,skip_independent = True)[0], 
                                                                   results[old_case][1]   ]
        else:
            # Otherwise, simulate both cases
            results[key] = simulation(case, args)
        
        # Save results
        if args.hdf5: save_results({ key : results[key] }, outfile)
        else: save_results({ key : results[key] }, outfile )
            
    return


if __name__ == '__main__':
    
    # Parse inline arguments
    args = parser()
    
    # Parse config dictionary from config file
    config = read_conf("config/landscape.pkl")

    # Run Main Cicle
    main( config, args )
