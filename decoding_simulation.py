#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 11:34:56 2025

@author: paolos
"""
from   utils.io_tools   import save_results_hdf5
from   utils.stat_tools import compute_RMSE
from   network import NeuralSystem, THETA
import scipy
import copy, os, sys
from   multiprocess import Pool
import h5py
from   tqdm import tqdm
import numpy as np
from   config.landscape import config
import cupy as cp

__version__ = 1.0

def compute_chunk_optimized(args):
    """Optimized worker that caches tuning curves per dtheta"""
    chunk_indices, config, N_theta_int = args
    
    theta_support = np.linspace(0, 2*np.pi, N_theta_int)
    
    log_d_f = lambda x: np.log(np.linalg.det(x))
    inv_f   = lambda x: np.linalg.inv(x)
    
    # Cache tuning curves by ix to avoid recomputation
    tuning_curve_cache = {}
    results = []
    
    for ix, iy in chunk_indices:
        dtheta = config['center_shift'][ix]
        alpha  = config['alpha'][iy]
        
        # Compute tuning curve only once per ix
        if ix not in tuning_curve_cache:
            local_config = copy.deepcopy(config)
            local_config['center_shift'] = dtheta
            local_config['alpha'] = 0.0
            system = NeuralSystem(local_config, N_trial=local_config['N'])
            
            mu_1, mu_2 = np.vectorize(system.mu[0]), np.vectorize(system.mu[1])
            tuning_curve = np.array([mu_1(theta_support), mu_2(theta_support)])
            tuning_curve_cache[ix] = (system, tuning_curve)
            ind_sigma   = np.array(list(map(system.sigma, theta_support)))      
            ind_inv_cov = np.array(list(map(inv_f, ind_sigma)))
            ind_log_det = np.array(list(map(log_d_f, ind_sigma)))
        
        # Update alpha and compute variance-dependent quantities
        system.alpha = alpha
        system.generate_variance_matrix()
        
        sigma   = np.array(list(map(system.sigma, theta_support)))
        inv_cov = np.array(list(map(inv_f, sigma)))
        log_det = np.array(list(map(log_d_f, sigma)))
        
        results.append((ix, iy, tuning_curve,
                        ind_sigma, ind_inv_cov, ind_log_det,
                        sigma,     inv_cov,     log_det))
    
    return results


def compute_signal_quantities_parallel(N_theta_int, n_processes=None, chunk_size=None):
    """Same as before, just use compute_chunk_optimized instead of compute_chunk"""
    from config.landscape import config
    
    N_rho_s = config['center_shift'].size
    N_rho_n = config['alpha'].size
    
    # Create meshgrid
    ix_grid, iy_grid = np.meshgrid(
        np.arange(N_rho_s), 
        np.arange(N_rho_n), 
        indexing='ij'
    )
    all_indices = list(zip(ix_grid.ravel(), iy_grid.ravel()))
    
    # Auto-determine chunk size
    if chunk_size is None:
        total_combinations = len(all_indices)
        n_procs = n_processes if n_processes else Pool()._processes
        # Larger chunks to benefit from tuning curve caching
        chunk_size = max(N_rho_n, total_combinations // (n_procs * 2))
    
    # Split into chunks
    chunks = [all_indices[i:i + chunk_size] 
              for i in range(0, len(all_indices), chunk_size)]
    
    args_list = [(chunk, config, N_theta_int) for chunk in chunks]
    
    # Initialize outputs
    tuning_curve_matrix = np.zeros((N_rho_s, 2, N_theta_int))
    cov_matrices        = np.zeros((N_rho_s, N_rho_n, N_theta_int, 2, 2))
    inv_cov_matrices    = np.zeros((N_rho_s, N_rho_n, N_theta_int, 2, 2))
    log_det_matrix      = np.zeros((N_rho_s, N_rho_n, N_theta_int))
    ind_cov_matrices        = np.zeros((N_rho_s, N_theta_int, 2, 2))
    ind_inv_cov_matrices    = np.zeros((N_rho_s, N_theta_int, 2, 2))
    ind_log_det_matrix      = np.zeros((N_rho_s, N_theta_int))

    # Parallel processing
    with Pool(processes=n_processes) as pool:
        chunk_results = pool.map(compute_chunk_optimized, args_list)
    
    # Collect results
    all_results = [item for chunk in chunk_results for item in chunk]
    
    for ix, iy, tuning_curve, ind_cov, ind_inv_cov, ind_log_det, cov, inv_cov, log_det in all_results:
        tuning_curve_matrix[ix, :, :]     = tuning_curve
        ind_cov_matrices[ix, :, :, :]     = ind_cov
        ind_inv_cov_matrices[ix, :, :, :] = ind_inv_cov
        ind_log_det_matrix[ix, :]         = ind_log_det
        cov_matrices[ix, iy, :, :, :]     = cov
        inv_cov_matrices[ix, iy, :, :, :] = inv_cov
        log_det_matrix[ix, iy, :]         = log_det

    # Flip last two axis of the tuning curve matrix ( new_size = [Nx,N_theta,2] )
    tuning_curve_matrix = np.moveaxis(tuning_curve_matrix, 1, -1)

    return tuning_curve_matrix, ind_cov_matrices, ind_inv_cov_matrices, ind_log_det_matrix, cov_matrices, inv_cov_matrices, log_det_matrix


def generate_response_chunk_indices_only(args):
    """Worker function that receives only indices and reconstructs what it needs"""
    chunk_indices, mu_subset, sigma_subset, n_trials, n_theta_subsample = args
    
    results = []
    
    for local_ix, local_iy, itheta in chunk_indices:
        # Use local indices to access the subset arrays
        response = np.random.multivariate_normal(
            mu_subset[local_ix, itheta * n_theta_subsample, :],
            sigma_subset[local_ix, local_iy, itheta * n_theta_subsample, :, :],
            size=n_trials
        )
        results.append((local_ix, local_iy, itheta, response))
    
    return results

def generate_responses_memory_efficient(mu, sigma, n_trials, n_theta_subsample, n_processes=Pool()._processes):
    """
    Memory-efficient version that splits work by ix (outer dimension)
    Better when nx is reasonably large
    """
    nx, ny, n_theta = sigma.shape[:3]
    
    if n_theta % n_theta_subsample != 0:
        sys.exit(f"Uneven angle resampling: trying to resample each {n_theta_subsample} "
                 f"angles from an array of {n_theta}")
    
    n_theta_reduced = n_theta // n_theta_subsample
    responses = np.zeros((nx, ny, n_theta_reduced, n_trials, 2))

    # Prepare work items: one per ix value
    work_items = []
    for ix in range(nx):
        # Create all (iy, itheta) combinations for this ix
        indices = [(0, iy, itheta)  # local_ix is always 0 since we're passing single slice
                   for iy in range(ny)
                   for itheta in range(n_theta_reduced)]
        
        work_items.append((
            indices,
            mu[ix:ix+1, :, :],      # Slice for this ix
            sigma[ix:ix+1, :, :, :, :],  # Slice for this ix
            n_trials,
            n_theta_subsample
        ))
    
    # Parallel processing
    with Pool(processes=n_processes) as pool:
        results_per_ix = pool.map(generate_response_chunk_indices_only, work_items)
    
    # Collect results
    for ix, chunk_results in enumerate(results_per_ix):
        for local_ix, iy, itheta, response in chunk_results:
            responses[ix, iy, itheta, :, :] = response
    
    return responses


def compute_bayesan_extimate(r, mu, inv_sigma, log_det, n_processes = None):
          
    theta_ext     = np.zeros( r.shape[:-1] )
    theta_support = np.linspace(0,2*np.pi,mu.shape[-2])
    sin_theta     = np.sin(theta_support)
    cos_theta     = np.cos(theta_support)
    n_stimuli     = r.shape[2]
    n_trials = r.shape[3]

    """
    args_list = [ (r[:,:,t,:,:], mu, inv_sigma, log_det_sigma, sin_theta, cos_theta, theta_support) for t in range(n_stimuli) ]

    # Parallel processing
    with Pool(processes=Pool()._processes) as pool:
        results = pool.map(bayesian_decoder, args_list)
        for t, result in enumerate(tqdm(  results, desc = 'Computing Theta ext') ):
            theta_ext[:,:,t,:] = result
    """
    
    # Determine batch size
    if n_processes is None:
        n_processes = Pool()._processes
    
    batch_size = max(1, n_trials // (n_processes))
    
    # Create batches
    args_list = []
    
    for t in tqdm(range(n_stimuli),desc='Decoding responses: '):
        for batch_id in range(0, n_trials, batch_size):
            end_idx = min(batch_id + batch_size, n_trials)
            trial_indices = list(range(batch_id, end_idx))
            r_batch = r[:, :, t, batch_id:end_idx, :]   
            args_list.append((
                batch_id // batch_size,
                trial_indices,
                r_batch,
                mu,
                inv_sigma,
                log_det,
                sin_theta,
                cos_theta,
                theta_support
            ))
        with Pool(processes=n_processes) as pool:
            for arg in args_list:
                batch_results = list(tqdm(
                    pool.imap(bayesian_decoder, args_list),
                    total=len(args_list),
                    desc='Decoding responses (batched)'
                ))

        for trial_index, trial_results in batch_results:
              theta_ext[:, :, t, trial_index] = trial_results
    
    
    return theta_ext

def bayesian_decoder(args):
    """
    r:  (Nx,Ny,N_trials,2)
    mu: (Nx,N_theta_int, 2)
    
    """
    _, trial_indices, r, mu, inv_sigma, log_det, sin_theta, cos_theta, theta_support = args
    dr = r[:,:,:,None,:] - mu[:,None,None,:,:] 

    # cost_function = np.einsum("ijkab,ijhka,ijhkb->ijkh",inv_sigma, dr,dr) + log_det[:,:,:,None]
    cost_function = np.einsum("abdij,abcdi,abcdj->abcd",inv_sigma, dr,dr) + log_det[:,:,None,:]

    # Compute posterior    
    P_post           = np.exp(-0.5*cost_function)

    # Define Integrand Functions
    P_post_sin = P_post*sin_theta[None,None,None,:]
    P_post_cos = P_post*cos_theta[None,None,None,:]
    del(P_post, cost_function, dr)

    # Integrate cartesian components
    # sin = scipy.integrate.simpson(P_post_sin, x = theta_support)
    # cos = scipy.integrate.simpson(P_post_cos, x = theta_support)
    sin = P_post_sin.sum(axis=-1)
    cos = P_post_cos.sum(axis=-1)
    del(P_post_sin, P_post_cos)

    # Compute stimulus extimate
    extimate = np.arctan2(sin, cos)

    return (trial_indices, np.mod( extimate , 2*np.pi ))

 
def compute_bayesan_extimate_gpu(r, mu, inv_sigma, log_det, batch_size = 50):
        
    theta_ext     = np.zeros( r.shape[:-1] )
    theta_support = cp.linspace(0,2*cp.pi,mu.shape[-2])
    sin_theta     = cp.sin(theta_support)
    cos_theta     = cp.cos(theta_support)
    n_stimuli     = r.shape[2]
    n_trials      = r.shape[3]

    cp_mu        = cp.asarray(mu)
    cp_inv_sigma = cp.asarray(inv_sigma)
    cp_log_det   = cp.asarray(log_det)

    for t in tqdm(range(n_stimuli),desc='Decoding responses: '):
        for batch_id in range(0, n_trials, batch_size):
            end_idx = min(batch_id + batch_size, n_trials)
            r_batch = r[:, :, t, batch_id:end_idx, :]   

            tmp = cp.asnumpy( 
                                cp_bayesian_decoder( cp.asarray(r_batch),
                                                     cp_mu,
                                                     cp_inv_sigma,
                                                     cp_log_det,
                                                     sin_theta,
                                                     cos_theta,
                                                     theta_support)  
                            ) 

            theta_ext[:,:,t,batch_id:end_idx ] = tmp
            cp.get_default_memory_pool().free_all_blocks()
                
    cp.get_default_memory_pool().free_all_blocks()
    return theta_ext

def cp_bayesian_decoder(r, mu, inv_sigma, log_det, sin_theta, cos_theta, theta_support):
    """
    r:  (Nx,Ny,N_trials,2)
    mu: (Nx,N_theta_int, 2)
    
    """
    dr = r[:,:,:,None,:] - mu[:,None,None,:,:] 

    # cost_function = np.einsum("ijkab,ijhka,ijhkb->ijkh",inv_sigma, dr,dr) + log_det[:,:,:,None]
    cost_function = cp.einsum("abdij,abcdi,abcdj->abcd",inv_sigma, dr,dr) + log_det[:,:,None,:]
    del(dr)
    
    # Compute posterior    
    P_post           = cp.exp(-0.5*cost_function)
    del(cost_function)
    
    # Define Integrand Functions
    P_post_sin = P_post*sin_theta[None,None,None,:]
    P_post_cos = P_post*cos_theta[None,None,None,:]
    
    del(P_post)
    # Integrate cartesian components
    # sin = scipy.integrate.simpson(P_post_sin, x = theta_support)
    # cos = scipy.integrate.simpson(P_post_cos, x = theta_support)
    sin = P_post_sin.sum(axis=-1)
    cos = P_post_cos.sum(axis=-1)
    
    del(P_post_cos, P_post_sin)
    
    # Compute stimulus extimate
    extimate = cp.arctan2(sin, cos)

    return cp.mod( extimate , 2*cp.pi )


if __name__ == '__main__':
    
    N_theta_int       = 180
    N_theta_subsample = 2
    N_trial           = config['N']
    beta              = config['beta']
    n_processes       = 30
    signal_dependency_file = "/home/paolo/data/signal_dependences_90points.hdf5"
    response_file          = "/home//paolo/data/responses_90points.hdf5"
    results_file           = "/home//paolo/data/bayesian_decoding_90points.hdf5"
    performance_file       = "/home//paolo/data/performance_90points.hdf5"
    attrs = {'beta' : beta, 'n_stimuli' : int(N_theta_int/N_theta_subsample) }
    
    """Compute Signal Manifolds and Cov. Matrices"""    
    if not os.path.isfile(signal_dependency_file):
        # Compute and save Tuning Curves, Inverse Covariance Matrices and determinants
        print("Computing Signal Quantities")
        mu, ind_sigma, ind_inv_sigma, ind_log_det_sigma,\
        sigma, inv_sigma, log_det_sigma = compute_signal_quantities_parallel(N_theta_int)    
        results = {'mu'                : mu,
                   'ind_sigma'         : ind_sigma,
                   'ind_inv_sigma'     : ind_inv_sigma,
                   'ind_log_det_sigma' : ind_log_det_sigma ,
                   'sigma'             : sigma,
                   'inv_sigma'         : inv_sigma,
                   'log_det_sigma'     : log_det_sigma,
                   'tuning_shift'      : config['center_shift'],
                   'noise_correlation' : config['alpha']       }
        save_results_hdf5(results, signal_dependency_file, attrs = attrs , version = __version__)
    else:
        # Read Input file with Tuning Curves, Inverse Covariance Matrices and determinants
        print("Reading file", signal_dependency_file)
        with h5py.File(signal_dependency_file,'r') as input_file:
            mu, sigma, inv_sigma, log_det_sigma         = input_file['mu'][:], input_file['sigma'][:], input_file['inv_sigma'][:], input_file['log_det_sigma'][:]
            ind_sigma, ind_inv_sigma, ind_log_det_sigma = input_file['ind_sigma'][:], input_file['ind_inv_sigma'][:], input_file['ind_log_det_sigma'][:]

    """Generate Neural Responses"""
    if not os.path.isfile(response_file):
        print("Generating Responses")        
        responses     = generate_responses_memory_efficient(mu, sigma, N_trial , N_theta_subsample, n_processes = n_processes) 
        ind_responses = generate_responses_memory_efficient(mu, ind_sigma[:,None,:,:,:], N_trial , N_theta_subsample, n_processes = n_processes)
        print('Saving results...')
        save_results_hdf5({'responses' : responses, 
                           'ind_responses'     : ind_responses,
                           'tuning_shift'      : config['center_shift'],
                           'noise_correlation' : config['alpha']       }, response_file, attrs = attrs, version = __version__)
    else:
        print("Reading Response File {}".format(response_file) )
        with h5py.File(response_file,'r') as input_file:
            responses, ind_responses = input_file['responses'][:], input_file['ind_responses'][:]
    del(sigma, ind_sigma) # Free memory
    
    """Compute Bayesian Extimates"""
    if not os.path.isfile(results_file):
        theta_ext     = compute_bayesan_extimate_gpu(responses, mu, inv_sigma, log_det_sigma)
        ind_theta_ext = compute_bayesan_extimate_gpu(ind_responses, mu, ind_inv_sigma[:,None,...], ind_log_det_sigma[:,None,...])    
        save_results_hdf5( {'theta_extimate'     : theta_ext,
                            'ind_theta_extimate' : ind_theta_ext,
                            'tuning_shift'       : config['center_shift'],
                            'noise_correlation'  : config['alpha']         }, results_file, attrs =attrs , version = __version__)
    else:
        print("Reading Decoding Extimates {}".format(results_file) )
        with h5py.File(results_file,'r') as input_file:
            theta_ext, ind_theta_ext = input_file['theta_extimate'][:], input_file['ind_theta_extimate'][:]
    del(responses, ind_responses)
    
    """Compute Decoding Error"""    
    RMSE = compute_RMSE(theta_ext, np.linspace(0,2*np.pi, theta_ext.shape[-2]) )
    ind_RMSE = compute_RMSE(ind_theta_ext, np.linspace(0,2*np.pi, ind_theta_ext.shape[-2]) )
    save_results_hdf5( {'RMSE'              : RMSE,
                        'ind_RMSE'          : ind_RMSE, 
                        'tuning_shift'      : config['center_shift'],
                        'noise_correlation' : config['alpha']     }, performance_file, attrs = attrs , version = __version__)
    
