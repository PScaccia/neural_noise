#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 11:34:56 2025

@author: paolos
"""
from   utils.io_tools   import save_results_hdf5
# from   utils.stat_tools import compute_RMSE_multiprocess
from   network import NeuralSystem
import copy, os, sys
from   multiprocess import Pool
import h5py
from   tqdm import tqdm
import numpy as np
from   config.landscape import config
import cupy as cp
from   datetime import datetime
import gc

__version__ = 1.0

def compute_chunk_optimized(args):
    """Optimized worker that caches tuning curves per dtheta"""
    chunk_indices, config, N_theta_int, mode = args
    
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
            local_config['rho']   = mode
            
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
    
    args_list = [(chunk, config, N_theta_int, args.mode) for chunk in chunks]
    
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
    np.random.seed( int(datetime.now().timestamp()) )
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

    for t in tqdm(range(n_stimuli),desc='Decoding responses: ', unit = 'stimulus'):
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


def compute_bayesan_extimate_gpu_in_batches(response_file, results_file, key, results_key, mu, inv_sigma, log_det, batch_size = 50, file_attrs = None , debug_mode = False):
        
    with h5py.File(response_file,'r') as handle:
        resp_shape = handle['responses'].shape
    print('Reading responses in ',response_file)
    
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)
    
    theta_support = cp.linspace(0, 2*cp.pi, mu.shape[-2])
    sin_theta     = cp.sin(theta_support)
    cos_theta     = cp.cos(theta_support)
    n_stimuli     = resp_shape[2]
    n_trials      = resp_shape[3]

    cp_mu        = cp.asarray(mu)
    cp_inv_sigma = cp.asarray(inv_sigma)
    cp_log_det   = cp.asarray(log_det)

    
    with h5py.File(response_file, 'r') as f:
       nx_points, ny_points = f[key].shape[:2]

    for batch_id in tqdm( range(0, n_trials, batch_size) , desc='Decoding responses: ', unit = 'batch'):
        
       if batch_id > 0 and debug_mode: continue 
        
       end_idx = min(batch_id + batch_size, n_trials)

       theta_ext     = np.zeros( ( nx_points, ny_points, end_idx - batch_id,  n_stimuli  )  )

       for t in tqdm(range(n_stimuli),leave = True, unit = 'stimulus'):
 
             theta_ext[:,:, :, t ] = cp.asnumpy( 
                                                 cp_bayesian_decoder( response_file,
                                                                      key,
                                                                      (t, batch_id, end_idx),
                                                                      cp_mu,
                                                                      cp_inv_sigma,
                                                                      cp_log_det,
                                                                      sin_theta,
                                                                      cos_theta,
                                                                      theta_support)  
                                               )
             cp.get_default_memory_pool().free_all_blocks()
       
       save_results_hdf5( { results_key  : theta_ext  }, results_file, silent_mode = True, attrs = file_attrs)
 
    cp.get_default_memory_pool().free_all_blocks()
    return


def cp_bayesian_decoder(file, key, indices, mu, inv_sigma, log_det, sin_theta, cos_theta, theta_support):
    """
    r:  (Nx,Ny,N_trials,2)
    mu: (Nx,N_theta_int, 2)
    
    """
    with h5py.File(file,'r') as handle:
        t, batch_id, end_idx = indices
        r = cp.asarray(handle[key][:, :, t, batch_id:end_idx, :])
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


def bayesian_decoder(args):
    """
    r:  (Nx,Ny,N_trials,2)
    mu: (Nx,N_theta_int, 2)
    
    """
    mu, inv_sigma, log_det, theta, sin_theta, cos_theta, t, batch_id, end_idx, file, key, results_file, results_key, attrs = args

    with h5py.File(response_file,'r') as handle:
        r = handle[key][:, :, t, batch_id:end_idx, :]

    dr = r[:,:,:, None,:] - mu[:,None,:,:] 

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
    extimate = np.mod( extimate , 2*np.pi )

    save_results_hdf5({results_key: extimate}, results_file, attrs = attrs, silent_mode=True)
    return 

def compute_bayesan_extimate_cpu_in_batches(response_file, results_file, key, results_key, mu, inv_sigma, log_det, batch_size = 50, file_attrs = None , debug_mode = False):
        
    with h5py.File(response_file,'r') as handle:
        resp_shape = handle['responses'].shape

    theta_support = np.linspace(0, 2*cp.pi, mu.shape[-2])
    sin_theta     = np.sin(theta_support)
    cos_theta     = np.cos(theta_support)
    n_stimuli     = resp_shape[2]
    n_trials      = resp_shape[3]
    indices       = range(0, n_trials, batch_size)
    n_processes   = len(indices)
    
    with h5py.File(response_file, 'r') as f:
       nx_points, ny_points = f[key].shape[:2]

    work_items = []

    for t in tqdm( range(n_stimuli) ):
        for batch_id in indices:
           end_idx    = min(batch_id + batch_size, n_trials)
           work_items.append( ( mu, inv_sigma, log_det, theta_support, sin_theta, cos_theta, t, batch_id, end_idx, response_file, key, results_file, results_key, file_attrs))

        # Parallel processing
        with Pool(processes=n_processes) as pool:
            for result in tqdm( pool.imap(bayesian_decoder, work_items)):
                 continue
    return


def circmean_gpu(samples, axis=None, high=2*cp.pi, low=0):
    """
    Compute the circular mean on GPU using CuPy
    
    Parameters:
    -----------
    samples : cp.ndarray
        Input array (on GPU)
    axis : int or None
        Axis along which to compute mean
    high : float
        Upper bound of circular range (default: 2π)
    low : float
        Lower bound of circular range (default: 0)
    
    Returns:
    --------
    mean : cp.ndarray or float
        Circular mean
    """
    # Convert samples to range [0, 2π]
    samples = cp.asarray(samples)
    
    if high != low:
        ang = (samples - low) * 2 * cp.pi / (high - low)
    else:
        ang = samples
    
    # Compute mean of unit vectors
    sin_mean = cp.mean(cp.sin(ang), axis=axis)
    cos_mean = cp.mean(cp.cos(ang), axis=axis)
    
    # Compute angle from mean vector
    mean_angle = cp.arctan2(sin_mean, cos_mean)
    
    # Convert back to original range
    if high != low:
        mean_angle = mean_angle * (high - low) / (2 * cp.pi) + low
    
    return mean_angle

def compute_RMSE( args ):
    import h5py
    from scipy.stats import circmean, circstd
    
    startx, endx, starty, endy, theta, file, key = args 
    
    with h5py.File(file,'r') as handle:
            theta_sampling = handle[key][startx:endx,starty:endy,...]

    diff = cp.asarray( theta[None,None,None,:] - theta_sampling )

    MSE  = circmean_gpu( cp.arctan2( cp.sin(diff), cp.cos(diff))**2, axis = 2 )    
    del(theta_sampling)
    gc.collect()
    return (startx, endx, starty, endy, cp.asnumpy( cp.sqrt(MSE)) )

def compute_RMSE_multiprocess(file, key, n_processes = None, batch_size = 20, debug_mode = False):
    import sys
    from   multiprocess import Pool
    from   tqdm import tqdm
    import h5py

    def iter_grid_batches(Nx, Ny, bx, by):
        for i in range(0, Nx, bx):
            for j in range(0, Ny, by):
                yield (i, min(i+bx, Nx), j, min(j+by, Ny))

    with h5py.File(file,'r') as handle:
        nx,ny,n_trials,n_stimuli = handle[key].shape
        
    theta = np.linspace(0,2*np.pi,n_stimuli) 
    RMSE  = np.empty( (nx,ny,n_stimuli)  )*np.nan

    results = []
    slices = [(startx, endx,starty, endy) for startx, endx, starty, endy in iter_grid_batches(nx, ny, batch_size, batch_size)]
    
    print('Computing error:')
    for startx, endx, starty, endy  in tqdm(slices, desc='Computing RMSE'):
        results.append(  compute_RMSE( (startx, endx, starty, endy, theta, file, key)) )
    
    print('Saving results')
    for startx, endx, starty, endy, mse in results:       
        RMSE[startx:endx, starty:endy, :] = mse
        
    return RMSE

def arg_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_proc',       type=int, required=False, default = 30,    help="Number of workers")
    parser.add_argument('--n_points_int', type=int, required=False, default = 180,   help="Number of integration points for baysian extimate")
    parser.add_argument('--decoding_batch_size', type=int, required=False, default = 60, help="Batch size for decoding")
    parser.add_argument('--n_subsample',  type=int, required=False, default = 2,     help="Number of input angles to be skipped for each point on the signal manifold")
    parser.add_argument('--wrkdir','-w',  type=str, required=False, default = "/home/paolo/data/",   help="Workdir")
    parser.add_argument('--mode','-m',  type=str, required=False, default = "poisson",   help="Type of covariance matrix")
    parser.add_argument('--generate_responses',action='store_true', default=False,   help="Generate and append responses to file (if it exists)")
    parser.add_argument('--decode_responses',  action='store_true', default=False,   help="Decode responses")
    parser.add_argument('--compute_error',     action='store_true', default=False,   help="Compute RMSE from decoded responses")
    parser.add_argument('--debug',             action='store_true', default=False,   help="Debug Mode")
    parser.add_argument('--extra_tag','-e',  type=str, required=False, default = "", help="Extra tag for the filename")
    return parser.parse_args()

def print_intro(args):
    
    for arg, value in args.__dict__.items():
        print(f"{arg:>20} : {repr(value):>10}")
    print()
    
if __name__ == '__main__':

    args = arg_parser()
    print_intro(args) 
    
    N_theta_int       = args.n_points_int
    N_theta_subsample = args.n_subsample
    N_trial           = config['N']
    n_processes       = args.n_proc
    n_stimuli         = int(N_theta_int/N_theta_subsample)
    
    signal_dependency_file = os.path.join( args.wrkdir, 'signal_dependences_'+args.extra_tag+'.h5' )
    response_file          = os.path.join( args.wrkdir, 'responses_'+args.extra_tag+'.h5' )
    results_file           = os.path.join( args.wrkdir, 'bayesian_extimate_'+args.extra_tag+'.h5' )
    performance_file       = os.path.join( args.wrkdir, 'improvement_'+args.extra_tag+'.h5' )
    attrs = {'beta' : config['beta'], 'n_stimuli' : n_stimuli, 'version' : __version__ }
    
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
    if args.generate_responses:
        print("Generating Responses")        
        responses     = generate_responses_memory_efficient(mu, sigma, N_trial , N_theta_subsample, n_processes = n_processes) 
        ind_responses = generate_responses_memory_efficient(mu, ind_sigma[:,None,:,:,:], N_trial , N_theta_subsample, n_processes = n_processes)
        print('Saving results...')
        save_results_hdf5({'responses' : responses, 
                           'ind_responses'     : ind_responses,
                           'tuning_shift'      : config['center_shift'],
                           'noise_correlation' : config['alpha']       }, response_file, attrs = attrs, version = __version__)
    del(sigma, ind_sigma) # Free memory
    
    if args.decode_responses:    
        """Compute Bayesian Extimates"""
        compute_bayesan_extimate_gpu_in_batches(response_file, results_file, 
                                                                'responses', 'theta_extimate', 
                                                                mu, inv_sigma, log_det_sigma, 
                                                                batch_size = args.decoding_batch_size, 
                                                                file_attrs = attrs, debug_mode = args.debug)
        compute_bayesan_extimate_gpu_in_batches(response_file,results_file, 
                                                                'ind_responses', 'ind_theta_extimate', 
                                                                mu, ind_inv_sigma[:,None,...], ind_log_det_sigma[:,None,...],
                                                                batch_size = args.decoding_batch_size, 
                                                                file_attrs = attrs , debug_mode = args.debug)
    """Compute Decoding Error"""            
    if args.compute_error:    
        RMSE     = compute_RMSE_multiprocess(results_file, 'theta_extimate',     batch_size = 6 , debug_mode = args.debug)
        ind_RMSE = compute_RMSE_multiprocess(results_file, 'ind_theta_extimate', batch_size = 6, debug_mode = args.debug)
        save_results_hdf5( {'RMSE'              : RMSE,
                            'ind_RMSE'          : ind_RMSE, 
                            'tuning_shift'      : config['center_shift'],
                            'noise_correlation' : config['alpha']     }, 
                          performance_file, attrs = attrs , version = __version__)
