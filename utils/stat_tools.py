#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:09:44 2024

@author: paolos
"""
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def metropolis_hastings(f, x0, num_samples, proposal_width):
    """
    Esegue l'algoritmo di Metropolis-Hastings per campionare dalla distribuzione proporzionale a f(x).
    
    Args:
        f (function): Funzione proporzionale alla distribuzione target (non normalizzata).
        x0 (float): Valore iniziale del campione.
        num_samples (int): Numero di campioni da generare.
        proposal_width (float): Deviazione standard della distribuzione di proposta (es. normale).
        
    Returns:
        np.array: Array dei campioni generati.
    """
    samples = []
    x_current = x0
    for _ in range(num_samples):
        # Proposta: un nuovo punto dalla distribuzione normale centrata su x_current
        x_proposed = np.random.normal(x_current, proposal_width)
        
        # Calcola il rapporto di accettazione
        acceptance_ratio = f(x_proposed) / f(x_current)
        
        # Accetta o rifiuta il campione proposto
        if np.random.rand() < acceptance_ratio:
            x_current = x_proposed  # Accetta il nuovo campione
        
        samples.append(x_current)  # Aggiungi il campione (anche se non accettato)

    return np.array(samples)

def integrate_metropolis(f, num_samples=10000, proposal_width=1.0):
    """
    Utilizza Metropolis-Hastings per stimare l'integrale di f(x).
    
    Args:
        f (function): Funzione target non normalizzata.
        num_samples (int): Numero di campioni da generare.
        proposal_width (float): Deviazione standard della distribuzione di proposta.
        
    Returns:
        float: Approssimazione dell'integrale di f(x).
    """
    # Inizializzazione
    x0 = 0.0  # Punto iniziale
    samples = metropolis_hastings(f, x0, num_samples, proposal_width)
    
    # Stima dell'integrale: usa la media della funzione target sui campioni generati
    integral_estimate = np.mean( np.array(list(map(f,samples)))) * (np.max(samples) - np.min(samples))
    return integral_estimate, samples

# Esempio: funzione da integrare
def target_function(x):
    # Funzione proporzionale a una distribuzione normale con media 0 e deviazione standard 1
    return np.exp(-x**2 / 2)

def compute_rho_star( system ):
    theta_space = np.linspace(0, 2*np.pi, 720)
    

    # Compute Noise Variance
    response1 = np.array([ system.neurons(t)[:,0].var()  for  t in theta_space ])
    response2 = np.array([ system.neurons(t)[:,1].var()  for  t in theta_space ])
    Vn=np.mean(np.sqrt(response1*response2))
    
    # Compute Total Variance
    var1 = np.array([ system.neurons(t)[:,0] for  t in theta_space ]).var()
    var2 = np.array([ system.neurons(t)[:,1] for  t in theta_space ]).var()
    Vtot = np.sqrt(var1*var2)

    # Compute Signal Covariance
    mu1 = np.array( list(map(system.mu[0], theta_space) ) )
    dmu1 = mu1 - mu1.mean()
    mu2 = np.array( list(map(system.mu[1], theta_space) ) )
    dmu2 = mu2 - mu2.mean()
    mu_cov = np.mean(dmu1*dmu2)
    
    return 2*Vn*mu_cov/(Vtot**2 - Vn**2)

def compute_MSE(theta_sampling, theta, mode = 'cos', errors = False):
    if mode == 'cos':
        error = np.array([  1 - np.cos(t - theta_sampling[i,:] )  for i,t in enumerate(theta) ])
        return np.mean( error, axis = 1)
    elif mode == 'mse':
        from scipy.stats import circmean, circstd
        error = np.array([ x - theta for x in theta_sampling.transpose()]) 
        tan_error = np.arctan2( np.sin(error), np.cos(error))
        MSE = circmean(tan_error**2,axis=0)
        if errors:
            MSE_std = np.sqrt(circstd(tan_error**2,axis=0))
            return MSE, MSE_std/np.sqrt(theta_sampling.size)
        else:
            return MSE, None

def compute_innovation(results, errors = False):
    from network import THETA
    from decoder import circular_moving_average
    from scipy.signal import savgol_filter

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
                    
                # # Smooth with Savitzky-Golay filter 
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

def d_prime_squared(vector, Sigma):
    invSigma = np.linalg.inv(Sigma)
    return vector@invSigma@vector

def compute_2AFC_mutual_information(vector, Sigma):
    from scipy.special import erf
    
    d = np.sqrt(d_prime_squared(vector, Sigma))/(2*np.sqrt(2))    
    pe = 0.5*(1-erf(d))    
    return 1 + pe*np.log(pe) + (1-pe)*np.log((1-pe))

def simulate_2AFC(V1 = 0.1, V2 = 0.1, n_theta_points = 360, n_rho_points = 500, fix_det = False):
    
    Sigma = np.array([[V1,0], [0, V2]])

    a = np.array([-np.sqrt(2), 0.])
    b = np.array([+np.sqrt(2), 0.])

    angles = np.linspace(-np.pi/4,np.pi/4,n_theta_points)
    rhos   = np.linspace(-0.99999,0.99999,n_rho_points)
    
    
    cov_matrix = lambda x : np.array([[V1, x*np.sqrt(V1*V2)],
                                   [x*np.sqrt(V1*V2), V2]])
    angle_grid, rho_grid = np.meshgrid(angles,rhos)
    synergy_grid = np.zeros_like(angle_grid)

    for i_row, (theta_row,rho_row) in enumerate(zip(angle_grid,rho_grid)):
        for i_col, (theta,rho) in enumerate( zip(theta_row, rho_row) ):
            Rotation = np.array([[np.cos(theta),-np.sin(theta)],
                                 [np.sin(theta),np.cos(theta)]])
            _local_a = Rotation@a
            _local_b = Rotation@b
            diff = _local_b - _local_a

            scale_factor = 1 if not fix_det else np.sqrt((1-rho**2))

            MI_correlated  = compute_2AFC_mutual_information(diff, cov_matrix(rho)/scale_factor)
            MI_independent = compute_2AFC_mutual_information(diff, Sigma)
            synergy_grid[i_row][i_col] = ((MI_correlated/MI_independent) - 1)*100
        
    return angle_grid, rho_grid, np.ma.masked_invalid(synergy_grid)

def compute_bar_experiment_MI( noise = 10, N_points_deltat = 200, N_points_rhon = 200, mode = 'poisson'):
    from network import NeuralSystem
    
    theta_support  = np.linspace(0, 2*np.pi, 1000)
    delta_theta_ax = np.linspace(-np.pi,np.pi,N_points_deltat)
    rho_ax         = np.linspace(-1,1,N_points_rhon)
    rhos_ax        = np.zeros_like(delta_theta_ax)
    
    delta_theta_grid, rho_grid = np.meshgrid(delta_theta_ax, rho_ax)
    synergy_grid = np.zeros_like(delta_theta_grid)

    for i_row, (theta_row,rho_row) in enumerate(zip(delta_theta_grid, rho_grid)):
        for i_col, (theta,rho) in enumerate( zip(theta_row, rho_row) ):
            label = mode if mode != 'constant' else rho

            config = {  'rho'           : label,
                        'alpha'         : rho,
                        'beta'          : noise,
                        'function'      : 'fvm',
                        'V'             : noise,
                        'A'             : 0.17340510276921817,
                        'width'         : 0.2140327993142855,
                        # 'width'         : 0.5140327993142855,
                        'flatness'      : 0.6585904840291591,
                        'b'             : 1.2731732385019432,
                        'center'        : 2.9616211149125977 - 0.21264734641020677,
                        'center_shift'  : theta,
                        'N'             : 100000,
                        'int_step'      : 100
                        }
            
            system     = NeuralSystem(config)                
            system_ind = NeuralSystem(config)
            system_ind.alpha = 0.0
            if mode == 'constant': system_ind.rho = 0.0
            system_ind.generate_variance_matrix()
            
            mus = np.array([ [system.mu[0](t),system.mu[1](t)] for t in theta_support ])
            
            Sigma_s = np.cov(mus,rowvar=False)
            Sigma_n = np.average(np.array(list( map(system.sigma, theta_support) )),axis=0)
            V_n     = np.average(np.array(list( map(system_ind.sigma, theta_support) )),axis=0)

            MI_correlated  = 0.5*np.log( np.linalg.det(Sigma_s + Sigma_n) / np.linalg.det(Sigma_n) ) 
            MI_independent = 0.5*np.log( np.linalg.det(Sigma_s + V_n)     / np.linalg.det(V_n)     )
            
            synergy_grid[i_row][i_col] = ((MI_correlated/MI_independent) - 1)*100

    for i,dt in enumerate(delta_theta_ax):
            config['center_shift'] = dt
            system = NeuralSystem(config)
            mus = np.array([ [system.mu[0](t),system.mu[1](t)] for t in theta_support ])
            rhos_ax[i] = np.corrcoef(mus, rowvar=False)[1,0]
            
    rhos_grid, rhon_grid = np.meshgrid(rhos_ax, rho_ax)
    
    return rhos_grid, rhon_grid, synergy_grid