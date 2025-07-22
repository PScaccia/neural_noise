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

# # Parametri
# num_samples = 10000
# proposal_width = 1.0

# # Calcola l'integrale
# integral, samples = integrate_metropolis(target_function, num_samples, proposal_width)

# # Visualizza i risultati
# print(f"Approssimazione dell'integrale: {integral}")

# # Istogramma dei campioni e la funzione target
# x = np.linspace(-4, 4, 500)