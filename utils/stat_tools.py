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

# # Parametri
# num_samples = 10000
# proposal_width = 1.0

# # Calcola l'integrale
# integral, samples = integrate_metropolis(target_function, num_samples, proposal_width)

# # Visualizza i risultati
# print(f"Approssimazione dell'integrale: {integral}")

# # Istogramma dei campioni e la funzione target
# x = np.linspace(-4, 4, 500)