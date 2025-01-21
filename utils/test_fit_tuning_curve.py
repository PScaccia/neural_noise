#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:02:30 2024

@author: paolos
"""

from scipy.optimize import curve_fit

# Funzione di fit: combinazione lineare di seno e coseno
def model(x, a, b, c, d, e):
    return a * np.cos(b * x + c) + d * np.sin(b * x + e)

# Dati di esempio (creiamo dei dati con rumore)
np.random.seed(0)
x_data = np.linspace(0, 10, 100)
y_data = 3 * np.cos(2 * x_data + 0.5) + 2 * np.sin(2 * x_data - 0.3) + 0.5 * np.random.normal(size=len(x_data))

# Stime iniziali per i parametri
initial_guess = [1, 1, 0, 1, 0]

# Fitting dei dati
params, params_covariance = curve_fit(model, x_data, y_data, p0=initial_guess)