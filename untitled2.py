#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:27:23 2026

@author: paolo
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
out = '/home/paolo/data/bayesian_extimate_beta0p1_info_limiting.h5'
THETA = np.linspace(0,2*np.pi,45)

with h5py.File(out, 'r') as handle:
    nc = handle['noise_correlation'][:]
    error     = handle['cosin_error'][:]
    ind_error = handle['ind_cosin_error'][:]
    
error.shape
I = (1 - (   error.mean(axis=-2)/ind_error.mean(axis=-2)))*100
I = I.mean(axis=-1).flatten()
    
out = '/home/paolo/data/bayesian_extimate_beta0p1_info_limiting.h5'
with h5py.File(out, 'r') as handle:
    nc = handle['noise_correlation'][:]
    error     = handle['cosin_error'][:]
    ind_error = handle['ind_cosin_error'][:]
    
I2 = (1 - (   error.mean(axis=-2)/ind_error.mean(axis=-2)))*100
I2 = I2.mean(axis=-1).flatten()
out = '/home/paolo/data/bayesian_extimate_beta0p5_info_limiting.h5'
with h5py.File(out, 'r') as handle:
    nc = handle['noise_correlation'][:]
    error     = handle['cosin_error'][:]
    ind_error = handle['ind_cosin_error'][:]
    
I3 = (1 - (   error.mean(axis=-2)/ind_error.mean(axis=-2)))*100
I3 = I3.mean(axis=-1).flatten()
out = '/home/paolo/data/bayesian_extimate_beta0p2_info_limiting.h5'
with h5py.File(out, 'r') as handle:
    nc = handle['noise_correlation'][:]
    error     = handle['cosin_error'][:]
    ind_error = handle['ind_cosin_error'][:]
    
I4 = (1 - (   error.mean(axis=-2)/ind_error.mean(axis=-2)))*100
I4 = I4.mean(axis=-1).flatten()
out = '/home/paolo/data/bayesian_extimate_beta0p05_info_limiting.h5'
with h5py.File(out, 'r') as handle:
    nc = handle['noise_correlation'][:]
    error     = handle['cosin_error'][:]
    ind_error = handle['ind_cosin_error'][:]
    
I5 = (1 - (   error.mean(axis=-2)/ind_error.mean(axis=-2)))*100
I5 = I5.mean(axis=-1).flatten()
out = '/home/paolo/data/bayesian_extimate_beta0p02_info_limiting.h5'
with h5py.File(out, 'r') as handle:
    nc = handle['noise_correlation'][:]
    error     = handle['cosin_error'][:]
    ind_error = handle['ind_cosin_error'][:]
    
I6 = (1 - (   error.mean(axis=-2)/ind_error.mean(axis=-2)))*100
I6 = I6.mean(axis=-1).flatten()
plt.plot(nc,I)
plt.plot(nc,I2)
plt.plot(nc,I3)
plt.plot(nc,I4)
plt.plot(nc,I5)
plt.plot(nc,I6)
plt.axhline();plt.show()
improvement_list = [I,I2,I3,I4,I5,I6]