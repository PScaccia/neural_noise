#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 15:45:15 2025

@author: paolos
"""


from glob import glob
from utils.stat_tools import read_landscape_simulation
from plot_tools       import plot_landscape

if __name__ == '__main__':
    wrkdir = '/home/paolos/repo/neural_noise/data'

    file_list = glob(wrkdir+'/landscape/*.npz')
    file_list += glob(wrkdir+'/*.hdf5')    
    
    rho_s, delta_theta, rho_n, Imp = read_landscape_simulation( file_list, n_workers = 10)
    
    plot_landscape(rho_s, rho_n, Imp ,
                   dx=0.08, dy = 0.05,
                   interp='nearest',
                   xmax = 0.92, xmin = -0.84,
                   vmin = -1, vmax = 1)
    