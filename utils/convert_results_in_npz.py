#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:01:45 2025

@author: paoloscaccia
"""

import numpy as np
from glob import glob
import sys
from run_decoding_simulation import save_results

DIRECTORIES =[ '/Users/paoloscaccia/Pictures/decoding/lab_bkp/POISSON_case13',
               '/Users/paoloscaccia/Pictures/decoding/lab_bkp/POISSON_case12',
               '/Users/paoloscaccia/Pictures/decoding/lab_bkp/POISSON_case9',
               '/Users/paoloscaccia/Pictures/decoding/lab_bkp/POISSON_case7',
               '/Users/paoloscaccia/Pictures/decoding/lab_bkp/POISSON_case1',
               '/Users/paoloscaccia/Pictures/decoding/lab_bkp/POISSON_case6',
               '/Users/paoloscaccia/Pictures/decoding/lab_bkp/POISSON_case8',
               '/Users/paoloscaccia/Pictures/decoding/lab_bkp/POISSON_case10',
               '/Users/paoloscaccia/Pictures/decoding/lab_bkp/POISSON_case11',
               '/Users/paoloscaccia/Pictures/decoding/lab_bkp/POISSON_case3',
               '/Users/paoloscaccia/Pictures/decoding/lab_bkp/POISSON_case4',
               '/Users/paoloscaccia/Pictures/decoding/lab_bkp/POISSON_case5',
               '/Users/paoloscaccia/Pictures/decoding/lab_bkp/POISSON_case2',
               '/Users/paoloscaccia/Pictures/decoding/POISSON_case7']


N = 10000
results = {}
LABEL   = 'simulation_POISSON'

for d in DIRECTORIES:
    files = glob( d + '/theta_sampling*.txt')
    print("Reading data in ",d)
    
    if len(files) < 2:
        continue
    if len(files) > 2:
        sys.exit(f"More than one file found in {d}")

    theta_file_control = [ x for x in files if 'control' in x][0]
    theta_file = [x for x in files if x not in theta_file_control][0]
    
    with open(theta_file) as f:
        label = "_".join(f.readlines()[8:10]).replace('\n','').replace('#','').replace(' ','').replace(':','_').replace('alpha','a').replace('beta','b')

    print(label)
    if label in results:
        print(f"Label {label} lready in results")
        continue
    
    corr_data =  np.loadtxt(theta_file)    
    if corr_data.shape[-1] != N:
        corr_data = corr_data.repeat(N//corr_data.shape[-1],axis=1)

    ind_data  =  np.loadtxt(theta_file_control)
    if ind_data.shape[-1] != N:
        ind_data = ind_data.repeat(N//ind_data.shape[-1],axis=1)

    results[label] = [ corr_data,
                       ind_data  ]
    print(label,
          results[label][0].shape,
          results[label][1].shape)
    
save_results(results, "./data/"+LABEL+'.npz')
