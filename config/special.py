#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 21:16:46 2025

@author: paolos
"""
import numpy as np
import pickle


config = {  'label'         : 'poisson_det',
            'rho'           : "poisson_det",
            #'alpha'         : np.array([-0.75,-0.5,-0.25,-0.1, 0.05,0.15, 0.25, 0.4, 0.5,0.6, 0.7 , 0.8, 0.95]),
            # 'alpha'         : np.linspace(0,1,21)[1:4],
            'alpha'         : np.array([ 0.2, 0.95 ]),
            'beta'          : np.array([ 2.0 ]),
            'function'      : 'fvm',
            'A'             : 0.17340510276921817,
            'width'         : 0.2140327993142855,
            'flatness'      : 0.6585904840291591,
            'b'             : 1.2731732385019432,
            'center'        : 2.9616211149125977 - 0.21264734641020677,
            'center_shift'  : np.pi/4,
            'N'             : 200000,
            'int_step'      : 100
            }

with open("config/special.pkl","wb") as ofile:
    pickle.dump(config, ofile, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved config file config/special.pkl")