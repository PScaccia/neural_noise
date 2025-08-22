#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 21:16:46 2025

@author: paolos
"""
import numpy as np
import pickle


config = {  'label'         : 'adaptive_selected',
            'rho'           : "adaptive",
            #'alpha'         : np.array([-0.75,-0.5,-0.25,-0.1, 0.05,0.15, 0.25, 0.4, 0.5,0.6, 0.7 , 0.8, 0.95]),
            # 'alpha'         : np.array([ 0.15, 0.35, 0.55, 0.75, 0.95, 0.99]),
            'alpha'         : np.arange(0.14,1.0,0.04),
            'beta'          : np.array([  0.2, 0.1 ]),
            'function'      : 'fvm',
            'A'             : 0.17340510276921817,
            'width'         : 0.2140327993142855,
            'flatness'      : 0.6585904840291591,
            'b'             : 1.2731732385019432,
            'center'        : 2.9616211149125977 - 0.21264734641020677,
            'center_shift'  : np.pi/4,
            'N'             : 10000,
            'int_step'      : 100
            }
pkl_name = __file__.replace('.py','.pkl')
with open(pkl_name,"wb") as ofile:
    pickle.dump(config, ofile, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved config file "+pkl_name)