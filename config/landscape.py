#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 21:16:46 2025

@author: paolos
"""
import numpy as np
import pickle


config = {  
            'label'         : 'landscape',
            'rho'           : "poisson" ,
            'alpha'         : np.arange(0.02,1.0,0.02),
            # 'alpha'         : np.arange(0.1,0.86,0.02),
            'beta'          :  1,
            'function'      : 'fvm',
            'A'             : 0.17340510276921817,
            'width'         : 0.2140327993142855,
            'flatness'      : 0.6585904840291591,
            'b'             : 1.2731732385019432,
            'center'        : 2.9616211149125977 - 0.21264734641020677,
            'center_shift'  : np.array([0.        , 0.34128722, 0.6913254 , 1.04136358, 1.39140176,  1.74143994, 2.09147812, 2.4415163 , 2.79155447, 3.14159265]),
            'N'             : 30000,
            'int_step'      : 360
            }
# delta_theta [0.        , 0.34128722, 0.6913254 , 1.04136358, 1.39140176,  1.74143994, 2.09147812, 2.4415163 , 2.79155447, 3.14159265]


with open("config/landscape.pkl","wb") as ofile:
    pickle.dump(config, ofile, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved config file config/landscape.pkl")
