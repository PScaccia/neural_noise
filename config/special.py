#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 21:16:46 2025

@author: paolos
"""
import numpy as np
import pickle


config = {  
            'label'         : 'poisson_selected_shift_0p34',
            'rho'           : "poisson" ,
            # 'alpha'         : np.arange(0.0,0.44,0.02),
            'alpha'         : np.arange(0.04,1.0,0.04),
            'beta'          : np.array([ 1.5, 0.8]),
            'function'      : 'fvm',
            'A'             : 0.17340510276921817,
            'width'         : 0.2140327993142855,
            'flatness'      : 0.6585904840291591,
            'b'             : 1.2731732385019432,
            'center'        : 2.9616211149125977 - 0.21264734641020677,
            # 'center_shift'  : np.pi/4,
            'center_shift'  :  0.34128722,
            'N'             : 40000,
            'int_step'      : 360
            }
# delta_theta [0.        , 0.34128722, 0.6913254 , 1.04136358, 1.39140176,  1.74143994, 2.09147812, 2.4415163 , 2.79155447, 3.14159265]

# test1 : N = 1000, int_step = 100
# test2 : N = 10000, int_step = 100 x
# test3 : N = 1000, int_step = 360
# test4 : N = 1000, int_step = 620
# test5 : N = 1000, int_step = 100, N_theta_points = 360
# test6 : N = 1000, int_step = 360, N_theta_points = 360
# test7 : N = 10000, int_step = 360, N_theta_points = 180 x winner
# test8 : N = 10000, int_step = 620, N_theta_points = 180 x
# test9 : N = 30000, int_step = 360, N_theta_points = 180 x winner


with open("config/special.pkl","wb") as ofile:
    pickle.dump(config, ofile, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved config file config/special.pkl")
