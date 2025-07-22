#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:56:54 2025

@author: paoloscaccia
"""

import numpy as np

config = {  'label'         : 'poisson_45deg_sim_N10000_simpson',
            'rho'           : "poisson",
            #'alpha'         : np.array([-0.75,-0.5,-0.25,-0.1, 0.05,0.15, 0.25, 0.4, 0.5,0.6, 0.7 , 0.8, 0.95]),
            'alpha'         : np.array([0.05,0.15, 0.25, 0.4]),
            'beta'          : np.array([ 0.7, 0.4,0.1]),
            'function'      : 'fvm',
            'A'             : 1.609874980656409e-10,
            'width'         : 0.04008507948420414,
            'center'        : 4.814400368872799 - 2.5,
            'flatness'      : 0.8999356386649207,
            'b'             : 0.7487998754100026,
            'center_shift'  : np.pi/4,
            'N'             : 10000,
            'int_step'      : 100
            }


"""
config = {  'label'         : 'adaptive_det_intermediate_sim_shift45deg',
            'rho'           : "adaptive_det",
            'alpha'         : np.append( [0.01],np.arange(1,10)*0.1),
            # 'alpha'         : np.arange(8,10)*0.1,
            'beta'          : np.array([0.9]),
            # 'beta'          : np.array([0.9]),
            'function'      : 'fvm',
            'A'             : 1.609874980656409e-10,
            'width'         : 0.04008507948420414,
            'center'        : 4.814400368872799 - 2.5,
            'flatness'      : 0.8999356386649207,
            'b'             : 0.7487998754100026,
            'center_shift'  : np.pi/4,
            'N'             : 10000,
            'int_step'      : 360
            }
"""

