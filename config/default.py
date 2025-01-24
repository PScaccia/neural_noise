#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:56:54 2025

@author: paoloscaccia
"""

import numpy as np


config = {  'label'         : 'default_sim',
            'rho'           : "adaptive",
            'alpha'         : np.array([0.1,0.01,0.1]),
            'beta'          : np.arange(2,9,1)*0.1,
            'function'      : 'fvm',
            'A'             : 1.609874980656409e-10,
            'width'         : 0.04008507948420414,
            'center'        : 4.814400368872799 - 2.5,
            'flatness'      : 0.8999356386649207,
            'b'             : 0.7487998754100026,
            'center_shift'  : np.pi/2,
            'N'             : 10,
            'int_step'      : 100
            }



