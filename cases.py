#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:54:57 2024

@author: paoloscaccia
"""
import numpy as np

CASE_1 = {  'A'              : 5,
            'width'          : .2,
            'center'         : 2.8,
            'function'       : 'fvm',
            'flatness'       : 0.1,
            'b'              : 0.0,
            'center_shift'   : np.pi/2,
            'V'              : 5e-1,
            'rho'            : 0.01 }

CASE_2 = {  'A'              : 5,
            'width'          : .2,
            'center'         : 2.8,
            'function'       : 'fvm',
            'flatness'       : 0.1,
            'b'              : 0.0,
            'center_shift'   : np.pi/2,
            'V'              : 5e-3,
            'rho'            : 0.01 }

CASE_3 = {  'A'              : 5,
            'width'          : .2,
            'center'         : 2.8,
            'function'       : 'fvm',
            'flatness'       : 0.1,
            'b'              : 0.0,
            'center_shift'   : np.pi/2,
            'V'              : 5e-1,
            'rho'            : 0.8 }

CASE_4 = {  'A'              : 5,
            'width'          : .2,
            'center'         : 2.8,
            'function'       : 'fvm',
            'flatness'       : 0.1,
            'b'              : 0.0,
            'center_shift'   : np.pi/2,
            'V'              : 5e-3,
            'rho'            : 0.8 }