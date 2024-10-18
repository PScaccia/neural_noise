#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:54:57 2024

@author: paoloscaccia
"""
import numpy as np

CASE_1 = {  'A'              : 4.185765887215448e-09,  # based on Carlo's fit
            'width'          : 0.0428942145079092,     # base on Carlo's fit
            'center'         : 1.7103939222623101,     # radiants, but base on Carlo's fit
            'function'       : 'fvm',
            'flatness'       : 0.768665715255412,      # based on Carlo's fit
            'b'              : 7.205646924554782,      # base on Carlo's fit
            'center_shift'   : np.pi/4,
            'V'              : 6400,
            'rho'            : 0.15 }

CASE_2 = {  'A'              : 4.185765887215448e-09,  # based on Carlo's fit
            'width'          : 0.0428942145079092,     # base on Carlo's fit
            'center'         : 1.7103939222623101,     # radiants, but base on Carlo's fit
            'function'       : 'fvm',
            'flatness'       : 0.768665715255412,      # based on Carlo's fit
            'b'              : 7.205646924554782,      # base on Carlo's fit
            'center_shift'   : np.pi/4,
            'V'              : 10,
            'rho'            : 0.15 }

CASE_3 = {  'A'              : 4.185765887215448e-09,  # based on Carlo's fit
            'width'          : 0.0428942145079092,     # base on Carlo's fit
            'center'         : 1.7103939222623101,     # radiants, but base on Carlo's fit
            'function'       : 'fvm',
            'flatness'       : 0.768665715255412,      # based on Carlo's fit
            'b'              : 7.205646924554782,      # base on Carlo's fit
            'center_shift'   : np.pi/4,
            'V'              : 6400,
            'rho'            : 0.96 }

CASE_4 = {  'A'              : 4.185765887215448e-09,  # based on Carlo's fit
            'width'          : 0.0428942145079092,     # base on Carlo's fit
            'center'         : 1.7103939222623101,     # radiants, but base on Carlo's fit
            'function'       : 'fvm',
            'flatness'       : 0.768665715255412,      # based on Carlo's fit
            'b'              : 7.205646924554782,      # base on Carlo's fit
            'center_shift'   : np.pi/4,
            'V'              : 10,
            'rho'            : 0.96 }

CASES = { i+1 : c for i,c in zip(range(4),[CASE_1,CASE_2,CASE_3,CASE_4]) }