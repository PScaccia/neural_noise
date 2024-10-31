#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:54:57 2024

@author: paoloscaccia
"""
import numpy as np


####################################################################################
#                 STIM-INDEPENDENT NOISE W CELL 2  (GAUSSIAN)                      #
####################################################################################

CASE_1 = {      'function' : 'fvm',
                'A'        : 0.6936204110768727,
                'width'    : 0.2140327993142855,
                'flatness' : 0.6585904840291591,
                'b'        : 1.2731732385019432,
                'center'   : 2.9616211149125977 - 0.64,
                'V'        : 2500,
                'center_shift'   : np.pi/2,
                'rho'      : 0.1 }
    
CASE_2 = {  'function' : 'fvm',
            'A'        : 0.6936204110768727,
            'width'    : 0.2140327993142855,
            'flatness' : 0.6585904840291591,
            'b'        : 1.2731732385019432,
            'center'   : 2.9616211149125977 - 0.64,
            'V'        : 25,
            'center_shift'   : np.pi/2,
            'rho'            : 0.1 }

CASE_3 = {      'function' : 'fvm',
                'A'        : 0.6936204110768727,
                'width'    : 0.2140327993142855,
                'flatness' : 0.6585904840291591,
                'b'        : 1.2731732385019432,
                'center'   : 2.9616211149125977 - 0.64,
                'V'        : 2500,
                'center_shift'   : np.pi/2,
                'rho'            : 0.95 }

CASE_4 = {      'function' : 'fvm',
                'A'        : 0.6936204110768727,
                'width'    : 0.2140327993142855,
                'flatness' : 0.6585904840291591,
                'b'        : 1.2731732385019432,
                'center'   : 2.9616211149125977 - 0.64,
                'V'        : 25,
                'center_shift'   : np.pi/2,
                'rho'            : 0.95 }

####################################################################################
#                 STIM-INDEPENDENT NOISE W CELL 2  (POISSON)                       #
####################################################################################

CASE_P1 = {     'function' : 'fvm',
                'A'        : 0.6936204110768727,
                'width'    : 0.2140327993142855,
                'flatness' : 0.6585904840291591,
                'b'        : 1.2731732385019432,
                'center'   : 2.9616211149125977 - 0.64,
                'center_shift'  : np.pi/2,
                'rho'      : "poisson",
                "alpha"    : 0.1,
                "beta"     : 80}
    
CASE_P2 = { 'function' : 'fvm',
            'A'        : 0.6936204110768727,
            'width'    : 0.2140327993142855,
            'flatness' : 0.6585904840291591,
            'b'        : 1.2731732385019432,
            'center'   : 2.9616211149125977 - 0.64,
            'center_shift'   : np.pi/2,
            'rho'      : "poisson",
            "alpha"    : 0.1,
            "beta"     : 0.8}

CASE_P3 = {     'function' : 'fvm',
                'A'        : 0.6936204110768727,
                'width'    : 0.2140327993142855,
                'flatness' : 0.6585904840291591,
                'b'        : 1.2731732385019432,
                'center'   : 2.9616211149125977 - 0.64,
                'V'        : 2500,
                'center_shift'   : np.pi/2,
                'rho'      : "poisson",
                "alpha"    : 0.95,
                "beta"     : 80}

CASE_P4 = {     'function' : 'fvm',
                'A'        : 0.6936204110768727,
                'width'    : 0.2140327993142855,
                'flatness' : 0.6585904840291591,
                'b'        : 1.2731732385019432,
                'center'   : 2.9616211149125977 - 0.64,
                'V'        : 25,
                'center_shift'   : np.pi/2,
                'rho'      : "poisson",
                "alpha"    : 0.95,
                "beta"     : 0.8}

####################################################################################
#                 STIM. DEPENDENT NOISE                                            #
####################################################################################


CASE_A1 = {      'function' : 'fvm',
                'A'        : 0.6936204110768727,
                'width'    : 0.2140327993142855,
                'flatness' : 0.6585904840291591,
                'b'        : 1.2731732385019432,
                'center'   : 2.9616211149125977 - 0.64,
                'V'        : 2500,
                'center_shift'   : np.pi/2,
                'rho'      : "adaptive",
                "alpha"    : 2,
                "beta"     : 1}

    
CASE_A2 = {  'A'             : 4.185765887215448e-09,  # based on Carlo's fit
            'width'          : 0.0428942145079092,     # base on Carlo's fit
            'center'         : 1.7103939222623101,     # radiants, but base on Carlo's fit
            'function'       : 'fvm',
            'flatness'       : 0.768665715255412,      # based on Carlo's fit
            'b'              : 7.205646924554782,      # base on Carlo's fit
            'center_shift'   : np.pi/4,
            'V'              : 25,
            'rho'            : "adaptive",
            'alpha'          : 10,
            'beta'           : 1}

CASE_A3 = {  'A'              : 4.185765887215448e-09,  # based on Carlo's fit
            'width'          : 0.0428942145079092,     # base on Carlo's fit
            'center'         : 1.7103939222623101,     # radiants, but base on Carlo's fit
            'function'       : 'fvm',
            'flatness'       : 0.768665715255412,      # based on Carlo's fit
            'b'              : 7.205646924554782,      # base on Carlo's fit
            'center_shift'   : np.pi/4,
            'V'              : 2500,
            'rho'            : "adaptive",
            'alpha'          : 10,
            'beta'           : 1}

CASE_A4 = {  'A'              : 4.185765887215448e-09,  # based on Carlo's fit
            'width'          : 0.0428942145079092,     # base on Carlo's fit
            'center'         : 1.7103939222623101,     # radiants, but base on Carlo's fit
            'function'       : 'fvm',
            'flatness'       : 0.768665715255412,      # based on Carlo's fit
            'b'              : 7.205646924554782,      # base on Carlo's fit
            'center_shift'   : np.pi/4,
            'V'              : 25,
            'rho'            : "adaptive",
            'alpha'          : 10,
            'beta'           : 1}

####################################################################################
#               TEST TUNING CURVE                                           #
####################################################################################

CASE_TEST = {      'function' : 'fvm',
                'A'        : 0.6936204110768727,
                'width'    : 0.2140327993142855,
                'flatness' : 0.6585904840291591,
                'b'        : 1.2731732385019432,
                'center'   : 2.9616211149125977 - 0.64,
                'V'        : 25,
                'center_shift'   : np.pi/4,
                'rho'            : 0.95 }

####################################################################################
#                 `INDEPENDENT NOISE  W CELL 1                                     #
####################################################################################

CASE_C1_1 = {  'A'              : 4.185765887215448e-09,  # based on Carlo's fit
            'width'          : 0.0428942145079092,     # based on Carlo's fit
            'center'         : 1.7103939222623101,     # radiants, but base on Carlo's fit
            'function'       : 'fvm',
            'flatness'       : 0.768665715255412,      # based on Carlo's fit
            'b'              : 7.205646924554782,      # based on Carlo's fit
            'center_shift'   : np.pi/4,
            'V'              : 2500,
            'rho'            : 0.1 }
    
CASE_C1_2 = {  'A'              : 4.185765887215448e-09,  # based on Carlo's fit
            'width'          : 0.0428942145079092,     # based on Carlo's fit
            'center'         : 1.7103939222623101,     # radiants, but base on Carlo's fit
            'function'       : 'fvm',
            'flatness'       : 0.768665715255412,      # based on Carlo's fit
            'b'              : 7.205646924554782,      # based on Carlo's fit
            'center_shift'   : np.pi/4,
            'V'              : 25,
            'rho'            : 0.1 }

CASE_C1_3 = {  'A'              : 4.185765887215448e-09,  # based on Carlo's fit
            'width'          : 0.0428942145079092,     # based on Carlo's fit
            'center'         : 1.7103939222623101,     # radiants, but base on Carlo's fit
            'function'       : 'fvm',
            'flatness'       : 0.768665715255412,      # based on Carlo's fit
            'b'              : 7.205646924554782,      # based on Carlo's fit
            'center_shift'   : np.pi/4,
            'V'              : 2500,
            'rho'            : 0.97 }

CASE_C1_4 = {  'A'           : 4.185765887215448e-09,  # based on Carlo's fit
            'width'          : 0.0428942145079092,     # based on Carlo's fit
            'center'         : 1.7103939222623101,     # radiants, but base on Carlo's fit
            'function'       : 'fvm',
            'flatness'       : 0.768665715255412,      # based on Carlo's fit
            'b'              : 7.205646924554782,      # based on Carlo's fit
            'center_shift'   : np.pi/4,
            'V'              : 25,
            'rho'            : 0.92 }

CASES = { i+1 : c for i,c in zip(range(4),[CASE_1,CASE_2,CASE_3,CASE_4]) }
A_CASES = { i+1 : c for i,c in zip(range(4),[CASE_A1,CASE_A2,CASE_A3,CASE_A4]) }
P_CASES = { i+1 : c for i,c in zip(range(4),[CASE_P1,CASE_P2,CASE_P3,CASE_P4]) }