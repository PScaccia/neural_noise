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

CASE_5 = {      'function' : 'fvm',
                'A'        : 0.6936204110768727,
                'width'    : 0.2140327993142855,
                'flatness' : 0.6585904840291591,
                'b'        : 1.2731732385019432,
                'center'   : 2.9616211149125977 - 0.64,
                'V'        : 625,
                'center_shift'   : np.pi/2,
                'rho'            : 0.1 }

CASE_6 = {      'function' : 'fvm',
                'A'        : 0.6936204110768727,
                'width'    : 0.2140327993142855,
                'flatness' : 0.6585904840291591,
                'b'        : 1.2731732385019432,
                'center'   : 2.9616211149125977 - 0.64,
                'V'        : 1875,
                'center_shift'   : np.pi/2,
                'rho'            : 0.1 }

CASE_7 = {      'function' : 'fvm',
                'A'        : 0.6936204110768727,
                'width'    : 0.2140327993142855,
                'flatness' : 0.6585904840291591,
                'b'        : 1.2731732385019432,
                'center'   : 2.9616211149125977 - 0.64,
                'V'        : 300,
                'center_shift'   : np.pi/2,
                'rho'            : 0.1 }

CASE_8 = {      'function' : 'fvm',
                'A'        : 0.6936204110768727,
                'width'    : 0.2140327993142855,
                'flatness' : 0.6585904840291591,
                'b'        : 1.2731732385019432,
                'center'   : 2.9616211149125977 - 0.64,
                'V'        : 100,
                'center_shift'   : np.pi/2,
                'rho'            : 0.1 }

CASE_9 = {      'function' : 'fvm',
                'A'        : 0.6936204110768727,
                'width'    : 0.2140327993142855,
                'flatness' : 0.6585904840291591,
                'b'        : 1.2731732385019432,
                'center'   : 2.9616211149125977 - 0.64,
                'V'        : 250,
                'center_shift'   : np.pi/2,
                'rho'            : 0.1 }

CASE_10 = {      'function' : 'fvm',
                'A'        : 0.6936204110768727,
                'width'    : 0.2140327993142855,
                'flatness' : 0.6585904840291591,
                'b'        : 1.2731732385019432,
                'center'   : 2.9616211149125977 - 0.64,
                'V'        : 60,
                'center_shift'   : np.pi/2,
                'rho'            : 0.1 }

CASE_11 = {      'function' : 'fvm',
                'A'        : 0.6936204110768727,
                'width'    : 0.2140327993142855,
                'flatness' : 0.6585904840291591,
                'b'        : 1.2731732385019432,
                'center'   : 2.9616211149125977 - 0.64,
                'V'        : 175,
                'center_shift'   : np.pi/2,
                'rho'            : 0.1 }


CASE_12 = {      'function' : 'fvm',
                'A'        : 0.6936204110768727,
                'width'    : 0.2140327993142855,
                'flatness' : 0.6585904840291591,
                'b'        : 1.2731732385019432,
                'center'   : 2.9616211149125977 - 0.64,
                'V'        : 170,
                'center_shift'   : np.pi/2,
                'rho'            : 0.1 }

CASE_13 = {      'function' : 'fvm',
                'A'        : 0.6936204110768727,
                'width'    : 0.2140327993142855,
                'flatness' : 0.6585904840291591,
                'b'        : 1.2731732385019432,
                'center'   : 2.9616211149125977 - 0.64,
                'V'        : 90,
                'center_shift'   : np.pi/2,
                'rho'            : 0.95 }

CASE_14 = {      'function' : 'fvm',
                'A'        : 0.6936204110768727,
                'width'    : 0.2140327993142855,
                'flatness' : 0.6585904840291591,
                'b'        : 1.2731732385019432,
                'center'   : 2.9616211149125977 - 0.64,
                'V'        : 97,
                'center_shift'   : np.pi/2,
                'rho'            : 0.95 }

CASE_15 = {      'function' : 'fvm',
                'A'        : 0.6936204110768727,
                'width'    : 0.2140327993142855,
                'flatness' : 0.6585904840291591,
                'b'        : 1.2731732385019432,
                'center'   : 2.9616211149125977 - 0.64,
                'V'        : 110,
                'center_shift'   : np.pi/2,
                'rho'            : 0.95 }


####################################################################################
#                 STIM-INDEPENDENT NOISE W CELL 2  (POISSON)                       #
####################################################################################

CASE_P1 = {     'function' : 'fvm',
                'A': 1.609874980656409e-10,
                 'width': 0.04008507948420414,
                 'center': 4.814400368872799 - 2.5,
                 'flatness': 0.8999356386649207,
                 'b': 0.7487998754100026,
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

CASE_TEST_2 = { 'function' : 'fvm',
                'A'        : 0.6936204110768727,
                'width'    : 0.2140327993142855,
                'flatness' : 0.6585904840291591,
                'b'        : 1.2731732385019432,
                'center'   : 2.9616211149125977 - 0.64,
                'V'        : 1250,
                'center_shift'   : np.pi/2,
                'rho'      : 0.1 }


####################################################################################
#                 `INDEPENDENT NOISE  W CELL 1                                     #
####################################################################################

CASE_C1_1 = {  'A'           : 4.185765887215448e-09,  # based on Carlo's fit
            'width'          : 0.0428942145079092,     # based on Carlo's fit
            'center'         : 1.7103939222623101,     # radiants, but base on Carlo's fit
            'function'       : 'fvm',
            'flatness'       : 0.768665715255412,      # based on Carlo's fit
            'b'              : 7.205646924554782,      # based on Carlo's fit
            'center_shift'   : np.pi/4,
            'V'              : 2500,
            'rho'            : 0.1 }
    
CASE_C1_2 = {  'A'           : 4.185765887215448e-09,  # based on Carlo's fit
            'width'          : 0.0428942145079092,     # based on Carlo's fit
            'center'         : 1.7103939222623101,     # radiants, but base on Carlo's fit
            'function'       : 'fvm',
            'flatness'       : 0.768665715255412,      # based on Carlo's fit
            'b'              : 7.205646924554782,      # based on Carlo's fit
            'center_shift'   : np.pi/4,
            'V'              : 25,
            'rho'            : 0.1 }

CASE_C1_3 = {  'A'           : 4.185765887215448e-09,  # based on Carlo's fit
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

CASES = { i+1 : c for i,c in zip(range(15),[CASE_1,CASE_2,CASE_3,CASE_4, CASE_5,CASE_6,CASE_7, CASE_8, CASE_9, CASE_10,CASE_11, CASE_12, CASE_13,CASE_14, CASE_15]) }
A_CASES = { i+1 : c for i,c in zip(range(4),[CASE_A1,CASE_A2,CASE_A3,CASE_A4]) }
P_CASES = { i+1 : c for i,c in zip(range(4),[CASE_P1,CASE_P2,CASE_P3,CASE_P4]) }