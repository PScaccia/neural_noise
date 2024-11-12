#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:50:33 2024

@author: paoloscaccia
"""

from network import THETA, NeuralSystem
from decoder import *
from case import *
import numpy as np

def parser():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--decoder',type=str,required=False,choices=['bayesian','MAP'],default='bayesian')
    parser.add_argument('--noise','-n',type=str,required=False,default='constant',choices=['constant','adaptive','poisson'],help="System Configuration")
    parser.add_argument('-N','--n_trials',type=int,required=False,default=1000,help="Numerosity of the sampling")
    parser.add_argument('--n_points',type=int,required=False,default=100,help="Number of stimuli around")
    parser.add_argument('-dt','--dtheta',type=float,required=False,default=0.1,help="Number of stimuli around")
    parser.add_argument('--central_point',type=float,required=False,default=2.3291,help="Fixed Stimulus [rad]")
    parser.add_argument('-o','--outdir',type=str,required=False,default='./',help="Output Folder")
    return parser.parse_args()


if __name__ == '__main__':
    
    args = parser()
    
    system_A = NeuralSystem(  CASES[4], N_trial = args.N_trial )
    system_B = NeuralSystem(  CASES_TEST, N_trial = args.N_trial )
    
    
    theta_sampling_A = sample_theta_ext( system_A, THETA, decoder = DECODER )
    theta_sampling_A = sample_theta_ext( system_A, THETA, decoder = DECODER )

    theta_space = np.linspace( args.central_point - args.dtheta*args.n_points,  args.central_point + args.dtheta*args.n_points, args.n_points )
    
