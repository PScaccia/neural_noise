#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 13:17:30 2025

@author: paolos
"""
import sys
sys.path.append("/home/paolos/repo/neural_noise/")
from   network        import NeuralSystem, THETA
from   plot_tools     import plot_simulation_single_panel
from   config.default import config
from   decoder        import sample_theta_ext, compute_MSE, bayesian_decoder, MAP_decoder
import copy
import numpy          as np
import matplotlib.pyplot as plt

case = copy.deepcopy(config)

case['rho']  = 'adaptive_det'
case_label   =  "Det."
color        =  "tab:green"


# case['rho']  = 'adaptive'
# case_label   =  "Trace"
# color        =  "tab:orange"


case['function'] = 'circle'
case['center']       = 0
case['center_shift'] = -np.pi/2
case['b']            = 2
case['A']            = 7.5
N_trials = 10000

alphas = np.append(np.append( [0.01],np.arange(1,10)*0.1), [0.99])
alphas = alphas

improvement = np.empty_like(alphas)
all_MSE         = np.empty_like(improvement)
all_MSE_control = np.empty_like(improvement)
target_angle    = 3*np.pi/4

case['beta']  = 0.01
decoder = 'bayesian'
decoder = 'MAP'
parallel = False

"""
case['beta']  = 0.05
case['alpha'] = 0.9

system = NeuralSystem( case, N_trial = N_trials )
r_sampling = system.neurons(np.pi)
f = lambda r : bayesian_decoder(system,r,np.pi,N_step=400)[::5]
theta_sampling = np.array(list(map(f,r_sampling)))

system.alpha = 0.0
system.generate_variance_matrix()
r_sampling_ind = system.neurons(np.pi)

f2 = lambda r : bayesian_decoder(system,r005,np.pi,N_step=400)
theta_sampling_control = np2.array(list(map(f2,r_sampling_ind)))
"""

for i, alpha in enumerate( alphas ):    
    case['alpha'] = alpha
    print("#"*20)
    print('rho: ',alpha)
    system = NeuralSystem( case, N_trial = N_trials )
    print("Decoding Correlated system...")
    theta_sampling = sample_theta_ext( system, np.array([target_angle]), decoder = decoder,  multi_thread=parallel, num_threads=20)
    print("Decoding Independent system...")
    system.alpha = 0.0
    system.generate_variance_matrix()
    theta_sampling_control = sample_theta_ext( system, np.array([target_angle]), decoder = decoder, multi_thread=parallel, num_threads=20)
    print()
    
    MSE = compute_MSE( theta_sampling, theta = np.array([target_angle]))
    MSE_control = compute_MSE( theta_sampling_control, theta = np.array([target_angle]))

    improvement[i] = ( 1 - MSE/MSE_control)*100

    all_MSE[i] = MSE
    all_MSE_control[i] = MSE_control

plt.plot(alphas, (1-all_MSE/all_MSE_control)*100, marker = 'o')
plt.axhline()
plt.title(f"NOISE: {system.beta} COSTANT: {case_label}",weight='bold')
plt.show()

system = NeuralSystem( case, N_trial = N_trials )
system.alpha = 0.99
system.generate_variance_matrix()
r_sampling = system.neurons(target_angle)
if decoder == 'bayesian':
    f = lambda r : bayesian_decoder(system,r,None,N_step=1000)
else:
    f = lambda r : MAP_decoder(system,r)

theta_sampling = np.array(list(map(f,r_sampling)))

system.alpha = 0.0
system.generate_variance_matrix()
r_sampling_ind = system.neurons(target_angle)
if decoder == 'bayesian':
    f2 = lambda r : bayesian_decoder(system,r,None,N_step=1000)
else:
    f2 = lambda r : MAP_decoder(system,r)

theta_sampling_control = np.array(list(map(f2,r_sampling_ind)))

def ang_dist(a,b):
    return min( np.abs(a-b), abs(2*np.pi - np.abs(a-b)))
d = lambda x : ang_dist(x, target_angle)
diff=np.array(list(map(d, theta_sampling.flatten())))
diff_control=np.array(list(map(d, theta_sampling_control.flatten())))

vmax = 0.08
vmin = 0
size = 1
delta = system.beta*150

plt.plot( system.mu[0](THETA), system.mu[1](THETA), c='black')
plt.scatter( r_sampling[:,0], r_sampling[:,1], c = diff, cmap = 'jet',vmin=vmin, vmax=vmax, s = size)
plt.colorbar()
plt.xlim(system.mu[0](target_angle) - delta, system.mu[0](target_angle) + delta)
plt.ylim(system.mu[1](target_angle) - delta, system.mu[1](target_angle) + delta)
plt.show()


plt.plot( system.mu[0](THETA), system.mu[1](THETA), c='black')
plt.scatter( r_sampling_ind[:,0], r_sampling_ind[:,1], c = diff_control, cmap = 'jet',vmin=vmin,vmax=vmax,s=size)
plt.colorbar()
plt.xlim(system.mu[0](target_angle) - delta, system.mu[0](target_angle) + delta)
plt.ylim(system.mu[1](target_angle) - delta, system.mu[1](target_angle) + delta)
plt.show()


bins = np.linspace(0, vmax*3, 25)
plt.hist( diff_control , weights = np.ones_like(diff_control)/diff_control.size, bins = bins, label = 'Indep.', zorder = 1)
plt.hist( diff , weights = np.ones_like(diff_control)/diff_control.size, color = color,bins = bins, alpha = 0.6, label = case_label, zorder = 2 )
plt.legend(prop={'size':20})
plt.xlabel('Angular Difference [rad]', weight='bold', size = 16)
plt.show()