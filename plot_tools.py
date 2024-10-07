#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:10:10 2024

@author: paoloscaccia
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_simulation( system, stimolus ):
    
    markers = ['o','+','s','^','*','v']
        
    
    # Plot Trials per Stimolous
    for i,t in enumerate(stimolus):
        data = system.neurons(t)
        j = i%len(markers)
        plt.scatter(data[:,0],data[:,1],
                    alpha=0.5,
                    marker=markers[ j ],
                    label = r"{:.2f}".format(  t/np.pi ) 
                    )
        
        
    # Plot Signal Manifold
    theta = np.linspace(stimolus[0], stimolus[-1] + np.pi,1000)
    plt.plot( list( map( system.mu[0], theta)), 
                 list( map( system.mu[1], theta)), 
                 c = 'black',
                 lw = 1 )
    
    plt.xlabel('Response 1', weight='bold',size=18)
    plt.ylabel('Response 2', weight='bold',size=18)
    plt.legend( title = r'Theta ($\pi$)')
    
    plt.show()
    

    
    return 