#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 13:51:50 2026

@author: paolo
"""

from glob import glob
import h5py, os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize


def trova_intersezioni_x(x, y):
    """
    Trova le intersezioni con l'asse X (y=0)
    usando interpolazione lineare tra punti consecutivi.
    
    Parametri:
        x : array-like
        y : array-like
        
    Ritorna:
        lista delle x dove la curva interseca y=0
    """
    x = np.array(x)
    y = np.array(y)
    
    intersezioni = []
    
    for i in range(len(y) - 1):
        # Controllo cambio di segno
        if y[i] == 0:
            intersezioni.append(x[i])
        elif y[i] * y[i+1] < 0:
            # Interpolazione lineare
            x0, x1 = x[i], x[i+1]
            y0, y1 = y[i], y[i+1]
            
            x_intersezione = x0 - y0 * (x1 - x0) / (y1 - y0)
            intersezioni.append(x_intersezione)
    
    return intersezioni

WRKDIR = '/home/paolo/data/'
WRKDIR = '/media/paolo/Tweedledee/'
filemask  = '/bayesian_extimate_beta*.h5'
tag      = ''

files = glob(WRKDIR + filemask)
# filefilter = lambda x: tag in x
filefilter = lambda x : x.count('_') <= 2

files = [f for f in files if filefilter(f) ]
files = ['bayesian_extimate_beta0p5.h5',
              'bayesian_extimate_beta0p6.h5',
              'bayesian_extimate_beta0p7.h5',
              'bayesian_extimate_beta0p8.h5',
              'bayesian_extimate_beta1.h5',                     
              'bayesian_extimate_beta1p2.h5',
              'bayesian_extimate_beta1p5.h5',
              'bayesian_extimate_beta2.h5',
              'bayesian_extimate_beta3.h5']
files = [ WRKDIR + '/' + f for f in files ]
files.append( '/home/paolo/data/bayesian_extimate_beta0p4.h5' )
files.append( '/home/paolo/data/bayesian_extimate_beta0p3.h5' )
files = [ f for f in files if os.path.isfile(f)]

# -------------------------
# Estrazione beta numerico
# -------------------------
beta_file_list = []

for file in files:
    beta = os.path.basename(file).split('_beta')[1].split('_')[0]
    beta = beta.replace('.h5','').replace('p','.')
    beta = float(beta)

    # if beta not in [1.0, 3.0, 5, 8, 12, 20]:
    #     continue

    beta_file_list.append((beta, file))

# -------------------------
# Ordinamento per beta
# -------------------------
beta_file_list.sort(key=lambda x: x[0])

# Riassegno
betas  = np.array([x[0] for x in beta_file_list])
files  = [x[1] for x in beta_file_list]

# -------------------------
# Colormap grey scale
# -------------------------
norm = Normalize(vmin=betas.min(), vmax=betas.max())
cmap = cm.winter

rho_stars = []
SNR       = []


x_points = []
y_points = []
beta_values = []

for beta, file in zip(betas, files):

    SNR.append(1 / beta)

    with h5py.File(file, 'r') as handle:
        nc = handle['noise_correlation'][:]
        error     = handle['cosin_error'][:]
        ind_error = handle['ind_cosin_error'][:]

    I = (1 - (error.mean(axis=-2) / ind_error.mean(axis=-2))) * 100
    I = I.mean(axis=-1).flatten()

    x = nc
    y = savgol_filter(I[::-1], 5, 3)

    zeri = trova_intersezioni_x(x, y)
    zero = np.max(zeri) if len(zeri) > 0 else np.nan

    # Colore funzione di beta
    color = cmap(norm(beta))

    plt.plot(x, y, label=f'{int(beta)}', color=color, lw = 6)
    # plt.scatter(zero, 0, marker='X', color=color)

    rho_stars.append(zero)
    
    x_points.append(x)
    y_points.append(y)
    beta_values.append( beta )
    

rho_stars = np.array(rho_stars)
SNR = np.array(SNR)

plt.xlim(0, 1)
plt.ylim(-5, 20)
plt.axhline(0, c='black')
plt.legend(title='beta')
plt.show()

# plt.scatter(SNR, rho_stars, s = 36)
# plt.xlabel('SNR', size = 15)
# plt.ylabel(r'$\text{Beneficial threshold } \rho^{*}$', size = 15)
# plt.xlim(0,None)
# plt.axhline(0,c='black', lw = .5)
# plt.show()



import pickle
plt.clf()
with open('/home/paolo/data/improvement_decoding_paper.pkl', 'rb') as handle:
    d = pickle.load(handle)
    print(d['beta'], 6.61/d['beta'])
    beta_values = d['beta']
    x_points = d['nc']
    y_points = d['improvement']
    del(d)
    
norm = Normalize(vmin=0.33, vmax=2.2)
cmap = cm.PiYG
cmap_discrete = cmap( np.linspace(0, 1, 12))

for i,beta in enumerate(beta_values):
    if beta == 0.4: continue
    label = 6.61/beta
    # c = cmap(norm(beta))
    c = cmap_discrete[i]
    plt.plot( x_points[i], y_points[i], label = int(label), color = c, lw = 6)
    
plt.xlim(0,1)
plt.ylim(-8, 20)
legend = plt.legend(title = 'SNR' , prop = {'size' : 15}, frameon = False,  bbox_to_anchor=(1.05, 1.05))
legend.get_title().set_fontsize('18')
plt.xlabel('Noise Correlation', size = 15, **{'fontname':'Arial'})
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
plt.xticks(fontsize=14)
ax.tick_params(width=1.2, size = 5)
plt.yticks(fontsize=14)
plt.axhline(0, c='black', lw = 1)
OUTFIG = "/home/paolo/Pictures/paper/figure2_panelC.pdf"
plt.savefig(OUTFIG, dpi = 300, bbox_inches = 'tight')
print("Saved plot ", OUTFIG)


# plt.show()


