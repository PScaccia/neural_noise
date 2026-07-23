import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import quad
import sys

def plot_and_compute_intersection(t, a, b, mode, outdir = None, tag = ''):
    
    # ---------------------------
    # Curva
    # ---------------------------
    def f(x, t ):
        return t*x**2 + 0.3*x
    
    def df_dx(x, t ):
        return 2*t*x + 0.3
    
    # ---------------------------
    # Centro ellisse sulla curva
    # ---------------------------
    
    x0 = 0.3
    xc = 0
    yc = 0
    
    # ---------------------------
    # Parametri ellisse
    # ---------------------------
    theta = np.deg2rad(30)
    ct, st = np.cos(theta), np.sin(theta)
    
    # ---------------------------
    # Equazione ellisse ruotata
    # ---------------------------
    def ellipse_eq(x, y):
        xr =  ct*(x-xc) + st*(y-yc)
        yr = -st*(x-xc) + ct*(y-yc)
        return (xr/a)**2 + (yr/b)**2 - 1
    
    # ---------------------------
    # Intersezioni curva-ellisse
    # ---------------------------
    intersection_eq = lambda x:  ellipse_eq(x, f(x,t))
    
    x1 = fsolve(intersection_eq, -2 )[0]
    x2 = fsolve(intersection_eq,  2 )[0]
    
    # ---------------------------
    # Lunghezza arco
    # ---------------------------
    def arc_integrand(x,t):
        return np.sqrt(1 + df_dx(x,t)**2)
    
    integrand = lambda x : arc_integrand(x, t)
    L, _ = quad(integrand, x1, x2)

    if mode == 'Trace':
        R = np.sqrt( (a**2 + b**2)/2 )
    elif mode == 'Det':
        R = np.sqrt(a*b)

    print(R)
    def circle_eq(x, y):
        xr =  ct*(x-xc) + st*(y-yc)
        yr = -st*(x-xc) + ct*(y-yc)
        return (xr/R)**2 + (yr/R)**2 - 1

    circle_intersection_eq = lambda x:  circle_eq(x, f(x,t))
    
    circle_x1 = fsolve(circle_intersection_eq, -2 )[0]
    circle_x2 = fsolve(circle_intersection_eq,  2 )[0]
    
    circle_L, _ = quad(integrand, circle_x1, circle_x2)

    
    # print(f"Punto centro ellisse sulla curva: ({xc:.3f}, {yc:.3f})")
    # print(f"x1 = {x1:.4f}, x2 = {x2:.4f}")
    # print(f"Lunghezza arco = {L:.4f}")
    
    
    
    
    # ---------------------------
    # GRAFICA
    # ---------------------------
    x_plot = np.linspace(x1 - 1, x2 + 1, 600)
    y_plot = f(x_plot, t)
    
    tfixed = t*1.0
    t = np.linspace(0, 2*np.pi, 600)
    
    xe = xc + a*np.cos(t)*ct - b*np.sin(t)*st
    ye = yc + a*np.cos(t)*st + b*np.sin(t)*ct
    
    xind = xc + R*np.cos(t)*ct - R*np.sin(t)*st
    yind = yc + R*np.cos(t)*st + R*np.sin(t)*ct
    
    mask = ellipse_eq(x_plot, y_plot) <= 0
    
    fig, axs = plt.subplots(1,2, figsize=(12,5))
    axs[0].plot(xe, ye, 'darkolivegreen', label='Ellisse')
    axs[0].plot(xind, yind, 'goldenrod', ls = '--', label='Ellisse')

    axs[0].plot(x_plot, y_plot, '-', color = 'grey', alpha=0.6, label='Curva', lw = 1)
    axs[0].plot(x_plot[mask], y_plot[mask], 'r', linewidth=2, label='Arco interno')
    axs[0].scatter([xc], [yc], color='green', s=80, label='Centro ellisse', lw = 1)
    axs[0].scatter([x1, x2], [f(x1,tfixed), f(x2,tfixed)], color='red')
    axs[0].scatter([circle_x1, circle_x2], [f(circle_x1,tfixed), f(circle_x2,tfixed)], color='red', marker = 'x')

    axs[0].axis('equal')
    axs[0].grid(True)
    # axs[0].legend()

    axs[0].set_ylim(-3,3)
    axs[0].set_xlim(-3,3)
    
    axs[1].bar(['Correlated'], L, color = 'darkolivegreen', width = 0.5)
    axs[1].bar(['Independent'], circle_L, color = 'goldenrod',width = 0.5)
    axs[1].set_ylim(0, 5)

    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].set_ylabel('Arc Length',size = 13)
    axs[1].set_xlabel(' ',size = 13)
    
    plt.sca(axs[0])
    axs[0].set_xlim(-3,3)
    if outdir is None:
        plt.show()
    else:
        outfile = outdir + f"/image{tag}.png"
        plt.savefig(outfile, dpi=100,bbox_inches = 'tight')
    return L, circle_L

t_support = np.linspace(0,1.5,100)
ellipses_lengths = np.zeros_like(t_support)
circle_lengths   = np.zeros_like(t_support)


mode = 'Det'
print(f'Mode: {mode} constant')


Rsquared = 1
rho      = 0.5
a = np.sqrt( Rsquared*(1+rho))

if mode == 'Trace':
    b = np.sqrt( (2*Rsquared) - a*a )
elif mode == 'Det':
    b = 1/a
else: 
    sys.exit(f'Unknown mode: {mode}')
    
for i,t in enumerate(t_support):
    ellipses_lengths[i], circle_lengths[i] = plot_and_compute_intersection(t, a , b, mode, outdir='/home/paolo/Desktop/GIF/', tag = repr(i+1))

plt.figure(figsize=(10,7))
plt.title(f'{mode} constant', size = 15)
plt.plot(t_support, ellipses_lengths, 'darkolivegreen', label = 'Correlated', lw = 3)
plt.plot(t_support, circle_lengths, 'goldenrod', label = 'Independent', lw = 3)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(prop = {'size' : 15}, loc = 'lower left')
plt.xlabel('Curvature', size = 13)
plt.ylabel('Arc Length', size = 13)
plt.savefig(f'/home/paolo/Desktop/GIF/long_plot_{rho:.2}.png',dpi=100)


