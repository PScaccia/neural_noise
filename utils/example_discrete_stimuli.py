import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.special import expit  # Sigmoid function

rho = -0.2

# Example data (replace with your actual means and covariance)
a = np.array([1, 2])
b = np.array([3, 4])
Sigma = np.array([[1, rho], [rho, 1]])
Sigma_inv = np.linalg.inv(Sigma)

def delta(x):
    diff = a - b
    constant = 0.5 * (a.T @ Sigma_inv @ a - b.T @ Sigma_inv @ b)
    return diff.T @ Sigma_inv @ x - constant

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an ellipse representing the covariance matrix 
    cov centered at pos, scaled to nstd standard deviations.
    """
    ax = ax or plt.gca()
    
    # Eigen-decomposition of covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort eigenvalues and eigenvectors
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    
    # Angle of ellipse rotation in degrees
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    
    # Width and height are "2 * nstd * sqrt(eigenvalue)"
    width, height = 2 * nstd * np.sqrt(eigvals)
    
    ellipse = Ellipse(xy=pos, width=width, height=height, angle=angle, **kwargs, zorder =13)
    
    ax.add_patch(ellipse)
    return ellipse

# Create a grid of points for contour plotting
x_vals = np.linspace(-2, 7, 400)
y_vals = np.linspace(-2, 7, 400)
X, Y = np.meshgrid(x_vals, y_vals)
points = np.vstack([X.ravel(), Y.ravel()])
Z = delta(points).reshape(X.shape)
plt.figure(figsize=(8, 6))
# Plot contour lines for Delta
# contour_levels = np.linspace(np.min(Z), np.max(Z), 100)
qmax = np.percentile(Z, 0.5)
qmax = 1
# plt.contour(X, Y, Z, levels=contour_levels, cmap='coolwarm',zorder = 12,vmin = -1000,vmax=1000)
cp=plt.pcolormesh(X, Y, Z, cmap='coolwarm_r',zorder = 1,vmin = -qmax,vmax=qmax)

plt.colorbar(cp, label=r'$\Delta(\mathbf{x})$')

# Plot means
plt.scatter(a[0], a[1], color='blue', label='State A Mean',zorder=13)
plt.scatter(b[0], b[1], color='red', label='State B Mean',zorder=13)

# Plot covariance ellipses at 2 standard deviations (~95% confidence)
plot_cov_ellipse(Sigma, a, nstd=2, ax=plt.gca(), edgecolor='blue', linestyle='--', lw=2, facecolor='none')
plot_cov_ellipse(Sigma, b, nstd=2, ax=plt.gca(), edgecolor='red', linestyle='--', lw=2, facecolor='none')

plt.title('Contour Lines of $\Delta(\mathbf{x})$ with Gaussian Confidence Ellipses')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(-2,7)
plt.ylim(-2,7)

# plt.grid(True)
plt.axis('equal')
plt.legend()

plt.show()