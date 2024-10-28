import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


mu_b = np.array([-7/6, -1/3])
cov_b = np.array([[11/12, 57/90], [57/90, 5/3]])

mu_c = np.array([-1.75, 2.5])
cov_c = np.array([[0.8, 0.375], [0.1, 0.05]])


def create_ellipse(mean, cov, ax, color):

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    largest_eigenval_index = eigenvalues.argmax()
    largest_eigenvec = eigenvectors[:, largest_eigenval_index]

    angle = np.arctan2(largest_eigenvec[1], largest_eigenvec[0])

    angle = np.degrees(angle)

    chi2_val = 5.991

    width, height = 2 * np.sqrt(chi2_val * eigenvalues)

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor=color, fc='None', lw=2, linestyle='--')

    ax.add_patch(ellipse)


mu_b_new = np.array([-7/6, -1/3])
cov_b_new = np.array([[11/12, 57/90], [57/90, 5/3]])

mu_c_new = np.array([-1.75, 2.5])
cov_c_new = np.array([[0.8, 0.375], [0.375, 2.5]])


fig, ax = plt.subplots(figsize=(8, 8))


create_ellipse(mu_b_new, cov_b_new, ax, 'blue')
create_ellipse(mu_c_new, cov_c_new, ax, 'red')


plt.xlim(-6, 4)
plt.ylim(-3, 7)


plt.xlabel('x1')
plt.ylabel('x2/x3')
plt.title('Isostatic Curves for Conditional Distributions')

# Draw the legend
plt.legend(['p(x1, x2 | x3 = 1)', 'p(x1, x3 | x2 = 1)'])

# Show the plot with grid
plt.grid(True)

# Add horizontal line at x2=0 with increased linewidth
plt.axhline(y=0, color='black', linewidth=2)

# Add vertical line at x1=0 with increased linewidth
plt.axvline(x=0, color='black', linewidth=2)

plt.show()