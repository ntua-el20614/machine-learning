import numpy as np
import matplotlib.pyplot as plt


mu1_a = np.array([-2, 0])
mu2_a = np.array([2, 1])
cov_a = np.identity(2) 


num_points = 200


class1_samples_a = np.random.multivariate_normal(mu1_a, cov_a, num_points)
class2_samples_a = np.random.multivariate_normal(mu2_a, cov_a, num_points)


def decision_boundary_a(x):
    return (8 * x - 1) / 2 


x_values_a = np.linspace(-6, 6, 400)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(class1_samples_a[:, 0], class1_samples_a[:, 1], c='red', label='Class 1')
plt.scatter(class2_samples_a[:, 0], class2_samples_a[:, 1], c='blue', label='Class 2')
plt.plot(x_values_a, decision_boundary_a(x_values_a), 'k-', label='Decision Boundary')
plt.scatter(mu1_a[0], mu1_a[1], s=100, c='black', marker='x', label='Mean of Class 1')
plt.scatter(mu2_a[0], mu2_a[1], s=100, c='black', marker='o', label='Mean of Class 2')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Part (a): Decision Boundary and Data Points')
plt.legend()
plt.grid(True)
plt.show()
