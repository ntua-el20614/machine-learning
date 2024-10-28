import numpy as np

mu1 = np.array([-2, 0])
mu2 = np.array([2, 1])

# Given covariance matrix for part (b)
cov_b = np.array([
    [1, -0.6],
    [-0.6, 1]
])


cov_b_inv = np.linalg.inv(cov_b)


# which simplifies to (mu1 - mu2)^T * Sigma^-1 * x = 1/2 * (mu2^T * Sigma^-1 * mu2 - mu1^T * Sigma^-1 * mu1)

# Coefficients for the linear decision boundary (A * x = C)
A = (mu1.T @ cov_b_inv - mu2.T @ cov_b_inv)
C = 0.5 * (mu2.T @ cov_b_inv @ mu2 - mu1.T @ cov_b_inv @ mu1)

print(A)
print(C)
