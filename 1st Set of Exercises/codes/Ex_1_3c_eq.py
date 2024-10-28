import numpy as np


mu1 = np.array([-2, 0])
mu2 = np.array([2, 1])
cov_b = np.array([[1, -0.6], [-0.6, 1]])


cov_b_inv = np.linalg.inv(cov_b)


lambda12 = 1
lambda21 = 0.5


A_c = -0.5 * (cov_b_inv @ (mu1 - mu2))
B_c = (mu1 - mu2).T @ cov_b_inv @ (mu1 + mu2) - 2 * np.log(lambda21 / lambda12)
C_c = -0.5 * (mu1.T @ cov_b_inv @ mu1 - mu2.T @ cov_b_inv @ mu2)


print(A_c)
print(B_c)
print(C_c)