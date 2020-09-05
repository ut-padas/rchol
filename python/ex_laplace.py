#import array
import numpy as np 
#import time
import scipy.sparse
from scipy.sparse import csc_matrix, csr_matrix, triu, linalg
#import importlib
import python_interface as pyi
from laplace_3d import *


# SDDM matrix from 3D constant Poisson equation
n = 16
A = laplace_3d(n)

# random RHS
N = A.shape[0]
b = np.random.rand(N)
#b = np.random.rand(N, dtype=np.double)
#b = b.astype(dtype=np.double)


p_vec = np.arange(A.shape[0] + 1, dtype=np.uint64)


# convert to laplacian, permute laplacian and extract upper triangular portion(csr), if lower triangular is used, then format needs to be csc
matrix_row = A.shape[0]
laplacian = triu(pyi.convert2laplacian(A)[p_vec[:, None], p_vec], format='csr') * -1

# call the wrapper function for C routine, which returns preconditioner and diagonal
L, D = pyi.python_factorization(laplacian)

# call pcg to solve Ax = b for x
epsilon = 1e-10
x = pyi.pcg(A[p_vec[0:-1, None], p_vec[0:-1]], b[p_vec[0:-1]], L, D, epsilon)
pt = np.zeros(p_vec.shape[0] - 1, dtype=np.uint64)
pt[p_vec[0:-1]] = np.arange(x.shape[0], dtype=np.uint64)
x = x[pt]

print('Verify relative residual: {:.2e}'.format(np.linalg.norm(A * x - b) / np.linalg.norm(b)))



