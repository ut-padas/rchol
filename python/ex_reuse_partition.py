import sys
sys.path.append('rchol/')
import numpy as np 
from scipy.sparse import identity
from numpy.linalg import norm
from rchol import *
from util import *


# Initial problem: 3D-Poisson
n = 20
A = laplace_3d(n) # see ./rchol/util.py

# random RHS
N = A.shape[0]
b = np.random.rand(N)
print("Initial problem:")

# compute preconditioner after reordering (multi thread)
nthreads = 2
G, perm, part = rchol(A, nthreads)
Aperm = A[perm[:, None], perm]
print('fill-in ratio: {:.2}'.format(2*G.nnz/A.nnz))

# solve with PCG
tol = 1e-6
maxit = 200
x, relres, itr = pcg(Aperm, b[perm], tol, maxit, G, G.transpose().tocsr())
print('# CG iterations: {}'.format(itr))
print('Relative residual: {:.2e}\n'.format(relres))

# perturb the original matrix
B = A + 1e-3*identity(N)
print('New problem (same sparsity) ...')

# compute preconditioner with existing permutation/partition
L = rchol(B, nthreads, perm, part)[0]
print('fill-in ratio: {:.2}'.format(2*L.nnz/A.nnz))

# solve the new problem
Bperm = B[perm[:, None], perm]
x, relres, itr = pcg(Bperm, b[perm], tol, maxit, L, L.transpose().tocsr())
print('# CG iterations: {}'.format(itr))
print('Relative residual: {:.2e}\n'.format(relres))



