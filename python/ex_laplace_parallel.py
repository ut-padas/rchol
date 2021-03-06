import sys
sys.path.append('rchol/')
import numpy as np 
from numpy.linalg import norm
from rchol import *
from util import *


# SDDM matrix from 3D constant Poisson equation
n = 20
A = laplace_3d(n) # see ./rchol/util.py

# random RHS
N = A.shape[0]
b = np.random.rand(N)

# compute preconditioner after reordering (multi thread)
nthreads = 2
G, p = rchol(A, nthreads)[0:2]
Aperm = A[p[:, None], p]
print('fill-in ratio: {:.2}'.format(2*G.nnz/A.nnz))


# solve with PCG
tol = 1e-6
maxit = 200
x, relres, itr = pcg(Aperm, b[p], tol, maxit, G, G.transpose().tocsr())
print('# CG iterations: {}'.format(itr))

# verify solution
y = np.zeros(N)
y[p] = x
print('Relative residual: {:.2e}'.format(norm(A*y-b)/norm(b)))


