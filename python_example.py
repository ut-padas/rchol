import array
import numpy as np 
import time
import scipy.sparse
from scipy.sparse import csc_matrix, csr_matrix, triu, linalg
import h5py
import os
from os.path import dirname, join as pjoin
import importlib
import python_interface as pyi




def example_problem():
    import sys
    sys.path.append(pjoin(os.path.dirname(os.path.realpath(__file__)), 'python'))
    import laplace_3d

    
    n = 10 # problem size
    thread = 2 # number of thread to use, should be powers of 2, and should not exceed problem size
    # generate the example matrix
    original = laplace_3d.laplace_3d(n)
    original = original.tocsr(copy=True)
    original.eliminate_zeros()
    
    
    
    # convert to laplacian, permute laplacian and extract content
    matrix_row = original.shape[0]
    test = pyi.convert2laplacian(original)
    logic = test.copy()
    logic.setdiag(0)
    logic.eliminate_zeros()
    p_vec, val, sep = pyi.recursive_separator1(logic, 1, np.log2(thread) + 1)
    result_idx = np.cumsum(np.append(0, val))
    M = triu(test[p_vec[:, None], p_vec], format='csr') * -1

    # call the wrapper function for C routine, which returns preconditioner and diagonal
    L, D = pyi.pass_to_C(M, matrix_row, thread, result_idx)

    # call pcg to solve Ax = b for x
    b = np.random.rand(matrix_row)
    b = b.astype(dtype=np.double)
    epsilon = 1e-10
    x = pyi.pcg(original[p_vec[0:-1, None], p_vec[0:-1]], b[p_vec[0:-1]], L, D, epsilon)
    pt = np.zeros(p_vec.shape[0] - 1, dtype=np.uint64)
    pt[p_vec[0:-1]] = np.arange(x.shape[0], dtype=np.uint64)
    x = x[pt]
    print('relative residual of system: ' + str(np.linalg.norm(original * x - b) / np.linalg.norm(b)))




# call the example
example_problem()