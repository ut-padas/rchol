cimport cython
from cpython cimport array
import array
import numpy as np 
cimport numpy as np 
import time
import scipy.sparse
from scipy.sparse import csc_matrix, csr_matrix, triu, linalg
import h5py
import os
from os.path import dirname, join as pjoin
from libc cimport stdint
import importlib
from libc.stdlib cimport malloc, free
cimport python_interface as pyi
import python_interface as pyi




cpdef example_problem():
    import sys
    sys.path.append(pjoin(os.path.dirname(os.path.realpath(__file__)), 'python'))
    import laplace_3d

    
    n = 10 # problem size
    cdef int thread = 2 # number of thread to use, should be powers of 2, and should not exceed problem size
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
    cdef np.ndarray[np.uint64_t, ndim=1] row = M.indices.astype(dtype=np.uint64)
    cdef np.ndarray[np.uint64_t, ndim=1] col = M.indptr.astype(dtype=np.uint64)
    cdef np.ndarray[np.double_t, ndim=1] data = M.data
    cdef np.ndarray[np.uint64_t, ndim=1] idx_data = result_idx.astype(dtype=np.uint64)


    # set up input to pass to C++
    cdef pyi.csc_form input
    input.row = &(row[0])
    input.col = &(col[0])
    input.val = &(data[0])
    input.nsize = M.shape[0]
    pyi.entrance(&input, &(idx_data[0]), idx_data.shape[0], thread) 
    np_ret_indptr = np.asarray(<np.uint64_t[:matrix_row + 1]> input.ret_col)
    np_ret_indices = np.asarray(<np.uint64_t[:np_ret_indptr[matrix_row]]> input.ret_row)
    np_ret_data = np.asarray(<np.double_t[:np_ret_indptr[matrix_row]]> input.ret_val)
    D = np.asarray(<np.double_t[:matrix_row]> input.ret_diag)
    L = csr_matrix((np_ret_data, np_ret_indices, np_ret_indptr), shape=(matrix_row, matrix_row))



    # call pcg to solve Ax = b for x
    b = np.random.rand(matrix_row)
    b = b.astype(dtype=np.double)
    epsilon = 1e-10
    x = pyi.pcg(original[p_vec[0:-1, None], p_vec[0:-1]], b[p_vec[0:-1]], L, D, epsilon)
    pt = np.zeros(p_vec.shape[0] - 1, dtype=np.uint64)
    pt[p_vec[0:-1]] = np.arange(x.shape[0], dtype=np.uint64)
    x = x[pt]
    print('relative residual of system: ' + str(np.linalg.norm(original * x - b) / np.linalg.norm(b)))