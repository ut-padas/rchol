cimport cython
import numpy as np 
cimport numpy as np 
import scipy.sparse
from scipy.sparse import csr_matrix
from libc cimport stdint 


cdef extern from "rchol_lap.cpp":
    void rchol(csc_form *input, stdint.uint64_t *idx_data, stdint.uint64_t idxdim, int thread)  
    ctypedef struct csc_form:
        stdint.uint64_t *row
        stdint.uint64_t *col
        double *val
        stdint.uint64_t *ret_row
        stdint.uint64_t *ret_col
        double *ret_val
        double *ret_diag
        stdint.uint64_t nsize


cpdef rchol_lap(M, matrix_row, thread, result_idx):

    cdef np.ndarray[np.uint64_t, ndim=1] row = M.indices.astype(dtype=np.uint64)
    cdef np.ndarray[np.uint64_t, ndim=1] col = M.indptr.astype(dtype=np.uint64)
    cdef np.ndarray[np.double_t, ndim=1] data = M.data
    cdef np.ndarray[np.uint64_t, ndim=1] idx_data = result_idx.astype(dtype=np.uint64)

    # set up input to pass to C++
    cdef csc_form input
    input.row = &(row[0])
    input.col = &(col[0])
    input.val = &(data[0])
    input.nsize = M.shape[0]
    rchol(&input, &(idx_data[0]), idx_data.shape[0], thread) 

    # create arrays to store answer and call c++
    np_ret_indptr = np.asarray(<np.uint64_t[:matrix_row + 1]> input.ret_col)
    np_ret_indices = np.asarray(<np.uint64_t[:np_ret_indptr[matrix_row]]> input.ret_row)
    np_ret_data = np.asarray(<np.double_t[:np_ret_indptr[matrix_row]]> input.ret_val)
    D = np.asarray(<np.double_t[:matrix_row]> input.ret_diag)
    L = csr_matrix((np_ret_data, np_ret_indices, np_ret_indptr), shape=(matrix_row, matrix_row))
    return L, D


cdef extern from "spcol.c":
    pass


