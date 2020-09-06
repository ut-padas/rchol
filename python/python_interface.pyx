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

cdef extern from "spcol.c":
    pass


cpdef sddm_to_laplacian(M):
    
    cdef np.ndarray[np.double_t, ndim=1] one_row = -np.squeeze(np.asarray(M.sum(axis=0)))
    np.where(np.abs(one_row) < 1e-9, 0, one_row)
    cdef double total = -one_row.sum()
    M = scipy.sparse.vstack([M, one_row], format='csr')
    one_col = np.append(one_row, total)
    M = scipy.sparse.hstack([M, one_col.reshape(-1, 1)], format='csr')
    return M


cpdef rchol_lap_cpp(M, matrix_row, thread, result_idx):

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


cpdef rchol_lap(laplacian):
    n = laplacian.shape[0]
    return rchol_lap_cpp(laplacian, n-1, 1, np.array([0, n], dtype=np.uint64))


# calculates the separator
"""
cpdef recursive_separator1(logic, depth, target):
    cdef stdint.uint64_t *sep_ptr
    cdef np.ndarray[np.uint64_t, ndim=1] row = logic.indices.astype(dtype=np.uint64)
    cdef np.ndarray[np.uint64_t, ndim=1] col = logic.indptr.astype(dtype=np.uint64)
    if (depth == target):
        size = logic.shape[0]
        val = size
        p = np.arange(size, dtype=np.uint64)
        separator = np.zeros(0, dtype=np.uint64)
        return p, val, separator
    elif (logic.shape[0] <= 1):
        size = logic.shape[0]
        p1, v1 = recursive_separator1([], depth + 1, target)
        p2, v2 = recursive_separator1(csr_matrix((size, size)), depth + 1, target)
        val = np.append(v1, np.append(v2, 0))
        p = np.append(p1, p2)
        separator = np.zeros(0, dtype=np.uint64)
        return p, val, separator
    else:
        sep_ptr = find_separator(logic.shape[0], &(row[0]), &(col[0]))
        sep = np.asarray(<np.uint64_t[:logic.shape[0]]> sep_ptr)

        if depth == 1:
            sep[-1] = 2

        l = np.where(sep == 0)[0]
        r = np.where(sep == 1)[0]
        s = np.where(sep == 2)[0]
        newleft = logic[l[:, None], l]
        newright = logic[r[:, None], r]
        

        [p1, v1, s1] = recursive_separator1(newleft, depth + 1, target)
        [p2, v2, s2] = recursive_separator1(newright, depth + 1, target)
        separator = np.append(l[s1], np.append(r[s2], s))
        val = np.append(v1, np.append(v2, s.shape[0]))
        p = np.append(l[p1], np.append(r[p2], s))
        return p, val, separator
"""



