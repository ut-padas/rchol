from libc cimport stdint
import importlib


cpdef convert2laplacian(M)
cpdef pcg(A, b, L, D, epsilon)
cpdef recursive_separator1(logic, depth, target)
cpdef python_factorization(laplacian)

ctypedef stdint.uint64_t size_t    



cdef extern from "python_source.cpp":
    int entrance(csc_form *input, stdint.uint64_t *idx_data, stdint.uint64_t idxdim, int thread)  
    ctypedef struct csc_form:
        stdint.uint64_t *row
        stdint.uint64_t *col
        double *val
        stdint.uint64_t *ret_row
        stdint.uint64_t *ret_col
        double *ret_val
        double *ret_diag
        stdint.uint64_t nsize

cdef extern from "separator.cpp":
    stdint.uint64_t * find_separator(stdint.uint64_t length, stdint.uint64_t *rpt1, stdint.uint64_t *cpt1)

cdef extern from "spcol.c":
    pass