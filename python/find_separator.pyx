cimport cython
import numpy as np
cimport numpy as np 
from libc cimport stdint

cdef extern from "metis_separator.cpp":
    stdint.uint64_t * metis_separator(stdint.uint64_t length, stdint.uint64_t *rpt1, stdint.uint64_t *cpt1)


cpdef find_separator(logic, depth, target):
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
        raise Exception("too many threads requested") 
        '''
        size = logic.shape[0]
        p1, v1 = find_separator([], depth + 1, target)
        p2, v2 = find_separator(csr_matrix((size, size)), depth + 1, target)
        val = np.append(v1, np.append(v2, 0))
        p = np.append(p1, p2)
        separator = np.zeros(0, dtype=np.uint64)
        return p, val, separator
        '''
    else:
        sep_ptr = metis_separator(logic.shape[0], &(row[0]), &(col[0]))
        sep = np.asarray(<np.uint64_t[:logic.shape[0]]> sep_ptr)

        if depth == 1:
            sep[-1] = 2

        l = np.where(sep == 0)[0]
        r = np.where(sep == 1)[0]
        s = np.where(sep == 2)[0]
        newleft = logic[l[:, None], l]
        newright = logic[r[:, None], r]
        

        [p1, v1, s1] = find_separator(newleft, depth + 1, target)
        [p2, v2, s2] = find_separator(newright, depth + 1, target)
        separator = np.append(l[s1], np.append(r[s2], s))
        val = np.append(v1, np.append(v2, s.shape[0]))
        p = np.append(l[p1], np.append(r[p2], s))
        return p, val, separator


