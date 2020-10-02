# randchol
Directory for randomized cholesky with MATLAB, Python and C++ interfaces. Each subdirectory contains a `makefile`, a `README` and examples. The core algorithm is written in C++. To use the parallel routine, the METIS library is needed.


# MATLAB
**MEX** is used to compile C++ codes.

# Python
**Cython** is used to wrap C++ codes. 


# input format
Python: scipy.sparse.csr_matrix

MATLAB: sparse matrix in MATLAB, which is internally a CSC format

C++: CSR format; see c++/sparse.hpp

