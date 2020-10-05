# rchol
randomized Cholesky factorization

# Sparse matrix format
`rchol()` accepts sarse matrices stored in the compressed sparse row (CSR) format; see `sparse.hpp`.

# Syntax and Description
```c++
void rchol(const SparseCSR &A, SparseCSR &G);
```

- **A**: SDDM sparse matrix 
- **G**: lower triangular matrix

This is a sequntial routine computing an approximate Cholesky factorization `G*G'~A`. Reordering of the input sparse matrix is recommended before calling this routine. See `ex_laplace.cpp` for an example.

```c++
void rchol(const SparseCSR &A, SparseCSR &G, std::vector<size_t> &p, int nthreads);
```

- **A**: SDDM sparse matrix 
- **nthreads**: number of threads
- **G**: lower triangular matrix
- **p**: permutation vector

This is a *parallel* routine computing an approximate Cholesky factorization `G*G'~A(p,p)`. The input sparse matrix is reordered inside the routine, and the permutation vector is returned as an output. See `ex_laplace_parallel.cpp` for an example. The METIS package is required for compilation; see [Compilation instructions](#compilation-instructions) for details.

<!--
# SDD matrix
For an SDD sparse matrix, we first create an extended SDDM matrix and then call `rchol`. See `ex_hyperbolic.m` for an example.
-->
# Compilation instructions
A makefile is available in the current directory.

The METIS library is required in order to compile c++ codes for the parallel rchol() routine. Please fill **METIS_INC** and **METIS_LIB** variables in the makefile with the METIS header directory and the library directory, respectively.

The Intel TBB memory allocator can be liked to automatically replace malloc and other C/C++ functions for better performance. Please fill **TBB_MALLOC_PROXY** variable in the makefile.

