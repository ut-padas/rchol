# rchol
randomized Cholesky factorization

# Sparse matrix format
`rchol()` accepts any sparse matrix format and converts it to a scipy.sparse.csr_matrix internally. We provide `util.py` script for generating standard 7-point finite difference discretization of Poisson equation in `./rchol/` directory.

# Syntax and description
```python
G = rchol(A)
```

- **A**: SDDM sparse matrix 
- **G**: lower triangular matrix

This is a sequntial routine computing an approximate Cholesky factorization `G*G'~A`. Reordering of the input sparse matrix is recommended before calling this routine. See `ex_laplace.m` for an example.

```python
G, p = rchol(A, nthreads)
```

- **A**: SDDM sparse matrix 
- **nthreads**: number of threads
- **G**: lower triangular matrix
- **p**: permutation vector

This is a *parallel* routine computing an approximate Cholesky factorization `G*G'~A(p,p)`. The input sparse matrix is reordered inside the routine, and the permutation vector is returned as an output. See `ex_laplace_parallel.m` for an example. The METIS package is required for compilation; see [Compilation instructions](#compilation-instructions) for details.


<!--
# SDD matrix
For an SDD sparse matrix, we first create an extended SDDM matrix and then call `rchol`. See `ex_hyperbolic.m` for an example.
-->

# Compilation instructions
A makefile is available in the current directory, which employs *Cython* to compile our c++ files. The makefile defaults to python3, so run the examples with python3, e.g., `python3 ex_laplace.py`.

The METIS library is required in order to compile c++ codes for the parallel rchol() routine. Please fill **METIS_INC** and **METIS_LIB** variables in the makefile with the METIS header directory and the library directory, respectively. (We suggest compile METIS to a static library.)



