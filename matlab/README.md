# rchol
randomized Cholesky factorization

# Sparse matrix format
rchol() accepts a matlab sparse matrix, which is internally a compressed sparse column (CSC) format. We provide `laplace_3d.m` script for generating standard 7-point finite difference discretization of Poisson equation in `./rchol/` directory.

# Syntax and description
```matlab
G = rchol(A)
```

- **A**: SDDM sparse matrix 
- **G**: lower triangular matrix

This is a sequntial routine computing an approximate Cholesky factorization `G*G'~A`. Reordering of the input sparse matrix is recommended before calling this routine. See `ex_laplace.m` for an example.

```matlab
[G, p] = rchol(A, nthreads)
```

- **A**: SDDM sparse matrix 
- **nthreads**: number of threads
- **G**: lower triangular matrix
- **p**: permutation vector

This is a *parallel* routine computing an approximate Cholesky factorization `G*G'~A(p,p)`. The input sparse matrix is reordered inside the routine, and the permutation vector is returned as an output. See `ex_laplace_parallel.m` for an example. The METIS package is required for compilation; see [Compilation instructions](#compilation-instructions) for details.

```matlab
[G, perm, part] = rchol(A, nthreads)
```

- **A**: SDDM sparse matrix 
- **nthreads**: number of threads
- **G**: lower triangular matrix
- **perm**: permutation vector
- **part**: partition used in parallel routine

The returned permutation and partition information can be stored for reuse. See below. 

```matlab
G = rchol(A, nthreads, perm, part)
```

- **A**: SDDM sparse matrix 
- **nthreads**: number of threads
- **perm**: permutation vector
- **part**: partition used in parallel routine
- **G**: lower triangular matrix

This routine uses an existing permutation/partition.


# SDD matrix
For an SDD sparse matrix, we first create an extended SDDM matrix and then call `rchol`. See `ex_hyperbolic.m` for an example.

# Compilation instructions
A makefile is available in the current directory, which employs *MEX* to compile our c++ files. Note that the gcc version currently supported with MEX is '6.3.x' as of September 2020.

The METIS library is required in order to compile c++ codes for the parallel rchol() routine. Please fill **METIS_INC** and **METIS_LIB** variables in the makefile with the METIS header directory and the library directory, respectively. (We suggest compile METIS to a static library.)

<!--

Matlab interface:

rchol inputs:

The rchol function takes two inputs. The first input is the sparse SDDM matrix, which is a mandatory input. If the second input is not given, then the factorization will be sequencial. In this case, if users want to use a particular permuation, then they should permute the first input before passing it into the function.

The second input (optional) is for multithreading purpose; it specifies the number of threads to be used during the execution of the parallel factorization. The second input should be strictly greater than 0 and a power of 2.

In the special case that the input thread number is 1, the method will be equivalent to the sequential method. In other words, the function behaves as if the second input is nonexistent at all. 


rchol outputs:

rchol returns two outputs. The first output is the Cholesky factor, the second output is the permutation used within rchol. If the thread number is not supplied or is equal to 1, then the returned permuation will simply be a vector from 1 to n, where n is the length of the matrix. However, if thread number is anything other than 1, meaning that parallelization is used, then the function will permute the given SDDM matrix first before factorization. This is necessary because in order to multi-thread the method, we would need to do graph partitioning. The permutation used internally within the function will be returned. Hence, when using pcg, the users will need to supply it with a permuted system, in which the permutation used should be the one returned by rchol.
-->



