# intro
`randchol` is a  C++ library that implements a randomized incomplete Cholesky factorization. `randchol` is provably effective for SDDM matrices (Symmetric and Diagonally Dominant M-matrices), but can be tested on any SPD matrix. It uses OpenMP for shared memory parallelism on x86 architectures. We do not support GPUs. 
Factorizing a 3D, one billion (1024^3) unknowns (a 1B-by-1B sparse matrix) correspondong on 7-point stencil Laplacian takes about 180 seconds on 64 threads. 

The underlying algorithm is based on  [Daniel Spielman's](http://www.cs.yale.edu/homes/spielman/) Julia implementation of a randomized incomplete factorization for  [graph Laplacians](https://github.com/danspielman/Laplacians.jl/blob/master/docs/src/usingSolvers.md#sampling-solvers-of-kyng-and-sachdeva). 

The corresponding paper that describes the details can be found in TODO add arxiv link. 

*Dependencies:*  The only dependency is the [METIS library](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview).  METIS is required for shared-memory parallelism. 

# Directory structure

* **C++**: This is the main directory for the source code. It also includes usage examples. 
* **MATLAB**: Provides an interface for MATLAB users; it is similar to MALTAB's `ichol`. We provide several examples for various matrices. In our experiments, `randchol` seems to be 3X faster than the thresholded `ichol` for the same sparsity pattern (as of September 2020). 
* **Python**: Similarly for Python.

At each directory, please check the README and makefile files.




