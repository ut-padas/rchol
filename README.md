# Overview
`rchol` is a  C++ library that implements a randomized incomplete Cholesky factorization. `rchol` is provably effective for SDDM matrices (Symmetric and Diagonally Dominant M-matrices), but can be tested on any SPD matrix. It uses OpenMP for shared memory parallelism on x86 architectures. We do not support GPUs. 
Factorizing the discrete Laplacian matrix on a 1024x1024x1024 grid with standard 7-point stencil, which has **one billion** unknowns, takes about 3 minutes/180 seconds using 64 threads. 

The underlying algorithm is based on  [Daniel Spielman's](http://www.cs.yale.edu/homes/spielman/) Julia implementation of a randomized incomplete factorization for  [graph Laplacians](https://github.com/danspielman/Laplacians.jl/blob/master/docs/src/usingSolvers.md#sampling-solvers-of-kyng-and-sachdeva). 

The corresponding paper that describes the details can be found [here](https://arxiv.org/abs/2011.07769). 

*Dependencies:*  The only dependency is the [METIS library](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview).  METIS is required for shared-memory parallelism. 

# Directory structure

* **C++**: This is the main directory for the source code. It also includes usage examples. 
* **MATLAB**: Provides an interface for MATLAB users; it is similar to MALTAB's `ichol`. We provide several examples for various matrices. In our experiments, `rchol` is 3X faster than the thresholded `ichol` for the same sparsity pattern (as of September 2020). 
* **Python**: Similarly for Python.

At each directory, please check the README and makefile files.

# Authors
Tianyu Liang  
Chao Chen  
George Biros (advisor)  

# Funding
This project is partially funded by NSF award CCF-1817048; and  by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research, Applied Mathematics program under Award Number DE-SC0019393, by the U.S. Department of Energy, National Nuclear Security Administration Award Number DE-NA000396 (PSAAP III).  Computing time on the Texas Advanced Computing Centers Stampede system was provided by an allocation from TACC and the NSF. 








