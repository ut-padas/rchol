# intro
`randchol` is a  C++ library that implements a randomized incomplete Cholesky decomposition and is based on 
[Daniel Spielman's](http://www.cs.yale.edu/homes/spielman/) Julia implementation of a randomized incomplete factorization for  [Graph Laplacians](https://danspielman.github.io/Laplacians.jl/latest/usingSolvers/#Sampling-Solvers-of-Kyng-and-Sachdeva-1). 

`randchol` can be provably effective for SDDM matrices (Symmetric and Diagonally Dominant M-matrices), but can be tested on any SPD matrix. It uses OpenMP for shared memory parallelism on x86 architecture. We do not support for GPU acceleration.  

The corresponding paper that describes the details can be found in TODO add arxiv link. 

*Dependencies:*  The only dependency is the [METIS library](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview).  METIS is required for shared-memory parallelism. 

# Directory structure

* **C++**: This is the main directory for the source code. It also includes usage examples. 
* **MATLAB**: Provides and interface for MATLAB matrices. We provide several examples for various matrices. 
* **Python**: Similar with MATLAB but for Python

At each directory, please check the README and makefile files.




