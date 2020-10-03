# intro
`randchol` is a  C++ library that implements a randomized incomplete Cholesky decomposition and is based on 
[Daniel Spielman's](http://www.cs.yale.edu/homes/spielman/) Julia implementation of a randomized [solver](https://danspielman.github.io/Laplacians.jl/latest/usingSolvers/#Sampling-Solvers-of-Kyng-and-Sachdeva-1)


# randchol

Directory for randomized cholesky with MATLAB, Python and C++ interfaces. Each subdirectory contains a `makefile`, a `README` and examples. The core algorithm is written in C++. To use the parallel routine, the METIS library is needed.


# MATLAB
**MEX** is used to compile C++ codes.

# Python
**Cython** is used to wrap C++ codes. 

# C++
This directory is intended to be used for obtaining high performance.


