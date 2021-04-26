#ifndef rchol_parallel_hpp
#define rchol_parallel_hpp

#include "sparse.hpp"

void rchol(const SparseCSR &A, SparseCSR &G, std::vector<size_t> &permutation, int threads);

#endif
