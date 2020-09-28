#ifndef rchol_hpp
#define rchol_hpp

#include "sparse.hpp"

void rchol(const SparseCSR &A, SparseCSR &G);
void rchol(const SparseCSR &A, SparseCSR &G, std::vector<size_t> &permutation, int threads);

#endif
