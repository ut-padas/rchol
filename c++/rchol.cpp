#include "rchol.hpp"
#include "util.hpp"
#include "rchol_lap.hpp"

#include <iostream>

void rchol(const SparseCSR &A, SparseCSR &G) {
  size_t N = A.size();
  size_t nnz = A.nnz();
  std::vector<size_t> rowPtr(A.rowPtr, A.rowPtr+N+1);
  std::vector<size_t> colIdx(A.colIdx, A.colIdx+nnz);
  std::vector<double> val(A.val, A.val+nnz);
  // upper triangular csr form
  std::vector<size_t> rowPtrU;
  std::vector<size_t> colIdxU;
  std::vector<double> valU;
  triu_csr(rowPtr, colIdx, val, rowPtrU, colIdxU, valU); 
  // convert to a laplacian
  std::vector<size_t> rowPtrL;
  std::vector<size_t> colIdxL;
  std::vector<double> valL;
  convert_to_laplace(rowPtrU, colIdxU, valU, 
      rowPtrL, colIdxL, valL, rowPtr, colIdx, val);
  // change edge values to positive and begin factorization
  change_to_positive_edge(valL);
  rchol_lap(rowPtrL, colIdxL, valL, G.rowPtr, G.colIdx, G.val, G.N);
}

