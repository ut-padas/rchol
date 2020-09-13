#include "sparse.hpp"


SparseCSR::SparseCSR() {
  N = 0;
  rowPtr = 0;
  colIdx = 0;
  val = 0;
}

SparseCSR::SparseCSR(const std::vector<size_t> &rowPtrA, const std::vector<size_t> &colIdxA,
    const std::vector<double>& valA) {
  this->N = rowPtrA.size()-1;
  this->rowPtr = new size_t[N+1];

  size_t nnz = rowPtrA[N];
  this->colIdx = new size_t[nnz];
  this->val = new double[nnz];
  std::copy(rowPtrA.begin(), rowPtrA.end(), rowPtr);
  std::copy(colIdxA.begin(), colIdxA.end(), colIdx);
  std::copy(valA.begin(), valA.end(), val);
}

size_t SparseCSR::size() const {
  return N;
}

size_t SparseCSR::nnz() const {
  return rowPtr[N];
}

SparseCSR::~SparseCSR() {
  if (N>0) {
    delete[] rowPtr, colIdx, val;
    rowPtr = 0;
    colIdx = 0;
    val = 0;
  }
}


