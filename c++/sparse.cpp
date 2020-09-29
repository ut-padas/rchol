#include "sparse.hpp"
#include <iostream>

SparseCSR::SparseCSR() {
  N = 0;
  rowPtr = 0;
  colIdx = 0;
  val = 0;
  ownMemory = false;
}

SparseCSR::SparseCSR(const std::vector<size_t> &rowPtrA, const std::vector<size_t> &colIdxA,
    const std::vector<double>& valA, bool mem) {
  this->N = rowPtrA.size()-1;
  this->rowPtr = new size_t[N+1];

  size_t nnz = rowPtrA[N];
  this->colIdx = new size_t[nnz];
  this->val = new double[nnz];
  std::copy(rowPtrA.begin(), rowPtrA.end(), rowPtr);
  std::copy(colIdxA.begin(), colIdxA.end(), colIdx);
  std::copy(valA.begin(), valA.end(), val);

  this->ownMemory = mem;
}

SparseCSR::SparseCSR(const SparseCSR &A) {
  this->N = A.size();
  this->rowPtr = new size_t[N+1];

  size_t nnz = A.nnz();
  this->colIdx = new size_t[nnz];
  this->val = new double[nnz];
  std::copy(A.rowPtr, A.rowPtr+N+1, this->rowPtr);
  std::copy(A.colIdx, A.colIdx+nnz, this->colIdx);
  std::copy(A.val, A.val+nnz, this->val);

  this->ownMemory = true;
}

size_t SparseCSR::size() const {
  return N;
}

size_t SparseCSR::nnz() const {
  return rowPtr[N];
}

SparseCSR::~SparseCSR() {
  if (N>0 && ownMemory) {
    delete[] rowPtr;
    delete[] colIdx;
    delete[] val;
    rowPtr = 0;
    colIdx = 0;
    val = 0;
    N = 0;
  }
}


void print(const SparseCSR &A, std::string name) {
  std::cout<<name<<" rowPtr:\n";
  for (size_t i=0; i<=A.N; i++)
    std::cout<<A.rowPtr[i]<<" ";
  std::cout<<std::endl;
  std::cout<<name<<" colIdx:\n";
  for (size_t i=0; i<A.rowPtr[A.N]; i++)
    std::cout<<A.colIdx[i]<<" ";
  std::cout<<std::endl;
  std::cout<<name<<" value:\n";
  for (size_t i=0; i<A.rowPtr[A.N]; i++)
    std::cout<<A.val[i]<<" ";
  std::cout<<std::endl;
}

