#include "rchol.hpp"
#include "util.hpp"
#include "rchol_lap.hpp"
#include <math.h> 
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
  std::vector<size_t> result_idx = {0, rowPtrL.size() - 1};
  rchol_lap(rowPtrL, colIdxL, valL, G.rowPtr, G.colIdx, G.val, G.N, result_idx);
}


SparseCSR remove_diagonal(const SparseCSR &A, std::vector<size_t> &rowPtr, std::vector<size_t> &colIdx, 
  std::vector<double> &val){
  rowPtr.push_back(0);
  for(size_t i = 0; i < A.N; i++)
  {
    size_t first = A.rowPtr[i];
    size_t last = A.rowPtr[i + 1];
    rowPtr.push_back(rowPtr[rowPtr.size()- 1]);
    size_t tempsize = rowPtr.size() - 1;
    for(size_t j = first; j < last; j++)
    {
      // add if not on diagonal
      if(A.colIdx[j] != i)
      {
        colIdx.push_back(A.colIdx[j]);
        val.push_back(A.val[j]);
        rowPtr[tempsize]++;
      }
      
    }
    
  }
  SparseCSR B(rowPtr, colIdx, val, true);
  return B;

}

// multithread version
void rchol(const SparseCSR &A, SparseCSR &G, std::vector<size_t> &permutation, int threads) {

  if(threads <= 0)
    throw std::invalid_argument( "thread number should be positive" );

  if((threads & (threads - 1)) != 0)
    throw std::invalid_argument( "thread number should be a power of 2" );

  size_t N = A.size();
  size_t nnz = A.nnz();
  std::vector<size_t> rowPtr;
  std::vector<size_t> colIdx;
  std::vector<double> val;

  // remove the diagonal
  std::vector<size_t> *rowPtr_no_diag = new std::vector<size_t>();
  std::vector<size_t> *colIdx_no_diag = new std::vector<size_t>();
  std::vector<double> *val_no_diag = new std::vector<double>();
  SparseCSR no_diag = remove_diagonal(A, *rowPtr_no_diag, *colIdx_no_diag, *val_no_diag);
  delete rowPtr_no_diag;
  delete colIdx_no_diag;
  delete val_no_diag;

  
  // calculate permuation
  Separator_info separator = find_separator(no_diag, 1, (size_t)(std::log2(threads) + 1));
  std::copy(separator.p->begin(), separator.p->end(), std::back_inserter(permutation));
  std::vector<size_t> result_idx;
  result_idx.push_back(0);
  for(size_t i = 0; i < separator.val->size(); i++)
  {
    result_idx.push_back(separator.val->at(i) + result_idx[result_idx.size() - 1]);
  }
  result_idx[result_idx.size() - 1]++;
  permute_matrix(A, rowPtr, colIdx, val, permutation);
  delete separator.p;
  delete separator.val;




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
  rchol_lap(rowPtrL, colIdxL, valL, G.rowPtr, G.colIdx, G.val, G.N, result_idx);
}