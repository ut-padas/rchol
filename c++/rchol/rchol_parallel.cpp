#include "rchol_parallel.hpp"
#include "util.hpp"
//#include "find_separator.hpp"
#include "partition_and_order.hpp"
#include "rchol_lap.hpp"
#include "timer.hpp"

#include <math.h> 
#include <iostream>


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
void rchol(const SparseCSR &A, SparseCSR &G, std::vector<size_t> &permutation, 
    std::vector<int> &S, int threads) {

  if(threads <= 0)
    throw std::invalid_argument( "thread number should be positive" );

  if((threads & (threads - 1)) != 0)
    throw std::invalid_argument( "thread number should be a power of 2" );

  Timer t, t1; t.start();

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
  t1.start();
  /*
  Separator_info separator = find_separator(no_diag, 1, (size_t)(std::log2(threads) + 1));
  std::copy(separator.p->begin(), separator.p->end(), std::back_inserter(permutation));
  S.resize( separator.val->size()+1, 0 );
  for(size_t i = 0; i < separator.val->size(); i++)
  {
    S[i+1] = separator.val->at(i) + S[i];
  }
  S.back()++; // artifitial vertex
  delete separator.p;
  delete separator.val;
  */
  
  partition_and_ordering(no_diag, threads, permutation, S);
  t.stop(); std::cout<<"find separator: "<<t.elapsed()<<" s\n";
  
  reorder(A, rowPtr, colIdx, val, permutation);
  t.stop(); std::cout<<"Compute ordering: "<<t.elapsed()<<" s\n";


  t.start();
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
  t.stop(); //std::cout<<"Before rchol_lap: "<<t.elapsed()<<" s\n";

  // change edge values to positive and begin factorization
  t.start();
  change_to_positive_edge(valL);
  rchol_lap(rowPtrL, colIdxL, valL, G.rowPtr, G.colIdx, G.val, G.N, S);
  t.stop(); std::cout<<"Call rchol_lap: "<<t.elapsed()<<" s\n";
}
