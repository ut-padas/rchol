#ifndef LAPLACE_3D_HPP
#define LAPLACE_3D_HPP

#include <cassert>
#include <vector>
#include "sparse.hpp"

template <typename T>
void laplace_3d(size_t n, std::vector<size_t> &rowPtr, std::vector<size_t> &colIdx, 
    std::vector<T> &val) {
  size_t N = n*n*n;
  size_t nnz = 0;
  rowPtr.reserve(N+1); rowPtr.push_back(nnz);
  colIdx.reserve(7*N);
  val.reserve(7*N);

  size_t n2 = n*n;
  for (size_t i=0; i<n; i++) {
    for (size_t j=0; j<n; j++) {
      for (size_t k=0; k<n; k++) {
        size_t idx = k+j*n+i*n2;
        if (i>0) {
          colIdx.push_back(idx-n2);
          val.push_back(-1);
          nnz++;
        }
        if (j>0) {
          colIdx.push_back(idx-n);
          val.push_back(-1);
          nnz++;
        }
        if (k>0) {
          colIdx.push_back(idx-1);
          val.push_back(-1);
          nnz++;
        }
        // self
        colIdx.push_back(idx);
        val.push_back(6);
        nnz++;
        if (k<n-1) {
          colIdx.push_back(idx+1);
          val.push_back(-1);
          nnz++;
        }
        if (j<n-1) {
          colIdx.push_back(idx+n);
          val.push_back(-1);
          nnz++;
        }
        if (i<n-1) {
          colIdx.push_back(idx+n2);
          val.push_back(-1);
          nnz++;
        }
        rowPtr.push_back(nnz);
      }
    }
  }
#if 0
  assert(rowPtr.size()==N+1);
  assert(colIdx.size()==val.size());
  assert(colIdx.size()==7*(n-2)*(n-2)*(n-2)+6*(n-2)*(n-2)*6+5*(n-2)*12+4*1*8);
#endif
}


SparseCSR laplace_3d(int n) {
  std::vector<size_t> rowPtr, colIdx;
  std::vector<double> val;
  
  std::cout<<"before laplace_3d ..."<<std::endl;
  laplace_3d(n, rowPtr, colIdx, val);
  std::cout<<"after laplace_3d ..."<<std::endl;
  
  SparseCSR A(rowPtr, colIdx, val, false);
  return A;
}

#endif

