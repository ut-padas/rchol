#ifndef LAPLACE_3D_HPP
#define LAPLACE_3D_HPP

#include <cassert>
#include <vector>

template <typename T>
void laplace_3d(int n, std::vector<int> &rowPtr, std::vector<int> &colIdx, 
    std::vector<T> &val) {
  size_t N = n*n*n;
  size_t nnz = 0;
  rowPtr.reserve(N+1); rowPtr.push_back(nnz);
  colIdx.reserve(7*N);
  val.reserve(7*N);

  size_t n2 = n*n;
  for (int i=0; i<n; i++) {
    for (int j=0; j<n; j++) {
      for (int k=0; k<n; k++) {
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


// input: sparse matrix in csr format
// output: the upper triangular submatrix (excluding diagonal) in csr format
template <typename T>
void triu_csr
(const std::vector<int> &rowPtrA, const std::vector<int> &colIdxA, const std::vector<T> &valA,
 std::vector<int> &rowPtrU, std::vector<int> &colIdxU, std::vector<T> &valU) {
  // get matrix size and nnz
  int n = rowPtrA.size()-1;
  int nnz = rowPtrA[n];
  // allocate memory
  rowPtrU.resize(n+1, 0);
  colIdxU.reserve((nnz-n)/2);
  valU.reserve((nnz-n)/2);
  // get upper triangular
  for (int r=0; r<n; r++) {
    int start = rowPtrA[r];
    int end = rowPtrA[r+1];
    for (int i=start; i<end; i++) {
      if (r <= colIdxA[i]) {
        colIdxU.push_back(colIdxA[i]);
        valU.push_back(valA[i]);
      }
    }
    rowPtrU[r+1] = valU.size();
  }
}


#endif
