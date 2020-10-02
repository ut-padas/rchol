
#include "util.hpp"
#include "laplace_3d.hpp"
#include <iostream>

SparseCSR laplace_3d(int n) {
  std::vector<size_t> rowPtr, colIdx;
  std::vector<double> val;
  laplace_3d(n, rowPtr, colIdx, val);
  SparseCSR A(rowPtr, colIdx, val, false);
  return A;
}


// permute the matrix
void reorder(const SparseCSR &A, std::vector<size_t> &rowPtr, std::vector<size_t> &colIdx, 
  std::vector<double> &val, const std::vector<size_t> &permutation){
  std::vector<size_t> transp;
  transp.resize(permutation.size());
  size_t N = permutation.size();
  for(size_t i = 0; i < N; i++)
  {
    transp[permutation[i]] = i;
  }

  rowPtr.push_back(0);
  for(size_t i = 0; i < N; i++)
  {
    size_t first = A.rowPtr[permutation[i]];
    size_t last = A.rowPtr[permutation[i] + 1];
    rowPtr.push_back(rowPtr[rowPtr.size()- 1] + last - first);
    Rearrange arrange[last - first];
    // permute elements in each row
    for(size_t j = first; j < last; j++)
    {
      arrange[j - first].row = transp[A.colIdx[j]];
      arrange[j - first].data = A.val[j];
      
    }
    // sort elements
    std::sort(arrange, arrange + last - first);

    for(size_t j = first; j < last; j++)
    {
      colIdx.push_back(arrange[j - first].row);
      val.push_back(arrange[j - first].data);
    }
  } 
}

void reorder(const SparseCSR &A, const std::vector<size_t> &P, SparseCSR &B) {
  std::vector<size_t> rowPtr;
  std::vector<size_t> colIdx;
  std::vector<double> val;
  reorder(A, rowPtr, colIdx, val, P);
  B.init(rowPtr, colIdx, val);
}


