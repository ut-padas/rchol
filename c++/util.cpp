#include "util.hpp"

SparseCSR laplace_3d(int n) {
  std::vector<size_t> rowPtr, colIdx;
  std::vector<double> val;
  laplace_3d(n, rowPtr, colIdx, val);
  SparseCSR A(rowPtr, colIdx, val);
  return A;
}

