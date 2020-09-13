#include "util.hpp"
#include "laplace_3d.hpp"

SparseCSR laplace_3d(int n) {
  std::vector<size_t> rowPtr, colIdx;
  std::vector<double> val;
  laplace_3d(n, rowPtr, colIdx, val);
  SparseCSR A(rowPtr, colIdx, val, false);
  return A;
}

