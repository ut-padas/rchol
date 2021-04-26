#include "sparse.hpp"

int main(int argc, char *argv[]) {
  SparseCSR A;
  A.read_mkt_file("/home1/06108/chaochen/matrices/example.mkt");
  print(A, "test");
  return 0;
}
