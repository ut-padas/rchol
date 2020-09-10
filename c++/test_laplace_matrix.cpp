#include <iostream>
#include <string>
#include <cstring>
#include "laplace_3d.hpp"
#include "mklsolve.hpp"

template <typename T>
void print(const T &x, std::string name) {
  std::cout<<name<<":"<<std::endl;
  for (size_t i=0; i<x.size(); i++)
    std::cout<<x[i]<<" ";
  std::cout<<std::endl;
}

int main(int argc, char *argv[]) {
  int n = 3;
  for (int i=0; i<argc; i++) {
    if (!strcmp(argv[i], "-n"))
      n = atoi(argv[i+1]);
  }
  std::vector<size_t> rowPtr;
  std::vector<size_t> colIdx;
  std::vector<double> val;
  laplace_3d(n, rowPtr, colIdx, val);
  //print(rowPtr, "rowPtr");
  //print(colIdx, "colIdx");
  //print(val, "val");
  
  // upper triangular csr form
  std::vector<size_t> rowPtrU;
  std::vector<size_t> colIdxU;
  std::vector<double> valU;
  triu_csr(rowPtr, colIdx, val, rowPtrU, colIdxU, valU);

  // print(rowPtrU, "rowPtrU");
  // print(colIdxU, "colIdxU");
  // print(valU, "valU");

  // convert to a laplacian
  std::vector<size_t> rowPtrL;
  std::vector<size_t> colIdxL;
  std::vector<double> valL;
  convert_to_laplace(rowPtrU, colIdxU, 
    valU, rowPtrL, colIdxL, valL, rowPtr, colIdx, val);

  // print(rowPtrL, "rowPtrL");
  // print(colIdxL, "colIdxL");
  // print(valL, "valL");
  

  // change edge values to positive and begin factorization
  change_to_positive_edge(valL);
  Sparse_storage_input input;
  Sparse_storage_output outputL;
  input.colPtr = &rowPtrL;
  input.rowIdx = &colIdxL;
  input.val = &valL;

  random_factorization_interface(&input, &outputL);

  // test with pcg
  
  SpMat A;
  SpMat L;
  Sparse_storage_output outputA;
  outputA.colPtr = rowPtrU.data();
  outputA.rowIdx = colIdxU.data();
  outputA.val = valU.data();
  outputA.N = rowPtrU.size() - 1;


  create_sparse(&outputL, L);
  create_sparse(&outputA, A);


  CG cg(100, 1e-10, rowPtrU.size() - 1);
  std::mt19937 gen(std::random_device{}());
  double *b = new double[rowPtrU.size() - 1]();
  for(size_t i = 0; i < rowPtrU.size() - 1; i++)
  {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    b[i] = dis(gen);
  }
     
  cg.solve(&A, b, &L);

  mkl_sparse_destroy(A);
  mkl_sparse_destroy(L);

  


  
  
  return 0;
}

