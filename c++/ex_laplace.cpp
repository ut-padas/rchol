#include <iostream>
#include <string>
#include <cstring>
#include "sparse.hpp" // define CSR sparse matrix type
#include "rchol.hpp"
#include "util.hpp"


int main(int argc, char *argv[]) {
  int n = 3; // DoF in every dimension
  for (int i=0; i<argc; i++) {
    if (!strcmp(argv[i], "-n"))
      n = atoi(argv[i+1]);
  }
 
  // SDDM matrix from 3D constant Poisson equation
  SparseCSR A;
  A = laplace_3d(n); // n x n x n grid

  // random RHS
  int N = A.size();
  std::vector<double> b(N); 
  rand(b);

  // compute preconditioner
  SparseCSR G;
  rchol(A, G);
  std::cout<<"Fill-in ratio: "<<2*G.nnz()/A.nnz()<<std::endl;

  // solve with PCG
  double tol = 1e-6;
  int maxit = 200;
  double relres;
  int itr;
  std::vector<double> x;
  pcg(A, b, tol, maxit, G, x, relres, itr);
  std::cout<<"# CG iterations: "<<itr<<std::endl;
  std::cout<<"Relative residual: "<<relres<<std::endl;

/*
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

  //random_factorization_interface(&input, &outputL);

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

  cg.solve(&A, b.data(), &L);

  mkl_sparse_destroy(A);
  mkl_sparse_destroy(L);
*/
  


  
  
  return 0;
}

