#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include "sparse.hpp" // define CSR sparse matrix type
#include "rchol_parallel.hpp"
#include "util.hpp"
#include "pcg.hpp"
#include "timer.hpp"


int main(int argc, char *argv[]) {
  int n = 64; // DoF in every dimension
  int threads = 1;
  int nitr = 10;
  for (int i=0; i<argc; i++) {
    if (!strcmp(argv[i], "-n"))
      n = atoi(argv[i+1]);
    if (!strcmp(argv[i], "-t"))
      threads = atoi(argv[i+1]);
    if (!strcmp(argv[i], "-i"))
      nitr = atoi(argv[i+1]);
  }
  std::cout<<std::setprecision(3);
 
  // SDDM matrix from 3D constant Poisson equation
  SparseCSR A;
  A = laplace_3d(n); // n x n x n grid
  //A.read_mkt_file("/home1/06108/chaochen/matrices/parabolic_fem.mtx");
  //A.read_mkt_file("/home1/06108/chaochen/matrices/ecology2.mtx");
  //A.read_mkt_file("/home1/06108/chaochen/matrices/apache2.mtx");
  //A.read_mkt_file("/home1/06108/chaochen/matrices/G3_circuit.mtx");
  //A.read_mkt_file("/home1/06108/chaochen/matrices/vc_laplace_128_1e5.mtx");
  //A.read_mkt_file("/home1/06108/chaochen/matrices/aniso_laplace_128.mtx");

  // random RHS
  int N = A.size();
  std::vector<double> b(N); 
  rand(b);

  // compute preconditioner (multithread) and solve
  Timer t; t.start();
  SparseCSR G;
  std::vector<size_t> P;
  std::vector<int> S;
  rchol(A, G, P, S, threads);
  t.stop();
  std::cout<<"Setup time: "<<t.elapsed()<<std::endl;
  std::cout<<"Fill-in ratio: "<<2.*G.nnz()/A.nnz()<<std::endl;

  // solve the reordered problem with PCG
  SparseCSR Aperm; reorder(A, P, Aperm);
  std::vector<double> bperm; reorder(b, P, bperm);
    
  double tol = 1e-10;
  double relres;
  int itr;
  std::vector<double> x;
  
  t.start();
  pcg(Aperm, bperm, S, threads, tol, nitr, G, x, relres, itr);
  t.stop();

  std::cout<<"Solve time: "<<t.elapsed()<<std::endl;
  std::cout<<"# CG iterations: "<<itr<<std::endl;
  std::cout<<"Relative residual: "<<relres<<std::endl;


  return 0;
}
