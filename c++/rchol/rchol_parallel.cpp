#include "rchol_parallel.hpp"
#include "partition_and_order.hpp"
#include "rchol_lap.hpp"
#include "timer.hpp"
#include "util.hpp"

#include <math.h> 
#include <iostream>
#include <fstream>
#include <sstream>


// multithread version
void rchol(const SparseCSR &A, SparseCSR &G, std::vector<size_t> &permutation, 
    std::vector<int> &S, int threads, std::string filename) {

  if(threads <= 0)
    throw std::invalid_argument( "thread number should be positive" );

  if((threads & (threads - 1)) != 0)
    throw std::invalid_argument( "thread number should be a power of 2" );

  Timer t; t.start();
  std::ifstream ifile(filename.c_str());
  if (ifile.is_open()) {
    permutation.clear(); permutation.reserve(A.size());
    S.clear(); S.reserve(2*threads);
    std::string line;
    int x;
    {
      std::getline(ifile, line);
      std::istringstream ss(line);
      while (ss >> x) {
        permutation.push_back(x);
      }
      assert(permutation.size()==A.size());
    }
    {
      std::getline(ifile, line);
      std::istringstream ss(line);
      while (ss >> x) {
        S.push_back(x);
      }
      assert(S.size()==unsigned(2*threads));
    }
    ifile.close();
  } else {
    partition_and_order(A, threads, permutation, S);
    
    assert(permutation.size()==A.size());
    assert(S.size()==unsigned(2*threads));
    std::ofstream ofile(filename.c_str());
    for (auto x : permutation)
      ofile << x << " ";
    ofile << std::endl;
    for (auto x : S)
      ofile << x << " ";
    ofile.close();
  }
  t.stop(); std::cout<<"find separator: "<<t.elapsed()<<" s\n";
  
  t.start();
  std::vector<size_t> rowPtr, colIdx;
  std::vector<double> val;
  reorder(A, rowPtr, colIdx, val, permutation);
  t.stop(); std::cout<<"permute matrix: "<<t.elapsed()<<" s\n";


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
