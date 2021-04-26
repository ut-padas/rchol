#include "sparse.hpp"
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>


SparseCSR::SparseCSR() {
  N = 0;
  rowPtr = 0;
  colIdx = 0;
  val = 0;
  ownMemory = false;
}

SparseCSR::SparseCSR(const std::vector<size_t> &rowPtrA, const std::vector<size_t> &colIdxA,
    const std::vector<double>& valA, bool mem) {
  this->init(rowPtrA, colIdxA, valA, mem);
}

void SparseCSR::init(const std::vector<size_t> &rowPtrA, const std::vector<size_t> &colIdxA,
    const std::vector<double>& valA, bool mem) {
  assert(this->N == 0 && "empty matrix");

  this->N = rowPtrA.size()-1;
  this->rowPtr = new size_t[N+1];

  size_t nnz = rowPtrA[N];
  this->colIdx = new size_t[nnz];
  this->val = new double[nnz];
  std::copy(rowPtrA.begin(), rowPtrA.end(), rowPtr);
  std::copy(colIdxA.begin(), colIdxA.end(), colIdx);
  std::copy(valA.begin(), valA.end(), val);

  this->ownMemory = mem;
}

void SparseCSR::read_mkt_file(std::string filename) {
  std::ifstream matFile(filename.c_str());
  assert(matFile.is_open() && "Can't open matrix file");

  std::string comment;
  std::getline(matFile, comment);

  // symmetric or not
  bool symmetric = false;
  if (comment.find("symmetric") != std::string::npos)
    symmetric = true;
  assert(symmetric == true && "Unsymmetric matrices NOT supported");

  // skip comment lines that start with %
  while (comment[0]=='%') {
    std::getline(matFile, comment);
  }
  int nrow, ncol, nline;
  std::stringstream ss(comment);
  ss >> nrow >> ncol >> nline;
  assert(nrow==ncol && "Non-square matrix");
  assert(nrow>0 && nline>0);

  // read a lower triangular in CSC
  std::vector<int> colPtrL(nrow+1,0), rowIdxL; 
  std::vector<double> valL; 
  colPtrL.reserve(N+1);
  rowIdxL.reserve(nline);
  valL.reserve(nline);

  int i, j, line;
  double x;
  for (line=0; line<nline; line++) {
    matFile >> i >> j >> x;
    colPtrL[j]++;
    rowIdxL.push_back(i);
    valL.push_back(x);
    assert(i>=j && "lower triangular");
  }
  matFile.close();
  for (int i=0; i<nrow+1; i++)
    colPtrL[i+1] += colPtrL[i];
  assert(rowIdxL.size() == colPtrL.back());

  int nnz = 2*nline-nrow;
  this->N = nrow;
  this->ownMemory = true;
  this->rowPtr = new size_t[N+1]();
  this->colIdx = new size_t[nnz];
  this->val = new double[nnz];

  // get nnz per row
  for (int c=1; c<N+1; c++) {
    for (int i=colPtrL[c-1]; i<colPtrL[c]; i++) {
      int r = rowIdxL[i]; rowPtr[ r ]++;
      if (r > c)          rowPtr[ c ]++;
    }
  }
  for (int i=0; i<N; i++)
    rowPtr[i+1] += rowPtr[i];

  //std::cout<<"N="<<N<<std::endl;
  //for (int i=0; i<N+1; i++)
    //std::cout<<rowPtr[i]<<" ";
  assert(rowPtr[N] == nnz);

  // fill the matrix
  std::vector<int> count(N, 0);
  for (int c=0; c<N; c++) {
    for (int i=colPtrL[c]; i<colPtrL[c+1]; i++) {
      int r = rowIdxL[i] - 1;
      int idx = rowPtr[r] + count[r];
    
      colIdx[idx] = c;
      val[idx] = valL[i];
      count[r]++; 

      if (symmetric && c!=r) {
        idx = rowPtr[c] + count[c];
        colIdx[idx] = r;
        val[idx] = valL[i];
        count[c]++;
      }
    }
  }

  std::cout<<"[Read sparse matrix] # rows: "<<nrow
     <<", # cols: "<<ncol<<", # nonezero: "<<nnz
     <<std::endl;
}

SparseCSR::SparseCSR(const SparseCSR &A) {
  this->N = A.size();
  this->rowPtr = new size_t[N+1];

  size_t nnz = A.nnz();
  this->colIdx = new size_t[nnz];
  this->val = new double[nnz];
  std::copy(A.rowPtr, A.rowPtr+N+1, this->rowPtr);
  std::copy(A.colIdx, A.colIdx+nnz, this->colIdx);
  std::copy(A.val, A.val+nnz, this->val);

  this->ownMemory = true;
}

size_t SparseCSR::size() const {
  return N;
}

size_t SparseCSR::nnz() const {
  return rowPtr[N];
}

SparseCSR::~SparseCSR() {
  if (N>0 && ownMemory) {
    delete[] rowPtr;
    delete[] colIdx;
    delete[] val;
    rowPtr = 0;
    colIdx = 0;
    val = 0;
    N = 0;
  }
}


void print(const SparseCSR &A, std::string name) {
  std::cout<<name<<" rowPtr:\n";
  for (size_t i=0; i<=A.N; i++)
    std::cout<<A.rowPtr[i]<<" ";
  std::cout<<std::endl;
  std::cout<<name<<" colIdx:\n";
  for (size_t i=0; i<A.rowPtr[A.N]; i++)
    std::cout<<A.colIdx[i]<<" ";
  std::cout<<std::endl;
  std::cout<<name<<" value:\n";
  for (size_t i=0; i<A.rowPtr[A.N]; i++)
    std::cout<<A.val[i]<<" ";
  std::cout<<std::endl;
}

