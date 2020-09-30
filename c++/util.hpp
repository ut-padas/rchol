#ifndef UTIL_HPP
#define UTIL_HPP

#include <cassert>
#include <vector>
#include <random>
#include <stdexcept>
#include "sparse.hpp"
#include <algorithm>
#include "metis.h"
#include <iostream>


struct Separator_info {

  std::vector<size_t> *p; 
  std::vector<size_t> *val;
  std::vector<size_t> *separator;
  Separator_info(std::vector<size_t> *input_p, std::vector<size_t> *input_val, std::vector<size_t> *input_separator)
  {
      p = input_p;
      val = input_val;
      separator = input_separator;
  }
  
};

struct Partition_info {

  std::vector<size_t> *zero_partition; 
  std::vector<size_t> *one_partition;
  std::vector<size_t> *second_partition;
  Partition_info(std::vector<size_t> *left_partition, std::vector<size_t> *right_partition, std::vector<size_t> *separator)
  {
      zero_partition = left_partition;
      one_partition = right_partition;
      second_partition = separator;
  }
  
};


struct Rearrange {
    size_t row;
    double data;
    Rearrange(size_t arg0, double arg1)
    {
        row = arg0;
        data = arg1;
    }
    Rearrange()
    {
        
    }
    bool operator<(Rearrange other) const
    {
        return row < other.row;
    }
};


SparseCSR laplace_3d(int);

Partition_info determine_parition(size_t *sep_idx, size_t N);

SparseCSR get_submatrix(std::vector<size_t> &par, size_t *sep_idx, const SparseCSR &A);

void reorder(const SparseCSR &A, const std::vector<size_t> &permutation, SparseCSR &);

void reorder(const SparseCSR &A, std::vector<size_t> &rowPtr, std::vector<size_t> &colIdx, 
      std::vector<double> &val, const std::vector<size_t> &permutation);

size_t * metis_separator(const SparseCSR &A);

Separator_info find_separator(const SparseCSR &A, int depth, int target);


template <typename T>
void print(const T &x, std::string name) {
  std::cout<<name<<":"<<std::endl;
  for (size_t i=0; i<x.size(); i++)
    std::cout<<x[i]<<" ";
  std::cout<<std::endl;
}


template <typename T>
void rand(std::vector<T> &x) {
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<T> dist(0.0, 1.0);
  for (size_t i=0; i<x.size(); i++)
    x[i] = dist(gen);
}


// input: sparse matrix in csr format
// output: the upper triangular submatrix (including diagonal) in csr format
template <typename T>
void triu_csr
(const std::vector<size_t> &rowPtrA, const std::vector<size_t> &colIdxA, const std::vector<T> &valA,
 std::vector<size_t> &rowPtrU, std::vector<size_t> &colIdxU, std::vector<T> &valU) {
  // get matrix size and nnz
  size_t n = rowPtrA.size()-1;
  size_t nnz = rowPtrA[n];
  // allocate memory
  rowPtrU.resize(n+1, 0);
  colIdxU.reserve((nnz-n)/2);
  valU.reserve((nnz-n)/2);
  // get upper triangular
  for (size_t r=0; r<n; r++) {
    size_t start = rowPtrA[r];
    size_t end = rowPtrA[r+1];
    for (size_t i=start; i<end; i++) {
      if (r <= colIdxA[i]) {
        colIdxU.push_back(colIdxA[i]);
        valU.push_back(valA[i]);
      }
    }
    rowPtrU[r+1] = valU.size();
  }
}

template <typename T>
void change_to_positive_edge(std::vector<T> &val) {
  for(size_t i = 0; i < val.size(); i++)
  {
    val[i] *= -1;
  }
}

template <typename T>
void convert_to_laplace(const std::vector<size_t> &rowPtr, const std::vector<size_t> &colIdx, 
    const std::vector<T> &val, std::vector<size_t> &rowPtrL, std::vector<size_t> &colIdxL, std::vector<T> &valL, 
    std::vector<size_t> &rowPtrA, std::vector<size_t> &colIdxA, std::vector<T> &valA) {
      
  // get matrix size and nnz
  size_t n = rowPtr.size() - 1;
  size_t nnz = rowPtr[n];
  // allocate memory
  rowPtrL.resize(n + 2, 0);
  colIdxL.reserve(nnz + n + 3);
  valL.reserve(nnz + n + 3);
  T overall = 0.0;
  for (size_t r = 0; r < n; r++) {
    size_t start = rowPtr[r];
    size_t end = rowPtr[r + 1];
    
    for (size_t i = start; i < end; i++) {
      colIdxL.push_back(colIdx[i]);
      valL.push_back(val[i]);
      
    }

    
    // add the term that makes it a laplace
    start = rowPtrA[r];
    end = rowPtrA[r + 1];
    T sum = 0.0;
    for (size_t i = start; i < end; i++) {
      sum += valA[i];
    }

    
    if(std::abs(sum) > 1e-09)
    {
      colIdxL.push_back(n);
      valL.push_back(-sum);
    }
    rowPtrL[r+1] = valL.size();
    overall -= sum;
  }

  if(std::abs(overall) > 1e-09)
  {
    colIdxL.push_back(n);
    valL.push_back(-overall);
  }
  rowPtrL[n + 1] = valL.size();

}





template <typename T>
std::vector<T> reorder(std::vector<T> &vec, std::vector<size_t> &p)
{
  std::vector<T> ret;
  ret.reserve(p.size());
  for(size_t i = 0; i < vec.size(); i++)
  {
    ret.push_back(vec[p[i]]);
  }
  return ret;
}




#endif
