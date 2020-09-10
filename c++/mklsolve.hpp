#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "spcol.h"
#include <typeinfo>
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sys/resource.h>
#include <string>
#include <sstream>
#include <future>
#include <thread>
#include <vector>
#define MKL_INT size_t

#include "mkl_spblas.h"
#include "mkl.h"
#include "mkl_types.h"


struct Sparse_storage_input {
    std::vector<size_t> *colPtr; 
    std::vector<size_t> *rowIdx; 
    std::vector<double> *val;
};


struct Sparse_storage_output {
    size_t *colPtr; 
    size_t *rowIdx; 
    double *val;
    size_t N;
};

typedef sparse_matrix_t SpMat;

void random_factorization_interface(Sparse_storage_input *input, Sparse_storage_output *output);
void create_sparse(const Sparse_storage_output *output, SpMat &mat);




class CG { 

public:

  CG(int numItr, double eps, size_t problem_size);
  CG(int nitr, double tol, size_t problem_size, const double *soln);

  void solve(const SpMat *A, const double *b, SpMat *lap);
  
  void print_results() const;

private:
  void matrix_vector_product(const SpMat *A, const double *b, double *q);
  void random_precond_solve(SpMat *lap, const double *b, double *x);

private:
  int maxSteps;
  double tolerance;
  double residual;
  size_t ps;
};










