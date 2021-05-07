#ifndef pcg_hpp
#define pcg_hpp

#include "sparse.hpp"
#include "timer.hpp"

#define MKL_INT size_t
#include "mkl_spblas.h"
#include "mkl.h"
#include "mkl_types.h"

typedef sparse_matrix_t SpMat;

class pcg{
public:
  pcg(const SparseCSR &A, const std::vector<double> &b, 
      const std::vector<int> &S, int nt, double tol, int maxit,
      const SparseCSR &G, std::vector<double> &x, double &relres, int &itr);

  ~pcg();

private:
  void create_sparse(size_t N, size_t *cpt, size_t *rpt, double *datapt, SpMat &mat);
  void iteration(const SpMat*, const double*, SpMat*, std::vector<double>&, double&, int&);
  void matrix_vector_product(const SpMat *A, const double *b, double *q);
  void matvec(const SpMat *A, const double *b, double *q);
  //void precond_solve(SpMat *lap, const double *b, double *ret, double*);
  void precond_solve(SpMat *lap, const double *b, double *ret);
  void upper_solve(double*, int, int, int, int, int, int);

  //void lower_solve(double*);
  void lower_solve(double*, int, int, int, int, int, int);

private:
  void copy(const double*, double*);
  void axpy(double, double*, double*);
  void xpay(double*, double, double*);
  double dot(double*, double*);
  double norm(const double*);

private:
  int N;
  int maxSteps;
  double tolerance;

  SparseCSR G;
  std::vector<int> S;
  int nThreads;

  matrix_descr MDA;
  matrix_descr MDG;

  Timer  timer;
  double t_pcg = 0;
  double t_itr = 0;
  double t_matvec = 0;
  double t_upper_solve = 0;
  double t_lower_solve = 0;
};


#endif

