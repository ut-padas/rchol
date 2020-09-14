#include "pcg.hpp"

#define MKL_INT size_t
#include "mkl_spblas.h"
#include "mkl.h"
#include "mkl_types.h"

typedef sparse_matrix_t SpMat;

#include <iostream>
#include <chrono>


pcg::pcg(const SparseCSR A, const std::vector<double> &b, double tol, int maxit,
    const SparseCSR G, std::vector<double> &x, double &relres, int &itr) {

  this->ps = A.N;
  this->tolerance = tol;
  this->maxSteps = maxit;

  SpMat Amat, Gmat;
  create_sparse(A.N, A.rowPtr, A.colIdx, A.val, Amat);
  create_sparse(G.N, G.rowPtr, G.colIdx, G.val, Gmat);
  this->iteration(&Amat, b.data(), &Gmat, x, relres, itr);

  mkl_sparse_destroy(Amat);
  mkl_sparse_destroy(Gmat);
}

  
void pcg::create_sparse(size_t N, size_t *cpt, size_t *rpt, double *datapt, SpMat &mat) {

    size_t *pointerB = new size_t[N + 1]();
    size_t *pointerE = new size_t[N + 1]();
    size_t *update_intend_rpt = new size_t[cpt[N]]();
    double *update_intend_datapt = new double[cpt[N]]();
    for(size_t i = 0; i < N; i++)
    {
        size_t start = cpt[i];
        size_t last = cpt[i + 1];
        pointerB[i] = start;
        for (size_t j = start; j < last; j++)
        {
            update_intend_datapt[j] = datapt[j];
            update_intend_rpt[j] = rpt[j];
        }
        pointerE[i] = last;    

    }
    
    mkl_sparse_d_create_csr(&mat, SPARSE_INDEX_BASE_ZERO, N, N, pointerB, pointerE, update_intend_rpt, update_intend_datapt);

    //delete pointerB, pointerE, update_intend_rpt, update_intend_datapt;
}


void pcg::iteration(const SpMat *A, const double *b, SpMat *lap) 
{
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    auto elapsed = end - start;


    start = std::chrono::steady_clock::now();
    // set initial
    double *x = new double[ps]();
    

    // residual
    double *r = new double[ps]();
    cblas_dcopy(ps, b, 1, r, 1);
    // iterations
    int n_iters = 0;


    // used for storing previous residual and preconditioner applied to r
    double *prev_r = new double[ps]();
    double *prev_cond = new double[ps]();
    double *p = new double[ps]();
    double *temp = new double[ps]();
    double *q = new double[ps]();
    while(cblas_dnrm2(ps, r, 1) > cblas_dnrm2(ps, b, 1) * tolerance && n_iters < this->maxSteps)
    {
        
        this->precond_solve(lap, r, temp);

        if (n_iters == 0)
        {
            cblas_dcopy(ps, temp, 1, p, 1);
        }
        else
        {
            double d1 = cblas_ddot(ps, r, 1, temp, 1);
            double d2 = cblas_ddot(ps, prev_r, 1, prev_cond, 1);
            cblas_dscal(ps, d1 / d2, p, 1);
            cblas_daxpy(ps, 1, temp, 1, p, 1);
        }

        
        this->matrix_vector_product(A, p, q);
        double d1 = cblas_ddot(ps, p, 1, r, 1);
        double d2 = cblas_ddot(ps, p, 1, q, 1);
        double alpha = d1 / d2;
        cblas_daxpy(ps, alpha, p, 1, x, 1);

        cblas_dcopy(ps, r, 1, prev_r, 1);
        cblas_dcopy(ps, temp, 1, prev_cond, 1);
        cblas_daxpy(ps, -alpha, q, 1, r, 1);

        n_iters++;
        std::cout << "current iters: " << n_iters << " current residual: " << cblas_dnrm2(ps, r, 1) / cblas_dnrm2(ps, b, 1) << "\n";
    }
    end = std::chrono::steady_clock::now();
    elapsed += end - start;
    std::cout << "total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " \n";
    this->matrix_vector_product(A, x, q);
    cblas_daxpy(ps, -1, b, 1, q, 1);
    std::cout << "pcg reached a relative residual of " << cblas_dnrm2(ps, q, 1) / cblas_dnrm2(ps, b, 1) << " after " << n_iters << " iterations\n";
    delete x, r, prev_r, prev_cond, p, temp, q;
}


void pcg::matrix_vector_product(const SpMat *A, const double *b, double *q)
{
    matrix_descr des;
    des.type = SPARSE_MATRIX_TYPE_GENERAL;
    //des.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
    //des.mode = SPARSE_FILL_MODE_UPPER;
    //des.diag = SPARSE_DIAG_NON_UNIT;
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, *A, des, b, 0, q);
}


void pcg::precond_solve(SpMat *lap, const double *b, double *ret)
{
    // lower solve
    size_t N = ps;
    double *x = new double[N]();
    matrix_descr des;
    des.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    des.mode = SPARSE_FILL_MODE_UPPER;
    des.diag = SPARSE_DIAG_NON_UNIT;

    mkl_sparse_d_trsv(SPARSE_OPERATION_TRANSPOSE, 1, *lap, des, b, x);

 
    // upper triangular solve
    mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1, *lap, des, x, ret);


    delete[] x;
}

