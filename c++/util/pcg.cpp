#include "pcg.hpp"
#include "timer.hpp"

#define MKL_INT size_t
#include "mkl_spblas.h"
#include "mkl.h"
#include "mkl_types.h"

typedef sparse_matrix_t SpMat;

#include <iostream>
#include <chrono>
#include <cassert>
#include <cmath>
#include <future>


pcg::pcg(const SparseCSR &A, const std::vector<double> &b, 
    const std::vector<int> &S, int nt, double tol, int maxit,
    const SparseCSR &G, std::vector<double> &x, double &relres, int &itr) {

  this->G = G;
  this->S = S; this->S.back()--; // remove artifitial vertex
  this->nThreads = nt;

  this->ps = A.N;
  this->tolerance = tol;
  this->maxSteps = maxit;

  SpMat Amat, Gmat;
  create_sparse(A.N, A.rowPtr, A.colIdx, A.val, Amat);
  create_sparse(G.N, G.rowPtr, G.colIdx, G.val, Gmat);

  // matrix descr for G
  MDG.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  MDG.mode = SPARSE_FILL_MODE_UPPER;
  MDG.diag = SPARSE_DIAG_NON_UNIT;

  Timer t; t.start();
  this->iteration(&Amat, b.data(), &Gmat, x, relres, itr);
  t.stop(); t_itr = t.elapsed();

  mkl_sparse_destroy(Amat);
  mkl_sparse_destroy(Gmat);
}

  
void pcg::create_sparse(size_t N, size_t *cpt, size_t *rpt, double *datapt, SpMat &mat) {

  /*
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
    */

    mkl_sparse_d_create_csr(&mat, SPARSE_INDEX_BASE_ZERO, N, N, cpt, cpt+1, rpt, datapt);

    //delete[] pointerB, pointerE, update_intend_rpt, update_intend_datapt;
}


void pcg::iteration(const SpMat *A, const double *b, SpMat *lap,
    std::vector<double> &x, double &relres, int &itr) 
{
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    auto elapsed = end - start;


    start = std::chrono::steady_clock::now();
    // set initial
    x.resize(ps);

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
    double *work = new double[ps](); // working memory for preconditioner solve
    while(cblas_dnrm2(ps, r, 1) > cblas_dnrm2(ps, b, 1) * tolerance && n_iters < this->maxSteps)
    {
        
        this->precond_solve(lap, r, temp, work);

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
        cblas_daxpy(ps, alpha, p, 1, x.data(), 1);

        cblas_dcopy(ps, r, 1, prev_r, 1);
        cblas_dcopy(ps, temp, 1, prev_cond, 1);
        cblas_daxpy(ps, -alpha, q, 1, r, 1);

        n_iters++;
        //std::cout << "current iters: " << n_iters << " current residual: " << cblas_dnrm2(ps, r, 1) / cblas_dnrm2(ps, b, 1) << "\n";
    }
    end = std::chrono::steady_clock::now();
    elapsed += end - start;
    //std::cout << "total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " \n";
    this->matrix_vector_product(A, x.data(), q);
    cblas_daxpy(ps, -1, b, 1, q, 1);
    relres = cblas_dnrm2(ps, q, 1) / cblas_dnrm2(ps, b, 1);
    itr = n_iters;
    //std::cout << "pcg reached a relative residual of " << cblas_dnrm2(ps, q, 1) / cblas_dnrm2(ps, b, 1) << " after " << n_iters << " iterations\n";
    delete[] r;
    delete[] prev_r;
    delete[] prev_cond;
    delete[] p;
    delete[] temp;
    delete[] q;
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


void pcg::precond_solve(SpMat *lap, const double *r, double *x, double *y)
{
    // lower solve
    Timer t; t.start();
    mkl_sparse_d_trsv(SPARSE_OPERATION_TRANSPOSE, 1, *lap, MDG, r, y);
    t.stop(); t_lower_solve += t.elapsed();

    // upper triangular solve
    //mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1, *lap, MDG, y, x);
    
    /*
    for (int r=G.N-1; r>=0; r--) {
      assert(G.colIdx[ G.rowPtr[r] ] == r);
      x[r] = y[r];
      for (int i=G.rowPtr[r]+1; i<G.rowPtr[r+1]; i++) {
        int c = G.colIdx[i];
        double v = G.val[i];
        x[r] -= x[c] * v;
        assert(c > r);
      }
      x[r] /= G.val[ G.rowPtr[r] ];
    }
    */

    t.start();
    this->upper_solve(y, 0, std::log2(nThreads), 0, S.size()-1, 0, nThreads);
    t.stop(); t_upper_solve += t.elapsed();

    for (int i=0; i<G.N; i++) x[i] = y[i];
}

void pcg::upper_solve(double *b, int depth, int target, 
    int start, int total_size, int core_begin, int core_end)
{

    // pin to a core
    if(sched_getcpu() != core_begin)
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_begin, &cpuset);
        sched_setaffinity(0, sizeof(cpuset), &cpuset);
    }

    /* base case */
    if(target == depth)
    {
        auto time_s = std::chrono::steady_clock::now();
        int  R0 = S.at(start);
        int  R1 = S.at(start + total_size);
        for (int r = R1-1; r >= R0; r--)
        {  
            assert(r == G.colIdx[ G.rowPtr[r] ]);
            for (int i = G.rowPtr[r]+1; i < G.rowPtr[r+1]; i++) 
            {
                int    c = G.colIdx[i];
                double v = G.val[i];
                b[r] -= b[c] * v;
                assert(c > r);
            }
            b[r] /= G.val[ G.rowPtr[r] ];
        }
        auto time_e = std::chrono::steady_clock::now();
        auto elapsed = time_e - time_s;
        int  cpu_num = sched_getcpu();
        std::cout << "depth: " << depth 
          << " thread " << std::this_thread::get_id() 
          << " cpu: " << cpu_num 
          << " time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() 
          << "\n";
    }
    else
    {
        /* separator portion */
        auto time_s = std::chrono::steady_clock::now();
        int  R0 = S.at(start + total_size - 1);
        int  R1 = S.at(start + total_size);
        for (size_t r = R1-1; r >= R0; r--)
        {
            assert(r == G.colIdx[ G.rowPtr[r] ]);
            for (int i = G.rowPtr[r]+1; i < G.rowPtr[r+1]; i++) 
            {
                int    c = G.colIdx[i];
                double v = G.val[i];
                b[r] -= b[c] * v;
                assert(c > r);
            }
            b[r] /= G.val[ G.rowPtr[r] ];
        }
        auto time_e = std::chrono::steady_clock::now();
        auto elapsed = time_e - time_s;
        
        int cpu_num = sched_getcpu();
        std::cout << "depth(separator): " << depth 
          << " thread " << std::this_thread::get_id() 
          << " cpu: " << cpu_num  
          << " time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() 
          << " length: " << S.at(start + total_size) - S.at(start + total_size - 1) 
          << "\n";

        /* recursive call */
        int core_id = (core_begin + core_end) / 2;
        /*
        this->upper_solve(b, depth + 1, target, (total_size - 1) / 2 + start, (total_size - 1) / 2, 
            core_id, core_end);
        
        this->upper_solve(b, depth + 1, target, start, (total_size - 1) / 2, 
            core_begin, core_id);
        */
        
        std::async(std::launch::async, &pcg::upper_solve, this,
            b, depth + 1, target, (total_size - 1) / 2 + start, (total_size - 1) / 2, 
            core_id, core_end);
        
        this->upper_solve(b, depth + 1, target, start, (total_size - 1) / 2, 
            core_begin, core_id);
    }
}

pcg::~pcg() {
  std::cout<<"\n-----------------------"
    <<"\nall iterations: "<<t_itr
    <<"\nlower solve: "<<t_lower_solve
    <<"\nupper solve: "<<t_upper_solve
    <<"\n-----------------------\n"
    <<std::endl;
}

