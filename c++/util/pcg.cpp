#include "pcg.hpp"

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

  this->N = A.N;
  this->tolerance = tol;
  this->maxSteps = maxit;

  SpMat Amat, Gmat;
  create_sparse(A.N, A.rowPtr, A.colIdx, A.val, Amat);
  create_sparse(G.N, G.rowPtr, G.colIdx, G.val, Gmat);

  mkl_sparse_optimize(Amat);
  mkl_sparse_optimize(Gmat);
  
  // matrix descr for A
  MDA.type = SPARSE_MATRIX_TYPE_GENERAL;
  //MDA.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
  //MDA.mode = SPARSE_FILL_MODE_UPPER;
  //MDA.diag = SPARSE_DIAG_NON_UNIT;

  // matrix descr for G
  MDG.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  MDG.mode = SPARSE_FILL_MODE_UPPER;
  MDG.diag = SPARSE_DIAG_NON_UNIT;

  Timer t; t.start();
  this->iteration(&Amat, b.data(), &Gmat, x, relres, itr);
  t.stop(); t_pcg = t.elapsed();

  mkl_sparse_destroy(Amat);
  mkl_sparse_destroy(Gmat);
}

  
void pcg::create_sparse(size_t N, size_t *cpt, size_t *rpt, double *datapt, SpMat &mat) {
    mkl_sparse_d_create_csr(&mat, SPARSE_INDEX_BASE_ZERO, N, N, cpt, cpt+1, rpt, datapt);
}

void pcg::iteration(const SpMat *Amat, const double *b, SpMat *Gmat,
    std::vector<double> &x, double &relres, int &itr) 
{
    // set initial
    x.resize(N, 0);

    // residual
    double *r = new double[N]();
    double *z = new double[N]();
    double *p = new double[N]();    
    double *q = new double[N](); // q = A * p

    Timer t; t.start();

    copy(b, r);
    precond_solve(Gmat, r, z);
    copy(z, p);
    
    int k = 0; // iteration count
    double a1, a2, rz, nr;
    double nb = norm(b);
    double err = nb * tolerance;
    while (k < maxSteps) {
        
      matvec(Amat, p, q);
      
      rz = dot(r, z);
      a1 = rz / dot(p, q);

      axpy(a1, p, x.data());
      axpy(-a1, q, r);

      nr = norm(r);
      if (nr < err) break;

      precond_solve(Gmat, r, z);

      a2 = dot(r, z) / rz;
      xpay(z, a2, p);
      
      k++;
    }

    relres = nr / nb;
    itr = k;
    t.stop(); t_itr = t.elapsed();

    // free memory
    delete[] r, z, p, q;
}

void pcg::copy(const double *src, double *des) {
    cblas_dcopy(N, src, 1, des, 1);
}

void pcg::axpy(double a, double *x, double *y) {
    cblas_daxpy(N, a, x, 1, y, 1);
}

void pcg::xpay(double *x, double a, double *y) {
    for (int i=0; i<N; i++) {
      y[i] = x[i] + a*y[i];
    }
}

double pcg::norm(const double *r) {
    return cblas_dnrm2(N, r, 1);
}

double pcg::dot(double *a, double *b) {
  return cblas_ddot(N, a, 1, b, 1);
}

void pcg::matvec(const SpMat *A, const double *b, double *q)
{
    timer.start();
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, *A, MDA, b, 0, q);
    timer.stop(); t_matvec += timer.elapsed();
}

void pcg::matrix_vector_product(const SpMat *A, const double *b, double *q)
{
    timer.start();
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, *A, MDA, b, 0, q);
    timer.stop(); t_matvec += timer.elapsed();
}

void pcg::precond_solve(SpMat *Gmat, const double *r, double *x)
{
#if 0
    // lower solve
    timer.start();
    mkl_sparse_d_trsv(SPARSE_OPERATION_TRANSPOSE, 1, *Gmat, MDG, r, x);
    timer.stop(); t_lower_solve += timer.elapsed();
#else
    // lower solve
    timer.start();
    this->copy(r, x);
    //this->lower_solve(x);
    this->lower_solve(x, 0, std::log2(nThreads), 0, S.size()-1, 0, nThreads);
    timer.stop(); t_lower_solve += timer.elapsed();
#endif

    // upper triangular solve
#if 0 
    // include '-lmkl_intel_ilp64' in linking flag
    timer.start();
    mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1, *Gmat, MDG, temp, x);
    timer.stop(); t_upper_solve += timer.elapsed();
#else

    timer.start();
    this->upper_solve(x, 0, std::log2(nThreads), 0, S.size()-1, 0, nThreads);
    timer.stop(); t_upper_solve += timer.elapsed();
#endif
}

/*
void pcg::lower_solve(double *b) {
  for (int c=0; c<G.N; c++) {
    assert(G.colIdx[G.rowPtr[c]] == c);
    b[c] /= G.val[ G.rowPtr[c] ];
    for (int i=G.rowPtr[c]+1; i<G.rowPtr[c+1]; i++) {
      int    r = G.colIdx[i];
      double v = G.val[i];
      b[r] -= b[c] * v;
    }
  }
}
*/

void pcg::lower_solve(double *b, int depth, int target, 
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
        int  C0 = S.at(start);
        int  C1 = S.at(start + total_size);
        for (int c = C0; c < C1; c++)
        {  
            assert(c == G.colIdx[ G.rowPtr[c] ]);
            b[c] /= G.val[ G.rowPtr[c] ];
            for (int i = G.rowPtr[c]+1; i < G.rowPtr[c+1]; i++) 
            {
                int    r = G.colIdx[i];
                double v = G.val[i];
                b[r] -= b[c] * v;
                assert(r > c);
            }
        }
        auto time_e = std::chrono::steady_clock::now();
        auto elaNed = time_e - time_s;
        int  cpu_num = sched_getcpu();
        std::cout << "depth: " << depth 
          << " thread " << std::this_thread::get_id() 
          << " cpu: " << cpu_num 
          << " time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elaNed).count() 
          << "\n";
    }
    else
    {
        /* separator portion */
        auto time_s = std::chrono::steady_clock::now();
        int  C0 = S.at(start + total_size - 1);
        int  C1 = S.at(start + total_size);
        for (int c = C0; c < C1; c++)
        {  
            assert(c == G.colIdx[ G.rowPtr[c] ]);
            b[c] /= G.val[ G.rowPtr[c] ];
            for (int i = G.rowPtr[c]+1; i < G.rowPtr[c+1]; i++) 
            {
                int    r = G.colIdx[i];
                double v = G.val[i];
                b[r] -= b[c] * v;
                assert(r > c);
            }
        }
        auto time_e = std::chrono::steady_clock::now();
        auto elaNed = time_e - time_s;
        
        int cpu_num = sched_getcpu();
        std::cout << "depth(separator): " << depth 
          << " thread " << std::this_thread::get_id() 
          << " cpu: " << cpu_num  
          << " time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elaNed).count() 
          << " length: " << S.at(start + total_size) - S.at(start + total_size - 1) 
          << "\n";

        /* recursive call */
        int core_id = (core_begin + core_end) / 2;
        
#if 0
        this->lower_solve(b, depth + 1, target, (total_size - 1) / 2 + start, (total_size - 1) / 2,
            core_id, core_end);
        
#elif 1   
        auto left = std::async(std::launch::async, &pcg::lower_solve, this,
            b, depth + 1, target, (total_size - 1) / 2 + start, (total_size - 1) / 2, 
            core_id, core_end);

#else
        std::thread t(&pcg::lower_solve, this,
            b, depth + 1, target, (total_size - 1) / 2 + start, (total_size - 1) / 2, 
            core_id, core_end);

#endif
        //auto right = std::async(std::launch::async, &pcg::lower_solve, this,
        //    b, depth + 1, target, start, (total_size - 1) / 2, 
        //    core_id, core_end);

        this->lower_solve(b, depth + 1, target, start, (total_size - 1) / 2, 
            core_begin, core_id);
    
        //t.join();
    }
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
        auto elaNed = time_e - time_s;
        int  cpu_num = sched_getcpu();
        std::cout << "depth: " << depth 
          << " thread " << std::this_thread::get_id() 
          << " cpu: " << cpu_num 
          << " time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elaNed).count() 
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
        auto elaNed = time_e - time_s;
        
        int cpu_num = sched_getcpu();
        std::cout << "depth(separator): " << depth 
          << " thread " << std::this_thread::get_id() 
          << " cpu: " << cpu_num  
          << " time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elaNed).count() 
          << " length: " << S.at(start + total_size) - S.at(start + total_size - 1) 
          << "\n";

        /* recursive call */
        int core_id = (core_begin + core_end) / 2;
        
#if 0
        this->upper_solve(b, depth + 1, target, (total_size - 1) / 2 + start, (total_size - 1) / 2,
            core_id, core_end);
        
#elif 0   
        auto left = std::async(std::launch::async, &pcg::upper_solve, this,
            b, depth + 1, target, (total_size - 1) / 2 + start, (total_size - 1) / 2, 
            core_id, core_end);

#else
        std::thread t(&pcg::upper_solve, this,
            b, depth + 1, target, (total_size - 1) / 2 + start, (total_size - 1) / 2, 
            core_id, core_end);

#endif
        //auto right = std::async(std::launch::async, &pcg::upper_solve, this,
        //    b, depth + 1, target, start, (total_size - 1) / 2, 
        //    core_id, core_end);

        this->upper_solve(b, depth + 1, target, start, (total_size - 1) / 2, 
            core_begin, core_id);
    
        t.join();
    }
}

pcg::~pcg() {
  std::cout<<"\n-----------------------"
    <<"\nPCG: "<<t_pcg
    <<"\niteration: "<<t_itr
    <<"\n\tmatvec: "<<t_matvec
    <<"\n\tlower solve: "<<t_lower_solve
    <<"\n\tupper solve: "<<t_upper_solve
    <<"\n\trest: "<<t_itr-t_matvec-t_lower_solve-t_upper_solve
    <<"\n-----------------------\n"
    <<std::endl;
}

