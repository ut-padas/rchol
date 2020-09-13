#ifndef pcg_hpp
#define pcg_hpp

#include "sparse.hpp"

class pcg{
public:
  pcg(const SparseCSR A, const std::vector<double> &b, double tol, int maxit,
      const SparseCSR G, std::vector<double> &x, double &relres, int &itr);
private:
  void iteration(const SpMat*, const double*, SpMat*);

};


#endif

