#ifndef sparse_hpp
#define sparse_hpp

#include <vector>


class SparseCSR {
public:
  SparseCSR();
  SparseCSR(const std::vector<size_t>&, const std::vector<size_t>&, const std::vector<double>&);

  size_t size() const;
  size_t nnz() const;

  ~SparseCSR();

private:
  size_t N = 0;
  size_t *rowPtr;
  size_t *colIdx;
  double *val;
};

#endif
