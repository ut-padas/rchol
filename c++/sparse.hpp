#ifndef sparse_hpp
#define sparse_hpp

#include <vector>
#include <string>


class SparseCSR {
public:
  SparseCSR();
  SparseCSR(const std::vector<size_t>&, const std::vector<size_t>&, const std::vector<double>&, 
      bool mem=true);
  SparseCSR(const SparseCSR &); // deep copy

  void init(const std::vector<size_t>&, const std::vector<size_t>&, const std::vector<double>&,
      bool mem=true);

  size_t size() const;
  size_t nnz() const;

  ~SparseCSR();

public:
  size_t N = 0;
  size_t *rowPtr;
  size_t *colIdx;
  double *val;
  bool ownMemory;
};


void print(const SparseCSR&, std::string);


#endif
