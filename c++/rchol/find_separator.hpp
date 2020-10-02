#ifndef find_separator_hpp
#define find_separator_hpp

#include "../sparse.hpp"


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

size_t * metis_separator(const SparseCSR &A);

Separator_info find_separator(const SparseCSR &A, int depth, int target);

#endif
