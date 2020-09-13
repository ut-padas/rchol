#ifndef rchol_lap_hpp
#define rchol_lap_hpp


#include <vector>
#include "spcol.h"


void random_factorization(Sparse_storage_input *input, Sparse_storage_output *output, std::vector<size_t> &result_idx, int thread);


#endif

