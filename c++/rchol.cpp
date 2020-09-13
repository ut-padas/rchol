#include "rchol.hpp"

void rchol(Sparse_storage_input *input, Sparse_storage_output *output)
{
    std::vector<size_t> result_idx;
    result_idx.push_back(0);
    result_idx.push_back(input->colPtr->size() - 1);
    random_factorization(input, output, result_idx, 1);
}


void rchol(const SparseCSR A, SparseCSR &G) {

}

