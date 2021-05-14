#include "partition_and_order.hpp"
#include "metis.h"
#include "util.hpp"
#include "timer.hpp"

#include <thread>

extern "C" {
#include "amd.h"
}

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


Partition_info determine_parition(size_t *sep_idx, size_t N){
  std::vector<size_t> *left = new std::vector<size_t>();
  std::vector<size_t> *right = new std::vector<size_t>();
  std::vector<size_t> *sep_part = new std::vector<size_t>();
  for(size_t i = 0; i < N; i++)
  {
    if(sep_idx[i] == 0)
      left->push_back(i);
    else if(sep_idx[i] == 1)
      right->push_back(i);
    else
      sep_part->push_back(i);
    
  }
  return Partition_info(left, right, sep_part);
}


SparseCSR get_submatrix(std::vector<size_t> &par, size_t *sep_idx, const SparseCSR &A){
   
  std::vector<size_t> transp;
  transp.resize(A.N, 0);
  
  for(size_t i = 0; i < par.size(); i++)
  {
    transp[par[i]] = i;
  }
  
  std::vector<size_t> rowPtr;
  std::vector<size_t> colIdx;
  std::vector<double> val;
  rowPtr.push_back(0);
  for(size_t i = 0; i < par.size(); i++)
  {
    size_t first = A.rowPtr[par[i]];
    size_t last = A.rowPtr[par[i] + 1];
    rowPtr.push_back(rowPtr[rowPtr.size()- 1]);
    size_t tempsize = rowPtr.size() - 1;
    for(size_t j = first; j < last; j++)
    {
      if(sep_idx[A.colIdx[j]] == sep_idx[par[0]])
      {
        colIdx.push_back(transp[A.colIdx[j]]);
        val.push_back(A.val[j]);
        rowPtr[tempsize]++;
      }
    }
  }

  SparseCSR ret(rowPtr, colIdx, val, true);
  return ret;
}


size_t * metis_separator(const SparseCSR &A) {

    idx_t nnz  = A.rowPtr[A.N] - A.N; // no diagonal
    idx_t *vtx = (idx_t*)calloc(A.N + 1, sizeof(idx_t));
    idx_t *adj = (idx_t*)calloc(nnz, sizeof(idx_t));
    idx_t *sep = (idx_t*)calloc(A.N, sizeof(idx_t));

    vtx[0] = 0;
    idx_t  k = 0;
    for (size_t i=0; i<A.N; i++) {
      for (size_t j=A.rowPtr[i]; j<A.rowPtr[i+1]; j++) {
        if (i != A.colIdx[j]) {
          adj[k++] = A.colIdx[j];
        }
      }
      vtx[i+1] = A.rowPtr[i+1] - i-1; // no diagonal
    }

    
    idx_t N = A.N;
    idx_t sepsize = 1;
    METIS_ComputeVertexSeparator(&N, vtx, adj, NULL, NULL, &sepsize, sep);
    

    size_t *separatorpt = (size_t*)calloc(A.N, sizeof(size_t));
    for(size_t i = 0; i < A.N; i++) {
        separatorpt[i] = (size_t)(sep[i]);
    }

    free(vtx);
    free(adj);
    free(sep);
    
    return separatorpt;
}


void find_separator(const SparseCSR &A, std::vector<size_t> &order, std::vector<int> &partition, 
    int nThreads, int core=0) {

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  sched_setaffinity(0, sizeof(cpuset), &cpuset);
  //std::cout<<"threads: "<<nThreads<<", core: "<<core<<std::endl;

  if (nThreads==1) {
   
    Timer t; t.start();
    partition.push_back(A.N);
    
    // AMD ordering
    order.resize(A.N);
    amd_l_order(A.N, (long *)A.rowPtr, (long *)A.colIdx, (long *)order.data(),
        (double*) NULL, (double*) NULL);
    t.stop(); //std::cout<<"Leaf node time: "<<t.elapsed()<<" s\n";
    
  } else {

    Timer t; t.start();
    size_t *sep_idx = metis_separator(A);
    t.stop(); //std::cout<<"Metis: "<<t.elapsed()<<" s\n";
    
    t.start();
    Partition_info par = determine_parition(sep_idx, A.N);
    SparseCSR newleft = get_submatrix(*(par.zero_partition), sep_idx, A);
    SparseCSR newright = get_submatrix(*(par.one_partition), sep_idx, A);
    t.stop(); //std::cout<<"Submatrix: "<<t.elapsed()<<" s\n";
    
    std::vector<size_t> lperm, rperm;
    std::vector<int> lpart, rpart;

    std::thread T(find_separator, newleft, std::ref(lperm), std::ref(lpart), nThreads/2, core);

    //find_separator(newleft, lperm, lpart, nThreads/2, core);
    find_separator(newright, rperm, rpart, nThreads/2, core+nThreads/2);

    T.join();

    t.start();
    std::vector<size_t> lorder = reorder(*(par.zero_partition), lperm);
    std::vector<size_t> rorder = reorder(*(par.one_partition), rperm);

    order.clear();
    order.reserve(lorder.size()+rorder.size()+par.second_partition->size());
    order.insert(order.end(), lorder.begin(), lorder.end());
    order.insert(order.end(), rorder.begin(), rorder.end());
    order.insert(order.end(), par.second_partition->begin(), par.second_partition->end());
  
    partition.clear();
    partition.reserve(lpart.size()+rpart.size()+1);
    partition.insert(partition.end(), lpart.begin(), lpart.end());
    partition.insert(partition.end(), rpart.begin(), rpart.end());
    partition.push_back(par.second_partition->size());


    delete par.zero_partition;
    delete par.one_partition;
    delete par.second_partition;
    t.stop(); //std::cout<<"After recursion: "<<t.elapsed()<<" s\n";

  }
}


void partition_and_order(const SparseCSR &A, int nThreads, 
    std::vector<size_t> &order, std::vector<int> &partition) {

  std::vector<int> part;
  find_separator(A, order, part, nThreads);

  partition.resize(nThreads*2, 0);
  std::partial_sum(part.begin(), part.end(), partition.begin()+1);
  partition.back()++;
}


