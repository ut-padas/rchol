#include "find_separator.hpp"
#include "metis.h"
#include "util.hpp"

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


size_t * metis_separator(const SparseCSR &A)
{

    idx_t N = A.N;
    size_t *cpt1 = A.rowPtr;
    size_t *rpt1 = A.colIdx;
    
    size_t nz = cpt1[N];
  
    
    idx_t *cpt = (idx_t*)calloc(N + 1, sizeof(idx_t));
    idx_t *rpt = (idx_t*)calloc(nz, sizeof(idx_t));

    //std::unique_ptr<idx_t> cpt(new idx_t(N + 1));
    //std::unique_ptr<idx_t> rpt(new idx_t(nz));

    for(idx_t i = 0; i < N+1; i++)
    {
        cpt[i] = (idx_t)(cpt1[i]);
    }

    for(idx_t i = 0; i < nz; i++)
    {
        rpt[i] = (idx_t)(rpt1[i]);
    }
    
    // vertex separator

    idx_t *vsep = (idx_t*)calloc(N, sizeof(idx_t));
    //std::unique_ptr<idx_t> vsep(new idx_t(N));
    idx_t sepsize = 1;
    METIS_ComputeVertexSeparator(&N, cpt, rpt, NULL, NULL, &sepsize, vsep);
    
    
    size_t *separatorpt = (size_t*)calloc(N, sizeof(size_t));
    for(idx_t i = 0; i < N; i++)
    {
        separatorpt[i] = (size_t)(vsep[i]);
    }

    free(cpt);
    free(rpt);
    free(vsep);
    
    return separatorpt;
}




Separator_info find_separator(const SparseCSR &A, int depth, int target){

  if(depth == target)
  {
    
    std::vector<size_t> *val = new std::vector<size_t>();
    val->push_back(A.N);
    std::vector<size_t> *p = new std::vector<size_t>();
    for(size_t i = 0; i < A.N; i++)
      p->push_back(i);
    return Separator_info(p, val, NULL);

  }
  else if (A.N <= 1)
  {

    throw std::invalid_argument( "too many threads requested" );

  }
  else
  {
    size_t *sep_idx = metis_separator(A);
    Partition_info par = determine_parition(sep_idx, A.N);
    SparseCSR newleft = get_submatrix(*(par.zero_partition), sep_idx, A);
    SparseCSR newright = get_submatrix(*(par.one_partition), sep_idx, A);
    
    Separator_info linfo = find_separator(newleft, depth + 1, target);
    Separator_info rinfo = find_separator(newright, depth + 1, target);

    std::vector<size_t> *val = new std::vector<size_t>();
    val->reserve(linfo.val->size() + rinfo.val->size() + 1);
    val->insert( val->end(), linfo.val->begin(), linfo.val->end() );
    val->insert( val->end(), rinfo.val->begin(), rinfo.val->end() );
    val->push_back(par.second_partition->size());

    std::vector<size_t> *p = new std::vector<size_t>();
    std::vector<size_t> l = reorder(*(par.zero_partition), *(linfo.p));
    std::vector<size_t> r = reorder(*(par.one_partition), *(rinfo.p));
    p->reserve(l.size() + r.size() + par.second_partition->size());
    p->insert(p->end(), l.begin(), l.end());
    p->insert(p->end(), r.begin(), r.end());
    p->insert(p->end(), par.second_partition->begin(), par.second_partition->end());


    delete linfo.p;
    delete linfo.val;
    delete rinfo.p;
    delete rinfo.val;
    delete par.zero_partition;
    delete par.one_partition;
    delete par.second_partition;

    return Separator_info(p, val, NULL);
  }
}

