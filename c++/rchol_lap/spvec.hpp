#ifndef sparse_vector_hpp
#define sparse_vector_hpp

#include <assert.h> 
#include <stdlib.h>
#include <stdexcept>


template <typename IT, typename FT>
class spvec {
  public:
    void set_element(IT idx, IT row, FT data);
    void scale(FT x);
    void set(IT row, IT col, FT val);
    void alloc_nzmax(IT n1, IT n2, IT nzmax);
    void clear();

  private:
    void spallocate(IT nz, IT nzmax);

  public:
    IT size1;
    IT size2;
    IT *i;   // row
    FT *data;
    IT *p;   // col
    IT nzmax;
    IT nz;
    IT split;
};


/* set the appropriate row and value at idx */
template <typename IT, typename FT>
void spvec<IT, FT>::set_element(IT idx, IT row, FT data)
{
    assert(idx < this->nz);
    this->i[idx] = row;
    this->data[idx] = data;
}


template <typename IT, typename FT>
void spvec<IT, FT>::scale(FT x)
{
  for (IT i = 0; i < this->nz; ++i)
    this->data[i] *= x;
} 


template <typename IT, typename FT>
void spvec<IT, FT>::set(IT row, IT col, FT val)
{
    if(col != 0)
        throw std::invalid_argument("only support sparse with 1 column\n");
    if(row >= this->size1)
        throw std::invalid_argument("row out of bound\n");
        
    this->nz += 1;
    IT nzmax = this->nzmax;
    IT nz = this->nz;
    // allocate new space
    if(nz > nzmax)
    {
        spallocate(nz, nzmax);
    }
    // set new data
    this->i[nz - 1] = row;
    this->data[nz - 1] = val;
    this->p[1] = nz;
    
}

// spallocate
template <typename IT, typename FT>
void spvec<IT, FT>::spallocate(IT nz, IT nzmax)
{
    // allocate new space
    IT allocate = std::max(nz, nzmax * 2);
    //allocate = std::min(allocate, this->size1);

    // set nzmax
    this->nzmax = allocate;
    // reallocate
    IT *tempi = (IT *)realloc(this->i, allocate * sizeof (IT));
    FT *tempdata = (FT *)realloc(this->data, allocate * sizeof (FT));
    assert(tempi);
    assert(tempdata);
    this->i = tempi;
    this->data = tempdata;
}

// create spcol
template <typename IT, typename FT>
void spvec<IT, FT>::alloc_nzmax(IT n1, IT n2, IT nzmax)
{
    if(nzmax < 1)
        throw std::invalid_argument("nzmax needs to be at least 1\n");
    if(n2 != 1)
        throw std::invalid_argument("only support sparse with 1 column\n");

    this->size1 = n1;
    this->size2 = n2;
    this->i = (IT *)calloc(nzmax + 1, sizeof (IT));
    this->data = (FT *)calloc(nzmax + 1, sizeof(FT));
    this->p = (IT *)calloc(n2 + 1, sizeof(IT));
    this->nzmax = nzmax + 1;
    this->nz = 0;
}

// free stuff
template <typename IT, typename FT>
void spvec<IT, FT>::clear()
{
    free(this->i);
    free(this->data);
    free(this->p);
}


#endif
