#include "spcol.h"


void sp_copy(const gsl_spmatrix *a, gsl_spmatrix *b);



/* set the appropriate row and value at idx */
void set_element(gsl_spmatrix *m, size_t idx, size_t row, double data)
{

    assert(idx < m->nz);
    m->i[idx] = row;
    m->data[idx] = data;
}





void gsl_spmatrix_scale(gsl_spmatrix *m, const double x)
{
  size_t i;

  for (i = 0; i < m->nz; ++i)
    m->data[i] *= x;

} /* gsl_spmatrix_scale() */

/* add a and b and put result in b, override what b has */
void gsl_spmatrix_add(const gsl_spmatrix *a, gsl_spmatrix *b)
{
    /* declare a temporary vector column THAT IS NOT MALLOCED(avoid overhead when running in parallel) */

    gsl_spmatrix temp;
    gsl_spmatrix *c = &temp;
    c->size1 = b->size1;
    c->size2 = 1;

    size_t i_array[ a->nz + 4 + b->nz] = {0};
    double data_array[ a->nz + 4 + b->nz] = {0};
    size_t p_array[2] = {0};
    c->i = i_array;
    c->data = data_array;
    c->p = p_array;

    c->nzmax = a->nz + 4 + b->nz;
    c->nz = 0;


    size_t size0 = a->nz;
    size_t size1 = b->nz;

    size_t *row = a->i;
    double *data = a->data;

    size_t *otherrow = b->i;
    double *otherdata = b->data;
    
    size_t *updaterow = c->i;
    double *updatedata = c->data;
    

    // index of this set
    size_t track = 0;
    // index of other set
    size_t trackOther = 0;
    // index of update
    size_t updateidx = 0;
    // while both list has not finished looking through all elements
    while (track < size0 && trackOther < size1)
    {
        // if smaller, move to next one
        if (row[track] < otherrow[trackOther])
        {
            updaterow[updateidx] = row[track];
            updatedata[updateidx] = data[track];
            track++;
        }
        // same logic as previous
        else if (row[track] > otherrow[trackOther])
        {
            updaterow[updateidx] = otherrow[trackOther];
            updatedata[updateidx] = otherdata[trackOther];
            trackOther++;
        }
            // if the same, move up one for both
        else
        {
            updaterow[updateidx] = row[track];
            updatedata[updateidx] = data[track] + otherdata[trackOther];
            track++;
            trackOther++;
        }
        updateidx++;
        c->nz = c->nz + 1;
    }
    // add the rest
    for (int i = track; i < size0; i++)
    {
        updaterow[updateidx] = row[i];
        updatedata[updateidx] = data[i];
        updateidx++;
        c->nz = c->nz + 1;
    }
        
    for (int i = trackOther; i < size1; i++)
    {
        updaterow[updateidx] = otherrow[i];
        updatedata[updateidx] = otherdata[i];
        updateidx++;
        c->nz = c->nz + 1;
    }
    c->p[1] = c->nz;
    sp_copy(c, b);

}

/* copy the content from a to b */
void sp_copy(const gsl_spmatrix *a, gsl_spmatrix *b)
{
    assert(a->nz >= b->nz);
    if(a->nz > b->nzmax)
    {
        spallocate(b, a->nz, b->nzmax);
    }
        

    for(size_t k = 0; k < a->nz; ++k)
    {
        b->i[k] = a->i[k];
        b->data[k] = a->data[k];
    }
    b->nz = a->nz;
    b->p[1] = a->nz;
}


void gsl_spmatrix_set(gsl_spmatrix *m, size_t row, size_t col, double val)
{
    if(col != 0)
        throw std::invalid_argument("only support sparse with 1 column\n");
    if(row >= m->size1)
    {
        throw std::invalid_argument("row out of bound\n");
    }
        
    m->nz = m->nz + 1;
    size_t nzmax = m->nzmax;
    size_t nz = m->nz;
    // allocate new space
    if(nz > nzmax)
    {
        spallocate(m, nz, nzmax);
    }
    // set new data
    m->i[nz - 1] = row;
    m->data[nz - 1] = val;
    m->p[1] = nz;
    
}

// spallocate
void spallocate(gsl_spmatrix *m, size_t nz, size_t nzmax)
{
    // allocate new space
    size_t allocate = std::max(nz, nzmax * 2);
    //allocate = std::min(allocate, m->size1);

    // set nzmax
    m->nzmax = allocate;
    // reallocate
    size_t *tempi = (size_t *)realloc(m->i, allocate * sizeof (size_t));
    double *tempdata = (double *)realloc(m->data, allocate * sizeof (double));
    assert(tempi);
    assert(tempdata);
    m->i = tempi;
    m->data = tempdata;
}

// create spcol
gsl_spmatrix * gsl_spmatrix_alloc_nzmax(size_t n1, size_t n2, size_t nzmax)
{
    if(nzmax < 1)
        throw std::invalid_argument("nzmax needs to be at least 1\n");
    if(n2 != 1)
        throw std::invalid_argument("only support sparse with 1 column\n");
    gsl_spmatrix *ret = (gsl_spmatrix *)malloc(sizeof (gsl_spmatrix));
    ret->size1 = n1;
    ret->size2 = n2;
    ret->i = (size_t *)calloc(nzmax + 1, sizeof (size_t));
    ret->data = (double *)calloc(nzmax + 1, sizeof(double));
    ret->p = (size_t *)calloc(n2 + 1, sizeof(size_t));
    ret->nzmax = nzmax + 1;
    ret->nz = 0;

    return ret;
}

// free stuff
void gsl_spmatrix_free(gsl_spmatrix *m)
{
    free(m->i);
    free(m->data);
    free(m->p);
    free(m);
}



