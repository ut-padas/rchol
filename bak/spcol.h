#include <assert.h> 

/* used to store sparse columns only */
typedef struct
{
  size_t size1;
  size_t size2;
  size_t *i;   // row
  double *data;
  size_t *p;   // col
  size_t nzmax;
  size_t nz;
  size_t split;
} gsl_spmatrix;



void gsl_spmatrix_scale(gsl_spmatrix *m, const double x);
void gsl_spmatrix_add(const gsl_spmatrix *a, gsl_spmatrix *b);
void gsl_spmatrix_set(gsl_spmatrix *m, size_t row, size_t col, double val);
gsl_spmatrix * gsl_spmatrix_alloc_nzmax(size_t n1, size_t n2, size_t nzmax);
void gsl_spmatrix_free(gsl_spmatrix *m);
void spallocate(gsl_spmatrix *m, size_t nz, size_t nzmax);
void set_element(gsl_spmatrix *m, size_t idx, size_t row, double data);
