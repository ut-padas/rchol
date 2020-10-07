#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include "mat.h"
#include <typeinfo>
#include "matrix.h"
#include <random>
#include "mex.h"
#include <chrono>
#include <algorithm>
#include <iostream>
#include <fstream>
#include "metis.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    const mxArray *arg0 = prhs[0];
    idx_t N = mxGetN(arg0);

    size_t *cpt1 = mxGetJc(arg0);
    size_t *rpt1 = mxGetIr(arg0);

    size_t nz = cpt1[N];

    idx_t *cpt = (idx_t*)calloc(N + 1, sizeof(idx_t));
    idx_t *rpt = (idx_t*)calloc(nz, sizeof(idx_t));

    for(idx_t i = 0; i < N+1; i++)
    {
        cpt[i] = cpt1[i];
 
    }
    for(idx_t i = 0; i < nz; i++)
    {
        rpt[i] = rpt1[i];
       
    }
  
    
    // vertex separator
    idx_t *vsep = (idx_t*)calloc(N, sizeof(idx_t));
    idx_t sepsize = 10;
    mxArray *separator = mxCreateNumericMatrix(1, N, mxDOUBLE_CLASS, mxREAL);
    double *separatorpt = (double *)mxGetData(separator);
    METIS_ComputeVertexSeparator(&N, cpt, rpt,
                                 NULL, NULL, &sepsize, vsep);
    for(idx_t i = 0; i < N; i++)
    {
        separatorpt[i] = (vsep[i]);
    }
    plhs[0] = separator;
    
    free(cpt);
    free(rpt);
    free(vsep);
    
}
