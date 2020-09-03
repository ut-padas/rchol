#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include "mex.h"
#include "spcol.c"
#include <set>
#include "mat.h"
#include <typeinfo>
#include "matrix.h"
#include <map>
#include <random>
#include <chrono>
//#include "omp.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include "metis.h"

int main()
{

    MATFile *mat = matOpen("c++.mat", "r");
    mxArray *arg0 = matGetVariable(mat, "graphtest");

    // idx_t options[METIS_NOPTIONS];
    
    // METIS_SetDefaultOptions(options);
    // options[METIS_OPTION_DBGLVL] = METIS_DBG_SEPINFO;
    

    
    idx_t N = mxGetN(arg0);
    
    
    
    idx_t ncon = 1;
    idx_t num_part = 2;
    idx_t info = 0;
    
    idx_t *part = (idx_t*)calloc(N, sizeof(idx_t));
    idx_t *ipart = (idx_t*)calloc(N, sizeof(idx_t));

    
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
    
//    idx_t p1[] = { 0, 2, 5, 8, 11, 13, 16, 20, 24, 28, 31, 33, 36, 39, 42, 44 };
//    idx_t p2[] = { 1, 5, 0, 2, 6, 1, 3, 7, 2, 4, 8, 3, 9, 0, 6, 10, 1, 5, 7, 11, 2, 6, 8, 12, 3, 7, 9, 13, 4, 8, 14, 5, 11, 6, 10, 12, 7, 11, 13, 8, 12, 14, 9, 13 };
//    idx_t K = 15;
    

    // METIS_PartGraphKway(&N, &ncon, (idx_t *)(mxGetJc(arg0)), 
    //               (idx_t *)(mxGetIr(arg0)), NULL, NULL, NULL, 
    //               &num_part, NULL, NULL, NULL, 
    //               &info, part);
    
    METIS_NodeND(&N, cpt, rpt, NULL, NULL, part, ipart);
    
    
    // vertex separator
    idx_t *vsep = (idx_t*)calloc(N, sizeof(idx_t));
    idx_t sepsize = 0;
    METIS_ComputeVertexSeparator(&N, cpt, rpt,
                                 NULL, NULL, &sepsize, vsep);
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    const mxArray *arg0 = prhs[0];
    idx_t N = mxGetN(arg0);


    idx_t ncon = 1;
    idx_t num_part = 2;
    idx_t info = 0;




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

//    idx_t *part = (idx_t*)calloc(N, sizeof(idx_t));
//    idx_t *ipart = (idx_t*)calloc(N, sizeof(idx_t));
//    mxArray *perm = mxCreateNumericMatrix(N, 1, mxDOUBLE_CLASS, mxREAL);
//    double *permpt = (double *)mxGetData(perm);
//
//    METIS_NodeND(&N, cpt, rpt, NULL, NULL, part, ipart);
//
//    for(idx_t i = 0; i < N; i++)
//    {
//        permpt[i] = (part[i] + 1);
//    }
//    plhs[0] = perm;
//
//    free(part);
//    free(ipart);
    
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
