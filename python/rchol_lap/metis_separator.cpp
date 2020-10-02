#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <typeinfo>
#include <map>
#include <random>
#include <chrono>
//#include "omp.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include "metis.h"

// int main()
// {
//     idx_t N = 9;
//     idx_t p1[] = { 0, 5, 10, 15, 20, 25, 30, 35, 40, 49 };
//     idx_t p2[] = { 0, 1, 2, 4, 8, 0, 1, 3, 5, 8, 0, 2, 3, 6, 8, 1, 2, 3, 7, 8, 0, 4, 5, 6, 8, 1, 4, 5, 7, 8, 2, 4, 6, 7, 8, 3, 5,
//  6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8};
//     // idx_t p1[] = { 0, 2, 5, 8, 11, 13, 16, 20, 24, 28, 31, 33, 36, 39, 42, 44 };
//     // idx_t p2[] = { 1, 5, 0, 2, 6, 1, 3, 7, 2, 4, 8, 3, 9, 0, 6, 10, 1, 5, 7, 11, 2, 6, 8, 12, 3, 7, 9, 13, 4, 8, 14, 5, 11, 6, 10, 12, 7, 11, 13, 8, 12, 14, 9, 13 };
//     // idx_t N = 15;
//     //idx_t p2[] = { 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
//     idx_t *vsep = (idx_t*)calloc(N, sizeof(idx_t));
//     idx_t sepsize = 10;
//     METIS_ComputeVertexSeparator(&N, p1, p2,
//                                  NULL, NULL, &sepsize, vsep);
// }



uint64_t * metis_separator(uint64_t length, uint64_t *rpt1, uint64_t *cpt1)
{

    idx_t N = length;
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
    
    
    uint64_t *separatorpt = (uint64_t*)calloc(N, sizeof(uint64_t));
    for(idx_t i = 0; i < N; i++)
    {
        separatorpt[i] = (uint64_t)(vsep[i]);
    }

    free(cpt);
    free(rpt);
    free(vsep);
    
    return separatorpt;
}
