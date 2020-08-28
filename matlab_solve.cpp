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
#include "omp.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sys/resource.h>
#include <string>
#include <sstream>
#include <future>
#include <thread>










/* used for random sampling */
struct Sample {
    size_t row;
    double data;
    Sample(size_t arg0, double arg1)
    {
        row = arg0;
        data = arg1;
    }
    Sample()
    {
        
    }
    bool operator<(Sample other) const
    {
        return row < other.row;
    }
};

/* for keeping track of edges in separator and other special use */
struct Edge_info {
    double val;
    size_t row;
    size_t col;
    Edge_info(double arg0, size_t arg1, size_t arg2)
    {
        val = arg0;
        row = arg1;
        col = arg2;
    }
    Edge_info()
    {
        
    }
    bool operator<(Edge_info other) const
    {
        return col < other.col;
    }
};



/* functions set up */
void process_array(const mxArray *arg0, std::vector<size_t> &result_idx, size_t depth, size_t target, std::vector<gsl_spmatrix *> &lap, size_t start, size_t total_size, int core_begin, int core_end); // read in matlab array to gsl sparray
std::vector<Edge_info> & recursive_calculation(std::vector<size_t> &result_idx, size_t depth, std::vector<gsl_spmatrix *> &lap, double *diagpt, size_t start, size_t total_size, size_t target, int core_begin, int core_end);
/* functions for factoring Cholesky */
void cholesky_factorization(std::vector<gsl_spmatrix *> &lap, std::vector<mxArray *> &result, std::vector<size_t> &result_idx);

void linear_update(gsl_spmatrix *b);
/* sampling algorithm */
double random_sampling0(gsl_spmatrix *cur, std::vector<gsl_spmatrix *> &lap, size_t curcol, std::vector<Edge_info> &sep_edge, size_t l_bound, size_t r_bound);
double random_sampling1(gsl_spmatrix *cur, std::vector<gsl_spmatrix *> &lap, size_t curcol, std::vector<Edge_info> &sep_edge, size_t l_bound, size_t r_bound);
double random_sampling2(gsl_spmatrix *cur, std::vector<gsl_spmatrix *> &lap, size_t curcol, std::vector<Edge_info> &sep_edge, size_t l_bound, size_t r_bound);

bool compare (Sample i, Sample j);
/* clean up memory */
void clear_memory(std::vector<gsl_spmatrix *> &lap);
/* used for matlab embedding */


int NUM_THREAD = 0;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    NUM_THREAD = (int)*mxGetPr(prhs[2]);
    cpu_set_t cpuset; 

    //the CPU we whant to use
    int cpu = 0;

    CPU_ZERO(&cpuset);       //clears the cpuset
    CPU_SET( cpu , &cpuset); //set CPU 2 on cpuset
    /*
    * cpu affinity for the calling thread 
    * first parameter is the pid, 0 = calling thread
    * second parameter is the size of your cpuset
    * third param is the cpuset in which your thread will be
    * placed. Each bit represents a CPU
    */
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
    omp_set_num_threads(NUM_THREAD);

    
    const mxArray *arg0 = prhs[0];
    const mxArray *arg1 = prhs[1];



    double *idx_data = (double *)mxGetData(arg1);
    std::vector<size_t> result_idx;
    size_t n = mxGetN(arg1);
    for(size_t i = 0; i < n; i++)
    {
        result_idx.push_back((size_t)(idx_data[i]));
    }

    n = mxGetN(arg0);
    std::vector<gsl_spmatrix *> *lap_val = new std::vector<gsl_spmatrix *>(n);
    std::vector<gsl_spmatrix *> &lap = *lap_val;
    process_array(arg0, result_idx, 1, (size_t)(std::log2(NUM_THREAD) + 1), lap, 0, result_idx.size() - 1, 0, NUM_THREAD);

    
    
    std::vector<mxArray *> result;
    
    auto start = std::chrono::steady_clock::now();
    cholesky_factorization(lap, result, result_idx);
    auto end = std::chrono::steady_clock::now();
    std::cout << "chol time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " \n";
    
    // free memory
    clear_memory(lap);
    // return value
    plhs[0] = result.at(0);
    plhs[1] = result.at(1);
    
}




int main(int argc, char *argv[])
{
    NUM_THREAD = std::stoi(argv[1]);
    cpu_set_t cpuset; 

    //the CPU we whant to use
    int cpu = 0;

    CPU_ZERO(&cpuset);       //clears the cpuset
    CPU_SET( cpu , &cpuset); //set CPU 2 on cpuset
    /*
    * cpu affinity for the calling thread 
    * first parameter is the pid, 0 = calling thread
    * second parameter is the size of your cpuset
    * third param is the cpuset in which your thread will be
    * placed. Each bit represents a CPU
    */
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
    omp_set_num_threads(NUM_THREAD);


    
    MATFile *mat = matOpen("../downloadproblems/give.mat", "r");
    mxArray *arg0 = matGetVariable(mat, "give");
    mat = matOpen("../downloadproblems/result_idx.mat", "r");
    mxArray *arg3 = matGetVariable(mat, "result_idx");
    

    double *idx_data = (double *)mxGetData(arg3);
    std::vector<size_t> result_idx;
    size_t n = mxGetN(arg3);
    for(size_t i = 0; i < n; i++)
    {
        result_idx.push_back((size_t)(idx_data[i]));
    }

    n = mxGetN(arg0);
    std::vector<gsl_spmatrix *> *lap_val = new std::vector<gsl_spmatrix *>(n);
    std::vector<gsl_spmatrix *> &lap = *lap_val;
    process_array(arg0, result_idx, 1, (size_t)(std::log2(NUM_THREAD) + 1), lap, 0, result_idx.size() - 1, 0, NUM_THREAD);
    



	mxDestroyArray(arg0);
	mxDestroyArray(arg3);

    std::vector<mxArray *> result;
    auto start = std::chrono::steady_clock::now();
    cholesky_factorization(lap, result, result_idx);
    auto end = std::chrono::steady_clock::now();
    std::cout << "chol time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "\n";




    // clear memory
    
    clear_memory(lap);
    if (result.size() == 1)
    {
        mxDestroyArray(result.at(0));
    }
    else
    {
        mxDestroyArray(result.at(0));
        mxDestroyArray(result.at(1));
    }
    

}







/* clear memory */
void clear_memory(std::vector<gsl_spmatrix *> &lap)
{
    size_t i;
    // clear lap
    for(i = 0; i < lap.size(); i++)
    {
        gsl_spmatrix_free(lap.at(i));
    }
    
    delete &lap;
    

}

#include<immintrin.h>

bool uni(Sample a, Sample b) 
{ 
    // Checking if both the arguments are same and equal 
    // to 'G' then only they are considered same 
    // and duplicates are removed 
    if (a.row == b.row) { 
        return 1; 
    } else { 
        return 0; 
    } 
} 




std::vector<Edge_info> & recursive_calculation(std::vector<size_t> &result_idx, size_t depth, std::vector<gsl_spmatrix *> &lap, double *diagpt, size_t start, size_t total_size, size_t target, int core_begin, int core_end)
{

    int core_id = (core_begin + core_end) / 2;
    if(sched_getcpu() != core_begin)
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_begin, &cpuset);
        sched_setaffinity(0, sizeof(cpuset), &cpuset);
    }
    
    

    /* base case */
    if(target == depth)
    {
        

        //std::cout << "thread id: " << omp_get_thread_num() << "\n";
		double density = 0;
        int cpu_num = sched_getcpu();
        auto time_s = std::chrono::steady_clock::now();
        auto time_e = std::chrono::steady_clock::now();
        auto elapsed = time_s - time_e;

        std::vector<Edge_info> *pt = new std::vector<Edge_info>();
        std::vector<Edge_info> &sep_edge = *pt;
        
        time_s = std::chrono::steady_clock::now();
        
        for (size_t i = result_idx.at(start); i < result_idx.at(start + total_size); i++)
        {
            
            size_t current = i;
            gsl_spmatrix *b = lap.at(current);
            if(b->nz - b->split > 0)
            {

                linear_update(b);
                
            }
        
            diagpt[current] = random_sampling0(lap.at(current), lap, current, sep_edge, result_idx.at(start), result_idx.at(start + total_size));
            density += lap.at(current)->nz;
        }

        time_e = std::chrono::steady_clock::now();
        elapsed += time_e - time_s;

        std::cout << "depth: " << depth << " thread " << omp_get_thread_num() << " cpu: " << cpu_num << " time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "\n";
        std::cout << "depth: " << depth << " length: " << result_idx.at(start) << " nztotal " << density << " density: " << density / (result_idx.at(start + total_size) - result_idx.at(start)) << "\n";
        std::cout << "depth: " << depth << "\n";
        //std::cout << omp_proc_bind_master << "  " << omp_get_proc_bind << "\n";
        return sep_edge;
    }
    else
    {
        auto time_s = std::chrono::steady_clock::now();
        auto time_e = std::chrono::steady_clock::now();
        auto elapsed = time_s - time_e;

        /* recursive call */
        std::vector<Edge_info> *l_pt;
        std::vector<Edge_info> *r_pt;

        // create new thread
        auto a1 = std::async(std::launch::async, recursive_calculation, std::ref(result_idx), depth + 1, std::ref(lap), diagpt, (total_size - 1) / 2 + start, (total_size - 1) / 2, target, core_id, core_end);
        
        //#pragma omp task shared(lap, r_pt)
        //{
        //    r_pt = &recursive_calculation(result_idx, depth + 1, lap, diagpt, (total_size - 1) / 2 + start, (total_size - 1) / 2, target, core_id, core_end);
        //}
        // run its own job
        l_pt = &recursive_calculation(result_idx, depth + 1, lap, diagpt, start, (total_size - 1) / 2, target, core_begin, core_id);
        //r_pt = &recursive_calculation(result_idx, depth + 1, lap, diagpt, (total_size - 1) / 2 + start, (total_size - 1) / 2, target, core_id, core_end);
        
        /* synchronize at this point */
        //#pragma omp taskwait
        r_pt = &a1.get();

        //auto a1 = std::async(std::launch::async, recursive_calculation, std::ref(result_idx), depth + 1, std::ref(lap), diagpt, start, (total_size - 1) / 2, target, core_id, core_end);
        //r_pt = &recursive_calculation(result_idx, depth + 1, lap, diagpt, (total_size - 1) / 2 + start, (total_size - 1) / 2, target, core_begin, core_id);
        //l_pt = &a1.get();
        
        std::vector<Edge_info> &l_edge = *l_pt;
        std::vector<Edge_info> &r_edge = *r_pt;
        
        /* process edges in separator on the current level */
        std::vector<Edge_info> *pt = new std::vector<Edge_info>();
        std::vector<Edge_info> &sep_edge = *pt;

        size_t l_bound = result_idx.at(start + total_size - 1);
        size_t r_bound = result_idx.at(start + total_size);
        for (size_t i = 0; i < l_edge.size(); i++)
        {

            Edge_info &temp = l_edge.at(i);
            if(temp.col >= l_bound && temp.col < r_bound)
            {
                gsl_spmatrix_set(lap.at(temp.col), temp.row, 0, temp.val);
            }
            else
            {
                sep_edge.push_back(Edge_info(temp.val, temp.row, temp.col));
            }
        }
        
        for (size_t i = 0; i < r_edge.size(); i++)
        {
            
            Edge_info &temp = r_edge.at(i);
            if(temp.col >= l_bound && temp.col < r_bound)
            {
                gsl_spmatrix_set(lap.at(temp.col), temp.row, 0, temp.val);
            }
            else
            {
                sep_edge.push_back(Edge_info(temp.val, temp.row, temp.col));
            }
        }
        delete &l_edge;
        delete &r_edge;

        /* separator portion */
        double density = 0;
        double before_density = 0;
/*
        for (size_t i = result_idx.at(start + total_size - 1); i < result_idx.at(start + total_size); i++)
        {
            
            std::vector<Sample> v2(relation.at(i)->receive);
            for(int j = 0; j < lap.at(i)->nz; j++)
            {
                v2.push_back(Sample(lap.at(i)->i[j], lap.at(i)->data[j]));
            }
            std::sort(v2.begin(), v2.end());
            auto ip = std::unique(v2.begin(), v2.end(), uni); 
            //assert(v2.size() == (ip - v2.begin()));
            before_density += (ip - v2.begin());
        }
*/

		
        time_s = std::chrono::steady_clock::now();
        for (size_t i = result_idx.at(start + total_size - 1); i < result_idx.at(start + total_size); i++)
        {
            
            size_t current = i;
            gsl_spmatrix *b = lap.at(current);
            if(b->nz - b->split > 0)
            {
                
                linear_update(b);
            }

            
            diagpt[current] = random_sampling0(lap.at(current), lap, current, sep_edge, l_bound, r_bound);
            density += lap.at(i)->nz;
        }
        time_e = std::chrono::steady_clock::now();
        elapsed = time_e - time_s;
        int cpu_num = sched_getcpu();
        std::cout << "depth(separator): " << depth << " thread " << omp_get_thread_num() << " cpu: " << cpu_num  << " time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " length: " << result_idx.at(start + total_size) - result_idx.at(start + total_size - 1) << " nztotal " << density << " before: " << before_density / (result_idx.at(start + total_size) - result_idx.at(start + total_size - 1)) << " density: " << density / (result_idx.at(start + total_size) - result_idx.at(start + total_size - 1)) << "\n";
		
        return sep_edge;
    }
    
}

auto start1 = std::chrono::steady_clock::now();
auto end1 = std::chrono::steady_clock::now();
auto elapsed1 = end1 - start1;

/* main routine for Cholesky */
void cholesky_factorization(std::vector<gsl_spmatrix *> &lap, std::vector<mxArray *> &result,  std::vector<size_t> &result_idx)
{
    // calculate nonzeros and create lower triangular matrix
    size_t m = lap.size();
    mxArray *diag = mxCreateNumericMatrix(m, 1, mxDOUBLE_CLASS, mxREAL);
    double *diagpt = (double *)mxGetData(diag);
    
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    auto elapsed = end - start;
    
    std::cout << omp_get_max_threads() << "\n";


    start = std::chrono::steady_clock::now();

    /* recursive call */
    //#pragma omp parallel
    //#pragma omp single
    recursive_calculation(result_idx, 1, lap,
        diagpt, 0, result_idx.size() - 1, (size_t)(std::log2(NUM_THREAD) + 1), 0, NUM_THREAD);
    // recursive_calculation(result_idx, 1, lap,
    //     diagpt, 0, result_idx.size() - 1, 3);
    end = std::chrono::steady_clock::now();
    elapsed = end - start;
    std::cout << "factor time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "\n";
    size_t nzmax = 0;
    size_t edge = 0;
    for (size_t i = 0; i < lap.size(); i++)
    {
        nzmax += lap.at(i)->nz;
	    edge += lap.at(i)->split;
    }
    std::cout<< "nzmax: " << nzmax <<"\n";
    std::cout<< "edge: " << edge <<"\n";
    std::cout << "size of matrix length: " << m - 1 << "\n";
    diagpt[m - 1] = 0;
    result.push_back(diag);

    
    
    
    mxArray *lower = mxCreateSparse(m, m, nzmax + 1, mxREAL);
    result.push_back(lower);
    
    size_t *cpt = mxGetJc(lower);
    size_t *rpt = mxGetIr(lower);
    double *datapt = (double *)mxGetData(lower);
    
    
    //    std::ofstream myfile;
    //    std::ofstream myfile1;
    //    myfile.open ("exampleL.txt");
    //    myfile1.open ("exampleD.txt");
    
    // set up return matrix
    
    size_t counter = 0;
    cpt[0] = 0;
    size_t i, j = 0;
    
    
    for (i = 0; i < m - 1; i++)
    {
        gsl_spmatrix *toCopy = lap.at(i);
        size_t nz = toCopy->nz;
        size_t *irow = toCopy->i;
        double *data = toCopy->data;
        cpt[i + 1] = cpt[i] + nz;
        for (j = counter; j < nz + counter; j++)
        {
            datapt[j] = data[j - counter];
            rpt[j] = irow[j - counter];
            if(rpt[j] != i)
                datapt[j] = datapt[j] * -1;
            //assert(myfile << rpt[j]+1 << " " << i+1 << " " << datapt[j] << "\n");
        }
        //assert(myfile1 << diagpt[i] << "\n");
        //std::cout.flush();
        counter = j;
    }
    // set last element
    datapt[j] = 1.0;
    rpt[j] = m - 1;
    cpt[m] = cpt[m - 1] + 1;
    diagpt[m - 1] = 0.0;
    lap.at(m - 1)->data[0] = 1.0;
    
    //    assert(myfile << m << " " << m << " " << 1.0 << "\n");
    //    assert(myfile1 << 0.0 << "\n");
    //    myfile.close();
    //    myfile1.close();

	/*
    // store it as a matlab file
    MATFile *r1 = matOpen("../downloadproblems/lower.mat", "w7.3");
    MATFile *r2 = matOpen("../downloadproblems/diag.mat", "w7.3");
    int status = matPutVariable(r1, "lower", lower);

    if (status != 0) {
        printf("Error using matPutVariable\n");
        exit(1);
    }  

    status = matPutVariable(r2, "diag", diag);
    if (status != 0) {
        printf("Error using matPutVariable\n");
        exit(1);
    }  

    if (matClose(r1) != 0)
    {
        printf("Error closing file %s\n", r1);
        exit(1);
    }

    if (matClose(r2) != 0)
    {
        printf("Error closing file %s\n", r2);
        exit(1);
    }
 	
    std::cout << "sampling time:" << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed1).count() << "\n";
    */
}


double random_sampling2(gsl_spmatrix *cur, std::vector<gsl_spmatrix *> &lap, size_t curcol, std::vector<Edge_info> &sep_edge, size_t l_bound, size_t r_bound)
{

    double *data = cur->data;
    size_t *row = cur->i;
    double sum = 0.0;
    size_t nz = cur->nz;
    // find sum of column
    for (size_t i = 1; i < nz; i++)
    {
        sum += data[i];
    }
    // run only if at least 3 elements, including diagonal

    
    if (nz > 2)
    {
        
        int size = nz - 1;

        // create sample vector for sampling
        Sample sample[size];
        // cumulative sum
        double cumsum[size];
        double csum = 0.0;
        for (size_t i = 0; i < size; i++)
        {
            sample[i].row = row[i + 1];
            sample[i].data = data[i + 1];
        }

        // sort first based on value
        std::sort(sample, sample + size, compare);
        for (size_t i = 0; i < size; i++)
        {
            csum += sample[i].data;
            cumsum[i] = csum;
        }
        
        
        // sampling
        // random number and edge values
        static thread_local std::mt19937 gen(std::random_device{}());
        static thread_local std::mt19937 gen_discrete;
        int num_sample = (size - 1);
        for (int i = 0; i < num_sample; i++)
        {
            // sample based on discrete uniform
            std::uniform_int_distribution<int> discrete_dis(0, size - 2);
            int uniform = discrete_dis(gen_discrete);

            // sample based on weight
            std::uniform_real_distribution<> uniform_dis(0.0, 1.0);
            double tar = cumsum[uniform];
            double search_num = uniform_dis(gen) * (csum - tar) + tar;
            double *weight = std::lower_bound (cumsum + uniform, cumsum + size, search_num);

            
            
            // edge weight
            size_t minl = std::min(sample[weight - cumsum].row, sample[uniform].row);
            size_t maxl = std::max(sample[weight - cumsum].row, sample[uniform].row);

            // if(uniform > size - 1 || (weight - cumsum) > size - 1)
            // {

            //     std::cout << "uniform: " << uniform << "\n";
            //     std::cout << "prev: " << cumsum[size - 2] << "\n";
            //     std::cout << "last: " << sample[size - 1].data << "\n";
            //     std::cout << "other: " << cumsum[size - 1] << "\n";
            //     std::cout << "search: " << search_num << "\n";
            //     std::cout << "size: " << size << "\n";
            //     assert(cumsum[size - 1] < search_num);
            //     assert(false);
            // }
            

            double setval = sample[uniform].data * (csum - cumsum[uniform]) / csum * (double)(size - 1) / (double)(num_sample);
            if(minl >= l_bound && minl < r_bound)
            {
                gsl_spmatrix_set(lap.at(minl), maxl, 0, setval);
            }
            else
            {
                sep_edge.push_back(Edge_info(setval, maxl, minl));
            }
            
        }
        
        
    }
    else
    {
        if(nz == 1)
            sum = -data[0];
    }
       
    
    // update column
    gsl_spmatrix_scale(cur, 1.0 / sum);
    cur->data[0] = 1.0;
    
    return sum;
}



double random_sampling1(gsl_spmatrix *cur, std::vector<gsl_spmatrix *> &lap, size_t curcol, std::vector<Edge_info> &sep_edge, size_t l_bound, size_t r_bound)
{

    double *data = cur->data;
    size_t *row = cur->i;
    double sum = 0.0;
    size_t nz = cur->nz;
    // find sum of column
    for (size_t i = 1; i < nz; i++)
    {
        sum += data[i];
    }
    // run only if at least 3 elements, including diagonal

    
    if (nz > 2)
    {
        
        int size = nz - 1;

        // create sample vector for sampling
        Sample sample[size];
        // cumulative sum
        double cumsum[size];
        double csum = 0.0;
        for (size_t i = 0; i < size; i++)
        {
            sample[i].row = row[i + 1];
            sample[i].data = data[i + 1];
            csum += data[i + 1];
            cumsum[i] = csum;
        }
        
        
        // sampling
        // random number and edge values
        static thread_local std::mt19937 gen(std::random_device{}());
        int num_sample = size;
        for (int i = 0; i < num_sample; i++)
        {
            // sample 1 based on weight
            std::uniform_real_distribution<> uniform_dis(0.0, csum);
            double *weight1 = std::lower_bound (cumsum, cumsum + size, uniform_dis(gen));

            // sample 2 based on weight
            double *weight2 = std::lower_bound (cumsum, cumsum + size, uniform_dis(gen));
            
            // edge weight
            size_t minl = std::min(sample[weight1 - cumsum].row, sample[weight2 - cumsum].row);
            size_t maxl = std::max(sample[weight1 - cumsum].row, sample[weight2 - cumsum].row);

            if (minl == maxl)
                continue;

            if(minl >= l_bound && minl < r_bound)
            {
                
                gsl_spmatrix_set(lap.at(minl), maxl, 0, csum / double((2 * num_sample)));
            }
            else
            {
                sep_edge.push_back(Edge_info(csum / double((2 * num_sample)), maxl, minl));
            }
            
        }
        
        
    }
    else
    {
        if(nz == 1)
           sum = -data[0];
    }
      


    // update column
    gsl_spmatrix_scale(cur, 1.0 / sum);
    cur->data[0] = 1.0;
    
    return sum;
}

double random_sampling0(gsl_spmatrix *cur, std::vector<gsl_spmatrix *> &lap, size_t curcol, std::vector<Edge_info> &sep_edge, size_t l_bound, size_t r_bound)
{

    double *data = cur->data;
    size_t *row = cur->i;
    double sum = 0.0;
    size_t nz = cur->nz;
    // find sum of column
    for (size_t i = 1; i < nz; i++)
    {
        sum += data[i];
    }
    // run only if at least 3 elements, including diagonal

    
    if (nz > 2)
    {
        
        int size = nz - 1;

        // create sample vector for sampling
        Sample sample[size];
        // cumulative sum
        double cumsum[size];
        double csum = 0.0;
        for (size_t i = 0; i < size; i++)
        {
            sample[i].row = row[i + 1];
            sample[i].data = data[i + 1];
        }
        // sort first based on value
        std::sort(sample, sample + size, compare);
        for (size_t i = 0; i < size; i++)
        {
            csum += sample[i].data;
            cumsum[i] = csum;
        }
        
        
        
        // sampling
        // random number and edge values
        static thread_local std::mt19937 gen(std::random_device{}());
        
        for (size_t i = 0; i < size - 1; i++)
        {
            std::uniform_real_distribution<> dis(0.0, 1.0);
            double tar = cumsum[i];
            double r = dis(gen) * (csum - tar) + tar;
            double *low = std::lower_bound (cumsum + i, cumsum + size, r);
            
            // edge weight
            size_t minl = std::min(sample[low - cumsum].row, sample[i].row);
            size_t maxl = std::max(sample[low - cumsum].row, sample[i].row);

            // if(curcol >= 0 && curcol < 275)
            // {
            //     if((minl >= 275 && minl < 550))
            //     {
            //         std::cout << "wrong: " << minl << "self: " << curcol << " \n";
            //         assert(0);
            //     }
                
            // }
                

            if(minl >= l_bound && minl < r_bound)
            {
                
                gsl_spmatrix_set(lap.at(minl), maxl, 0, sample[i].data * (csum - cumsum[i]) / csum);
            }
            else
            {
                sep_edge.push_back(Edge_info(sample[i].data * (csum - cumsum[i]) / csum, maxl, minl));
            }
            
        }
        
/*
        if (omp_get_thread_num() == 0)
	                start1 = std::chrono::steady_clock::now();
        if (omp_get_thread_num() == 0)
        {
            end1 = std::chrono::steady_clock::now();
	        elapsed1 = elapsed1 + end1 - start1;
        }
*/
        
    }
    
    // update column
    gsl_spmatrix_scale(cur, 1.0 / sum);
    cur->data[0] = 1.0;
    
    return sum;
}


// uses int for some loops
void linear_update(gsl_spmatrix *b)
{
    // sort
    size_t size = b->nz;
    Sample sample[size];
    for (size_t i = 0; i < size; i++)
    {
        sample[i].row = b->i[i];
        sample[i].data = b->data[i];
    }

    std::sort(sample, sample + size);



    // override original vector
    size_t rowp = sample[0].row;
    size_t addidx = 0;
    set_element(b, 0, sample[0].row, sample[0].data);
    for (size_t i = 1; i < size; i++)
    {
        if(sample[i].row != rowp)
        {
            addidx++;
            rowp = sample[i].row;
            set_element(b, addidx, sample[i].row, sample[i].data);
        }
        else
        {
            b->data[addidx] += sample[i].data;
        }
    }

    // set nonzero and split
    
    b->split = b->nz - b->split; 
    b->nz = addidx + 1;
}

bool compare1 (Edge_info *i, Edge_info *j)
{
    return (i->col < j->col);
}
/* read in matlab array */
void process_array(const mxArray *arg0, std::vector<size_t> &result_idx, size_t depth, size_t target, std::vector<gsl_spmatrix *> &lap, size_t start, size_t total_size, int core_begin, int core_end)
{
    int core_id = (core_begin + core_end) / 2;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_begin, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
    
    // bottom level
    if(target == depth)
    {
        size_t i, j;
        size_t max_element = mxGetNzmax(arg0);
        size_t N = mxGetN(arg0);
        size_t *cpt = mxGetJc(arg0);
        size_t *rpt = mxGetIr(arg0);
        double *datapt = (double *)mxGetData(arg0);
        for(i = result_idx.at(start); i < result_idx.at(start + total_size); i++)
        {
            size_t start = cpt[i];
            size_t last = cpt[i + 1];
            gsl_spmatrix *one_col = gsl_spmatrix_alloc_nzmax(N, 1, 40); // CSC format
            size_t count = 0;
            for (j = start; j < last; j++)
            {
                
                if(rpt[j] == lap.size() - 1 && datapt[j] < 0 && i != lap.size() - 1)
                {
                    //std::cout << "set to 0 at i: " << i << "\n";
                    count++;
                    continue;
                }
                gsl_spmatrix_set(one_col, rpt[j], 0, datapt[j]);
            }
            one_col->split = last - start - count;
            lap.at(i) = one_col;
        }
    }
    else
    {
        /* code */   
        // std::async(std::launch::async, process_array, arg0, std::ref(result_idx), depth + 1, target, std::ref(lap), (total_size - 1) / 2 + start, (total_size - 1) / 2, core_id, core_end); 
        #pragma omp parallel
        #pragma omp single
        #pragma omp task shared(lap)
        {
            process_array(arg0, result_idx, depth + 1, target, lap, (total_size - 1) / 2 + start, (total_size - 1) / 2, core_id, core_end);
        }
        process_array(arg0, result_idx, depth + 1, target, lap, start, (total_size - 1) / 2, core_begin, core_id);

        // synchronize
        #pragma omp taskwait

        // separator
        size_t i, j;
        size_t max_element = mxGetNzmax(arg0);
        size_t N = mxGetN(arg0);
        size_t *cpt = mxGetJc(arg0);
        size_t *rpt = mxGetIr(arg0);
        double *datapt = (double *)mxGetData(arg0);
        for (i = result_idx.at(start + total_size - 1); i < result_idx.at(start + total_size); i++)
        {
            size_t start = cpt[i];
            size_t last = cpt[i + 1];
            gsl_spmatrix *one_col = gsl_spmatrix_alloc_nzmax(N, 1, 90); // CSC format
            size_t count = 0;
            for (j = start; j < last; j++)
            {
                
                if(rpt[j] == lap.size() - 1 && datapt[j] < 0 && i != lap.size() - 1)
                {
                    //std::cout << "set to 0 at i: " << i << "\n";
                    count++;
                    continue;
                }
                    
                gsl_spmatrix_set(one_col, rpt[j], 0, datapt[j]);
            }
            one_col->split = last - start - count;
            lap.at(i) = one_col;
        }

        
    }
    
}




bool compare (Sample i, Sample j)
{
    return (i.data < j.data);
}





