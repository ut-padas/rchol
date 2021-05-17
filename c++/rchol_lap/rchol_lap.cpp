#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <future>
#include <thread>
#include <vector>
#include "rchol_lap.hpp"
#include "spvec.hpp"


//typedef float real;
typedef double real;
typedef spvec<int, real> edges;


struct Sparse_storage_input {
    std::vector<size_t> *colPtr; 
    std::vector<size_t> *rowIdx; 
    std::vector<double> *val;
};


struct Sparse_storage_output {
    size_t *colPtr; 
    size_t *rowIdx; 
    double *val;
    size_t N;
};

/* used for random sampling */
struct Sample {
    int row;
    real data;
    Sample() {}
    Sample(int arg0, real arg1)
    {
        row = arg0;
        data = arg1;
    }
    bool operator<(Sample other) const
    {
        return row < other.row;
    }
};

/* for keeping track of edges in separator and other special use */
struct Edge_info {
    real val;
    int row;
    int col;
    Edge_info(){}
    Edge_info(real arg0, int arg1, int arg2)
    {
        val = arg0;
        row = arg1;
        col = arg2;
    }
    bool operator<(Edge_info other) const
    {
        return col < other.col;
    }
};



/* functions set up */
void process_array(const Sparse_storage_input *input, std::vector<int> &partition, int depth, int target, std::vector<edges> &lap, int start, int total_size, int core_begin, int core_end);

/* clean up memory */
void clear_memory(std::vector<edges> &lap, std::vector<int> &partition, int depth, int target, int start, int total_size, int core_begin, int core_end);

/* functions for factoring Cholesky */
void cholesky_factorization(std::vector<edges> &lap, std::vector<int> &partition, Sparse_storage_output *output);

std::vector<Edge_info> & recursive_calculation(std::vector<int> &partition, int depth, std::vector<edges> &lap, real *diagpt, int start, int total_size, int target, int core_begin, int core_end);

void linear_update(edges *b);

real random_sampling(edges *cur, std::vector<edges> &lap, int curcol, std::vector<Edge_info> &sep_edges, int l_bound, int r_bound);

bool compare (Sample i, Sample j);



void rchol_lap(Sparse_storage_input *input, Sparse_storage_output *output, std::vector<int> &partition, int thread)
{

    //the CPU we whant to use
    int cpu = 0;
    cpu_set_t cpuset; 
    CPU_ZERO(&cpuset);       //clears the cpuset
    CPU_SET(cpu , &cpuset);  //set CPU 2 on cpuset
    /*
    * cpu affinity for the calling thread 
    * first parameter is the pid, 0 = calling thread
    * second parameter is the size of your cpuset
    * third param is the cpuset in which your thread will be
    * placed. Each bit represents a CPU
    */
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
    

    auto start = std::chrono::steady_clock::now();
    std::vector<edges> lap(input->colPtr->size() - 1);
    process_array(input, partition, 0, int(std::log2(thread)), lap, 0, partition.size() - 1, 0, thread);
    auto end = std::chrono::steady_clock::now();
    std::cout << "alloc time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/1000. << " s\n";
    

    start = std::chrono::steady_clock::now();
    cholesky_factorization(lap, partition, output);
    end = std::chrono::steady_clock::now();
    std::cout << "chol time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/1000. << " s\n";


    // clear memory
    clear_memory(lap, partition, 0, int(std::log2(thread)), 0, partition.size() - 1, 0, thread);
}


void rchol_lap(std::vector<size_t> &rowPtrA, std::vector<size_t> &colIdxA, std::vector<double> &valA, size_t* &colPtrG, size_t* &rowIdxG, double* &valG, size_t &sizeG, std::vector<int> &partition) {
  Sparse_storage_input input;
  input.colPtr = &rowPtrA;
  input.rowIdx = &colIdxA;
  input.val = &valA;
  Sparse_storage_output output;
  rchol_lap(&input, &output, partition, partition.size() / 2);
  colPtrG = output.colPtr;
  rowIdxG = output.rowIdx;
  valG = output.val;
  sizeG = output.N;
}


void clear_memory(std::vector<edges> &lap, std::vector<int> &partition, 
    int depth, int target, int start, int total_size, int core_begin, int core_end)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_begin, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
    
    // bottom level
    if(target == depth)
    {
        for(int i = partition.at(start); i < partition.at(start + total_size); i++)
        {
            lap[i].clear();
        }
    }
    else
    {
        /* code */   
        int core_id = (core_begin + core_end) / 2;
        auto future = std::async(std::launch::async, clear_memory, std::ref(lap), 
            std::ref(partition), 
            depth + 1, target, (total_size - 1) / 2 + start, (total_size - 1) / 2, 
            core_id, core_end); 
        
        clear_memory(lap, partition, depth + 1, target, 
            start, (total_size - 1) / 2, core_begin, core_id);

        for (int i = partition.at(start + total_size - 1); i < partition.at(start + total_size); i++)
        {
            lap[i].clear();
        }
    }
    
}


std::vector<Edge_info> & 
recursive_calculation(std::vector<int> &partition, int depth, std::vector<edges> &lap, 
    real *diagpt, int start, int total_size, int target, int core_begin, int core_end)
{

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
		    //double density = 0;
        //int cpu_num = sched_getcpu();
        auto time_s = std::chrono::steady_clock::now();
        auto time_e = std::chrono::steady_clock::now();
        auto elapsed = time_s - time_e;

        std::vector<Edge_info> *pt = new std::vector<Edge_info>();
        std::vector<Edge_info> &sep_edges = *pt;
        
        time_s = std::chrono::steady_clock::now();
        for (int i = partition.at(start); i < partition.at(start + total_size); i++)
        {
            int current = i;
            edges *b = &lap.at(current);
            if(b->nz - b->split > 0)
            {
                linear_update(b);
            }
        
            diagpt[current] = random_sampling(&lap.at(current), lap, current, sep_edges, partition.at(start), partition.at(start + total_size));
        }

        time_e = std::chrono::steady_clock::now();
        elapsed += time_e - time_s;

        //std::cout << "depth: " << depth << " thread " << std::this_thread::get_id() << " cpu: " << cpu_num << " time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "\n";
        //std::cout << "depth: " << depth << " length: " << partition.at(start) << " nztotal " << density << " density: " << density / (partition.at(start + total_size) - partition.at(start)) << "\n";
        //std::cout << "depth: " << depth << "\n";
        //std::cout << omp_proc_bind_master << "  " << omp_get_proc_bind << "\n";
        return sep_edges;
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
        int core_id = (core_begin + core_end) / 2;
        auto a1 = std::async(std::launch::async, recursive_calculation, std::ref(partition), depth + 1, std::ref(lap), diagpt, (total_size - 1) / 2 + start, (total_size - 1) / 2, target, core_id, core_end);
        
        // run its own job
        l_pt = &recursive_calculation(partition, depth + 1, lap, diagpt, start, (total_size - 1) / 2, target, core_begin, core_id);
        
        /* synchronize at this point */
        r_pt = &a1.get();

        std::vector<Edge_info> &l_edges = *l_pt;
        std::vector<Edge_info> &r_edges = *r_pt;
        
        /* process edgess in separator on the current level */
        std::vector<Edge_info> *pt = new std::vector<Edge_info>();
        std::vector<Edge_info> &sep_edges = *pt;
        sep_edges.reserve(l_edges.size()+r_edges.size());

        int l_bound = partition.at(start + total_size - 1);
        int r_bound = partition.at(start + total_size);
        for (int i = 0; i < l_edges.size(); i++)
        {

            Edge_info &temp = l_edges.at(i);
            if(temp.col >= l_bound && temp.col < r_bound)
            {
                lap.at(temp.col).set(temp.row, 0, temp.val);
            }
            else
            {
                sep_edges.push_back(Edge_info(temp.val, temp.row, temp.col));
            }
        }
        
        for (int i = 0; i < r_edges.size(); i++)
        {
            
            Edge_info &temp = r_edges.at(i);
            if(temp.col >= l_bound && temp.col < r_bound)
            {
                lap.at(temp.col).set(temp.row, 0, temp.val);
            }
            else
            {
                sep_edges.push_back(Edge_info(temp.val, temp.row, temp.col));
            }
        }
        delete &l_edges;
        delete &r_edges;

        /* separator portion */
        time_s = std::chrono::steady_clock::now();
        for (int i = partition.at(start + total_size - 1); i < partition.at(start + total_size); i++)
        {
            
            int current = i;
            edges *b = &lap.at(current);
            if(b->nz - b->split > 0)
            {
                
                linear_update(b);
            }

            
            diagpt[current] = random_sampling(&lap.at(current), lap, current, sep_edges, l_bound, r_bound);
            //density += lap.at(i)->nz;
        }
        time_e = std::chrono::steady_clock::now();
        elapsed = time_e - time_s;
        //int cpu_num = sched_getcpu();
        //std::cout << "depth(separator): " << depth << " thread " << std::this_thread::get_id() << " cpu: " << cpu_num  << " time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " length: " << partition.at(start + total_size) - partition.at(start + total_size - 1) << " nztotal " << density << " before: " << before_density / (partition.at(start + total_size) - partition.at(start + total_size - 1)) << " density: " << density / (partition.at(start + total_size) - partition.at(start + total_size - 1)) << "\n";
		
        return sep_edges;
    }
    
}


void coalesce(std::vector<edges> &lap, size_t *cpt, size_t *rpt, double *datapt, real *diagpt)
{
    size_t counter = 0;
    int  m = lap.size() - 1;
    cpt[0] = 0;
    
    
    for (int i = 0; i < m; i++)
    {
        edges *toCopy = &lap.at(i);
        int nz = toCopy->nz;
        int *irow = toCopy->i;
        real *data = toCopy->data;
        
        for (int j = 0; j < nz; j++)
        {
            if (irow[j] == m)
                continue;
            datapt[counter] = data[j];
            rpt[counter] = irow[j];
            if(rpt[counter] != i)
                datapt[counter] = -datapt[counter];
            datapt[counter] *= std::sqrt(diagpt[i]);
            counter++;
        }
        cpt[i + 1] = counter;
    }
}


/* main routine for Cholesky */
void cholesky_factorization(std::vector<edges> &lap, std::vector<int> &partition, Sparse_storage_output *output)
{
    // calculate nonzeros and create lower triangular matrix
    int  m = lap.size();
    real *diagpt = new real[m]();

    auto start = std::chrono::steady_clock::now();
    recursive_calculation(partition, 0, lap, diagpt, 0, partition.size() - 1, 
                          int(std::log2(partition.size()/2)), 0, partition.size()/2);
    auto end = std::chrono::steady_clock::now();
    auto elapsed = end - start;
    std::cout << "factor time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()/1000. << " s\n";

    size_t nzmax = 0;
    //size_t edges = 0;
    for (size_t i = 0; i < lap.size(); i++)
    {
        nzmax += lap.at(i).nz;
	      // edges += lap.at(i)->split;
    }
    //std::cout<< "nzmax: " << nzmax <<"\n";
    //std::cout<< "edges: " << edges <<"\n";
    //std::cout << "size of matrix length: " << m - 1 << "\n";

    // return results back 
    size_t *cpt = new size_t[m]();
    size_t *rpt = new size_t[nzmax]();
    double *datapt = new double[nzmax]();
    
    coalesce(lap, cpt, rpt, datapt, diagpt);
    output->colPtr = cpt;
    output->rowIdx = rpt;
    output->val = datapt;
    output->N = m - 1;
    delete[] diagpt;
}


real random_sampling(edges *cur, std::vector<edges> &lap, int curcol, 
    std::vector<Edge_info> &sep_edges, int l_bound, int r_bound)
{

    real *data = cur->data;
    int *row = cur->i;
    real sum = 0.0;
    int nz = cur->nz;
    // find sum of column
    for (int i = 1; i < nz; i++)
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
        real cumsum[size];
        real csum = 0.0;
        for (int i = 0; i < size; i++)
        {
            sample[i].row = row[i + 1];
            sample[i].data = data[i + 1];
        }
        // sort first based on value
        std::sort(sample, sample + size, compare);
        for (int i = 0; i < size; i++)
        {
            csum += sample[i].data;
            cumsum[i] = csum;
        }
        
        // sampling
        // random number and edges values
        static thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        for (int i = 0; i < size - 1; i++)
        {
            real tar = cumsum[i];
            real r = dis(gen) * (csum - tar) + tar;
            real *low = std::lower_bound (cumsum + i, cumsum + size, r);
            
            // edges weight
            int minl = std::min(sample[low - cumsum].row, sample[i].row);
            int maxl = std::max(sample[low - cumsum].row, sample[i].row);

            if (minl >= l_bound && minl < r_bound)
            {
                lap.at(minl).set(maxl, 0, sample[i].data * (csum - cumsum[i]) / csum);
            }
            else
            {
                sep_edges.push_back(Edge_info(sample[i].data * (csum - cumsum[i]) / csum, maxl, minl));
            }
            
        }
        
        
    }
    
    // update column
    cur->scale(1.0 / sum);
    cur->data[0] = 1.0;
    
    return sum;
}


// uses int for some loops
void linear_update(edges *b)
{
    // sort
    int size = b->nz;
    Sample sample[size];
    for (int i = 0; i < size; i++)
    {
        sample[i].row = b->i[i];
        sample[i].data = b->data[i];
    }

    std::sort(sample, sample + size);

    // override original vector
    int rowp = sample[0].row;
    int addidx = 0;
    b->set_element(0, sample[0].row, sample[0].data);
    for (int i = 1; i < size; i++)
    {
        if(sample[i].row != rowp)
        {
            addidx++;
            rowp = sample[i].row;
            b->set_element(addidx, sample[i].row, sample[i].data);
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


void process_array(const Sparse_storage_input *input, std::vector<int> &partition, 
    int depth, int target, std::vector<edges> &lap, int start, int total_size, 
    int core_begin, int core_end)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_begin, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
    
    // bottom level
    if(target == depth)
    {
        int N = input->colPtr->size() - 1;
        std::vector<size_t> &cpt = *(input->colPtr);
        std::vector<size_t> &rpt = *(input->rowIdx);
        std::vector<double> &datapt = *(input->val);
        for(int i = partition.at(start); i < partition.at(start + total_size); i++)
        {
            size_t start = cpt[i];
            size_t last  = cpt[i + 1];
            int count = 0;
            lap[i].alloc_nzmax(N, 1, 40);
            for (size_t j = start; j < last; j++)
            {
                
                if(rpt[j] == lap.size() - 1 && datapt[j] < 0 && i != lap.size() - 1)
                {
                    //std::cout << "set to 0 at i: " << i << "\n";
                    count++;
                    continue;
                }
                lap[i].set(rpt[j], 0, datapt[j]);
            }
            lap[i].split = last - start - count;
        }
    }
    else
    {
        int core_id = (core_begin + core_end) / 2;
        auto future = std::async(std::launch::async, process_array, input, std::ref(partition), 
            depth + 1, target, std::ref(lap), (total_size - 1) / 2 + start, (total_size - 1) / 2, 
            core_id, core_end); 
        
        process_array(input, partition, depth + 1, target, lap, 
            start, (total_size - 1) / 2, core_begin, core_id);

        // separator
        size_t N = input->colPtr->size() - 1;
        std::vector<size_t> &cpt = *(input->colPtr);
        std::vector<size_t> &rpt = *(input->rowIdx);
        std::vector<double> &datapt = *(input->val);
        for (int i = partition.at(start + total_size - 1); i < partition.at(start + total_size); i++)
        {
            size_t start = cpt[i];
            size_t last = cpt[i + 1];
            int count = 0;
            lap[i].alloc_nzmax(N, 1, 90);
            for (size_t j = start; j < last; j++)
            {
                
                if(rpt[j] == lap.size() - 1 && datapt[j] < 0 && i != lap.size() - 1)
                {
                    //std::cout << "set to 0 at i: " << i << "\n";
                    count++;
                    continue;
                }
                    
                lap[i].set(rpt[j], 0, datapt[j]);
            }
            lap[i].split = last - start - count;
        }
    }
}


bool compare (Sample i, Sample j)
{
    return (i.data < j.data);
}


