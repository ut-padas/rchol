Matlab interface:

rchol inputs:

The rchol function takes two inputs. The first input is the sparse SDDM matrix, which is a mandatory input. If the second input is not given, then the factorization will be sequencial. In this case, if users want to use a particular permuation, then they should permute the first input before passing it into the function.

The second input (optional) is for multithreading purpose; it specifies the number of threads to be used during the execution of the parallel factorization. The second input should be strictly greater than 0 and a power of 2.

In the special case that the input thread number is 1, the method will be equivalent to the sequential method. In other words, the function behaves as if the second input is nonexistent at all. 


rchol outputs:

rchol returns two outputs. The first output is the Cholesky factor, the second output is the permutation used within rchol. If the thread number is not supplied or is equal to 1, then the returned permuation will simply be a vector from 1 to n, where n is the length of the matrix. However, if thread number is anything other than 1, meaning that parallelization is used, then the function will permute the given SDDM matrix first before factorization. This is necessary because in order to multi-thread the method, we would need to do graph partitioning. The permutation used internally within the function will be returned. Hence, when using pcg, the users will need to supply it with a permuted system, in which the permutation used should be the one returned by rchol.
