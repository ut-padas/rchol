
METIS_INC ?= $(TACC_METIS_INC)
METIS_LIB ?= $(TACC_METIS_LIB)

# METIS is required
ifeq ($(METIS_INC),)
$(error METIS_INC (metis header directory) is not set)
endif
ifeq ($(METIS_LIB),)
$(error METIS_LIB (metis library directory) is not set)
endif


CFLAG = -Wall -O3 -I$(METIS_INC)
LDFLAG = -lmkl_intel_ilp64 -mkl -L$(METIS_LIB) -lmetis

all: driver

objects := $(patsubst %.cpp,%.o,$(wildcard *.cpp))

%.o: %.cpp %.hpp
	icc -c $< -o $@ $(CFLAG)

%.o: %.cpp
	icc -c $< -o $@ $(CFLAG)

driver: $(objects)
	icpc $^ -o $@ $(CFLAG) $(LDFLAG)
	
	
	#icpc test_laplace_matrix.cpp mklsolve.cpp spcol.c -o driver -I/home1/apps/matlab/2019b/extern/include -I/home1/apps/intel19/eigen/3.3.7/include -L/home1/apps/matlab/2019b/bin/glnxa64 -lmx -lmat -lmex -O3 -m64 -msse3 -march=native -fopenmp -L/opt/intel/parallel_studio_xe_2020.1.102/compilers_and_libraries_2020/linux/tbb/lib/intel64/gcc4.8/ -I/opt/intel/parallel_studio_xe_2020.1.102/compilers_and_libraries_2020/linux/tbb/lib/intel64/gcc4.8/ -Wl,-rpath,/opt/intel/parallel_studio_xe_2020.1.102/compilers_and_libraries_2020/linux/tbb/lib/intel64/gcc4.8/ -ltbbmalloc_proxy -lmkl_core -lmkl_intel_ilp64 -lmkl_intel_thread -liomp5 -lpthread -lm -ldl

run: driver
	./driver

clean:
	rm -f driver *.o

