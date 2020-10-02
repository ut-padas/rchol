EXE = driver driver_parallel
OBJS := $(patsubst %.cpp,%.o,$(wildcard *.cpp))
OBJS := $(filter-out ex_laplace.o ex_laplace_parallel.o,$(OBJS))

CFLAG = -Wall -O3
LDFLAG = -lmkl_intel_ilp64 -mkl


METIS_INC =
METIS_LIB = 

# METIS is required for parallel
ifneq ($(and $(METIS_INC),$(METIS_LIB)),)
CFLAG += -I$(METIS_INC)
LDFLAG += -L$(METIS_LIB) -lmetis
else
POBJ = find_separator.o rchol_parallel.o
OBJS := $(filter-out $(POBJ),$(OBJS))
EXE := $(filter-out driver_parallel,$(EXE))
endif

all: $(EXE)

%.o: %.cpp %.hpp
	icc -c $< -o $@ $(CFLAG)

%.o: %.cpp
	icc -c $< -o $@ $(CFLAG)

driver: ex_laplace.o $(OBJS)
	icpc $^ -o $@ $(CFLAG) $(LDFLAG)
	
driver_parallel: ex_laplace_parallel.o $(OBJS)
	icpc $^ -o $@ $(CFLAG) $(LDFLAG)
	
	
clean:
	rm -f driver* *.o

