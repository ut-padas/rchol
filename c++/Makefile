EXE = driver driver_parallel
OBJS := $(patsubst %.cpp,%.o,$(wildcard *.cpp rchol/*.cpp rchol_lap/*.cpp util/*.cpp))
OBJS := $(filter-out ex_laplace.o ex_laplace_parallel.o,$(OBJS))

CFLAG = -Wall -O3 -Irchol/ -Irchol_lap/ -Iutil/
LDFLAG = -lmkl_intel_ilp64 -mkl


METIS_INC = $(TACC_METIS_INC)
METIS_LIB = $(TACC_METIS_LIB)

# METIS is required for parallel
ifneq ($(and $(METIS_INC),$(METIS_LIB)),)
CFLAG += -I$(METIS_INC)
LDFLAG += -L$(METIS_LIB) -lmetis
else
POBJ = rchol/find_separator.o rchol/rchol_parallel.o
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
	rm -f driver* *.o rchol/*.o rchol_lap/*.o util/*.o

