
PYTHON_INC?=/usr/include/python3.6m


all: pyrchol_lap.so


METIS_INC = $(TACC_METIS_INC)
METIS_LIB = $(TACC_METIS_LIB)

# METIS is required for parallel
ifneq ($(and $(METIS_INC),$(METIS_LIB)),)
pyrchol_lap.so: pyrchol_lap.cpp find_seperator.cpp metis_separator.cpp
	g++ $^ -I$(PYTHON_INC) -I$(METIS_INC) -L$(METIS_LIB) -lmetis -fPIC -shared -O3 -o $@
else
pyrchol_lap.so: pyrchol_lap.cpp
	g++ $< -I$(PYTHON_INC) -fPIC -shared -O3 -o $@
endif


pyrchol_lap.cpp: pyrchol_lap.pyx rchol_lap.cpp 
	cython $< --cplus -3

find_seperator.cpp: find_seperator.pyx
	cython $< --cplus -3

run: pyrchol_lap.so
	python3 ex_laplace.py

clean:
	rm -f pyrchol_lap.cpp pyrchol_lap.so find_separator.cpp
	
