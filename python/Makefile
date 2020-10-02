
PYLIB = pyrchol_lap.so

# need Python.h
PYTHON_INC ?= /usr/include/python3.6m

# METIS 
METIS_INC = $(TACC_METIS_INC)
METIS_LIB = $(TACC_METIS_LIB)


ifeq ($(PYTHON_INC),)
$(error PYTHON_INC (Python header directory) is not set)
endif

# METIS is required for parallel
ifneq ($(and $(METIS_INC),$(METIS_LIB)),)
PYLIB += find_separator.so
endif


all: $(PYLIB)

DIR = rchol_lap

pyrchol_lap.so: pyrchol_lap.cpp
	g++ $< -I$(PYTHON_INC) -I$(DIR) -fPIC -shared -O3 -o $@

find_separator.so: find_separator.cpp
	g++ $< -I$(PYTHON_INC) -I$(DIR) -I$(METIS_INC) -L$(METIS_LIB) -lmetis -fPIC -shared -O3 -o $@

pyrchol_lap.cpp: $(DIR)/pyrchol_lap.pyx
	cython $< --cplus -3 -o $@

find_separator.cpp: $(DIR)/find_separator.pyx
	cython $< --cplus -3 -o $@

run: pyrchol_lap.so
	python3 ex_laplace.py

clean:
	rm -f pyrchol_lap.cpp *.so find_separator.cpp
	
