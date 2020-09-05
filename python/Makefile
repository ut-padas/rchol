
CC?=g++
PYTHON_INC?=/usr/include/python2.7

ifeq ($(findstring frontera,$(shell uname -n)),frontera)
	PYTHON_INC=/usr/include/python3.6m
endif


all: python_interface.so


python_interface.so: python_interface.cpp
	g++ $< -I$(PYTHON_INC) -fPIC -shared -O3 -o $@

python_interface.cpp: python_interface.pyx python_interface.pxd rchol_lap.cpp 
	cython $< --cplus


run:
	python3 ex_laplace.py

clean:
	rm -f python_interface.cpp
	rm -f python_interface.so
