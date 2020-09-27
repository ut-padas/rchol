
CC?=g++
PYTHON_INC?=/usr/include/python3.6m


all: pyrchol_lap.so

pyrchol_lap.so: pyrchol_lap.cpp
	g++ $< -I$(PYTHON_INC) -I/home/ty/Documents/metis-5.1.0/include -L/home/ty/Documents/metis-5.1.0/build/Linux-x86_64/libmetis -lmetis -fPIC -shared -O3 -o $@

pyrchol_lap.cpp: pyrchol_lap.pyx rchol_lap.cpp 
	cython $< --cplus -3


run: pyrchol_lap.so
	python3 ex_laplace.py

clean:
	rm -f pyrchol_lap.cpp
	rm -f pyrchol_lap.so
