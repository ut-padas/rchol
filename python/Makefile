
CC?=g++
PYTHON_INC?=/usr/include/python3.6m


all: pyrchol_lap.so

pyrchol_lap.so: pyrchol_lap.cpp
	g++ $< -I$(PYTHON_INC) -fPIC -shared -O3 -o $@

pyrchol_lap.cpp: pyrchol_lap.pyx rchol_lap.cpp 
	cython $< --cplus -3


run: pyrchol_lap.so
	python3 ex_laplace.py

clean:
	rm -f pyrchol_lap.cpp
	rm -f pyrchol_lap.so
