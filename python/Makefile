
CC?=g++
PYTHON_INC?=/usr/include/python3.6m


all: python_interface.so

python_interface.so: python_interface.cpp
	g++ $< -I$(PYTHON_INC) -fPIC -shared -O3 -o $@

python_interface.cpp: python_interface.pyx rchol_lap.cpp 
	cython $< --cplus -3


run: python_interface.so
	python3 ex_laplace.py

clean:
	rm -f python_interface.cpp
	rm -f python_interface.so
